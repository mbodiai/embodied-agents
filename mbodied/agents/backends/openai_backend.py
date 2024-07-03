# Copyright 2024 mbodi ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, List, Optional

import backoff
from anthropic import RateLimitError as AnthropicRateLimitError
from openai._exceptions import RateLimitError as OpenAIRateLimitError

from mbodied.agents.backends.serializer import Serializer
from mbodied.types.message import Message
from mbodied.types.sense.vision import Image

ERRORS = (
    OpenAIRateLimitError,
    AnthropicRateLimitError,
)


class OpenAISerializer(Serializer):
    """Serializer for OpenAI-specific data formats."""

    @classmethod
    def serialize_image(cls, image: Image) -> dict[str, Any]:
        """Serializes an image to the OpenAI format.

        Args:
            image: The image to be serialized.

        Returns:
            A dictionary representing the serialized image.
        """
        return {
            "type": "image_url",
            "image_url": {
                "url": image.url,
            },
        }

    @classmethod
    def serialize_text(cls, text: str) -> dict[str, Any]:
        """Serializes a text string to the OpenAI format.

        Args:
            text: The text to be serialized.

        Returns:
            A dictionary representing the serialized text.
        """
        return {"type": "text", "text": text}


class OpenAIBackendMixin:
    """Backend for interacting with OpenAI's API.

    Attributes:
        api_key: The API key for the OpenAI service.
        client: The client for the OpenAI service.
        serialized: The serializer for the OpenAI backend.
        response_format: The format for the response.
    """

    INITIAL_CONTEXT = [
        Message(role="system", content="You are a robot with advanced spatial reasoning."),
    ]
    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, api_key: str | None = None, client: Any | None = None, response_format: str = None, **kwargs):
        """Initializes the OpenAIBackend with the given API key and client.

        Args:
            api_key: The API key for the OpenAI service.
            client: An optional client for the OpenAI service.
            response_format: The format for the response.
            **kwargs: Additional keyword arguments.
        """
        self.api_key = api_key
        self.client = client
        if self.client is None:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, **kwargs)
        self.serialized = OpenAISerializer
        self.response_format = response_format

    def _create_completion(self, messages: List[Message], model: str = "gpt-4o", stream: bool = False, **kwargs) -> str:
        """Creates a completion for the given messages using the OpenAI API standard.

        Args:
            messages: A list of messages to be sent to the completion API.
            model: The model to be used for the completion.
            stream: Whether to stream the response. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The content of the completion response.
        """
        serialized_messages = [self.serialized(msg) for msg in messages]

        completion = self.client.chat.completions.create(
            model=model,
            messages=serialized_messages,
            temperature=0,
            max_tokens=1000,
            stream=stream,
            response_format=self.response_format,
            **kwargs,
        )
        return completion.choices[0].message.content

    @backoff.on_exception(
        backoff.expo,
        ERRORS,
        max_tries=3,
    )
    def act(self, message: Message, context: List[Message] | None = None, model: Optional[Any] = None, **kwargs) -> str:
        """Create a completion based on the given message and context.

        Args:
            message (Message): The message to process.
            context (Optional[List[Message]]): The context of messages.
            model (Optional[Any]): The model used for processing the messages.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The result of the completion.
        """
        if context is None:
            context = []

        if model is not None:
            kwargs["model"] = model
        return self._create_completion(context + [message], **kwargs)
