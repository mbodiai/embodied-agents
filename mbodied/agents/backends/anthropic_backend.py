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

from typing import Any, List

import anthropic

from mbodied.agents.backends.openai_backend import OpenAIBackendMixin
from mbodied.agents.backends.serializer import Serializer
from mbodied.types.message import Message
from mbodied.types.sense.vision import Image


class AnthropicSerializer(Serializer):
    """Serializer for Anthropic-specific data formats."""

    @classmethod
    def serialize_image(cls, image: Image) -> dict[str, Any]:
        """Serializes an image to the Anthropic format.

        Args:
            image: The image to be serialized.

        Returns:
            A dictionary representing the serialized image.
        """
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": f"image/{image.encoding}",
                "data": image.base64,
            },
        }

    @classmethod
    def serialize_text(cls, text: str) -> dict[str, Any]:
        """Serializes a text string to the Anthropic format.

        Args:
            text: The text to be serialized.

        Returns:
            A dictionary representing the serialized text.
        """
        return {"type": "text", "text": text}


class AnthropicBackend(OpenAIBackendMixin):
    """Backend for interacting with Anthropic's API.

    Attributes:
        api_key: The API key for the Anthropic service.
        client: The client for the Anthropic service.
        serialized: The serializer for Anthropic-specific data formats.
    """
    serializer = AnthropicSerializer()
    DEFAULT_MODEL = "claude-3-5-sonnet-20240620"
    INITIAL_CONTEXT = [
        Message(role="user", content="Imagine you are a robot with advanced spatial reasoning."),
        Message(role="assistant", content="Got it!"),
    ]

    def __init__(self, api_key: str | None, client: anthropic.Anthropic | None = None, **kwargs):
        """Initializes the AnthropicBackend with the given API key and client.

        Args:
            api_key: The API key for the Anthropic service.
            client: An optional client for the Anthropic service.
            kwargs: Additional keyword arguments.
        """
        self.api_key = api_key
        self.client = client

        self.model = kwargs.pop("model", self.DEFAULT_MODEL)
        if self.client is None:
            self.client = anthropic.Anthropic(api_key=self.api_key)


    def predict(
        self,
        message: Message,
        context: List[Message] | None = None,
        model: str | None = "claude-3-5-sonnet-20240620",
        **kwargs,
    ) -> str:
        """Creates a completion for the given messages using the Anthropic API.

        Args:
            message: the message to be sent to the completion API.
            context: The context for the completion.
            model: The model to be used for the completion.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The content of the completion response.
        """
        if model is None:
            model = self.DEFAULT_MODEL
        serialized_messages = [self.serialized(msg) for msg in context + [message]]
        completion = self.client.messages.create(
            model=model,
            messages=serialized_messages,
            max_tokens=1024,
            **kwargs,
        )
        return completion.content[0].text

    async def async_predict(
        self, message: Message, context: List[Message] | None = None, model: Any | None = None
    ) -> str:
        """Asynchronously predict the next message in the conversation."""
        # For now, we'll use the synchronous method since Anthropic doesn't provide an async API
        return self.predict(message, context, model)

    def stream(
        self, message: Message, context: List[Message] = None, model: str = "claude-3-5-sonnet-20240620", **kwargs
    ):
        """Streams a completion for the given messages using the Anthropic API standard.

        Args:
            message: Message to be sent to the completion API.
            context: The context of the messages.
            model: The model to be used for the completion.
            **kwargs: Additional keyword arguments.
        """
        model = model or self.DEFAULT_MODEL
        serialized_messages = [self.serialized(msg) for msg in context + [message]]
        with self.client.messages.stream(
            max_tokens=1024,
            messages=serialized_messages,
            model=model,
            **kwargs,
        ) as stream:
            for text in stream.text_stream:
                yield text or ""
