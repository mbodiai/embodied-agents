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


import os
from typing import Any, List

import backoff
import httpx
from anthropic import RateLimitError as AnthropicRateLimitError
from openai._exceptions import RateLimitError as OpenAIRateLimitError

from mbodied.agents.backends.backend import Backend
from mbodied.agents.backends.serializer import Serializer
from mbodied.types.message import Message
from mbodied.types.sense.vision import Image
from mbodied.types.tool import Tool, ToolCall

ERRORS = (
    OpenAIRateLimitError,
    AnthropicRateLimitError,
    httpx.HTTPError,
    ConnectionError,
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


class OpenAIBackendMixin(Backend):
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

    def __init__(
        self,
        api_key: str | None = None,
        client: Any | None = None,
        response_format: str = None,
        aclient=False,
        **kwargs,
    ):
        """Initializes the OpenAIBackend with the given API key and client.

        Args:
            api_key: The API key for the OpenAI service.
            client: An optional client for the OpenAI service.
            response_format: The format for the response.
            aclient: Whether to use the asynchronous client.
            **kwargs: Additional keyword arguments.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = client
        if self.client is None:
            from openai import AsyncOpenAI, OpenAI

            kwargs.pop("model_src", None)
            self.client = OpenAI(api_key=self.api_key or "any_key", **kwargs)
            if aclient:
                self.aclient = AsyncOpenAI(api_key=self.api_key or "any_key", **kwargs)

        self.serialized = OpenAISerializer
        self.response_format = response_format

    @backoff.on_exception(
        backoff.expo,
        ERRORS,
        max_tries=3,
        on_backoff=lambda details: print(f"Backing off {details['wait']:.1f} seconds after {details['tries']} tries."),  # noqa
    )
    def predict(
        self,
        message: Message,
        context: List[Message] | None = None,
        model: Any | None = None,
        tools: List[Tool] | None = None,
        **kwargs,
    ) -> str | tuple[str, List[ToolCall]]:
        """Create a completion based on the given message and context.

        Args:
            message (Message): The message to process.
            context (Optional[List[Message]]): The context of messages.
            model (Optional[Any]): The model used for processing the messages.
            tools (Optional[List[Tool]]): The tools to make available for function calling.
            **kwargs: Additional keyword arguments.

        Returns:
            str | tuple[str, List[ToolCall]]:
                When tools are not provided: Just the text response
                When tools are provided: A tuple of (text_response, tool_calls)
        """
        context = context or self.INITIAL_CONTEXT
        model = model or self.DEFAULT_MODEL
        serialized_messages = [self.serialized(msg).serialize() for msg in context + [message]]

        completion = self.client.chat.completions.create(
            model=model,
            messages=serialized_messages,
            temperature=0,
            max_tokens=1000,
            tools=tools,
            **kwargs,
        )
        if tools:
            tool_calls = []
            if completion.choices[0].message.tool_calls:
                for tool_call in completion.choices[0].message.tool_calls:
                    tool_calls.append(ToolCall.model_validate(tool_call))
            return completion.choices[0].message.content, tool_calls

        return completion.choices[0].message.content

    def stream(
        self, message: Message, context: List[Message] = None, model: str = "gpt-4o", tools: List[Tool] = None, **kwargs
    ):
        """Streams a completion for the given messages using the OpenAI API standard.

        Args:
            message: Message to be sent to the completion API.
            context: The context of the messages.
            model: The model to be used for the completion.
            tools: Optional list of tools (function calls) available to the model.
            **kwargs: Additional keyword arguments.

        Yields:
            When tools is None:
                str: Content delta chunks
            When tools is provided:
                tuple[str, Any]: Tuples of (content_delta, tool_call_delta) where either may be None
        """
        model = model or self.DEFAULT_MODEL
        context = context or self.INITIAL_CONTEXT
        serialized_messages = [self.serialized(msg).serialize() for msg in context + [message]]
        stream = self.client.chat.completions.create(
            messages=serialized_messages,
            model=model,
            temperature=0,
            stream=True,
            tools=tools,
            **kwargs,
        )

        if not tools:
            for chunk in stream:
                yield chunk.choices[0].delta.content or ""
        else:
            for chunk in stream:
                delta = chunk.choices[0].delta
                content = delta.content or ""
                tool_calls = delta.tool_calls
                yield content, tool_calls

    async def astream(
        self, message: Message, context: List[Message] = None, model: str = "gpt-4o", tools: List[Tool] = None, **kwargs
    ):
        """Streams a completion asynchronously for the given messages using the OpenAI API standard.

        Args:
            message: Message to be sent to the completion API.
            context: The context of the messages.
            model: The model to be used for the completion.
            tools: Optional list of tools (function calls) available to the model.
            **kwargs: Additional keyword arguments.

        Yields:
            When tools is None:
                str: Content delta chunks
            When tools is provided:
                tuple[str, Any]: Tuples of (content_delta, tool_call_delta) where either may be None
        """
        if not hasattr(self, "aclient"):
            raise AttributeError("AsyncOpenAI client not initialized. Pass in aclient=True to the constructor.")
        model = model or self.DEFAULT_MODEL
        context = context or self.INITIAL_CONTEXT
        serialized_messages = [self.serialized(msg).serialize() for msg in context + [message]]
        stream = await self.aclient.chat.completions.create(
            messages=serialized_messages,
            model=model,
            temperature=0,
            stream=True,
            tools=tools,
            **kwargs,
        )

        if not tools:
            async for chunk in stream:
                yield chunk.choices[0].delta.content or ""
        else:
            async for chunk in stream:
                delta = chunk.choices[0].delta
                content = delta.content or ""
                tool_calls = delta.tool_calls
                yield content, tool_calls
