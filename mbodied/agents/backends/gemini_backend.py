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

from google import genai
from google.genai import types

from mbodied.agents.backends.openai_backend import OpenAIBackendMixin
from mbodied.agents.backends.serializer import Serializer
from mbodied.types.message import Message
from mbodied.types.sense.vision import Image


class GeminiSerializer(Serializer):
    """Serializer for Gemini-specific data formats."""

    @classmethod
    def serialize_image(cls, image: Image) -> Any:
        """Serializes an image to the Gemini format.

        Args:
            image: The image to be serialized.

        Returns:
            A serialized image for the Gemini API.
        """
        # For Gemini, we can pass the PIL image object directly
        return image.pil

    @classmethod
    def serialize_text(cls, text: str) -> str:
        """Serializes a text string to the Gemini format.

        Args:
            text: The text to be serialized.

        Returns:
            The text string (Gemini accepts raw text).
        """
        return text

    def serialize(self) -> Any:
        """Serializes a message for the Gemini API.

        Returns:
            A serialized message for the Gemini API.
        """
        if isinstance(self.wrapped, Message):
            # Gemini expects raw content in a list
            content = []

            for item in self.wrapped.content:
                if isinstance(item, str):
                    content.append(self.serialize_text(item))
                elif isinstance(item, Image):
                    content.append(self.serialize_image(item))
                else:
                    raise ValueError(f"Unsupported content type: {type(item)}")

            # If we only have one content item, return it directly
            return content[0] if len(content) == 1 else content

        # For other types, use default serialization
        return super().serialize()


class GeminiBackend(OpenAIBackendMixin):
    """Backend for interacting with Google's Gemini API.

    Attributes:
        api_key: The API key for the Gemini service.
        client: The client for the Gemini service.
        serialized: The serializer for Gemini-specific data formats.
        model: The default model to use.
    """

    DEFAULT_MODEL = "gemini-2.0-flash"
    INITIAL_CONTEXT = [
        Message(role="system", content="You are a robot with advanced spatial reasoning."),
    ]

    def __init__(
        self,
        api_key: str | None = None,
        client: Any | None = None,
        **kwargs,
    ):
        """Initializes the GeminiBackend with the given API key and client.

        Args:
            api_key: The API key for the Gemini service.
            client: An optional client for the Gemini service.
            **kwargs: Additional keyword arguments.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("MBODI_API_KEY")
        self.client = client

        self.model = kwargs.pop("model", self.DEFAULT_MODEL)
        if self.client is None:
            self.client = genai.Client(api_key=self.api_key)

        self.serialized = GeminiSerializer

    def predict(
        self, message: Message, context: List[Message] | None = None, model: str | None = None, **kwargs
    ) -> str:
        """Create a completion based on the given message and context.

        Args:
            message (Message): The message to process.
            context (Optional[List[Message]]): The context of messages.
            model (Optional[str]): The model used for processing the messages.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The result of the completion.
        """
        context = context or self.INITIAL_CONTEXT
        model = model or self.model or self.DEFAULT_MODEL

        # For system instructions, extract from context if available
        system_instruction = None
        messages_to_process = context + [message]

        # Check if first message is a system message
        if messages_to_process and messages_to_process[0].role == "system":
            system_instruction = self.serialized(messages_to_process[0]).serialize()
            messages_to_process = messages_to_process[1:]

        # Prepare the content from the message
        contents = []
        for msg in messages_to_process:
            contents.append(self.serialized(msg).serialize())

        # Create config with system instruction if available
        config = None
        if system_instruction:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=kwargs.pop("temperature", 0),
                max_output_tokens=kwargs.pop("max_tokens", 1000),
                **kwargs,
            )
        else:
            config = types.GenerateContentConfig(
                temperature=kwargs.pop("temperature", 0), max_output_tokens=kwargs.pop("max_tokens", 1000), **kwargs
            )

        # Make the API call
        response = self.client.models.generate_content(model=model, contents=contents, config=config)

        return response.text

    def stream(self, message: Message, context: List[Message] = None, model: str = None, **kwargs):
        """Streams a completion for the given messages using the Gemini API.

        Args:
            message: Message to be sent to the completion API.
            context: The context of the messages.
            model: The model to be used for the completion.
            **kwargs: Additional keyword arguments.
        """
        context = context or self.INITIAL_CONTEXT
        model = model or self.model or self.DEFAULT_MODEL

        # For system instructions, extract from context if available
        system_instruction = None
        messages_to_process = context + [message]

        # Check if first message is a system message
        if messages_to_process and messages_to_process[0].role == "system":
            system_instruction = self.serialized(messages_to_process[0]).serialize()
            messages_to_process = messages_to_process[1:]

        # Prepare the content from the message
        contents = []
        for msg in messages_to_process:
            contents.append(self.serialized(msg).serialize())

        # Create config with system instruction if available
        config = None
        if system_instruction:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=kwargs.pop("temperature", 0),
                max_output_tokens=kwargs.pop("max_tokens", 1000),
                **kwargs,
            )
        else:
            config = types.GenerateContentConfig(
                temperature=kwargs.pop("temperature", 0), max_output_tokens=kwargs.pop("max_tokens", 1000), **kwargs
            )

        # Make the streaming API call
        stream = self.client.models.generate_content_stream(model=model, contents=contents, config=config)

        for chunk in stream:
            yield chunk.text or ""


if __name__ == "__main__":
    backend = GeminiBackend(context=["You are a robot with advanced spatial reasoning."])
    message = Message(role="user", content=[Image("resources/bridge_example.jpeg"), "What do you see in this image?"])
    response = backend.predict(message)
    print(response)
