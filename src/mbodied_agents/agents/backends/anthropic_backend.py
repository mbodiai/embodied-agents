import os
from typing import Any

import anthropic

from mbodied_agents.base.backend import Backend
from mbodied_agents.base.serializer import Serializer
from mbodied_agents.types.message import Message
from mbodied_agents.types.vision import Image


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


class AnthropicBackend(Backend):
    """Backend for interacting with Anthropic's API."""

    DEFAULT_MODEL = "claude-3-opus-20240229"
    INITIAL_CONTEXT = [
        Message(
            role="user", content="Imagine you are a robot with advanced spatial reasoning."),
        Message(role="assistant", content="Got it!"),
    ]

    def __init__(self, api_key: str | None, client: anthropic.Anthropic | None = None):
        """Initializes the AnthropicBackend with the given API key and client.

        Args:
            api_key: The API key for the Anthropic service.
            client: An optional client for the Anthropic service.
        """
        super().__init__(api_key or os.getenv("ANTHROPIC_API_KEY"), client=client)
        if self.client is None:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        self.serialialized = AnthropicSerializer

    def _create_completion(self, messages: list[Message], model: str = "claude-3-opus-20240229", **kwargs) -> str:
        """Creates a completion for the given messages using the Anthropic API.

        Args:
            messages: A list of messages to be sent to the completion API.
            model: The model to be used for the completion.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The content of the completion response.
        """
        if model is None:
            model = self.DEFAULT_MODEL
        completion = self.client.messages.create(
            model=model, max_tokens=1024, messages=self.serialialized(messages), **kwargs,
        )
        return completion.content[0].text
