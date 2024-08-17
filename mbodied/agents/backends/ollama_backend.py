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

import json
from typing import Any

from mbodied.agents.backends.httpx_backend import HttpxBackend
from mbodied.agents.backends.serializer import Serializer
from mbodied.types.message import Message
from mbodied.types.sense.image import Image


class OllamaSerializer(Serializer):
    """Serializer for Ollama-specific data formats."""

    @classmethod
    def serialize_image(cls, image: Image) -> str:
        """Serializes an image to the Ollama format."""
        return image.base64

    @classmethod
    def serialize_text(cls, text: str) -> str:
        """Serializes a text string to the Ollama format."""
        return text

    @classmethod
    def serialize_msg(cls, message: Message) -> dict[str, Any]:
        """Serializes a message to the Ollama format."""
        images = [cls.serialize_image(im) for im in message.content if isinstance(im, Image)]
        texts = ".".join([txt for txt in message.content if isinstance(txt, str)])
        return {
            "role": message.role,
            "content": texts,
            "images": images,
        }

    @classmethod
    def extract_response(cls, response: dict[str, Any]) -> str:
        """Extracts the response from the Ollama format."""
        if isinstance(response, str):
            return json.loads(response)["message"]["content"]
        return response["message"]["content"]

    @classmethod
    def extract_stream(cls, response):
        try:
            parsed = json.loads(response)
            return cls.extract_response(parsed)
        except json.JSONDecodeError:
            # If it's not valid JSON, return the raw response
            return response


class OllamaBackend(HttpxBackend):
    """Backend for interacting with Ollama's API."""

    INITIAL_CONTEXT = [
        Message(role="system", content="You are a robot with advanced spatial reasoning."),
    ]
    DEFAULT_MODEL = "llava"
    SERIALIZER = OllamaSerializer
    DEFAULT_SRC = "http://localhost:11434/api/chat/"

    def __init__(self, api_key: str | None = None, endpoint: str = None):
        """Initializes an OllamaBackend instance."""
        endpoint = endpoint or self.DEFAULT_SRC
        super().__init__(api_key, endpoint=endpoint)


if __name__ == "__main__":
    # Usage
    import asyncio

    client = OllamaBackend()
    image_url = "https://v0.docs.reka.ai/_images/000000245576.jpg"
    text = "What animal is this? Answer briefly."
    response = ""
    # for chunk in client._stream_completion([Message(role="user", content=[text, Image(url=image_url)])], "llava"):
    #     if chunk.strip():
    #         response += chunk
    #         print(f"Response: {response}")
    # run = client._astream_completion([Message(role="user", content=[text, Image(url=image_url)])], "llama3")

    # async def runner():
    #     async for response in run:
    #         print(response)
    # asyncio.run(runner())
