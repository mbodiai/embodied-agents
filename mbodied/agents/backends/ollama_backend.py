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

from typing import Any

from mbodied.agents.backends.httpx_backend import HttpxBackend
from mbodied.agents.backends.serializer import Serializer
from mbodied.types.message import Message
from mbodied.types.sense.vision import Image


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
        return response["message"]["content"]


class OllamaBackend(HttpxBackend):
    """Backend for interacting with Ollama's API."""

    INITIAL_CONTEXT = [
        Message(role="system", content="You are a robot with advanced spatial reasoning."),
    ]
    DEFAULT_MODEL = "llava"
    SERIALIZER = OllamaSerializer
    DEFAULT_SRC = "http://localhost:11434/api/chat/"

    def __init__(self, api_key: str | None = None, model_src: str = None):
        """Initializes an OllamaBackend instance."""
        model_sr = model_src or self.DEFAULT_SRC
        super().__init__(api_key, model_src=model_sr)

