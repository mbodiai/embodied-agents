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

from mbodied.agents.backends.openai_backend import OpenAIBackendMixin
from mbodied.agents.backends.serializer import Serializer
from mbodied.types.message import Message
from mbodied.types.sense.vision import Image

import httpx
import backoff

from typing import Any, List, Optional

class OllamaSerializer(Serializer):
    """Serializer for Ollama-specific data formats."""

    @classmethod
    def serialize_image(cls, image: 'Image') -> str:
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
        texts = [txt for txt in message.content if isinstance(txt, str)]
        return {
            "role": message.role,
            "content": texts,
            "images": images,
        }
        
        

class OllamaBackend(OpenAIBackendMixin):
    """Backend for interacting with Ollama's API."""

    INITIAL_CONTEXT = [
        Message(role="system", content="You are a robot with advanced spatial reasoning."),
    ]
    DEFAULT_MODEL = "llava"  # Changed to "llava" as it's mentioned in the example

    def __init__(self, model_src: str = "http://localhost:11434", **kwargs):
        """Initializes the OllamaBackend.

        Args:
            base_url: The base URL for the Ollama API.
            **kwargs: Additional keyword arguments.
        """
        self.base_url = model_src
        self.client = httpx.Client(base_url=self.base_url)
        self.serializer = OllamaSerializer

    def _create_completion(self, messages: List[Message], model: str = DEFAULT_MODEL, stream: bool = False, **kwargs) -> str:
        """Creates a completion for the given messages using the Ollama API.

        Args:
            messages: A list of messages to be sent to the completion API.
            model: The model to be used for the completion.
            stream: Whether to stream the response. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The content of the completion response.
        """
        serialized_messages = [self.serialized(msg) for msg in messages]
        
        data = {
            "model": model,
            "messages": serialized_messages,
            "stream": stream,
            **kwargs
        }

        response = self.client.post("/api/chat", json=data)
        response.raise_for_status()
        
        if stream:
            return self._handle_stream(response)
        else:
            return response.json()["message"]["content"]

    def _handle_stream(self, response):
        """Handles streaming responses from Ollama API."""
        for line in response.iter_lines():
            if line:
                yield line.decode()
