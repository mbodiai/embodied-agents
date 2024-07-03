import httpx
import backoff
from typing import Any, List, Optional

from mbodied.agents.backends.openai_backend import OpenAIBackendMixin
from mbodied.agents.backends.serializer import Serializer
from mbodied.types.message import Message
from mbodied.types.sense.vision import Image

import httpx
import backoff

from typing import Any, List, Optional

class RekaSerializer(Serializer):
    """Serializer for Reka-specific data formats."""

    @classmethod
    def serialize_image(cls, image: Image) -> dict[str, Any]:
        """Serializes an image to the Reka format."""
        return {
            "type": "image_url",
            "image_url": image.url
        }


class RekaBackend(OpenAIBackendMixin):
    """Backend for interacting with Reka's API."""

    INITIAL_CONTEXT = [
        Message(role="system", content="You are a helpful AI assistant."),
    ]
    DEFAULT_MODEL = "reka-core"

    def __init__(self, api_key: str, model_src: str = "https://api.reka.ai/v1", **kwargs):
        """Initializes the RekaBackend."""
        self.base_url = model_src
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={"X-Api-Key": api_key, "Content-Type": "application/json"}
        )
        self.serialized = RekaSerializer

    def _create_completion(self, messages: List[Message], model: str = DEFAULT_MODEL, stream: bool = False, use_search_engine: bool =False,**kwargs) -> str:
        """Creates a completion for the given messages using the Reka API."""
        serialized_messages = [self.serialized(msg) for msg in messages]
        
        data = {
            "messages": serialized_messages,
            "model": model,
            "stream": stream,
            "use_search_engine": use_search_engine,
            
            **kwargs
        }

        response = self.client.post("/chat", json=data)
        response.raise_for_status()
        
        if stream:
            return self._handle_stream(response)
        else:
            return response.json()["choices"][0]["message"]["content"]

    def _handle_stream(self, response):
        """Handles streaming responses from Reka API."""
        for line in response.iter_lines():
            if line:
                yield line.decode()

