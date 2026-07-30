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

"""MiniMax backend for the language agent.

MiniMax exposes OpenAI-compatible and Anthropic-compatible chat completion
endpoints. This backend reuses the OpenAI-compatible request/response flow from
``OpenAIBackendMixin`` and only customizes the default model, the regional base
URLs and the multimodal image serialization. Text and image inputs are supported
so multimodal messages can be sent to MiniMax models that accept them.
"""

import os
from typing import Any

from mbodied.agents.backends.openai_backend import OpenAIBackendMixin
from mbodied.agents.backends.serializer import Serializer
from mbodied.types.message import Message
from mbodied.types.sense.vision import Image

# Regional OpenAI-compatible base URLs for the MiniMax API.
# ``api.minimax.io`` serves the global endpoint and ``api.minimaxi.com`` the
# China endpoint; both are OpenAI-compatible chat completion surfaces.
MINIMAX_OPENAI_BASE_URLS = {
    "global": "https://api.minimax.io/v1",
    "cn": "https://api.minimaxi.com/v1",
}
DEFAULT_MINIMAX_OPENAI_BASE_URL = MINIMAX_OPENAI_BASE_URLS["global"]


class MiniMaxSerializer(Serializer):
    """Serializer for MiniMax OpenAI-compatible data formats."""

    @classmethod
    def serialize_image(cls, image: Image) -> dict[str, Any]:
        """Serializes an image to the MiniMax OpenAI-compatible format.

        MiniMax accepts either a publicly reachable image URL or an inline base64
        data URL. Prefer the URL when available and fall back to a base64 data URL
        so locally loaded images are still supported.

        Args:
            image: The image to be serialized.

        Returns:
            A dictionary representing the serialized image.
        """
        url = image.url if image.url is not None else f"data:image/{image.encoding};base64," + image.base64
        return {
            "type": "image_url",
            "image_url": {
                "url": url,
            },
        }

    @classmethod
    def serialize_text(cls, text: str) -> dict[str, Any]:
        """Serializes a text string to the MiniMax format.

        Args:
            text: The text to be serialized.

        Returns:
            A dictionary representing the serialized text.
        """
        return {"type": "text", "text": text}


class MiniMaxBackend(OpenAIBackendMixin):
    """Backend for interacting with the MiniMax API.

    MiniMax provides OpenAI-compatible chat completion endpoints for both the
    global (``api.minimax.io``) and China (``api.minimaxi.com``) regions. This
    backend points the OpenAI client at the requested regional base URL and uses
    the MiniMax serializer so multimodal (text and image) messages are supported.

    Attributes:
        api_key: The API key for the MiniMax service.
        client: The OpenAI-compatible client for the MiniMax service.
        serialized: The serializer for MiniMax-specific data formats.
    """

    DEFAULT_MODEL = "MiniMax-M3"
    INITIAL_CONTEXT = [
        Message(role="system", content="You are a robot with advanced spatial reasoning."),
    ]

    def __init__(
        self,
        api_key: str | None = None,
        client: Any | None = None,
        region: str = "global",
        base_url: str | None = None,
        **kwargs,
    ):
        """Initializes the MiniMaxBackend with the given API key and client.

        Args:
            api_key: The API key for the MiniMax service. Defaults to the
                ``MINIMAX_API_KEY`` environment variable.
            client: An optional pre-configured OpenAI-compatible client.
            region: The regional endpoint to use, either ``"global"`` or ``"cn"``.
                Ignored when ``base_url`` is provided.
            base_url: An explicit OpenAI-compatible base URL. Takes precedence over
                ``region``.
            **kwargs: Additional keyword arguments forwarded to the OpenAI client.
        """
        self.api_key = api_key or os.getenv("MINIMAX_API_KEY")
        self.client = client
        self.region = region
        self.base_url = base_url or MINIMAX_OPENAI_BASE_URLS.get(region, DEFAULT_MINIMAX_OPENAI_BASE_URL)

        if self.client is None:
            from openai import AsyncOpenAI, OpenAI

            kwargs.pop("model_src", None)
            self.client = OpenAI(api_key=self.api_key or "any_key", base_url=self.base_url, **kwargs)
            use_async = kwargs.pop("aclient", False)
            if use_async:
                self.aclient = AsyncOpenAI(api_key=self.api_key or "any_key", base_url=self.base_url, **kwargs)

        self.serialized = MiniMaxSerializer
