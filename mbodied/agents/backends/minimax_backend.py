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
endpoints. This backend selects the requested API format and regional base URL
while keeping text and image message serialization compatible with each API.
"""

import os
from collections.abc import AsyncGenerator, Generator
from typing import Any, Literal

from mbodied.agents.backends.anthropic_backend import AnthropicSerializer
from mbodied.agents.backends.openai_backend import OpenAIBackendMixin
from mbodied.agents.backends.serializer import Serializer
from mbodied.types.message import Message
from mbodied.types.sense.vision import Image

MINIMAX_OPENAI_BASE_URLS = {
    "global_en": "https://api.minimax.io/v1",
    "cn_zh": "https://api.minimaxi.com/v1",
}
MINIMAX_ANTHROPIC_BASE_URLS = {
    "global_en": "https://api.minimax.io/anthropic",
    "cn_zh": "https://api.minimaxi.com/anthropic",
}
MINIMAX_REGION_ALIASES = {
    "global": "global_en",
    "cn": "cn_zh",
}
DEFAULT_MINIMAX_OPENAI_BASE_URL = MINIMAX_OPENAI_BASE_URLS["global_en"]


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

    MiniMax provides OpenAI-compatible and Anthropic-compatible chat completion
    endpoints for both the global and China regions. The OpenAI-compatible route
    uses ``MiniMaxSerializer`` for text and image messages, while the
    Anthropic-compatible route uses ``AnthropicSerializer``.

    Attributes:
        api_key: The API key for the MiniMax service.
        client: The client for the selected API format.
        serialized: The serializer for the selected API format.
    """

    DEFAULT_MODEL = "MiniMax-M3.1-Flash"
    # Recommended models; explicit legacy and custom model IDs remain accepted.
    SUPPORTED_MODELS = ("MiniMax-M3.1-Flash", "MiniMax-M3")
    INITIAL_CONTEXT = [
        Message(role="system", content="You are a robot with advanced spatial reasoning."),
    ]
    ANTHROPIC_INITIAL_CONTEXT = [
        Message(role="user", content="Imagine you are a robot with advanced spatial reasoning."),
        Message(role="assistant", content="Got it!"),
    ]

    def __init__(
        self,
        api_key: str | None = None,
        client: Any | None = None,
        region: str = "global_en",
        api_format: Literal["openai", "anthropic"] = "openai",
        base_url: str | None = None,
        response_format: str | None = None,
        aclient: bool = False,
        **kwargs,
    ):
        """Initializes the MiniMaxBackend with the given API key and client.

        Args:
            api_key: The API key for the MiniMax service. Defaults to the
                ``MINIMAX_API_KEY`` environment variable.
            client: An optional pre-configured client for the selected API format.
            region: The regional endpoint to use, either ``"global_en"`` or
                ``"cn_zh"``. The aliases ``"global"`` and ``"cn"`` are also
                accepted. Ignored when ``base_url`` is provided.
            api_format: The compatible API format, either ``"openai"`` or
                ``"anthropic"``.
            base_url: An explicit base URL. Takes precedence over ``region``.
            response_format: The OpenAI-compatible response format.
            aclient: Whether to initialize the asynchronous OpenAI client.
            **kwargs: Additional keyword arguments forwarded to the selected client.
        """
        if api_format not in ("openai", "anthropic"):
            raise ValueError("api_format must be either 'openai' or 'anthropic'.")

        normalized_region = MINIMAX_REGION_ALIASES.get(region, region)
        regional_urls = MINIMAX_OPENAI_BASE_URLS if api_format == "openai" else MINIMAX_ANTHROPIC_BASE_URLS
        if base_url is None and normalized_region not in regional_urls:
            raise ValueError("region must be one of 'global_en', 'cn_zh', 'global', or 'cn'.")

        self.api_key = api_key or os.getenv("MINIMAX_API_KEY")
        self.client = client
        self.region = normalized_region
        self.api_format = api_format
        self.base_url = base_url or regional_urls[normalized_region]
        self.response_format = response_format

        if self.client is None:
            kwargs.pop("model_src", None)
            if self.api_format == "openai":
                from openai import AsyncOpenAI, OpenAI

                self.client = OpenAI(api_key=self.api_key or "any_key", base_url=self.base_url, **kwargs)
                if aclient:
                    self.aclient = AsyncOpenAI(api_key=self.api_key or "any_key", base_url=self.base_url, **kwargs)
            else:
                import anthropic

                self.client = anthropic.Anthropic(api_key=self.api_key, base_url=self.base_url, **kwargs)

        self.serialized = MiniMaxSerializer if self.api_format == "openai" else AnthropicSerializer

    def predict(
        self,
        message: Message,
        context: list[Message] | None = None,
        model: Any | None = None,
        tools: list[Any] | None = None,
        **kwargs,
    ) -> Any:
        """Create a completion using the selected MiniMax-compatible API."""
        if self.api_format == "openai":
            return super().predict(message, context=context, model=model, tools=tools, **kwargs)
        if tools:
            raise NotImplementedError("Tool calls are not supported by the MiniMax Anthropic-compatible route.")

        context = context or self.ANTHROPIC_INITIAL_CONTEXT
        serialized_messages = [self.serialized(msg).serialize() for msg in context + [message]]
        completion = self.client.messages.create(
            model=model or self.DEFAULT_MODEL,
            messages=serialized_messages,
            max_tokens=kwargs.pop("max_tokens", 1024),
            **kwargs,
        )
        return completion.content[0].text

    def stream(
        self,
        message: Message,
        context: list[Message] | None = None,
        model: str | None = None,
        tools: list[Any] | None = None,
        **kwargs,
    ) -> Generator[str | tuple[str, Any], None, None]:
        """Stream a completion using the selected MiniMax-compatible API."""
        if self.api_format == "openai":
            yield from super().stream(message, context=context, model=model, tools=tools, **kwargs)
            return
        if tools:
            raise NotImplementedError("Tool calls are not supported by the MiniMax Anthropic-compatible route.")

        context = context or self.ANTHROPIC_INITIAL_CONTEXT
        serialized_messages = [self.serialized(msg).serialize() for msg in context + [message]]
        with self.client.messages.stream(
            model=model or self.DEFAULT_MODEL,
            messages=serialized_messages,
            max_tokens=kwargs.pop("max_tokens", 1024),
            **kwargs,
        ) as stream:
            yield from stream.text_stream

    async def astream(
        self,
        message: Message,
        context: list[Message] | None = None,
        model: str | None = None,
        tools: list[Any] | None = None,
        **kwargs,
    ) -> AsyncGenerator[str | tuple[str, Any], None]:
        """Stream asynchronously using the MiniMax default when no model is supplied."""
        async for chunk in super().astream(message, context=context, model=model, tools=tools, **kwargs):
            yield chunk
