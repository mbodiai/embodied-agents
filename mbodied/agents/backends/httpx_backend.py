import json
import os
from typing import AsyncGenerator, Generator, List, overload

import httpx

from mbodied.agents.backends.openai_backend import OpenAIBackendMixin
from mbodied.agents.backends.serializer import Serializer
from mbodied.types.message import Message
from mbodied.types.sense import Image


class HttpxSerializer(Serializer):
    def __call__(self, messages: List[Message]) -> List[dict]:
        return [self.serialize_message(message) for message in messages]

    @classmethod
    def serialize_message(cls, message: Message) -> dict:
        content = message.content
        if isinstance(content, list):
            serialized_content = [cls.serialize_content(item) for item in content]
        else:
            serialized_content = cls.serialize_content(content)
        return {
            "role": message.role,
            "content": serialized_content[0] if len(serialized_content) == 1 else serialized_content,
        }

    @classmethod
    def serialize_content(cls, content) -> dict:
        if isinstance(content, Image):
            return cls.serialize_image(content)
        return cls.serialize_text(content)

    @classmethod
    def serialize_image(cls, image: Image) -> dict:
        return {"type": "image_url", "image_url": image.url if image.url is not None else ""}

    @classmethod
    def serialize_text(cls, text) -> dict:
        return {"type": "text", "text": text}

    @classmethod
    def extract_response(cls, response) -> str:
        return response["responses"][0]["message"]["content"]

    @classmethod
    def extract_stream(cls, response) -> str:
        if response:
            response = json.loads(response.split("data:")[-1].strip())
            if response:
                return response["responses"][0]["chunk"]["content"]
        return response


class HttpxBackend(OpenAIBackendMixin):
    SERIALIZER = HttpxSerializer
    DEFAULT_SRC = "https://api.reka.ai/v1/chat"
    DEFAULT_MODEL = "reka-core-20240501"

    def __init__(
        self, api_key=None, endpoint: str | None = None, serializer: Serializer | None = None, **kwargs
    ) -> None:
        """Initializes the CompleteBackend. Defaults to using the API key from the environment and.

        Args:
            api_key (Optional[str]): The API key for the Complete service.
            endpoint (str): The base URL for the Complete API.
            serializer (Optional[Serializer]): The serializer to use for serializing messages.
        """
        self.base_url = endpoint or self.DEFAULT_SRC
        self.api_key = api_key or os.getenv("MBODI_API_KEY")
        self.headers = {"X-Api-Key": self.api_key, "Content-Type": "application/json"}
        self.serialized = serializer or self.SERIALIZER
        self.kwargs = kwargs
        self.DEFAULT_MODEL = kwargs.get("model", self.DEFAULT_MODEL)

    @overload
    def predict(self, messages: List[Message], model: str | None = None, **kwargs) -> str: ...

    def predict(self, message: Message, context: List[Message] | None = None, model: str | None = None, **kwargs) -> str:
        if isinstance(message, list):
            messages = message
        elif isinstance(message, Message):
            messages = context + [message]
        else:
            messages = context
        if isinstance(context, str):
            model = context
        model = model or self.DEFAULT_MODEL
        messages = [message] + messages
        data = {
            "messages": [self.serialized(msg).serialize() for msg in messages],
            "model": model,
            "stream": False,
            **kwargs,
        }
        data.update(kwargs)
        with httpx.Client(trust_env=True) as client:
            response = client.post(
                self.base_url, headers=self.headers, json=data, timeout=kwargs.get("timeout", 60), follow_redirects=True
            )
            if response.status_code == 200:
                response_data = response.json()
                return self.serialized.extract_response(response_data)
            response.raise_for_status()
            return response.text

    @overload
    def stream(self, messages: List[Message], model: str | None = None, **kwargs) -> str: ...

    def stream(
        self, message: Message, context: List[Message] | None = None, model: str | None = None, **kwargs
    ) -> Generator[str, None, None]:
        if isinstance(message, list):
            messages = message
        elif isinstance(message, Message):
            messages = context + [message]
        else:
            messages = context
        if isinstance(context, str):
            model = context
        else:
            messages = context
        yield from self._stream_completion(messages, model, **kwargs)

    @overload
    async def astream(self, messages: List[Message], model: str | None = None, **kwargs) -> str: ...

    async def astream(
        self, message: Message, context: List[Message] | None = None,  model: str | None = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        if isinstance(message, list):
            messages = message
        elif isinstance(message, Message):
            messages = context + [message]
        else:
            messages = context
        if isinstance(messages, str):
            model = messages
        else:
            messages = context

        async for chunk in self._astream_completion(messages, model, **kwargs):
            yield chunk

    def _stream_completion(
        self,
        messages: List[Message],
        model: str | None = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        model = model or self.DEFAULT_MODEL
        data = {
            "messages": [self.serialized(msg).serialize() for msg in messages],
            "model": model,
            "stream": True,
            **kwargs,
        }
        data.update(kwargs)
        with (
            httpx.Client(follow_redirects=True) as client,
            client.stream(
                "post", self.base_url, headers=self.headers, json=data, timeout=kwargs.get("timeout", 60)
            ) as stream,
        ):
            for chunk in stream.iter_text():
                yield self.serialized.extract_stream(chunk)

    async def _acreate_completion(self, messages: List[Message], model: str | None = None, **kwargs) -> str:
        model = model or self.DEFAULT_MODEL
        data = {
            "messages": [self.serialized(msg).serialize() for msg in messages],
            "model": model,
            "stream": False,
            **kwargs,
        }
        data.update(kwargs)

        async with httpx.AsyncClient(timeout=-1) as client:
            response = await client.post(
                self.base_url, headers=self.headers, json=data, timeout=kwargs.get("timeout", 60)
            )
            if response.status_code == 200:
                response_data = response.json()
                return self.serialized.extract_response(response_data)
            return response.text

    async def _astream_completion(
        self, messages: List[Message], model: str | None = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        model = model or self.DEFAULT_MODEL
        data = {
            "messages": [self.serialized(msg).serialize() for msg in messages],
            "model": model,
            "stream": True,
            **kwargs,
        }
        data.update(kwargs)
        async with httpx.AsyncClient(follow_redirects=True) as client:
            stream = client.stream("POST", self.base_url, headers=self.headers, json=data, timeout=60)
            async with stream as response:
                async for chunk in response.aiter_text():
                    yield self.serialized.extract_stream(chunk)


if __name__ == "__main__":
    # Usage
    import asyncio

    client = HttpxBackend()
    image_url = "https://v0.docs.reka.ai/_images/000000245576.jpg"
    text = "What animal is this? Answer briefly."
    run = client.predict([Message(role="user", content=[text, Image(url=image_url)])])
    print(run)

    async def runner():
        async for response in run:
            print(response)

    print(asyncio.run(run))
