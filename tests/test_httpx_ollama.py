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
from typing import Any, Dict
from unittest import mock
import pytest
from mbodied.agents.backends import OllamaBackend
from mbodied.agents.backends.httpx_backend import HttpxSerializer
from mbodied.types.message import Message
from mbodied.types.sense.vision import Image
from pydantic import BaseModel

# Mock responses for the API callss
mock_response = "OpenAI response text"


from mbodied.types.sense.vision import Image


class Response(BaseModel):
    message: Dict[str, Any]
    status_code: int
    text: str

    def __init__(self, content, *args, **kwargs):
        kwargs = dict(message={"content": content}, status_code=200, text=json.dumps({"message": {"content": content}}))
        super().__init__(*args, **kwargs)

    def json(self):
        return self.model_dump_json()


class FakeHttpxClient:
    def __init__(self, *args, **kwargs):
        pass

    def post(self, *args, **kwargs):
        return Response(content=mock_response)

    def stream(self, *args, **kwargs):
        return self

    def iter_bytes(self):
        yield mock_response.encode()

    def iter_text(self):
        yield json.dumps({"message": {"content": "Ollama response text"}})

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass


class FakeAsyncHttpxClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args, **kwargs):
        pass

    async def post(self, *args, **kwargs):
        return Response(content=mock_response)

    def stream(self, *args, **kwargs):
        class StreamContextManager:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args, **kwargs):
                pass

            async def aiter_text(self):
                yield json.dumps({"message": {"content": mock_response}})

        return StreamContextManager()


@mock.patch("httpx.Client", FakeHttpxClient)
def test_completion():
    wrapper = OllamaBackend()
    text = "What animal is this? Answer briefly."
    print("Sending message to Ollama model...")
    # Synchronous usage
    response = wrapper.predict([Message(role="user", content=text)], model="llama3")
    assert response == mock_response


@mock.patch("httpx.AsyncClient", FakeAsyncHttpxClient)
def test_async_stream():
    wrapper = OllamaBackend()
    for chunk in wrapper._stream_completion([Message(role="user", content="Hello")], "llama3"):
        assert chunk == mock_response


# Mock responses for the API calls
mock_response = "Ollama response text"


@mock.patch("httpx.Client", FakeHttpxClient)
def test_completion():
    wrapper = OllamaBackend()
    text = "What is the capital of France?"
    response = wrapper.predict([Message(role="user", content=text)], model="llama2")
    assert response == mock_response


@mock.patch("httpx.Client", FakeHttpxClient)
def test_stream_completion():
    wrapper = OllamaBackend()
    chunks = list(wrapper._stream_completion([Message(role="user", content="Hello")], "llama2"))
    assert len(chunks) == 1
    assert chunks[0] == mock_response


@mock.patch("httpx.AsyncClient", FakeAsyncHttpxClient)
@pytest.mark.asyncio
async def test_async_stream():
    wrapper = OllamaBackend()
    chunks = []
    async for chunk in wrapper._astream_completion([Message(role="user", content="Hello")], "llama3"):
        chunks.append(chunk)
    assert len(chunks) == 1
    assert chunks[0] == mock_response


@mock.patch("httpx.AsyncClient", FakeAsyncHttpxClient)
@pytest.mark.asyncio
async def test_acreate_completion():
    wrapper = OllamaBackend()
    response = await wrapper._acreate_completion([Message(role="user", content="Hello")], "llama2")
    assert response == mock_response


@mock.patch("httpx.AsyncClient", FakeAsyncHttpxClient)
@pytest.mark.asyncio
async def test_astream_completion():
    wrapper = OllamaBackend()
    chunks = []
    async for chunk in wrapper._astream_completion([Message(role="user", content="Hello")], "llama2"):
        chunks.append(chunk)
    assert len(chunks) == 1
    assert chunks[0] == mock_response


def test_serializer():
    serializer = HttpxSerializer()
    messages = [Message(role="user", content="Hello"), Message(role="assistant", content="Hi there!")]
    serialized = serializer(messages)
    assert len(serialized) == 2
    assert serialized[0] == {"role": "user", "content": {"type": "text", "text": "Hello"}}
    assert serialized[1] == {"role": "assistant", "content": {"type": "text", "text": "Hi there!"}}


def test_serializer_with_image():
    serializer = HttpxSerializer()
    image = Image(url="http://example.com/image.jpg", size=(224, 224))
    messages = [Message(role="user", content=["Describe this image", image])]
    serialized = serializer(messages)
    assert len(serialized) == 1
    assert serialized[0] == {
        "role": "user",
        "content": [{"type": "text", "text": "Describe this image"}, {"type": "image_url", "image_url": image.url}],
    }

@pytest.mark.slow
def test_ollama_agent():                                                                                                                
    from mbodied.agents.language import LanguageAgent                                                                 
    agent = LanguageAgent(context="You are a robot agent.", model_src="ollama")                                       
    response = agent.act("Hello, how are you?")                                                                       
                                                    
if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
