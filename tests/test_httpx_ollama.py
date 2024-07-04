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

from unittest import mock
import pytest
from mbodied.agents.backends import OllamaBackend
from mbodied.types.message import Message

# Mock responses for the API callss
mock_response = "OpenAI response text"


class FakeHttpx:
    class Response:
        def __init__(self, content, *args, **kwargs):
            self.content = content
            self.status_code = 200

        def json(self):
            return {"message": {"content": self.content}}

        def __call__(self, *args, **kwargs):
            return self

    def stream(self, *args, **kwargs):
        return self

    def iter_bytes(self):
        yield mock_response.encode()

    def iter_text(self):
        yield mock_response

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass

    def __init__(self, *args, **kwargs):
        self.post = self.Response(content=mock_response)


@mock.patch("httpx.Client", FakeHttpx)
def test_completion():
    print("Initializing OllamaBackend")
    # api_key = os.getenv("MBB_API_KEY")
    wrapper = OllamaBackend()
    image_url = "https://v0.docs.reka.ai/_images/000000245576.jpg"
    text = "What animal is this? Answer briefly."
    print("Sending message to Ollama model...")
    # Synchronous usage
    response = wrapper._create_completion([Message(role="user", content=text)], model="llama3")
    assert response == mock_response


@mock.patch("httpx.Client", FakeHttpx)
def test_async_stream():
    wrapper = OllamaBackend()
    for chunk in wrapper._stream_completion([Message(role="user", content="Hello")], "llama3"):
        assert chunk == mock_response


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
