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

import pytest
from mbodied.agents.backends import AnthropicBackend
from mbodied.types.message import Message
from mbodied.types.sense.vision import Image


mock_anthropic_response = "Anthropic response text"


class FakeAnthropic:
    class Content:
        def __init__(self, text):
            self.text = text

    class Message:
        def __init__(self, content):
            self.content = [FakeAnthropic.Content(text=content)]

    class Messages:
        @staticmethod
        def create(*args, **kwargs):
            return FakeAnthropic.Message(content=mock_anthropic_response)

    def __init__(self):
        self.messages = self.Messages()


@pytest.fixture
def anthropic_api_key():
    return "fake_anthropic_api_key"


@pytest.fixture
def anthropic_backend(anthropic_api_key):
    return AnthropicBackend(api_key=anthropic_api_key, client=FakeAnthropic())


def test_anthropic_backend_create_completion(anthropic_backend):
    response = anthropic_backend.predict(Message("hi"), context=[])
    assert response == mock_anthropic_response


def test_anthropic_backend_with_context(anthropic_backend):
    context = [Message("Hello"), Message("How are you?")]
    response = anthropic_backend.predict(Message("What's the weather like?"), context=context)
    assert response == mock_anthropic_response


def test_anthropic_backend_with_image(anthropic_backend):
    image = Image(size=(224, 224))
    response = anthropic_backend.predict(Message(["Describe this image", image]), context=[])
    assert response == mock_anthropic_response


def test_anthropic_backend_stream(anthropic_backend):
    class MockStream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

        @property
        def text_stream(self):
            return iter(["Streaming response part 1", "Streaming response part 2", ""])

    def mock_stream(*args, **kwargs):
        return MockStream()

    anthropic_backend.client.messages.stream = mock_stream
    stream_generator = anthropic_backend.stream(Message("Stream this message"), context=[])
    results = list(stream_generator)
    expected_results = ["Streaming response part 1", "Streaming response part 2", ""]

    assert results == expected_results


if __name__ == "__main__":
    pytest.main([__file__])
