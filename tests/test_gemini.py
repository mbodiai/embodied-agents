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
from mbodied.agents.backends import GeminiBackend
from mbodied.types.message import Message
from mbodied.types.sense.vision import Image


mock_gemini_response = "Gemini response text"


class FakeGenAI:
    class Content:
        def __init__(self, text):
            self.text = text

    class Response:
        def __init__(self, text):
            self.text = text

    class StreamResponse:
        def __init__(self, text_chunks):
            self.chunks = [self.Chunk(text=chunk) for chunk in text_chunks]

        class Chunk:
            def __init__(self, text):
                self.text = text

        def __iter__(self):
            return iter(self.chunks)

    class Models:
        @staticmethod
        def generate_content(*args, **kwargs):
            return FakeGenAI.Response(text=mock_gemini_response)

        @staticmethod
        def generate_content_stream(*args, **kwargs):
            return FakeGenAI.StreamResponse(["Streaming response part 1", "Streaming response part 2", ""])

    def __init__(self):
        self.models = self.Models()


@pytest.fixture
def gemini_api_key():
    return "fake_gemini_api_key"


@pytest.fixture
def gemini_backend(gemini_api_key):
    return GeminiBackend(api_key=gemini_api_key, client=FakeGenAI())


def test_gemini_backend_create_completion(gemini_backend):
    response = gemini_backend.predict(Message("hi"), context=[])
    assert response == mock_gemini_response


def test_gemini_backend_with_context(gemini_backend):
    context = [Message("Hello"), Message("How are you?")]
    response = gemini_backend.predict(Message("What's the weather like?"), context=context)
    assert response == mock_gemini_response


def test_gemini_backend_with_system_message(gemini_backend):
    context = [Message(role="system", content="You are a helpful assistant"), Message("Hello")]
    response = gemini_backend.predict(Message("What's the weather like?"), context=context)
    assert response == mock_gemini_response


def test_gemini_backend_with_image(gemini_backend):
    image = Image(size=(224, 224))
    response = gemini_backend.predict(Message(["Describe this image", image]), context=[])
    assert response == mock_gemini_response


def test_gemini_backend_stream(gemini_backend):
    stream_generator = gemini_backend.stream(Message("Stream this message"), context=[])
    results = list(stream_generator)
    expected_results = ["Streaming response part 1", "Streaming response part 2", ""]
    assert results == expected_results


def test_gemini_serializer_text():
    from mbodied.agents.backends.gemini_backend import GeminiSerializer

    serializer = GeminiSerializer("Hello, world")
    result = serializer.serialize()
    assert result == "Hello, world"


def test_gemini_serializer_image():
    from mbodied.agents.backends.gemini_backend import GeminiSerializer

    image = Image(size=(224, 224))
    serializer = GeminiSerializer(image)
    result = serializer.serialize()
    assert result == image.pil


def test_gemini_serializer_message_with_text():
    from mbodied.agents.backends.gemini_backend import GeminiSerializer

    message = Message("Hello, world")
    serializer = GeminiSerializer(message)
    result = serializer.serialize()
    assert result == "Hello, world"


def test_gemini_serializer_message_with_image():
    from mbodied.agents.backends.gemini_backend import GeminiSerializer

    image = Image(size=(224, 224))
    message = Message([image])
    serializer = GeminiSerializer(message)
    result = serializer.serialize()
    assert result == image.pil


def test_gemini_serializer_message_with_mixed_content():
    from mbodied.agents.backends.gemini_backend import GeminiSerializer

    image = Image(size=(224, 224))
    message = Message(["Describe this", image])
    serializer = GeminiSerializer(message)
    result = serializer.serialize()
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == "Describe this"
    assert result[1] == image.pil


if __name__ == "__main__":
    pytest.main([__file__])
