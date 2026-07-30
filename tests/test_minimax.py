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
from mbodied.agents.backends import MiniMaxBackend
from mbodied.agents.backends.minimax_backend import MiniMaxSerializer
from mbodied.types.message import Message
from mbodied.types.sense.vision import Image

mock_minimax_response = "MiniMax response text"


class FakeOpenAI:
    class Message:
        def __init__(self, content):
            self.content = content

    class Choice:
        def __init__(self, message):
            self.message = FakeOpenAI.Message(content=message)

    class Completion:
        def __init__(self, text):
            self.choices = [FakeOpenAI.Choice(message=text)]

    class Completions:
        def create(*args, **kwargs):
            return FakeOpenAI.Completion(text=mock_minimax_response)

    class Chat:
        def __init__(self):
            self.completions = FakeOpenAI.Completions()

    def __init__(self):
        self.chat = self.Chat()


@pytest.fixture
def minimax_api_key():
    return "fake_minimax_api_key"


@pytest.fixture
def minimax_backend(minimax_api_key):
    return MiniMaxBackend(api_key=minimax_api_key, client=FakeOpenAI())


def test_minimax_default_model_and_base_url():
    assert MiniMaxBackend.DEFAULT_MODEL == "MiniMax-M3"
    assert MiniMaxBackend(api_key=minimax_api_key, client=FakeOpenAI()).base_url == "https://api.minimax.io/v1"


def test_minimax_regional_base_urls():
    assert (
        MiniMaxBackend(api_key=minimax_api_key, client=FakeOpenAI(), region="global").base_url
        == "https://api.minimax.io/v1"
    )
    assert (
        MiniMaxBackend(api_key=minimax_api_key, client=FakeOpenAI(), region="cn").base_url
        == "https://api.minimaxi.com/v1"
    )


def test_minimax_backend_create_completion(minimax_backend):
    response = minimax_backend.predict(Message("hi"), context=[])
    assert response == mock_minimax_response


def test_minimax_backend_with_context(minimax_backend):
    context = [Message("Hello"), Message("How are you?")]
    response = minimax_backend.predict(Message("What's the weather like?"), context=context)
    assert response == mock_minimax_response


def test_minimax_backend_with_image(minimax_backend):
    image = Image(size=(224, 224))
    response = minimax_backend.predict(Message(["Describe this image", image]), context=[])
    assert response == mock_minimax_response


def test_minimax_serializer_text():
    assert MiniMaxSerializer.serialize_text("hello") == {"type": "text", "text": "hello"}


def test_minimax_serializer_image_url():
    image = Image(url="http://example.com/image.jpg", size=(224, 224))
    serialized = MiniMaxSerializer.serialize_image(image)
    assert serialized == {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}}


def test_minimax_serializer_image_base64_fallback():
    image = Image(size=(224, 224))
    serialized = MiniMaxSerializer.serialize_image(image)
    assert serialized["type"] == "image_url"
    assert serialized["image_url"]["url"].startswith("data:image/jpeg;base64,")


@mock.patch("openai.OpenAI", return_value=FakeOpenAI())
def test_language_backend_language_agent_act_minimax(minimax_api_key):
    from mbodied.agents.language import LanguageAgent

    agent = LanguageAgent(api_key=minimax_api_key, model_src="minimax")
    response = agent.act("Hello, MiniMax!", context=[])
    assert response == mock_minimax_response


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
