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
from mbodied.agents.backends.anthropic_backend import AnthropicSerializer
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
            return FakeAnthropic.Message(content=mock_minimax_response)

        @staticmethod
        def stream(*args, **kwargs):
            class Stream:
                text_stream = iter(["Streaming response part 1", "Streaming response part 2"])

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc_value, traceback):
                    pass

            return Stream()

    def __init__(self):
        self.messages = self.Messages()


@pytest.fixture
def minimax_api_key():
    return "fake_minimax_api_key"


@pytest.fixture
def minimax_backend(minimax_api_key):
    return MiniMaxBackend(api_key=minimax_api_key, client=FakeOpenAI())


def test_minimax_default_model_and_base_url(minimax_api_key):
    assert MiniMaxBackend.DEFAULT_MODEL == "MiniMax-M3.1-Flash"
    assert MiniMaxBackend.SUPPORTED_MODELS == ("MiniMax-M3.1-Flash", "MiniMax-M3")
    assert MiniMaxBackend(api_key=minimax_api_key, client=FakeOpenAI()).base_url == "https://api.minimax.io/v1"


@pytest.mark.parametrize(
    ("region", "api_format", "expected_url"),
    [
        ("global_en", "openai", "https://api.minimax.io/v1"),
        ("cn_zh", "openai", "https://api.minimaxi.com/v1"),
        ("global_en", "anthropic", "https://api.minimax.io/anthropic"),
        ("cn_zh", "anthropic", "https://api.minimaxi.com/anthropic"),
        ("global", "openai", "https://api.minimax.io/v1"),
        ("cn", "openai", "https://api.minimaxi.com/v1"),
    ],
)
def test_minimax_regional_base_urls(minimax_api_key, region, api_format, expected_url):
    client = FakeOpenAI() if api_format == "openai" else FakeAnthropic()
    backend = MiniMaxBackend(api_key=minimax_api_key, client=client, region=region, api_format=api_format)
    assert backend.base_url == expected_url


def test_minimax_backend_create_completion(minimax_backend):
    response = minimax_backend.predict(Message("hi"), context=[])
    assert response == mock_minimax_response


def test_minimax_backend_supports_m2_7(minimax_backend):
    minimax_backend.client.chat.completions.create = mock.Mock(
        return_value=FakeOpenAI.Completion(text=mock_minimax_response)
    )
    response = minimax_backend.predict(Message("hi"), context=[], model="MiniMax-M2.7")
    assert response == mock_minimax_response
    assert minimax_backend.client.chat.completions.create.call_args.kwargs["model"] == "MiniMax-M2.7"


def test_minimax_anthropic_backend_create_completion(minimax_api_key):
    backend = MiniMaxBackend(api_key=minimax_api_key, client=FakeAnthropic(), api_format="anthropic")
    backend.client.messages.create = mock.Mock(return_value=FakeAnthropic.Message(content=mock_minimax_response))
    response = backend.predict(Message("hi"), context=[])
    assert response == mock_minimax_response
    assert backend.serialized is AnthropicSerializer
    assert backend.client.messages.create.call_args.kwargs["model"] == "MiniMax-M3.1-Flash"


def test_minimax_anthropic_backend_stream(minimax_api_key):
    backend = MiniMaxBackend(api_key=minimax_api_key, client=FakeAnthropic(), api_format="anthropic")
    response = list(backend.stream(Message("hi"), context=[]))
    assert response == ["Streaming response part 1", "Streaming response part 2"]


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


def test_language_backend_language_agent_act_minimax_anthropic(minimax_api_key):
    from mbodied.agents.language import LanguageAgent

    agent = LanguageAgent(
        api_key=minimax_api_key,
        model_src="minimax",
        model_kwargs={"api_format": "anthropic", "client": FakeAnthropic()},
    )
    response = agent.act("Hello, MiniMax!", context=[])
    assert response == mock_minimax_response


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
