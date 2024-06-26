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
from mbodied.agents.backends import OpenAIBackend
from mbodied.types.message import Message
from mbodied.agents.language import LanguageAgent
from mbodied.types.sense.vision import Image
from pathlib import Path
from importlib_resources import files

# Mock responses for the API callss
mock_openai_response = "OpenAI response text"


@pytest.fixture
def openai_api_key():
    return "fake_openai_api_key"


@pytest.fixture
def openai_backend(openai_api_key):
    return OpenAIBackend(api_key=openai_api_key, client=FakeOpenAI())


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
            return FakeOpenAI.Completion(text=mock_openai_response)

    class Chat:
        def __init__(self):
            self.completions = FakeOpenAI.Completions()

    def __init__(self):
        self.chat = self.Chat()


# Test the OpenAIBackend create_completion method
@mock.patch("mbodied.agents.backends.OpenAIBackend.act", return_value=mock_openai_response)
def test_openai_backend_create_completion(mock_openai_act):
    backend = OpenAIBackend(api_key="fake_openai_api_key")
    response = backend.act(Message("hi"), context=[])
    assert response == mock_openai_response


# Test the LanguageAgent act method with OpenAI backend
@mock.patch("openai.OpenAI", return_value=FakeOpenAI())
def test_language_backend_language_agent_act_openai(openai_api_key):
    agent = LanguageAgent(api_key=openai_api_key)

    response = agent.act("Hello, OpenAI!", context=[])
    assert response == mock_openai_response


# Test the LanguageAgent act method with an image input
@mock.patch("openai.OpenAI", return_value=FakeOpenAI())
def test_language_backend_language_agent_act_with_image(openai_api_key):
    agent = LanguageAgent(api_key=openai_api_key)
    resource = Path("resources") / "xarm.jpeg"
    test_image = Image(path=resource)
    response = agent.act("Hi", test_image, context=[])
    assert response == mock_openai_response


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
