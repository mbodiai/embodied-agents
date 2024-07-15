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
from mbodied.agents.language.language_agent import LanguageAgent
from mbodied.types.sense.vision import Image

# Mock responses for the API calls
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
        @staticmethod
        def create(*args, **kwargs):
            return FakeOpenAI.Completion(text=mock_openai_response)

    class Chat:
        def __init__(self):
            self.completions = FakeOpenAI.Completions()

    def __init__(self):
        self.chat = self.Chat()


# Mock the initialization of OpenAIBackend within LanguageAgent
@mock.patch("mbodied.agents.backends.OpenAIBackend.__init__", return_value=None)
@mock.patch("mbodied.agents.backends.OpenAIBackend.predict", return_value=mock_openai_response)
def test_language_agent_initialization(mock_openai_init, mock_openai_act):
    agent = LanguageAgent()
    assert agent.reminders == []
    assert agent.context == []


@mock.patch("mbodied.agents.backends.OpenAIBackend.__init__", return_value=None)
@mock.patch("mbodied.agents.backends.OpenAIBackend.predict", return_value=mock_openai_response)
def test_language_agent_forget_last(mock_openai_init, mock_openai_act):
    agent = LanguageAgent(context=["Hello", Image(size=(224, 224)), "How are you?"])
    last_message = agent.forget_last()
    assert last_message.content == ["How are you?"]
    assert agent.context[0] == Message(content="Hello")
    assert agent.context[1].content[0].base64 == Image(size=(224, 224)).base64


@mock.patch("mbodied.agents.backends.OpenAIBackend.__init__", return_value=None)
@mock.patch("mbodied.agents.backends.OpenAIBackend.predict", return_value=mock_openai_response)
def test_language_agent_remind_every(mock_openai_init, mock_openai_act):
    agent = LanguageAgent()
    agent.remind_every("Please provide more details", 3)
    assert len(agent.reminders) == 1
    assert agent.reminders[0].prompt.content == ["Please provide more details"]
    assert agent.reminders[0].n == 3


@mock.patch("mbodied.agents.backends.OpenAIBackend.__init__", return_value=None)
@mock.patch("mbodied.agents.backends.OpenAIBackend.predict", return_value=mock_openai_response)
def test_language_agent_forget(mock_openai_init, mock_openai_act):
    agent = LanguageAgent(context=["Hello", "How are you?", "What's your name?"])
    agent.forget(last_n=2)
    assert agent.context == [Message(content="Hello")]


@mock.patch("mbodied.agents.backends.OpenAIBackend.__init__", return_value=None)
@mock.patch("mbodied.agents.backends.OpenAIBackend.predict", return_value=mock_openai_response)
def test_language_agent_forget_after(mock_openai_init, mock_openai_act):
    agent = LanguageAgent(context=["Hello", "How are you?", "What's your name?"])
    agent.forget_after(1)
    assert agent.context == [Message(content="Hello")]


@mock.patch("mbodied.agents.backends.OpenAIBackend.__init__", return_value=None)
@mock.patch("mbodied.agents.backends.OpenAIBackend.predict", return_value=mock_openai_response)
def test_language_agent_forget_everything(mock_openai_init, mock_openai_act):
    agent = LanguageAgent(context=["Hello", "How are you?", "What's your name?"])
    agent.forget(everything=True)
    assert agent.context == []


@mock.patch("mbodied.agents.backends.OpenAIBackend.__init__", return_value=None)
@mock.patch("mbodied.agents.backends.OpenAIBackend.predict", return_value=mock_openai_response)
def test_language_agent_history(mock_openai_init, mock_openai_act):
    agent = LanguageAgent(context=["Hello", "How are you?"])
    agent.act("What's your name?")
    history = agent.history()
    assert len(history) == 4
    assert history[0].content == ["Hello"]
    assert history[1].content == ["How are you?"]
    assert history[2].content == ["What's your name?"]
    assert history[3].role == "assistant"


@mock.patch("mbodied.agents.backends.OpenAIBackend.__init__", return_value=None)
@mock.patch("mbodied.agents.backends.OpenAIBackend.predict", return_value=mock_openai_response)
def test_language_agent_forget_after(mock_openai_init, mock_openai_act):
    agent = LanguageAgent(context=["Hello", "How are you?", "What's your name?"])
    agent.forget_after(first_n=1)
    assert agent.context == [Message(content="Hello")]


@mock.patch("mbodied.agents.backends.OpenAIBackend.__init__", return_value=None)
@mock.patch("mbodied.agents.backends.OpenAIBackend.predict", return_value=mock_openai_response)
def test_language_agent_act(mock_openai_init, mock_openai_act):
    agent = LanguageAgent()
    response = agent.act("Hello, world!")
    assert len(agent.context) == 2
    assert agent.context[1].role == "assistant"
    assert agent.context[1].content[0] == response


@mock.patch("mbodied.agents.backends.OpenAIBackend.__init__", return_value=None)
@mock.patch("mbodied.agents.backends.OpenAIBackend.predict", return_value=mock_openai_response)
def test_language_agent_act_with_image(mock_openai_init, mock_openai_act):
    agent = LanguageAgent()
    response = agent.act("Hello, world!", image=Image(size=(224, 224)))
    assert len(agent.context) == 2
    assert isinstance(agent.context[0].content[1], Image)
    assert agent.context[1].role == "assistant"
    assert agent.context[1].content[0] == response


@mock.patch("mbodied.agents.backends.OpenAIBackend.__init__", return_value=None)
@mock.patch("mbodied.agents.backends.OpenAIBackend.predict", return_value=mock_openai_response)
def test_language_agent_act_with_context(mock_openai_init, mock_openai_act):
    agent = LanguageAgent(context=["Hello"])
    response = agent.act("How are you?", context=["Nice weather today"])
    assert len(agent.context) == 3
    assert agent.context[2].role == "assistant"
    assert agent.context[2].content[0] == response


@mock.patch("mbodied.agents.backends.OpenAIBackend.__init__", return_value=None)
@mock.patch("mbodied.agents.backends.OpenAIBackend.predict", return_value=mock_openai_response)
@pytest.mark.asyncio
async def test_language_agent_async_act(mock_openai_init, mock_openai_act):
    agent = LanguageAgent()
    response = await agent.async_act("Hello, async world!")
    assert len(agent.context) == 2
    assert agent.context[1].role == "assistant"
    assert agent.context[1].content[0] == response


@mock.patch("mbodied.agents.backends.OpenAIBackend.__init__", return_value=None)
@mock.patch("mbodied.agents.backends.OpenAIBackend.predict", return_value=mock_openai_response)
@pytest.mark.asyncio
async def test_language_agent_async_act_with_image(mock_openai_init, mock_openai_act):
    agent = LanguageAgent()
    response = await agent.async_act("Hello, world!", image=Image(size=(224, 224)))
    assert len(agent.context) == 2
    assert isinstance(agent.context[0].content[1], Image)
    assert agent.context[1].role == "assistant"
    assert agent.context[1].content[0] == response


@mock.patch("mbodied.agents.backends.OpenAIBackend.__init__", return_value=None)
@mock.patch("mbodied.agents.backends.OpenAIBackend.predict", return_value=mock_openai_response)
@pytest.mark.asyncio
async def test_language_agent_async_act_with_context(mock_openai_init, mock_openai_act):
    agent = LanguageAgent(context=["Hello"])
    response = await agent.async_act("How are you?", context=["Nice weather today"])
    assert len(agent.context) == 3
    assert agent.context[2].role == "assistant"
    assert agent.context[2].content[0] == response


@mock.patch("mbodied.agents.backends.OpenAIBackend.__init__", return_value=None)
@mock.patch("mbodied.agents.backends.OpenAIBackend.predict", return_value='{"key": "value"}')
def test_language_agent_act_and_parse(mock_openai_init, mock_openai_act):
    from mbodied.types.sample import Sample

    class TestSample(Sample):
        key: str

    agent = LanguageAgent()
    response = agent.act_and_parse("Parse this", parse_target=TestSample)
    assert isinstance(response, TestSample)
    assert response.key == "value"


@mock.patch("mbodied.agents.backends.OpenAIBackend.__init__", return_value=None)
@mock.patch("mbodied.agents.backends.OpenAIBackend.predict", return_value='{"key": "value"}')
@pytest.mark.asyncio
async def test_language_agent_async_act_and_parse(mock_init, mock_openai_act):
    from mbodied.types.sample import Sample

    class TestSample(Sample):
        key: str

    agent = LanguageAgent()
    response = await agent.async_act_and_parse("Parse this", parse_target=TestSample)
    assert isinstance(response, TestSample)
    assert response.key == "value"


@mock.patch(
    "mbodied.agents.language.language_agent.LanguageAgent.act", side_effect=['{"invalid": "json"}', '{"key": "value"}']
)
def test_language_agent_act_and_parse_retry(mock_act):
    from mbodied.types.sample import Sample

    class TestSample(Sample):
        key: str

    agent = LanguageAgent()
    response = agent.act_and_parse("Parse this", parse_target=TestSample, max_retries=1)
    assert isinstance(response, TestSample)
    assert response.key == "value"
    assert mock_act.call_count == 2, f"Expected 2 calls, but got {mock_act.call_count}"


@mock.patch("mbodied.agents.backends.OpenAIBackend.predict", side_effect=['{"invalid": "json"}', '{"key": "value"}'])
def test_language_agent_act_and_parse_retry_history(mock_act):
    from mbodied.types.sample import Sample

    class TestSample(Sample):
        key: str

    agent = LanguageAgent()
    response = agent.act_and_parse("Parse this", parse_target=TestSample, max_retries=1)

    assert isinstance(response, TestSample)
    assert response.key == "value"
    assert mock_act.call_count == 2, f"Expected 2 calls, but got {mock_act.call_count}"

    history = agent.history()
    assert len(history) == 2, f"Expected 2 messages in history, but got {len(history)}"
    assert history[0].content == [
        "Parse this. Avoid the following error: Error parsing response: 1 validation error for TestSample\nkey\n  Field required [type=missing, input_value={'invalid': 'json'}, input_type=dict]\n    For further information visit https://errors.pydantic.dev/2.8/v/missing"
    ]
    assert history[1].content == ['{"key": "value"}']


if __name__ == "__main__":
    pytest.main(["-vv", __file__])
