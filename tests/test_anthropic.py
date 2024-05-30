import pytest
from mbodied_agents.agents.backends.anthropic_backend import AnthropicBackend
from mbodied_agents.types.message import Message


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
    response = anthropic_backend.create_completion(Message("hi"), context=[])
    assert response == mock_anthropic_response


if __name__ == "__main__":
    pytest.main([__file__])
