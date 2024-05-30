# Imports for the tests
import pytest
from unittest.mock import patch
from mbodied_agents.agents.backends.openai_backend import OpenAIBackend
from mbodied_agents.types.message import Message

# Mock response for the API call
mock_openai_response = {"choices": [
    {"message": {"content": "Mocked OpenAI response"}}]}

mock_return_value = "test"


@patch("mbodied_agents.agents.backends.openai_backend.OpenAIBackend._create_completion", return_value=mock_return_value)
def test_openai_backend_create_completion_success(mock_create):
    api_key = "test"
    backend = OpenAIBackend(api_key=api_key)
    result = backend.create_completion(Message(content="test"), [
                                       Message(content="Test message")])
    assert result == mock_return_value
    mock_create.assert_called_once()


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
