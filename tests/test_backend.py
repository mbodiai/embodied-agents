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

# Imports for the tests
import pytest
from unittest.mock import patch
from mbodied.agents.backends import OpenAIBackend
from mbodied.types.message import Message

# Mock response for the API call
mock_openai_response = {"choices": [{"message": {"content": "Mocked OpenAI response"}}]}

mock_return_value = "test"


@patch("mbodied.agents.backends.OpenAIBackend._create_completion", return_value=mock_return_value)
def test_openai_backend_create_completion_success(mock_create):
    api_key = "test"
    backend = OpenAIBackend(api_key=api_key)
    result = backend.act(Message(content="test"), [Message(content="Test message")])
    assert result == mock_return_value
    mock_create.assert_called_once()


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
