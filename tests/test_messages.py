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
from pydantic import ValidationError
from mbodied.types.message import Message, Sample
from mbodied.agents.backends.openai_backend import OpenAISerializer as OpenAISerializable


def test_openai_serializable_with_message():
    message = Message(role="user", content=[Sample(datum="Hello")])
    serializer = OpenAISerializable(message=message)
    serialized_data = serializer.model_dump_json()
    assert serialized_data == '{"role":"user","content":[{"type":"text","text":"Hello"}]}'


def test_openai_serializable_with_invalid_type():
    with pytest.raises(ValidationError):
        OpenAISerializable(wrapped=123)  # Invalid type


def test_openai_serializable_list_of_messages():
    messages = [
        Message(role="user", content=[Sample(datum="Hello")]),
        Message(role="user", content=[Sample(datum="Bye")]),
    ]
    serializer = OpenAISerializable(wrapped=messages)
    serialized_data = serializer.model_dump_json()
    assert (
        serialized_data
        == '[{"role":"user","content":[{"type":"text","text":"Hello"}]},{"role":"user","content":[{"type":"text","text":"Bye"}]}]'
    )


# Run the tests
if __name__ == "__main__":
    pytest.main(["-vv", __file__])
