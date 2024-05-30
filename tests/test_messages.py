import pytest
from pydantic import ValidationError
from mbodied_agents.types.message import Message, Sample
from mbodied_agents.agents.backends.openai_backend import OpenAISerializer as OpenAISerializable


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
