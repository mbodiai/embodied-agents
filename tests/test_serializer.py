import numpy as np
import pytest
from pydantic import ValidationError
from mbodied_agents.base.sample import Sample
from mbodied_agents.types.message import Message
from mbodied_agents.types.vision import Image
from mbodied_agents.base.serializer import Serializer


def test_serialize_string():
    serializer = Serializer(wrapped="Hello, World!")
    serialized = serializer.serialize()
    assert serialized == {"type": "text",
                          "text": "Hello, World!"}, "Failed to serialize string"


def test_serialize_sample():
    sample = Sample("Sample text")
    serializer = Serializer(wrapped=sample)
    serialized = serializer.serialize()
    assert serialized == {"type": "text",
                          "text": "Sample text"}, "Failed to serialize Sample"


def test_serialize_message():
    message = Message(role="user", content=[Sample("Message content")])
    serializer = Serializer(wrapped=message)
    serialized = serializer.serialize()
    assert serialized == {
        "role": "user",
        "content": [{"type": "text", "text": "Message content"}],
    }, "Failed to serialize Message"


def test_serialize_image():
    image = Image(np.ones((10, 10, 3), dtype=np.uint8))
    serializer = Serializer(wrapped=image)
    serialized = serializer.serialize()
    assert serialized["type"] == "image", "Failed to serialize Image"


def test_validate_model_rejection():
    with pytest.raises(ValidationError):
        Serializer(wrapped=123)  # Invalid type, should raise ValidationError


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
