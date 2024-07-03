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

from typing import Any

from pydantic import ConfigDict, model_serializer, model_validator

from mbodied.types.message import Message
from mbodied.types.sample import Sample
from mbodied.types.sense.vision import Image


class Serializer(Sample):
    """A class to serialize messages and samples.

    This class provides a mechanism to serialize messages and samples into a dictionary format
    used by i.e. OpenAI, Anthropic, or other APIs.

    Attributes:
        wrapped: The message or sample to be serialized.
        model_config: The Pydantic configuration for the Serializer model.
    """
    wrapped: Any | None = None
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        wrapped: Message | Sample | list[Message] | None = None,
        *,
        message: Message | None = None,
        sample: Sample | None = None,
        **data,
    ):
        """Initializes the Serializer with various possible wrapped types.

        Args:
            wrapped: An instance of Message, Sample, a list of Messages, or None.
            message: An optional Message to be wrapped.
            sample: An optional Sample to be wrapped.
            **data: Additional data to initialize the Sample base class.

        """
        if wrapped is not None:
            data["wrapped"] = wrapped
        elif message is not None:
            data["wrapped"] = message
        elif sample is not None:
            data["wrapped"] = sample
        super().__init__(**data)

    @model_validator(mode="before")
    @classmethod
    def validate_model(cls, values: dict[str, Any]) -> dict[str, Any] | list[Any]:
        """Validates the 'wrapped' field of the model.

        Args:
            values: A dictionary of field values to validate.

        Returns:
            The validated values dictionary.

        Raises:
            ValueError: If the 'wrapped' field contains an invalid type.

        """
        if ("wrapped" in values and values["wrapped"] is not None
                and not isinstance(
                    values["wrapped"],
                    Message | Sample | list | str | Image,
                )):
            raise ValueError(
                f"Invalid wrapped type {type(values['wrapped'])}", )
        return values

    def serialize_sample(self, sample: Any) -> dict[str, Any]:
        """Serializes a given sample.

        Args:
            sample: The sample to be serialized.

        Returns:
            A dictionary representing the serialized sample.

        """
        if isinstance(sample, Message):
            return self.serialize_msg(sample)
        if not isinstance(sample, Sample):
            sample = Sample(sample)
        if isinstance(sample, Image):
            return self.serialize_image(sample)
        if Image.supports(sample):
            return self.serialize_image(Image(sample))
        if hasattr(sample, "datum") and isinstance(sample.datum, str):
            return self.serialize_text(sample.datum)

        return self.serialize_text(str(sample))

    @model_serializer(when_used="always")
    def serialize(self) -> dict[str, Any] | list[Any]:
        """Serializes the wrapped content of the Serializer instance.

        Returns:
            A dictionary representing the serialized wrapped content.

        """
        if isinstance(self.wrapped, Message):
            return self.serialize_msg(self.wrapped)
        if isinstance(self.wrapped, list):
            if all(isinstance(m, Message) for m in self.wrapped):
                return [self.serialize_msg(m) for m in self.wrapped]
            return [self.serialize_sample(m) for m in self.wrapped]

        return self.serialize_sample(self.wrapped)

    def serialize_msg(self, message: Message) -> dict[str, Any]:
        """Serializes a Message instance.

        Args:
            message: The Message to be serialized.

        Returns:
            A dictionary representing the serialized Message.

        """
        return {
            "role": message.role,
            "content": [self.serialize_sample(c) for c in message.content],
        }

    @classmethod
    def serialize_image(cls, image: Image) -> dict[str, Any]:
        """Serializes an Image instance.

        Args:
            image: The Image to be serialized.

        Returns:
            A dictionary representing the serialized Image.

        """
        return {
            "type": "image",
            "image_url": f"data:image/{image.encoding};base64," + image.base64,
        }

    @classmethod
    def serialize_text(cls, text: str) -> dict[str, Any]:
        """Serializes a text string.

        Args:
            text: The text to be serialized.

        Returns:
            A dictionary representing the serialized text.

        """
        return {"type": "text", "text": text}
