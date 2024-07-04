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

from typing import Any, Literal

from pydantic import Field

from mbodied.types.sample import Sample
from mbodied.types.sense.vision import Image

Role = Literal["user", "assistant", "system"]


class Message(Sample):
    """Single completion sample space.

    Message can be text, image, list of text/images, Sample, or other modality.

    Attributes:
        role: The role of the message sender (user, assistant, or system).
        content: The content of the message, which can be of various types.
    """

    role: Role = "user"
    content: Any | None = Field(default_factory=list)

    @classmethod
    def supports(cls, arg: Any) -> bool:
        """Checks if the argument type is supported by the Message class.

        Args:
            arg: The argument to be checked.

        Returns:
            True if the argument type is supported, False otherwise.
        """
        return Image.supports(arg) or isinstance(arg, str | list | Sample | tuple | dict)

    def __init__(
        self,
        content: Any | None = None,
        role: Role = "user",
    ):
        """Initializes a Message instance.

        Args:
            content: The content of the message, which can be of various types.
            role: The role of the message sender (default is "user").
        """
        data = {"role": role}
        if content is not None and not isinstance(content, list):
            content = [content]
        data["content"] = content
        super().__init__(**data)
