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
"""Wrap any common image representation in an Image class to convert to any other common format.

The following image representations are supported:
- NumPy array
- PIL Image
- Base64 encoded string
- File path
- URL
- Bytes object

The image can be resized to and from any size, compressed, and converted to and from any supported format:

```python
image = Image("path/to/image.png", size=new_size_tuple).save("path/to/new/image.jpg")
image.save("path/to/new/image.jpg", quality=5)"""

import base64 as base64lib
import io
from functools import cached_property, reduce
from pathlib import Path
from typing import Any, ClassVar, Dict, List, SupportsBytes, Tuple, TypeAlias

import numpy as np
from datasets.features import Features
from datasets.features import Image as HFImage
from gymnasium import spaces
from PIL.Image import Image as PILImage
from pydantic import (
    AnyUrl,
    Base64Str,
    ConfigDict,
    Field,
    FilePath,
    InstanceOf,
    ValidationError,
    computed_field,
    model_serializer,
    model_validator,
    validate_call,
)
from typing_extensions import Literal

from mbodied.types.ndarray import NumpyArray
from mbodied.types.sample import Sample
from mbodied.types.sense.image_utils import dispatch_arg, init_base64, load_url, pil_to_data
from mbodied.types.sense.image_utils import open as open_image

ImageLikeArray = NumpyArray[3, Any, Any, np.uint8] | NumpyArray[Any, Any, 3, np.uint8]
SupportsImage: TypeAlias = (
    NumpyArray[3, ..., np.uint8]
    | InstanceOf[PILImage]
    | InstanceOf[Base64Str]
    | InstanceOf[AnyUrl]
    | InstanceOf[FilePath]
    | InstanceOf[bytes]
    | InstanceOf[io.BytesIO]
)

HIGHEST_QUALITY = 100

PIL_IMAGE_ORDER = ["width", "height"]
NUMPY_IMAGE_ORDER = ["height", "width", "channel"]
TORCH_IMAGE_ORDER = ["channel", "height", "width"]


class Image(Sample):
    """An image sample that can be represented in various formats.

    The image can be represented as a NumPy array, a base64 encoded string, a file path, a PIL Image object,
    or a URL. The image can be resized to and from any size and converted to and from any supported format.

    Attributes:
        array (Optional[np.ndarray]): The image represented as a NumPy array.
        base64 (Optional[Base64Str]): The base64 encoded string of the image.
        path (Optional[FilePath]): The file path of the image.
        pil (Optional[PILImage]): The image represented as a PIL Image object.
        url (Optional[AnyUrl]): The URL of the image.
        size (Optional[tuple[int, int]]): The size of the image as a (width, height) tuple.
        encoding (Optional[Literal["png", "jpeg", "jpg", "bmp", "gif"]]): The encoding of the image.

    Example:
        >>> image = Image("https://example.com/image.jpg")
        >>> image = Image("/path/to/image.jpg")
        >>> image = Image("data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/4Q3zaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wLwA")

        >>> jpeg_from_png = Image("path/to/image.png", encoding="jpeg")
        >>> resized_image = Image(image, size=(224, 224))
        >>> pil_image = Image(image).pil
        >>> array = Image(image).array
        >>> base64 = Image(image).base64
    """

    SOURCE_TYPES: ClassVar[List[str]] = ["array", "base64", "path", "url", "bytes"]
    DEFAULT_MODE: ClassVar[str] = "RGB"
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True, extras="forbid", validate_assignment=False)

    size: tuple[int, int] | tuple[int, int, int] | None = None
    mode: Literal["RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "I", "F"] = "RGB"
    pil: InstanceOf[PILImage] | None = Field(
        default=None,
        repr=False,
        exclude=True,
        description="The image represented as a PIL Image object.",
    )
    encoding: Literal["png", "jpeg", "jpg", "bmp", "gif"] = "jpeg"
    path: str | FilePath | None = None

    @staticmethod
    def supports(arg: SupportsImage) -> bool:  # type: ignore # noqa
        """Check if the input argument is a supported image type."""

        @validate_call
        def _supports(arg: SupportsImage) -> bool:  # type: ignore # noqa
            """Check if the input argument is a supported image type."""
            return True

        try:
            return _supports(arg)
        except (ValidationError, TypeError, AttributeError):
            return False

    @computed_field(return_type=ImageLikeArray)
    @cached_property
    def array(self) -> ImageLikeArray:
        """The image represented as a NumPy array."""
        return np.array(self.pil)

    @computed_field
    @cached_property
    def base64(self) -> Base64Str:
        """The base64 encoded string of the image."""
        buffer = io.BytesIO()
        image = self.pil.convert(self.mode)
        image.save(buffer, format=self.encoding.upper())
        return base64lib.b64encode(buffer.getvalue()).decode("utf-8")

    @computed_field
    @cached_property
    def url(self) -> str:
        """The URL of the image."""
        if self._url is not None:
            return self._url
        return f"data:image/{self.encoding};base64,{self.base64}"

    @computed_field
    @cached_property
    def bytes(self) -> io.BytesIO:
        """The bytes object of the image."""
        buffer = io.BytesIO()
        return self.pil.save(buffer, format=self.encoding.upper())

    @model_serializer(when_used="json")
    def serialize_for_json(self) -> dict:
        """Serialize the image for JSON mode."""
        return {
            "base64": self.base64,
            "path": self.path,
            "url": self.url,
            "size": self.size,
            "encoding": self.encoding,
        }

    @classmethod
    def fromarray(cls, array: ImageLikeArray) -> "Image":
        """Convert a NumPy array to a PIL image."""
        return cls(array=array)

    @classmethod
    def load(cls, path: str | Path | AnyUrl) -> "Image":
        """Load an image from a file path and create an Image instance."""
        if isinstance(path, AnyUrl):
            return cls(pil=load_url(path))
        return cls(path=path)

    def __init__(  # noqa
        self,
        arg: SupportsImage | None = None,
        path: str | FilePath | None = None,
        array: np.ndarray | None = None,
        base64: Base64Str | None = None,
        encoding: str = "jpeg",
        size: Tuple[int, ...] | None = None,
        bytes: SupportsBytes | None = None,  # noqa
        mode: Literal["RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "I", "F"] | None = DEFAULT_MODE,
        **kwargs,
    ):
        """Initializes an image. Either one source argument or size tuple must be provided.

        Args:
            arg (SupportsImage, optional): The primary image source.
            url (Optional[str], optional): The URL of the image.
            path (Optional[str], optional): The file path of the image.
            base64 (Optional[str], optional): The base64 encoded string of the image.
            array (Optional[np.ndarray], optional): The numpy array of the image.
            pil (Optional[PILImage], optional): The PIL image object.
            encoding (Optional[str], optional): The encoding format of the image. Defaults to 'jpeg'.
            size (Optional[Tuple[int, int]], optional): The size of the image as a (width, height) tuple.
            bytes (Optional[bytes], optional): The bytes object of the image.
            mode (Optional[str], optional): The mode to use for the image. Defaults to RGB.
            **kwargs: Additional keyword arguments.
        """
        kwargs["encoding"] = encoding or "jpeg"
        kwargs["path"] = path
        kwargs["size"] = size[:2] if isinstance(size, Tuple) else size
        kwargs["mode"] = mode
        kwargs["array"] = array
        kwargs["base64"] = base64
        kwargs["bytes"] = bytes
        arg = arg.dict() if isinstance(arg, Sample) else arg

        if isinstance(arg, Image):
            kwargs.update(arg.model_dump())
            arg = None
        elif isinstance(arg, dict):
            kwargs.update(arg)
            arg = None
        if arg is None:
            for k, v in kwargs.items():
                if k in self.SOURCE_TYPES and v is not None:
                    arg = kwargs.pop(k)
                    break
            if arg is None and kwargs.get("size") is not None:
                arg = np.zeros(kwargs["size"] + (3,), dtype=np.uint8)

        kwargs = dispatch_arg(arg, **kwargs)
        super().__init__(**kwargs)

    @model_validator(mode="before")
    @classmethod
    def ensure_pil(cls, values: Dict[str, SupportsImage]) -> None:
        """Ensure the image is represented as a PIL Image object."""
        sources = ["array", "base64", "path", "url", "bytes"]
        url = values.get("url")
        if values.get("pil") is None:
            arg = reduce(lambda x, y: x if x is not None else y, [values.get(key) for key in sources])
            values.update(dispatch_arg(arg, **values))
            if url is not None:
                values["url"] = url
        return {key: value for key, value in values.items() if key is not None}

    def convert(self, mode: Literal["RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "I", "F"]) -> "Image":
        """Convert the image to a different mode.

        Args:
            mode (str): The mode to convert the image to.

        Returns:
            Image: The converted image.
        """
        return Image(**pil_to_data(self.pil.convert(mode), self.encoding, self.size, mode))

    @classmethod
    def open(
        cls,
        path: Path | str,
        encoding: str = "jpeg",
        size: Tuple[int] | None = None,
        mode: Literal["RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "I", "F"] | None = DEFAULT_MODE,
        **kwargs,
    ) -> "Image":
        """Opens an image from a file path and creates an Image instance."""
        return cls(**open_image(path, encoding, size, mode, **kwargs))

    def save(self, path: str | Path, encoding: str | None = None, quality: int = 100) -> str:
        """Save the image to the specified path.

        If the image is a JPEG, the quality parameter can be used to set
        the quality of the saved image. For other formats, the quality
        parameter is ignored.

        Args:
            path (str): The path to save the image to.
            encoding (Optional[str]): The encoding to use for saving the image.
                                      If None, uses the current image encoding.
            quality (int): The quality to use for saving the image (0-10).
                           Only applies to JPEG format. Defaults to 10.

        Raises:
            ValueError: If trying to set quality for PNG images.

        Example:
            >>> image = Image.open("input.jpg")
            >>> image.save("output.png", encoding="png")
            >>> print(f"Image saved to {image.path}")

            >>> jpeg_image = Image.open("input.jpg")
            >>> jpeg_image.save("output.jpg", quality=8.5)
            >>> print(f"JPEG image saved with quality 8.5/10 to {jpeg_image.path}")
        """
        if encoding == "png" and quality < HIGHEST_QUALITY:
            msg = "Quality can only be set for JPEG images."
            raise ValueError(msg)

        encoding = encoding or self.encoding
        if quality < HIGHEST_QUALITY:
            encoding = "jpeg"

        pil_image = self.pil
        if encoding != self.encoding:
            pil_image = Image(self.array, encoding=encoding).pil

        pil_image.save(path, encoding, quality=quality)
        self.path = str(path)  # Update the path attribute to the new file path
        return path

    def show(self) -> None:
        import platform

        import matplotlib as mpl

        if platform.system() == "Darwin":
            mpl.use("TkAgg")
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(self.array)

    def space(self) -> spaces.Box:
        """Returns the space of the image."""
        if self.size is None:
            msg = "Image size is not defined."
            raise ValueError(msg)
        return spaces.Box(low=0, high=255, shape=(*self.size, 3), dtype=np.uint8)

    def numpy(self) -> np.ndarray:
        """Return the image as a NumPy array."""
        return self.array

    def torch(self) -> Any:
        """Return the image as a PyTorch tensor."""
        import torch

        return torch.from_numpy(self.array)

    def dump(self, *_args, as_field: str | None = None, **_kwargs) -> dict | Any:
        """Return a dict or a field of the image."""
        if as_field is not None:
            return getattr(self, as_field)
        out = {
            "size": self.size,
            "mode": self.mode,
            "encoding": self.encoding,
            "base64": self.base64,
        }
        if self.path is not None:
            out["path"] = self.path
        if self.url not in self.base64 and len(self.url) < 120:
            out["url"] = self.url
        return out

    def infer_features_dict(self) -> Features:
        """Infer features of the image."""
        return HFImage()

    def __eq__(self, other: object) -> bool:
        """Check if the image is equal to another image."""
        if not isinstance(other, Image):
            return False
        return np.allclose(self.array, other.array)
