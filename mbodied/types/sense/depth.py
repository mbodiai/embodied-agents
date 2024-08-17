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

from functools import cached_property, reduce, wraps
from pathlib import Path
from typing import Any, ClassVar, Dict, List, SupportsBytes, Tuple, Union

import cv2
import numpy as np
from PIL.Image import Image as PILImage
from pydantic import (
    AnyUrl,
    Base64Str,
    FilePath,
    PrivateAttr,
    computed_field,
    model_serializer,
    model_validator,
)
from typing_extensions import Literal

from mbodied.types.ndarray import NumpyArray
from mbodied.types.sample import Sample
from mbodied.types.sense.image import Image
from mbodied.types.sense.image_utils import dispatch_arg

SupportsImage = Union[np.ndarray, PILImage, Base64Str, AnyUrl, FilePath]  # noqa: UP007

DepthImageLike = NumpyArray[1, Any, Any, np.uint16] | NumpyArray[Any, Any, np.uint16]

PointCloudLike = NumpyArray[Any, 3, np.float32]

LinearUnit = Literal["m", "cm", "mm", "km", "in", "ft", "yd", "mi"]


class Depth(Sample):
    """A class for representing depth images and points."""

    DEFAULT_MODE: ClassVar[str] = "I"
    SOURCE_TYPES: ClassVar[List[str]] = ["path", "array", "base64", "bytes", "url", "pil", "points"]
    mode: Literal["RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "I", "F"] = DEFAULT_MODE
    points: NumpyArray[Any, 3, np.float32] | None = None
    encoding: Literal["png"] = "png"
    size: Tuple[int, int] | None = None
    # camera: Camera | None = None
    path: str | Path | FilePath | None = None

    _url: AnyUrl | None = PrivateAttr(default=None)
    _array: NumpyArray[Any, Any, np.uint16] | None = PrivateAttr(default=None)
    _rgb: Image | None = PrivateAttr(default=None)

    def __eq__(self, other: Any) -> bool:
        """Check if two depth images are equal."""
        if not isinstance(other, Depth):
            return False
        return np.allclose(self.array, other.array)

    @computed_field(return_type=NumpyArray[Any, Any, 3, np.uint16])
    @property
    def array(self) -> NumpyArray[Any, Any, 3, np.uint16]:
        """The raw depth image represented as a NumPy array."""
        return self._array if self._array is not None else None

    @array.setter
    def array(self, value: NumpyArray[Any, Any, np.uint16] | None) -> NumpyArray[Any, Any, 3, np.uint16]:
        self._array = value
        if self._array is not None:
            self.size = (self._array.shape[1], self._array.shape[0])
        return self._array if value is not None else None

    @computed_field(return_type=Image)
    @property
    def rgb(self) -> Image:
        """Convert the depth image to an RGB image."""
        if self._rgb is None:
            if self.array is None:
                msg = "The depth array must be set to convert to an RGB image."
                raise ValueError(msg)
            normalized_array = cv2.normalize(self.array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            rgb_image = Image(normalized_array, mode="RGB")
            self._rgb = rgb_image
        return self._rgb

    @rgb.setter
    def rgb(self, value: Image) -> Image:
        if not isinstance(value, Image):
            msg = "rgb must be an instance of Image"
            raise TypeError(msg)
        if self.array is not None and value.size != (self.array.shape[1], self.array.shape[0]):
            msg = "The size of the RGB image must match the size of the depth image."
            raise ValueError(msg)
        self._rgb = value
        return self._rgb

    @classmethod
    def supports_pointcloud(cls, arg: Any) -> bool:
        """Check if the argument is a point cloud."""
        return isinstance(arg, np.ndarray) and arg.ndim == 2 and arg.shape[1] == 3

    @computed_field(return_type=NumpyArray[Any, 3, np.uint16])
    @cached_property
    def pil(self) -> PILImage:
        """The PIL image object."""
        if self.array is None:
            msg = "The array must be set to convert to a PIL image."
            raise ValueError(msg)
        return Image(self.array, mode="I", encoding="png").pil

    @computed_field(return_type=Base64Str)
    @cached_property
    def base64(self) -> Base64Str:
        """The base64 encoded string of the image."""
        if self.array is None:
            msg = "The array must be set to convert to a base64 string."
            raise ValueError(msg)
        return Image(self.array, mode="I", encoding="png").base64

    @computed_field
    @cached_property
    def url(self) -> str:
        """The URL of the image."""
        if self._url is not None:
            return self._url
        return f"data:image/{self.encoding};base64,{self.base64}"

    def __init__(  # noqa
        self,
        arg: SupportsImage | DepthImageLike | None = None,
        points: DepthImageLike | None = None,
        path: str | Path | FilePath | None = None,
        array: np.ndarray | None = None,
        base64: Base64Str | None = None,
        rgb: Image | None = None,
        # camera: Camera | None = None,
        encoding: str = "png",
        size: Tuple[int, ...] | None = None,
        bytes: SupportsBytes | None = None,  # noqa
        unit: LinearUnit | None = None,
        mode: Literal["RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "I", "F"] | None = "I",
    ):
        """Initializes a Depth representation. Unlike the Image class, an empty array is used as the default image.

        Args:
            arg (SupportsImage, optional): The primary image source.
            points (DepthImageLike, optional): The point cloud data.
            path (str | Path | FilePath, optional): The path to the image file.
            array (np.ndarray, optional): The raw image data.
            base64 (Base64Str, optional): The base64 encoded image data.
            rgb (Image, optional): The RGB image.
            encoding (str, optional): The image encoding.
            size (Tuple[int, ...], optional): The image size.
            bytes (SupportsBytes, optional): The raw image bytes.
            unit (LinearUnit, optional): The linear unit of the image.
            mode (Literal["RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "I", "F"], optional): The image mode.
        """
        kwargs = {}
        kwargs["path"] = path
        kwargs["array"] = array
        kwargs["base64"] = base64
        kwargs["bytes"] = bytes
        kwargs["encoding"] = encoding
        kwargs["size"] = size
        kwargs["mode"] = mode
        kwargs["points"] = points
        # kwargs["camera"] = camera
        kwargs["unit"] = unit
        if not self.supports_pointcloud(arg):
            points = kwargs.pop("points", None)

            kwargs["mode"] = "I"
            kwargs["encoding"] = "png"
            num_keys = 0

            for (
                k,
                v,
            ) in kwargs.items():
                if v is not None and k in self.SOURCE_TYPES:
                    num_keys += 1
            if num_keys > 1:
                msg = (
                    "Only one of the following arguments can be provided: path, array, base64, bytes, url, pil, points."
                )
                raise ValueError(msg)

            if rgb or isinstance(arg, Image) and arg.mode == "RGB":
                # case 1: arg is an RGB image
                rgb = arg if isinstance(arg, Image) else rgb
            elif isinstance(arg, Image | PILImage):
                # case 2: arg is a depth image
                if arg.mode == "RGB":
                    msg = "The RGB image must be provided as the 'rgb' argument."
                    raise ValueError(msg)
                points = kwargs.pop("points", None)
                rgb = Image(arg, **kwargs)
                kwargs["points"] = points

        super().__init__(**kwargs)
        # Set self.array directly if array is provided as the source
        if array is not None:
            self._array = array
        elif any(kwargs.get(k) is not None for k in self.SOURCE_TYPES):
            # Otherwise, initialize from another source type if provided
            self._array = Image(arg, **kwargs).array
        else:
            self._array = None

        if rgb is not None:
            self.rgb = rgb

        self._url = kwargs.get("url")

    @model_validator(mode="before")
    @classmethod
    def ensure_pil(cls, values: Dict[str, DepthImageLike] | DepthImageLike) -> Dict[str, Any]:
        """Ensure the image is represented as a PIL Image object."""
        sources = ["array", "base64", "path", "url", "bytes"]
        if not isinstance(values, dict):
            values = dispatch_arg(values, encoding="png", mode="I")
        url = values.get("url")
        if values.get("pil") is None:
            arg = reduce(lambda x, y: x if x is not None else y, [values.get(key) for key in sources])
            arg = arg if arg is not None else np.zeros((224, 224), dtype=np.uint16)
            values.update(dispatch_arg(arg, **values))
            if url is not None:
                values["url"] = url
        return {key: value for key, value in values.items() if key is not None}

    @model_serializer(when_used="json")
    def omit_array(self) -> dict:
        """Omit the array when serializing the object."""
        out = {
            "encoding": self.encoding,
            "size": self.size,
            "camera": self.camera,
        }
        if self._url is not None and self.base64 not in self.url:
            out["url"] = self.url
        elif self.base64 is not None:
            out["base64"] = self.base64
        return out

    @classmethod
    def from_pil(cls, pil: PILImage, **kwargs) -> "Depth":
        """Create an image from a PIL image."""
        array = np.array(pil.convert("I"), dtype=np.uint16)
        kwargs.update({"encoding": "png", "size": pil.size, "mode": cls.DEFAULT_MODE})
        return cls(array=array, **kwargs)

    def show(self) -> None:
        Image(self.colormap()).show()

    @wraps(Image.save, assigned=("__doc__"))
    def save(self, *args, **kwargs) -> None:
        """Save the image to a file."""
        self.colormap().save(*args, **kwargs)

    def dump(self, *_args, as_field: str | None = None, **_kwargs) -> dict | Any:
        """Return a dict or a field of the image."""
        if as_field is not None:
            return getattr(self, as_field)
        out = {
            "size": self.size,
            "mode": self.mode,
            "encoding": self.encoding,
        }
        if self.path is not None:
            out["path"] = self.path
        if self.base64 is not None:
            out["base64"] = self.base64
        if self.url not in self.base64 and len(self.url) < 120:
            out["url"] = self.url
        return out
