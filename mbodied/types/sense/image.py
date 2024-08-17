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
image.save("path/to/new/image.jpg", quality=5)
"""

import base64 as base64lib
import io
import logging
from functools import cached_property, reduce, singledispatchmethod
from pathlib import Path
from typing import Any, ClassVar, Dict, List, SupportsBytes, Tuple

import numpy as np
from datasets.features import Features
from datasets.features import Image as HFImage
from gymnasium import spaces
from PIL import Image as PILModule
from PIL import ImageOps
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

ImageLikeArray = NumpyArray[3, Any, Any, np.uint8] | NumpyArray[Any, Any, 3, np.uint8]
SupportsImage = (
    NumpyArray[3, ..., np.uint8]
    | InstanceOf[PILImage]
    | InstanceOf[Base64Str]
    | InstanceOf[AnyUrl]
    | InstanceOf[FilePath]
    | InstanceOf[bytes]
    | "Image"
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
    path: FilePath | None = None

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
        return f"data:image/{self.encoding};base64,{self.base64}"

    @computed_field
    @cached_property
    def bytes(self) -> io.BytesIO:
        """The bytes object of the image."""
        buffer = io.BytesIO()
        return self.pil.save(buffer, format=self.encoding.upper())

    @model_serializer(when_used="json")
    def omit_array_pil(self) -> dict:
        """Return a dictionary with the array and PIL image omitted."""
        return {
            "base64": self.base64,
            "path": self.path,
            "url": self.url,
            "size": self.size,
            "encoding": self.encoding,
        }

    def __init__(  # noqa
        self,
        arg: SupportsImage | None = None,  # type: ignore
        path: str | None = None,
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
        if isinstance(arg, Image):
            kwargs.update(arg.model_dump())
            arg = None
        if arg is None:
            for k, v in kwargs.items():
                if k in self.SOURCE_TYPES and v is not None:
                    arg = kwargs.pop(k)
                    break
            if arg is None and kwargs.get("size") is not None:
                arg = np.zeros(kwargs["size"] + (3,), dtype=np.uint8)
        kwargs = Image.dispatch_arg(arg, **kwargs)
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        """Return a string representation of the image."""
        out = self.dump()
        out.update({"array": f"{np.round(self.array, 2)[:2, :2]}...{self.array.shape}"})
        return out

    @singledispatchmethod
    @classmethod
    def dispatch_arg(cls, arg: SupportsImage | None = None, **kwargs) -> dict:
        msg = f"Unsupported argument type: {type(arg)}"
        raise ValueError(msg)

    @model_validator(mode="before")
    @classmethod
    def ensure_pil(cls, values: Dict[str, SupportsImage]) -> None:
        """Ensure the image is represented as a PIL Image object."""
        sources = ["array", "base64", "path", "url", "bytes"]
        if values.get("pil") is None:
            arg = reduce(lambda x, y: x if x is not None else y, [values.get(key) for key in sources])
            values.update(cls.dispatch_arg(arg, **values))
        return {key: value for key, value in values.items() if key is not None}

    @staticmethod
    def pil_to_data(image: PILImage, encoding: str, size=None, mode: str | None = DEFAULT_MODE) -> dict:
        """Creates an Image instance from a PIL image.

        Args:
            image (PIL.Image.Image): The source PIL image from which to create the Image instance.
            encoding (str): The format used for encoding the image when converting to base64.
            size (Optional[Tuple[int, int]]): The size of the image as a (width, height) tuple.
            mode (Optional[str]): The mode to use for the image. Defaults to "RGB".

        Returns:
            Image: An instance of the Image class with populated fields.
        """
        if encoding.lower() == "jpg":
            encoding = "jpeg"
        buffer = io.BytesIO()
        image = image.convert(mode)
        image.save(buffer, format=encoding.upper())
        base64_encoded = base64lib.b64encode(buffer.getvalue()).decode("utf-8")
        data_url = f"data:image/{encoding};base64,{base64_encoded}"
        if size is not None:
            image = ImageOps.fit(image, size, PILModule.Resampling.LANCZOS)
        else:
            size = image.size
        return {
            "array": np.array(image),
            "base64": base64_encoded,
            "pil": image,
            "size": size,
            "url": data_url,
            "encoding": encoding.lower(),
        }

    @dispatch_arg.register
    @classmethod
    def init_dict(cls, arg: dict, **kwargs) -> None:
        kwargs.update(arg)
        if "pil" not in kwargs:
            kwargs.update(cls.dispatch_arg(**kwargs))
        return kwargs

    @dispatch_arg.register(io.BytesIO)
    @classmethod
    def init_bytesio(
        cls,
        arg: SupportsImage | None = None,  # type: ignore
        size: Tuple[int, int] | None = None,
        encoding="jpeg",
        mode: Literal["RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "I", "F"] | None = DEFAULT_MODE,
        **kwargs,
    ) -> None:
        kwargs.update(cls.pil_to_data(PILModule.open(arg, formats=[encoding.upper()]), encoding, size, mode))
        return kwargs

    @dispatch_arg.register(SupportsBytes)
    @classmethod
    def init_bytes(
        cls,
        arg: SupportsImage | None = None,  # type: ignore
        size: Tuple[int, int] | None = None,
        encoding="jpeg",
        mode: Literal["RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "I", "F"] | None = None,
        **kwargs,
    ) -> None:
        io.BytesIO(arg)
        kwargs.update(cls.pil_to_data(PILModule.frombytes(arg, mode="r", size=size, data=arg), encoding, size, mode))
        return kwargs

    @dispatch_arg.register(np.ndarray)
    @classmethod
    def fromarray(
        cls,
        arg: NumpyArray[3, Any, Any, np.uint8] | NumpyArray[Any, Any, 3, np.uint8] | None = None,
        size: Tuple[int, int] | None = None,
        encoding="jpeg",
        mode: Literal["RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "I", "F"] | None = None,
        **kwargs,
    ) -> None:
        """Creates an Image instance from a NumPy array."""
        if not isinstance(arg, np.ndarray) or arg.ndim != 3:
            msg = "Input must be a 3D NumPy array representing an image."
            raise ValueError(msg)
        if arg.shape[0] != 3 and arg.shape[2] != 3:
            msg = "Input must be a 3D NumPy array representing an image."
            raise ValueError(msg)
        kwargs.update(cls.pil_to_data(PILModule.fromarray(arg), encoding, size, mode))
        return kwargs

    @dispatch_arg.register(str)
    @classmethod
    def init_str(
        cls,
        arg: str,
        encoding: str = "jpeg",
        action: Literal["download", "set"] = "set",
        size: Tuple[int, int] | None = None,
        mode: Literal["RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "I", "F"] | None = DEFAULT_MODE,
        **kwargs,
    ) -> None:
        """Decodes a base64 string to create an Image instance.

        This method takes a base64 encoded string representation of an image,
        decodes it, and creates an Image instance from it. It's useful when
        you have image data in base64 format and want to work with it as an Image object.

        Args:
            arg (str): The base64 string to decode.
            encoding (str): The format used for encoding the image when converting to base64.
            action (str): Either "download" or "set" if the URL is a download link.
            size (Optional[Tuple[int, int]]): The size of the image as a (width, height) tuple.
            mode (Optional[str]): The mode to use for the image. Defaults to "
            **kwargs: Additional keyword arguments.

        Returns:
            Image: An instance of the Image class with populated fields.

        Example:
            >>> base64_str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
            >>> image = Image.from_base64(base64_str, encoding="png", size=(1, 1))
            >>> print(image.size)
            (1, 1)

            # Example with complex nested structure
            >>> nested_data = {
            ...     "image": Image.from_base64(base64_str, encoding="png"),
            ...     "metadata": {"text": "A small red square", "tags": ["red", "square", "small"]},
            ... }
            >>> print(nested_data["image"].size)
            (1, 1)
            >>> print(nested_data["metadata"]["text"])
            A small red square
        """
        if arg and Path(arg[:120]).exists():
            return cls.dispatch_arg(Path(arg), encoding, size, mode, **kwargs)
        if arg.startswith(("data:image", "http")):
            image = cls.load_url(arg, action=action, **kwargs)
            kwargs.update(cls.pil_to_data(image, encoding, size, mode))
            return kwargs
        return cls.init_base64(arg, encoding, size, mode, **kwargs)

    @classmethod
    def init_base64(
        cls,
        arg: Base64Str,
        encoding: str = "jpeg",
        size: Tuple[int, int] | None = None,
        mode: Literal["RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "I", "F"] | None = "RGB",
        **kwargs,
    ) -> None:
        """Decodes a base64 string to create an Image instance."""
        image_data = base64lib.b64decode(arg)
        image = PILModule.open(io.BytesIO(image_data)).convert(mode)
        kwargs.update(cls.pil_to_data(image, encoding, size, mode))
        return kwargs

    @dispatch_arg.register(PILImage)
    @classmethod
    def init_pil(
        cls,
        arg: PILImage,
        encoding: str = "jpeg",
        size: Tuple[int, int] | None = None,
        mode: Literal["RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "I", "F"] | None = DEFAULT_MODE,
        **kwargs,
    ) -> None:
        """Creates an Image instance from a PIL image.

        Args:
            arg (PIL.Image.Image): The source PIL image from which to create the Image instance.
            encoding (str): The format used for encoding the image when converting to base64.
            size (Optional[Tuple[int, int]]): The size of the image as a (width, height) tuple.
            mode (Optional[str]): The mode to use for the image. Defaults to "RGB".
            **kwargs: Additional keyword arguments.

        Returns:
            Image: An instance of the Image class with populated fields.
        """
        kwargs.update(cls.pil_to_data(arg, encoding, size, mode))
        return kwargs

    @dispatch_arg.register(Path)
    @classmethod
    def open(
        cls,
        arg: Path,
        encoding: str = "jpeg",
        size: Tuple[int] | None = None,
        mode: Literal["RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "I", "F"] | None = DEFAULT_MODE,
        **kwargs,
    ) -> Dict[str, Any]:
        """Opens an image from a file path and creates an Image instance.

        This method reads an image file from the specified path,
        and creates an Image instance from it. It's a convenient way to load images from
        your local file system.

        Args:
            arg (str): The path to the image file.
            encoding (str): The format used for encoding the image when converting to base64.
                            Defaults to "jpeg".
            size (Optional[Tuple[int, int]]): The size of the image as a (width, height) tuple.
                                              If provided, the image will be resized.
            mode (Optional[str]): The mode to use for the image.
            **kwargs: Additional keyword arguments.

        Returns:
            Image: An instance of the Image class with populated fields.

        Example:
            >>> image = Image.open("/path/to/image.jpg", encoding="jpeg", size=(224, 224))
            >>> print(image.size)
            (224, 224)
        """
        image = PILModule.open(arg)
        image = image.convert(mode)
        kwargs.update(cls.pil_to_data(image, encoding, size, mode))
        return kwargs

    @staticmethod
    def load_url(
        url: str,
        action: Literal["download", "set"] = "set",
        mode: Literal["RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "I", "F"] | None = DEFAULT_MODE,
        **kwargs,
    ) -> PILImage | None:
        """Downloads an image from a URL or decodes it from a base64 data URI.

        This method can handle both regular image URLs and base64 data URIs.
        For regular URLs, it downloads the image data. For base64 data URIs,
        it decodes the data directly. It's useful for fetching images from
        the web or working with inline image data.

        Args:
            url (str): The URL of the image to download, or a base64 data URI.
            action (str): Either "download" or "set" to prompt the user before downloading.
            mode (Optional[str]): The mode to use for the image. Defaults to None

        Returns:
            PIL.Image.Image | None: The downloaded and decoded image as a PIL Image object,
                                    or None if the download fails or is cancelled.

        Example:
            >>> image = Image.load_url("https://example.com/image.jpg")
            >>> if image:
            ...     print(f"Image size: {image.size}")
            ... else:
            ...     print("Failed to load image")

            >>> data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
            >>> image = Image.load_url(data_uri)
            >>> if image:
            ...     print(f"Image size: {image.size}")
            ... else:
            ...     print("Failed to load image")
        """
        if url.startswith("data:image"):
            base64_str = url.split(";base64", 1)[1]
            image_data = base64lib.b64decode(base64_str)
            image = PILModule.open(io.BytesIO(image_data))
            return image.convert(mode)

        from urllib.request import Request, urlopen

        user_agent = "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7"
        headers = {"User-Agent": user_agent}
        if not url.startswith(("http:", "https:")):
            msg = "URL must start with 'http' or 'https'."
            raise ValueError(msg)

        if not url.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")) and not url.split("?")[0].endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".gif")
        ):
            if url.find("huggingface.co") != -1:
                logging.warning("URL not ending with a valid image extension.")
            else:
                msg = f"URL must end with a valid image extension: {url[:20]}...{url[-20:]}"
                raise ValueError(msg)
        with urlopen(Request(url, None, headers)) as response:  # noqa
            data = response.read()
            image = PILModule.open(io.BytesIO(data))
        return image.convert(mode)

    def save(self, path: str, encoding: str | None = None, quality: int = 100) -> str:
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
        self.path = path  # Update the path attribute to the new file path
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

    def dump(self, *args, as_field: str | None = None, **kwargs) -> dict | Any:
        """Return a dict or a field of the image."""
        if as_field is not None:
            return getattr(self, as_field)
        return {
            "size": self.size,
            "mode": self.mode,
            "encoding": self.encoding,
            "path": self.path,
            "base64": self.base64,
        }

    def infer_features_dict(self) -> Features:
        """Infer features of the image."""
        return HFImage()
