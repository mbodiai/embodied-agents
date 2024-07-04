# Copyright 2024 Mbodi AI
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

TODO: Implement Lazy attribute loading for the image data.
"""
import base64 as base64lib
import importlib
import io
import logging
from pathlib import Path
from typing import Annotated, Any, Optional, Tuple, Union
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
from datasets.features import Features
from datasets.features import Image as HFImage
from gymnasium import spaces
from PIL import Image as PILModule
from PIL.Image import Image as PILImage
from pydantic import (
    AnyUrl,
    Base64Str,
    ConfigDict,
    Field,
    FilePath,
    InstanceOf,
    PrivateAttr,
    field_serializer,
    model_serializer,
    model_validator,
)
from typing_extensions import Literal

from mbodied.types.ndarray import NumpyArray
from mbodied.types.sample import Sample

SupportsImage = Union[np.ndarray, PILImage, Base64Str, AnyUrl, FilePath]  # noqa: UP007



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
    model_config: ConfigDict = ConfigDict(
        arbitrary_types_allowed=True, extras="forbid", validate_assignment=False)

    array: NumpyArray
    size: tuple[int, int] 
    
    pil: InstanceOf[PILImage] | None = Field(
        None,
        repr=False,
        exclude=True,
        description="The image represented as a PIL Image object.",
    )
    encoding: Literal["png", "jpeg", "jpg",
                      "bmp", "gif"] 
    base64: Optional[InstanceOf[Base64Str]] = None
    url: Optional[InstanceOf[AnyUrl] | str] = None
    path: FilePath | None = None


    @classmethod
    def supports(cls, arg: SupportsImage) -> bool:
        if not isinstance(arg, np.ndarray | PILImage | AnyUrl | str):
            return False
        return Path(arg).exists() or arg.startswith("data:image")

    def __init__(
        self,
        arg: SupportsImage = None,
        url: str | None = None,
        path: str | None = None,
        base64: str | None = None,
        array: np.ndarray | None = None,
        pil: PILImage | None = None,
        encoding: str | None = "jpeg",
        size: Tuple | None = None,
        bytes_obj: bytes | None = None,
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
          **kwargs: Additional keyword arguments.
        """
        kwargs["encoding"] = encoding or "jpeg"
        kwargs["size"] = size
        if arg is not None:
            if isinstance(arg, bytes):
                kwargs["bytes"] = arg
            elif isinstance(arg, str):
                if isinstance(arg, AnyUrl):
                    kwargs["url"] = arg
                elif Path(arg).exists():
                    kwargs["path"] = arg
                else:
                    kwargs["base64"] = arg
            elif isinstance(arg, Path):
                kwargs["path"] = str(arg)
            elif isinstance(arg, np.ndarray):
                kwargs["array"] = arg
            elif isinstance(arg, PILImage):
                kwargs["pil"] = arg
            elif isinstance(arg, Image):
                # Overwrite an Image instance with the new kwargs
                kwargs.update({"array": arg.array})
            elif isinstance(arg, Tuple) and len(arg) == 2:
                kwargs["size"] = arg
            else:
                raise ValueError(f"Unsupported argument type '{type(arg)}'.")
        else:
            if url is not None:
                kwargs["url"] = url
            elif path is not None:
                kwargs["path"] = path
            elif base64 is not None:
                kwargs["base64"] = base64
            elif array is not None:
                kwargs["array"] = array
            elif pil is not None:
                kwargs["pil"] = pil
            elif bytes_obj is not None:
                kwargs["bytes"] = bytes_obj
        super().__init__(**kwargs)

    def __repr__(self):
        """Return a string representation of the image."""
        if self.base64 is None:
            return f"Image(encoding={self.encoding}, size={self.size})"
        return f"Image(base64={self.base64[:10]}..., encoding={self.encoding}, size={self.size})"

    def __str__(self):
        """Return a string representation of the image."""
        return f"Image(base64={self.base64[:10]}..., encoding={self.encoding}, size={self.size})"

    @staticmethod
    def from_base64(base64_str: str, encoding: str, size=None) -> 'Image':
        """Decodes a base64 string to create an Image instance.

        Args:
            base64_str (str): The base64 string to decode.
            encoding (str): The format used for encoding the image when converting to base64.
            size (Optional[Tuple[int, int]]): The size of the image as a (width, height) tuple.

        Returns:
            Image: An instance of the Image class with populated fields.
        """
        image_data = base64lib.b64decode(base64_str)
        image = PILModule.open(io.BytesIO(image_data)).convert("RGB")
        return Image(image, encoding, size)

    @staticmethod
    def open(path: str, encoding: str = 'jpeg', size=None) -> 'Image':
        """Opens an image from a file path.

        Args:
            path (str): The path to the image file.
            encoding (str): The format used for encoding the image when converting to base64.
            size (Optional[Tuple[int, int]]): The size of the image as a (width, height) tuple.

        Returns:
            Image: An instance of the Image class with populated fields.
        """
        image = PILModule.open(path).convert("RGB")
        return Image(image, encoding, size)

    @staticmethod
    def pil_to_data(image: PILImage, encoding: str, size=None) -> dict:
        """Creates an Image instance from a PIL image.

        Args:
            image (PIL.Image.Image): The source PIL image from which to create the Image instance.
            encoding (str): The format used for encoding the image when converting to base64.
            size (Optional[Tuple[int, int]]): The size of the image as a (width, height) tuple.

        Returns:
            Image: An instance of the Image class with populated fields.
        """
        if encoding.lower() == "jpg":
            encoding = "jpeg"
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format=encoding.upper())
        base64_encoded = base64lib.b64encode(buffer.getvalue()).decode("utf-8")
        data_url = f"data:image/{encoding};base64,{base64_encoded}"
        if size is not None:
            image = image.resize(size)
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

    @staticmethod
    def load_url(url: str) -> PILImage | None:
        """Downloads an image from a URL or decodes it from a base64 data URI.

        Args:
            url (str): The URL of the image to download, or a base64 data URI.

        Returns:
            PIL.Image.Image: The downloaded and decoded image as a PIL Image object.
        """
        if url.startswith("data:image"):
            # Extract the base64 part of the data URI
            base64_str = url.split(";base64", 1)[1]
            image_data = base64lib.b64decode(base64_str)
            return PILModule.open(io.BytesIO(image_data)).convert("RGB")
        else:
            try:
                # Open the URL and read the image data
                import urllib.request

                user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
                headers={'User-Agent':user_agent,} 

                request=urllib.request.Request(url,None,headers) #The assembled request
                response = urllib.request.urlopen(request)
                data = response.read() # The data u need
                return PILModule.open(io.BytesIO(data)).convert("RGB")
            except Exception as e:
                logging.warning(f"Failed to load image from URL: {url}. {e}")
                logging.warning(f"Not validating the Image data")
                return None

    @classmethod
    def from_bytes(cls, bytes_data: bytes, encoding: str = "jpeg", size=None) -> 'Image':
        """Creates an Image instance from a bytes object.

        Args:
            bytes_data (bytes): The bytes object to convert to an image.
            encoding (str): The format used for encoding the image when converting to base64.
            size (Optional[Tuple[int, int]]): The size of the image as a (width, height) tuple.

        Returns:
            Image: An instance of the Image class with populated fields.
        """
        image = PILModule.open(io.BytesIO(bytes_data)).convert("RGB")
        return cls(image, encoding, size)
    
    @staticmethod
    def bytes_to_data(bytes_data: bytes, encoding: str = "jpeg", size=None) -> dict:
        """Creates an Image instance from a bytes object.

        Args:
            bytes_data (bytes): The bytes object to convert to an image.
            encoding (str): The format used for encoding the image when converting to base64.
            size (Optional[Tuple[int, int]]): The size of the image as a (width, height) tuple.

        Returns:
            Image: An instance of the Image class with populated fields.
        """
        image = PILModule.open(io.BytesIO(bytes_data)).convert("RGB")
        return Image.pil_to_data(image, encoding, size)
    
    

    @model_validator(mode="before")
    @classmethod
    def validate_kwargs(cls, values) -> dict:
        # Ensure that exactly one image source is provided
        provided_fields = [
            k for k in values if values[k] is not None and k in ["array", "base64", "path", "pil", "url"]
        ]
        if len(provided_fields) > 1:
            raise ValueError(f"Multiple image sources provided; only one is allowed but got: {provided_fields}")

        # Initialize all fields to None or their default values
        validated_values = {
            "array": None,
            "base64": None,
            "encoding": values.get("encoding", "jpeg").lower(),
            "path": None,
            "pil": None,
            "url": None,
            "size": values.get("size", None),
        }

        # Validate the encoding first
        if validated_values["encoding"] not in ["png", "jpeg", "jpg", "bmp", "gif"]:
            raise ValueError(
                "The 'encoding' must be a valid image format (png, jpeg, jpg, bmp, gif).")

        if "bytes" in values and values["bytes"] is not None:
            validated_values.update(cls.bytes_to_data(
                values["bytes"], values["encoding"], values["size"]))
            return validated_values
            
        if "pil" in values and values["pil"] is not None:
            validated_values.update(
                cls.pil_to_data(
                    values["pil"], values["encoding"], values["size"]),
            )
            return validated_values
        # Process the provided image source
        if "path" in provided_fields:
            image = PILModule.open(values["path"]).convert("RGB")
            validated_values["path"] = values["path"]
            validated_values.update(cls.pil_to_data(
                image, validated_values["encoding"], validated_values["size"]))

        elif "array" in provided_fields:
            image = PILModule.fromarray(values["array"]).convert("RGB")
            validated_values.update(cls.pil_to_data(
                image, validated_values["encoding"], validated_values["size"]))

        elif "pil" in provided_fields:
            validated_values.update(
                cls.pil_to_data(
                    values["pil"], validated_values["encoding"], validated_values["size"]),
            )

        elif "base64" in provided_fields:
            validated_values.update(
                cls.from_base64(
                    values["base64"], validated_values["encoding"], validated_values["size"]),
            )

        elif "url" in provided_fields:
            url_path = urlparse(values["url"]).path
            file_extension = (
                Path(url_path).suffix[1:].lower() if Path(
                    url_path).suffix else validated_values["encoding"]
            )
            validated_values["encoding"] = file_extension
            image = cls.load_url(values["url"])
            if image is None:
                validated_values["array"] = np.zeros((224, 224, 3), dtype=np.uint8)
                validated_values["size"] = (224, 224)
                return validated_values
         
            validated_values.update(cls.pil_to_data(
                image, file_extension, validated_values["size"]))
            validated_values["url"] = values["url"]
        
        elif "size" in values and values["size"] is not None:
            array = np.zeros((values["size"][0], values["size"][1], 3), dtype=np.uint8)
            image = PILModule.fromarray(array).convert("RGB")
            validated_values.update(cls.pil_to_data(
                image, validated_values["encoding"], validated_values["size"]))
        if any(validated_values[k] is None for k in ["array", "base64", "pil", "url"]):
            logging.warning(
                f"Failed to validate image data. Could only fetch {[k for k in validated_values if validated_values[k] is not None]}",
            )
        return validated_values
    

    def save(self, path: str, encoding: str | None = None, quality: int = 10) -> None:
        """Save the image to the specified path.

        If the image is a JPEG, the quality parameter can be used to set the quality of the saved image.
        The path attribute of the image is updated to the new file path.

        Args:
            path (str): The path to save the image to.
            encoding (Optional[str]): The encoding to use for saving the image.
            quality (int): The quality to use for saving the image.
        """
        if encoding == "png" and quality < 10:
            raise ValueError("Quality can only be set for JPEG images.")

        encoding = encoding or self.encoding
        if quality < 10:
            encoding = "jpeg"

        pil_image = self.pil
        if encoding != self.encoding:
            pil_image = Image(self.array, encoding=encoding).pil

        pil_image.save(path, encoding, quality=quality)
        self.path = path  # Update the path attribute to the new file path

    def show(self) -> None:
        import matplotlib
        import platform
        if platform.system() == "Darwin":
            matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        plt.imshow(self.array)

    def space(self) -> spaces.Box:
        """Returns the space of the image."""
        if self.size is None:
            raise ValueError("Image size is not defined.")
        return spaces.Box(low=0, high=255, shape=(*self.size, 3), dtype=np.uint8)
    

    @model_serializer(mode="plain", when_used="json")
    def exclude_pil(self) -> dict:
        """Convert the image to a base64 encoded string."""
        if self.base64 in self.url:
            return {
                "size": self.size,
                "url": self.url,
                "encoding": self.encoding
            }
        return {
            "base64": self.base64,
            "size": self.size,
            "url": self.url,
            "encoding": self.encoding
        }
    
    def dump(self, *args, as_field: str|None = None, **kwargs) -> dict | Any:
        """Return a dict or a field of the image."""
        if as_field is not None:
            return getattr(self, as_field)
        return {
            "array": self.array,
            "base64": self.base64,
            "size": self.size,
            "url": self.url,
            "encoding": self.encoding
        }

    def infer_features_dict(self) -> Features:
        """Infer features of the image."""
        return HFImage()



    