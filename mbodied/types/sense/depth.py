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
import tempfile
from functools import cached_property, wraps
from typing import Any, ClassVar, List, SupportsBytes, Tuple, Union

import cv2
import numpy as np
from PIL.Image import Image as PILImage
from pydantic import (
    AnyUrl,
    Base64Str,
    FilePath,
    computed_field,
)
from sklearn.cluster import KMeans
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from typing_extensions import Literal

from mbodied.types.ndarray import NumpyArray
from mbodied.types.sample import Sample
from mbodied.types.sense.image import Image

# TODO: Migrate camera class.
# from mbodied.types.sense.camera import Camera

SupportsImage = Union[np.ndarray, PILImage, Base64Str, AnyUrl, FilePath]  # noqa: UP007

DepthArrayLike = NumpyArray[1, Any, Any, np.uint16] | NumpyArray[Any, Any, np.uint16]


class Depth(Sample):
    """A class for representing depth images and points."""

    DEFAULT_MODE: ClassVar[str] = "I"
    SOURCE_TYPES: ClassVar[List[str]] = ["path", "array", "base64", "bytes", "url", "pil", "points"]
    mode: Literal["RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "I", "F"] = DEFAULT_MODE
    points: NumpyArray[Any, 3, np.float32] | None = None
    encoding: Literal["png"] = "png"
    size: Tuple[int, int] | None = None
    # camera_params: Camera | None = None
    array: NumpyArray[Any, Any, np.uint16] | None = None
    """The raw depth image represented as a NumPy array."""

    @computed_field(return_type=NumpyArray[Any, 3, np.uint16])
    @cached_property
    def pil(self) -> PILImage:
        """The PIL image object."""
        return self.rgb.pil

    @computed_field(return_type=NumpyArray[Any, Any, 3, np.uint8])
    @cached_property
    def rgb(self) -> Image:
        """The rgb image represented as a NumPy array."""
        self.array = self.array if self.array is not None else np.zeros((224, 224), dtype=np.uint16)
        return Image(cv2.normalize(self.array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), mode="RGB")

    @computed_field(return_type=Base64Str)
    @cached_property
    def base64(self) -> Base64Str:
        return self.rgb.base64

    def __init__(  # noqa
        self,
        arg: SupportsImage | DepthArrayLike | None = None,
        points: DepthArrayLike | None = None,
        path: str | None = None,
        array: np.ndarray | None = None,
        base64: Base64Str | None = None,
        encoding: str = "png",
        size: Tuple[int, ...] | None = None,
        bytes: SupportsBytes | None = None,  # noqa
        mode: Literal["RGB", "RGBA", "L", "P", "CMYK", "YCbCr", "I", "F"] | None = "I",
        **kwargs,
    ):
        """Initializes a Depth representation. Unlike the Image class, an empty array is used as the default image.

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
            mode (Optional[str], optional): The mode to use for the image. Defaults to 'RGB'.
            **kwargs: Additional keyword arguments.
        """
        kwargs["path"] = path
        kwargs["array"] = array
        kwargs["base64"] = base64
        kwargs["bytes"] = bytes
        kwargs["encoding"] = encoding
        kwargs["size"] = size
        kwargs["mode"] = mode
        kwargs["points"] = points
        if isinstance(arg, Image):
            arg = arg.array

        if isinstance(arg, np.ndarray) and arg.ndim == 3 and arg.shape[2] == 3:
            dest = np.zeros_like(arg).astype(np.uint16)
            cv2.normalize(arg, dest, 0, 65535, cv2.NORM_MINMAX)
            arg = dest

        if arg is None:
            arg = np.zeros((224, 224), dtype=np.uint16)

        super().__init__(**kwargs)

    @classmethod
    def from_pil(cls, pil: PILImage, **kwargs) -> "Depth":
        """Create an image from a PIL image."""
        array = np.array(pil.convert("I"), dtype=np.uint16)
        kwargs.update({"encoding": "png", "size": pil.size, "mode": cls.DEFAULT_MODE})
        return cls(array=array, **kwargs)

    def cluster_points(self, n_clusters: int = 3) -> List[int]:
        """Cluster the points using KMeans.

        Args:
            n_clusters (int): The number of clusters to form.

        Returns:
            List[int]: The cluster labels for each point.
        """
        kmeans = KMeans(n_clusters=n_clusters)
        return kmeans.fit_predict(self.points.T)

    def segment_plane(
        self, min_samples=3, threshold: float = 0.01, max_trials: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Segment the largest plane using RANSAC."""
        ransac = RANSACRegressor(min_samples=min_samples, residual_threshold=threshold, max_trials=max_trials)
        ransac.fit(self.points[:2].T, self.points[2])
        inlier_mask = ransac.inlier_mask_
        plane_coefficients = np.append(ransac.estimator_.coef_, ransac.estimator_.intercept_)
        return inlier_mask, plane_coefficients

    def colormap(self, depth_scale=1.0, path=None, **kwargs) -> Image:
        """Postprocess the predicted depth tensor."""
        depth_normalized = cv2.normalize(self.array, None, 0, 255, cv2.NORM_MINMAX)

        depth_8bit = depth_normalized.astype("uint8")

        import platform

        import matplotlib as mpl

        if platform.system() == "Darwin":
            mpl.use("TkAgg")
        import matplotlib.pyplot as plt

        colormap_image = plt.cm.inferno(depth_8bit / 255.0)
        colormap_image_rgb = (colormap_image[..., :3] * 255).astype(np.uint8)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            plt.imsave(f.name, colormap_image_rgb)
            saved_image_path = f.name

            return Image(saved_image_path, mode="RGB")

    def show(self) -> None:
        Image(self.colormap()).show()

    @wraps(Image.save, assigned=("__doc__"))
    def save(self, *args, **kwargs) -> None:
        """Save the image to a file."""
        self.colormap().save(*args, **kwargs)

    def segment_cylinder(
        self, min_samples=3, threshold: float = 0.01, max_trials: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Segment the largest cylinder using RANSAC.

        Args:
            min_samples (int): The minimum number of data points to fit a model.
            threshold (float): The maximum distance for a point to be considered as an inlier.
            max_trials (int): The maximum number of iterations for RANSAC.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The inlier points and their indices.
        """
        poly = PolynomialFeatures(degree=2)
        ransac = make_pipeline(
            poly, RANSACRegressor(min_samples=min_samples, residual_threshold=threshold, max_trials=max_trials)
        )

        x = self.points[:, :2]  # Assuming self.points is of shape (n_points, 3)
        y = self.points[:, 2]

        ransac.fit(x, y)

        inlier_mask = ransac.named_steps["ransacregressor"].inlier_mask_
        inlier_points = self.points[inlier_mask]
        inlier_indices = np.where(inlier_mask)[0]

        return inlier_points, inlier_indices
