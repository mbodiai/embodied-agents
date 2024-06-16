from typing import Tuple

import matplotlib
import numpy as np
import open3d as o3d
from mbodied_agents.base.sample import Sample
from mbodied_agents.types.ndarray import NumpyArray
from pydantic import Field

matplotlib.use("Agg")
from matplotlib import pyplot as plt

class PointCloud(Sample):
    points: np.ndarray = Field(default_factory=lambda: np.zeros((0, 3), dtype=np.float64))
    colors: np.ndarray = Field(default_factory=lambda: np.zeros((0, 3), dtype=np.float64))
    size: int = 0

    @classmethod
    def init_from(cls, depth_image_sample, intrinsic_matrix, depth_scale=1000.0, depth_trunc=1000.0):
        """Converts a DepthImageSample to a PointCloudSample using given camera intrinsics."""
        depth_image = o3d.geometry.Image(depth_image_sample.depth_image)
        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        camera_intrinsic.set_intrinsics(
            width=depth_image_sample.size[1],
            height=depth_image_sample.size[0],
            fx=intrinsic_matrix[0, 0],
            fy=intrinsic_matrix[1, 1],
            cx=intrinsic_matrix[0, 2],
            cy=intrinsic_matrix[1, 2],
        )
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_image,
            camera_intrinsic,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
        )
        return cls(points=np.asarray(pcd.points), colors=np.asarray(pcd.colors))

    def __init__(self, **data):
        super().__init__(**data)
        self.size = (self.points.shape[0], 3)

    def visualize(self) -> None:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(self.points)
        if self.colors.any():
            pc.colors = o3d.utility.Vector3dVector(self.colors)
        o3d.visualization.draw_geometries([pc])


class DepthImage(Sample):
    array: NumpyArray | None = Field(None, repr=False, description="The depth image represented as a NumPy array.")
    size: Tuple[int, int] = Field(default_factory=lambda: (480, 640))

    def __init__(self, **data):
        super().__init__(**data)
        self.size = self.pixels.shape[:2]

    def visualize(self) -> None:
        plt.imshow(self.pixels, cmap="gray")
        plt.show()

    @classmethod
    def init_from(cls, point_cloud, intrinsic_matrix, image_size=(480, 640), depth_scale=1000.0):
        points = np.asarray(point_cloud.points)

        # Assume intrinsic_matrix is [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        fx, fy, cx, cy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

        # Project points to 2D plane
        zs = points[:, 2]
        xs = (points[:, 0] * fx / zs + cx).astype(np.int32)
        ys = (points[:, 1] * fy / zs + cy).astype(np.int32)

        # Initialize depth image
        pixel_depths = np.zeros(image_size, dtype=np.float32)

        # Assign depth values to the depth image, choosing the nearest point for each pixel
        for i in range(len(points)):
            x, y, z = xs[i], ys[i], zs[i]
            if (
                0 <= x < image_size[1]
                and 0 <= y < image_size[0]
                and (pixel_depths[y, x] == 0 or pixel_depths[y, x] > z)
            ):
                pixel_depths[y, x] = z

        # Optionally, scale the depth values
        pixel_depths *= depth_scale

        return cls(pixel_depths=pixel_depths)
