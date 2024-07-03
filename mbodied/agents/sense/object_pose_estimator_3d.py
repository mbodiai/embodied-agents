from typing import Dict, List

import numpy as np
from gradio_client import Client, handle_file
from PIL import Image as PILImage

from mbodied.agents.sense.sensory_agent import SensoryAgent
from mbodied.types.geometry import Pose6D
from mbodied.types.sample import Sample


class ObjectPoseEstimator3D(SensoryAgent):
    """3D object pose estimation class to interact with a Gradio server for image processing.

    Attributes:
        server_url (str): URL of the Gradio server.
        client (Client): Gradio client to interact with the server.
    """

    def __init__(self, server_url: str = "https://api.mbodi.ai/3d-object-pose-detection") -> None:
        """Initialize the ObjectPoseEstimator3D with the server URL.

        Args:
            server_url (str): The URL of the Gradio server.
        """
        self.server_url = server_url
        self.client = Client(self.server_url)

    @staticmethod
    def save_data(
        color_image_array: np.ndarray,
        depth_image_array: np.ndarray,
        color_image_path: str,
        depth_image_path: str,
        intrinsic_matrix: np.ndarray,
    ) -> None:
        """Save color and depth images as PNG files.

        Args:
            color_image_array (np.ndarray): The color image array.
            depth_image_array (np.ndarray): The depth image array.
            color_image_path (str): The path to save the color image.
            depth_image_path (str): The path to save the depth image.
            intrinsic_matrix (np.ndarray): The intrinsic matrix.

        Example:
            >>> color_image = np.zeros((480, 640, 3), dtype=np.uint8)
            >>> depth_image = np.zeros((480, 640), dtype=np.uint16)
            >>> intrinsic_matrix = np.eye(3)
            >>> ObjectPoseEstimator3D.save_data(color_image, depth_image, "color.png", "depth.png", intrinsic_matrix)
        """
        color_image = PILImage.fromarray(color_image_array, mode="RGB")
        depth_image = PILImage.fromarray(depth_image_array.astype("uint16"), mode="I;16")
        color_image.save(color_image_path, format="PNG")
        depth_image.save(depth_image_path, format="PNG")
        np.save("resources/intrinsic_matrix.npy", intrinsic_matrix)

    def act(
        self,
        rgb_image_path: str,
        depth_image_path: str,
        camera_intrinsics: List[float] | np.ndarray,
        distortion_coeffs: List[float] | None = None,
        aruco_pose_world_frame: Pose6D | None = None,
        object_classes: List[str] | None = None,
        confidence_threshold: float | None = None,
        using_realsense: bool = False,
    ) -> Dict:
        """Capture images using the RealSense camera, process them, and send a request to estimate object poses.

        Args:
            rgb_image_path (str): Path to the RGB image.
            depth_image_path (str): Path to the depth image.
            camera_intrinsics (List[float] | np.ndarray): Path to the camera intrinsics or the intrinsic matrix.
            distortion_coeffs (Optional[List[float]]): List of distortion coefficients.
            aruco_pose_world_frame (Optional[Pose6D]): Pose of the ArUco marker in the world frame.
            object_classes (Optional[List[str]]): List of object classes.
            confidence_threshold (Optional[float]): Confidence threshold for object detection.
            using_realsense (bool): Whether to use the RealSense camera.

        Returns:
            Dict: Result from the Gradio server.

        Example:
            >>> estimator = ObjectPoseEstimator3D()
            >>> result = estimator.act(
            ...     "resources/color_image.png",
            ...     "resources/depth_image.png",
            ...     [911, 911, 653, 371],
            ...     [0.0, 0.0, 0.0, 0.0, 0.0],
            ...     [0.0, 0.2032, 0.0, -90, 0, -90],
            ...     ["Remote Control", "Basket", "Fork", "Spoon", "Red Marker"],
            ...     0.5,
            ...     False,
            ... )
        """
        camera_source = "realsense" if using_realsense else "webcam"

        result = self.client.predict(
            image=handle_file(rgb_image_path),
            depth=handle_file(depth_image_path),
            camera_intrinsics={
                "headers": ["fx", "fy", "cx", "cy"],
                "data": [Sample(camera_intrinsics).to("list")],
                "metadata": None,
            },
            distortion_coeffs={
                "headers": ["k1", "k2", "p1", "p2", "k3"],
                "data": [Sample(distortion_coeffs).to("list")],
                "metadata": None,
            },
            aruco_to_base_offset={
                "headers": ["Z(m)", "Y(m)", "X(m)", "Roll(degrees)", "Pitch(degrees)", "Yaw(degrees)"],
                "data": [Sample(aruco_pose_world_frame).to("list")],
                "metadata": None,
            },
            object_classes={"headers": ["1"], "data": [Sample(object_classes).to("list")], "metadata": None},
            confidence_threshold=confidence_threshold,
            camera_source=camera_source,
        )
        return result  # noqa: RET504
