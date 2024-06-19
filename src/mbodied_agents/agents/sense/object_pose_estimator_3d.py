from typing import Optional, Dict, List, Union
import numpy as np
from gradio_client import Client, file
from mbodied_agents.types.sense.vision import Image
from mbodied_agents.agents.sense.sensory_agent import SensoryAgent
from mbodied_agents.types.geometry import Pose6D


class ObjectPoseEstimator3D(SensoryAgent):
    """A client class to interact with a Gradio server for image processing.

    Attributes:
        server_url (str): URL of the Gradio server.
    """

    def __init__(self, server_url: str = "http://80.188.223.202:11103/") -> None:
        """Initialize the ObjectPoseEstimator3D with the server URL.

        Args:
            server_url (str): The URL of the Gradio server.
        """
        self.server_url = server_url
        self.client = Client(self.server_url)

    def format_parameters(self, parameter: Union[np.ndarray, list], param_type: str) -> dict:
        """Format the given parameter based on its type to pass it to the Gradio server.

        Args:
            parameter (Union[np.ndarray, list]): The parameter to be formatted.
            param_type (str): The type of the parameter ("intrinsics", "distortion_coeffs", "object_classes", "target_frame_offset").

        Returns:
            dict: Formatted parameter.

        Example:
            >>> estimator = ObjectPoseEstimator3D()
            >>> param = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            >>> estimator.format_parameters(param, "intrinsics")
            {'headers': ['1', '2', '3'], 'data': [[1, 2, 3], [4, 5, 6], [7, 8, 9]], 'metadata': None}
        """
        if param_type == "intrinsics":
            return {"headers": ["1", "2", "3"], "data": parameter.tolist(), "metadata": None}
        elif param_type == "distortion_coeffs":
            return {"headers": ["1", "2", "3", "4", "5"], "data": [parameter], "metadata": None}
        elif param_type == "object_classes":
            return {"headers": ["1", "2", "3", "4", "5"], "data": [parameter], "metadata": None}
        elif param_type == "target_frame_offset":
            return {
                "headers": ["X(m)", "Y(m)", "Z(m)", "Roll(degrees)", "Pitch(degrees)", "Yaw(degrees)"],
                "data": [parameter],
                "metadata": None,
            }
        else:
            raise ValueError("Invalid parameter type specified")

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
        color_image = Image.fromarray(color_image_array, mode="RGB")
        depth_image = Image.fromarray(depth_image_array.astype("uint16"), mode="I;16")
        color_image.save(color_image_path, format="PNG")
        depth_image.save(depth_image_path, format="PNG")
        np.save("intrinsic_matrix.npy", intrinsic_matrix)

    def act(
        self,
        rgb_image_path: str,
        depth_image_path: str,
        camera_intrinsics: Union[str, np.ndarray],
        distortion_coeffs: Optional[List[float]] = None,
        aruco_pose_world_frame: Optional[Pose6D] = None,
        object_classes: Optional[List[str]] = None,
        confidence_threshold: Optional[float] = None,
        using_realsense: bool = False,
    ) -> Dict:
        """Capture images using the RealSense camera, process them, and send a request to estimate object poses.

        Args:
            rgb_image_path (str): Path to the RGB image.
            depth_image_path (str): Path to the depth image.
            camera_intrinsics (Union[str, np.ndarray]): Path to the camera intrinsics or the intrinsic matrix.
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
            ...     "color_image.png",
            ...     "depth_image.png",
            ...     "intrinsic_matrix.npy",
            ...     [0.0, 0.0, 0.0, 0.0, 0.0],
            ...     [0.0, 0.2032, 0.0, -90, 0, -90],
            ...     ["Remote Control", "Basket", "Fork", "Spoon", "Red Marker"],
            ...     0.5,
            ...     False,
            ... )
        """
        if isinstance(camera_intrinsics, str):
            intrinsic_matrix = np.load(camera_intrinsics)
        else:
            intrinsic_matrix = camera_intrinsics

        intrinsics_list = self.format_parameters(intrinsic_matrix, "intrinsics")
        distortion_coeffs_list = self.format_parameters(distortion_coeffs, "distortion_coeffs")
        object_classes_list = self.format_parameters(object_classes, "object_classes")
        aruco_pose_world_frame_list = self.format_parameters(aruco_pose_world_frame, "target_frame_offset")

        camera_source = "realsense" if using_realsense else "webcam"

        result = self.client.predict(
            image=file(rgb_image_path),
            depth=file(depth_image_path),
            camera_intrinsics=intrinsics_list,
            distortion_coeffs=distortion_coeffs_list,
            target_frame_offset=aruco_pose_world_frame_list,
            object_classes=object_classes_list,
            confidence_threshold=confidence_threshold,
            camera_source=camera_source,
            api_name="/predict",
        )
        print(result)
        return result


if __name__ == "__main__":
    import doctest

    doctest.testmod()
