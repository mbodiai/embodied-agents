from typing import List, Union
from PIL import Image as PILImage
import numpy as np
from gradio_client import Client, handle_file
from mbodied.base.sample import Sample
from mbodied.agents.sense.sensory_agent import SensoryAgent
from mbodied.types.geometry import Pose6D
from mbodied.types.sense.vision import Image
from mbodied.types.sense.camera_params import IntrinsicParameters, DistortionParameters
from mbodied.types.sense.world import SceneData, SceneObjects, ObjectsPose


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
        """
        Save color and depth images as PNG files.

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
        rgb_image_path: str = "resources/color_image.png",
        depth_image_path: str = "resources/depth_image.png",
        camera_intrinsics: IntrinsicParameters = IntrinsicParameters(focal_length_x=911.0, focal_length_y=911.0, optical_center_x=653.0, optical_center_y=371.0),
        distortion_coeffs: DistortionParameters = DistortionParameters(k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0),
        aruco_pose_world_frame: Pose6D = Pose6D(z=0.0, y=0.2032, x=0.0, roll=-90.0, pitch=0.0, yaw=-90.0),
        object_names: List[str] = ["Remote Control", "Basket", "Fork", "Spoon", "Red Marker"],
        confidence_threshold: float = 0.3,
        using_realsense: bool = True,
    ) -> SceneData:
        """
        Capture images using the RealSense camera, process them, and send a request to estimate object poses.

        Args:
            rgb_image_path (str): Path to the RGB image.
            depth_image_path (str): Path to the depth image.
            camera_intrinsics (IntrinsicParameters): Camera intrinsic parameters.
            distortion_coeffs (DistortionParameters): Camera distortion coefficients.
            aruco_pose_world_frame (Pose6D): Pose of the ArUco marker in the world frame.
            object_names (List[str]): List of object names in the scene.
            confidence_threshold (float): Confidence threshold for object detection.
            using_realsense (bool): Whether to use the RealSense camera.

        Returns:
            SceneData: Result from the Gradio server.

        Example:
            >>> estimator = ObjectPoseEstimator3D()
            >>> camera_intrinsics = IntrinsicParameters(
            ...     focal_length_x=911.0,
            ...     focal_length_y=911.0,
            ...     optical_center_x=653.0,
            ...     optical_center_y=371.0
            ... )
            >>> distortion_params = DistortionParameters(
            ...     k1=0.0,
            ...     k2=0.0,
            ...     p1=0.0,
            ...     p2=0.0,
            ...     k3=0.0
            ... )
            >>> aruco_pose_world_frame = Pose6D(
            ...     z=0.0,
            ...     y=0.2032,
            ...     x=0.0,
            ...     roll=-90.0,
            ...     pitch=0.0,
            ...     yaw=-90.0
            ... )
            >>> object_names = ["Remote Control", "Basket", "Fork", "Spoon", "Red Marker"]
            >>> result = estimator.act(
            ...     "resources/color_image.png",
            ...     "resources/depth_image.png",
            ...     camera_intrinsics=camera_intrinsics,
            ...     distortion_coeffs=distortion_params,
            ...     aruco_pose_world_frame=aruco_pose_world_frame,
            ...     object_names=object_names,
            ...     confidence_threshold=0.5,
            ...     using_realsense=False
            ... )
        """
        
        if camera_intrinsics is None:
            raise ValueError("Camera intrinsics are required.")
        if distortion_coeffs is None:
            raise Warning("Distortion coefficients not provided.")
        if aruco_pose_world_frame is None:
            raise Warning("ArUco pose not provided.")
        if object_names is None:
            raise ValueError("Object names are required.")
        
        camera_source = "realsense" if using_realsense else "webcam"

        result = self.client.predict(
            image=handle_file(rgb_image_path),
            depth=handle_file(depth_image_path),
            camera_intrinsics={
                "headers": ["fx", "fy", "cx", "cy"],
                "data": [camera_intrinsics.to('list')],
                "metadata": None,
            },
            distortion_coeffs={
                "headers": ["k1", "k2", "p1", "p2", "k3"],
                "data": [distortion_coeffs.to('list')],
                "metadata": None,
            },
            aruco_to_base_offset={
                "headers": ["Z(m)", "Y(m)", "X(m)", "Roll(degrees)", "Pitch(degrees)", "Yaw(degrees)"],
                "data": [aruco_pose_world_frame.to('list')],
                "metadata": None,
            },
            object_classes={
                "headers": ["object_name"],
                "data": [Sample(object_names).to('list')],
                "metadata": None,
            },
            confidence_threshold=confidence_threshold if confidence_threshold else 0.3,
            camera_source=camera_source if camera_source else "realsense",
        )

        annotated_image = result[0]
        object_poses_data = result[1]

        # Parse the object poses
        object_names = []
        object_poses = []

        for object_name, pose_data in object_poses_data.items():
            pose = Pose6D(x=pose_data[0][0], y=pose_data[1][0], z=pose_data[2][0])
            object_names.append(object_name.strip('.'))
            object_poses.append(pose)

        scene_objects = SceneObjects(object_name=object_names)
        objects_pose = ObjectsPose(object_pose=object_poses)

        scene_data = SceneData(image=Image(annotated_image), objects=scene_objects, object_poses=objects_pose)

        return scene_data

    
if __name__ == "__main__":
    import doctest
    doctest.testmod()

    # Create intrinsic parameters
    intrinsic_params = IntrinsicParameters(
        focal_length_x=911.0,
        focal_length_y=911.0,
        optical_center_x=653.0,
        optical_center_y=371.0
    )

    # Create distortion parameters
    distortion_params = DistortionParameters(
        k1=0.0,
        k2=0.0,
        p1=0.0,
        p2=0.0,
        k3=0.0
    )

    # Create aruco pose world frame
    aruco_pose_world_frame = Pose6D(
        z=0.0,
        y=0.2032,
        x=0.0,
        roll=-90.0,
        pitch=0.0,
        yaw=-90.0
    )

    # Create object classes
    object_names = ["Remote Control", "Basket", "Fork", "Spoon", "Red Marker"]

    # Initialize the estimator
    estimator = ObjectPoseEstimator3D()
    
    # Call the act method
    scene_data = estimator.act(
        "resources/color_image.png",
        "resources/depth_image.png",
        camera_intrinsics=intrinsic_params,
        distortion_coeffs=distortion_params,
        aruco_pose_world_frame=aruco_pose_world_frame,
        object_names=object_names,
        confidence_threshold=0.5,
        using_realsense=False
    )

    # Display the annotated image
    scene_data.image.show()

    # Print object names and poses
    for obj, pose in zip(scene_data.objects.object_name, scene_data.object_poses.object_pose):
        print(f"Object: {obj}, Pose: {pose}")
