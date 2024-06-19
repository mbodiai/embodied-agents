import cv2
import numpy as np
import logging

try:
    import pyrealsense2 as rs
except ImportError:
    logging.warning("pyrealsense2 is not installed.")
from cv2 import aruco
from mbodied_agents.hardware.realsense_camera import RealsenseCamera
from mbodied_agents.agents.sense.utils.aruco_marker_pose_estimation import ArucoMarkerPoseEstimation


class ObjectPoseEstimation:
    """A class to estimate object poses using ArUco markers.

    Attributes:
        color_image (np.ndarray): The color image captured.
        depth_image (np.ndarray): The depth image captured.
        intrinsics (object): Realsense Camera intrinsics object.
        centroids (dict): Centroids of detected objects.
        marker_size (float): The size of the ArUco marker in meters.
        depth_scale (float): Depth scale factor for the RealSense camera.
        intrinsic_matrix (np.ndarray): Camera intrinsic matrix.
    """

    def __init__(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        intrinsic_matrix: np.ndarray,
        centroids: dict = None,
        marker_size: float = 0.1,
        depth_scale: float = 0.001,
        distortion_coeffs: np.ndarray = None,
        aruco_to_base_offset: np.ndarray = np.array([0, 0.2032, 0, -np.pi / 2, 0, -np.pi / 2]),
        camera_source: str = "realsense",
    ) -> None:
        """Initialize the ArucoMarkerBasedObjectPoseEstimation with images, intrinsics, and centroids.

        Args:
            color_image (np.ndarray): The color image captured.
            depth_image (np.ndarray): The depth image captured.
            intrinsic_matrix (np.ndarray): Camera intrinsic matrix.
            centroids (dict, optional): Centroids of detected object names and their centroids in the image frame. Defaults to None.
            marker_size (float): The size of the ArUco marker in meters. Defaults to 0.1.
            depth_scale (float): The depth scale factor for the RealSense camera. Defaults to 0.001.
            distortion_coeffs (np.ndarray, optional): Distortion coefficients for the camera. Defaults to None.
            pose_offset (np.ndarray, optional): Offset for the pose. Defaults to np.array([0, 0.2032, 0, -np.pi/2, 0, -np.pi/2]).
            camera_source (str): The device type. Defaults to "realsense".
        """
        self.color_image = color_image
        self.depth_image = depth_image
        self.intrinsic_matrix = intrinsic_matrix
        self.centroids = centroids if centroids is not None else {}
        self.marker_size = marker_size
        self.depth_scale = depth_scale
        self.camera_source = camera_source
        self.distortion_coeffs = distortion_coeffs
        self.aruco_to_base_offset = aruco_to_base_offset

        image_height, image_width = self.color_image.shape[:2]

        if self.camera_source == "realsense":
            self.distortion_coeffs = np.zeros((5,))
            self.realsense_instrinsics = RealsenseCamera.matrix_and_distortion_to_intrinsics(
                image_height=image_height,
                image_width=image_width,
                matrix=self.intrinsic_matrix,
                coeffs=self.distortion_coeffs,
            )

    def pose_estimation(self) -> dict:
        """Detect the pose of objects using ArUco markers.

        Returns:
            dict: Differences in object positions relative to the ArUco marker.

        Example:
            >>> estimator = ArucoMarkerBasedObjectPoseEstimation(color_image, depth_image, intrinsic_matrix)
            >>> estimator.pose_detector()
        """
        translational_offset = self.aruco_to_base_offset[:3]
        rotation_z_offset = np.radians(self.aruco_to_base_offset[3])
        rotation_y_offset = np.radians(self.aruco_to_base_offset[4])
        rotation_x_offset = np.radians(self.aruco_to_base_offset[5])

        corners, aruco_centroid, rvecs, tvecs = ArucoMarkerPoseEstimation.detect_markers(
            self.color_image, self.intrinsic_matrix, self.distortion_coeffs, self.marker_size
        )
        if not corners:
            print("No ArUco markers detected.")
            return None

        depth_aruco_marker = self.get_depth(self.depth_image, self.depth_scale, aruco_centroid[0])

        rotation_vector_z = np.array([rotation_z_offset, 0, 0])
        rotation_vector_y = np.array([0, rotation_y_offset, 0])
        rotation_vector_x = np.array([0, 0, rotation_x_offset])

        camera_to_aruco_rotation = cv2.Rodrigues(rvecs[0])[0]
        rotation_matrix_z = cv2.Rodrigues(rotation_vector_z)[0]
        rotation_matrix_y = cv2.Rodrigues(rotation_vector_y)[0]
        rotation_matrix_x = cv2.Rodrigues(rotation_vector_x)[0]

        aruco_to_base_rotation = rotation_matrix_z @ rotation_matrix_y @ rotation_matrix_x
        camera_to_base = camera_to_aruco_rotation @ aruco_to_base_rotation
        new_rvecs = cv2.Rodrigues(camera_to_base)[0]

        translation_world_frame = translational_offset.reshape((3, 1))
        translation_camera_frame = camera_to_base @ translation_world_frame

        if self.camera_source == "realsense":
            world_coords = self.pixel_to_3dpoint_realsense(
                aruco_centroid[0], depth_aruco_marker, self.realsense_instrinsics
            )
        else:
            fx, fy, cx, cy = (
                self.intrinsic_matrix[0, 0],
                self.intrinsic_matrix[1, 1],
                self.intrinsic_matrix[0, 2],
                self.intrinsic_matrix[1, 2],
            )
            world_coords = self.pixel_to_3dpoint_generic(aruco_centroid[0], depth_aruco_marker, fx, fy, cx, cy)

        new_tvecs = world_coords.reshape((3, 1)) + translation_camera_frame

        object_depths = {
            name: self.get_depth(self.depth_image, self.depth_scale, centroid)
            for name, centroid in self.centroids.items()
        }

        if self.camera_source == "realsense":
            object_3d_points = {
                name: self.pixel_to_3dpoint_realsense(centroid, depth, self.realsense_instrinsics)
                for name, centroid, depth in zip(self.centroids.keys(), self.centroids.values(), object_depths.values())
            }
        else:
            object_3d_points = {
                name: self.pixel_to_3dpoint_generic(centroid, depth, fx, fy, cx, cy)
                for name, centroid, depth in zip(self.centroids.keys(), self.centroids.values(), object_depths.values())
            }

        if len(object_3d_points) < 1:
            print("Not enough centroids detected.")
            return None

        world_coords_object = {name: point.reshape((3, 1)) for name, point in object_3d_points.items()}

        rotation_matrix = cv2.Rodrigues(new_rvecs)[0]
        transformed_tvecs = rotation_matrix @ new_tvecs
        transformed_world_coords_object = {name: rotation_matrix @ coord for name, coord in world_coords_object.items()}

        base_to_objects = {
            name: (transformed - transformed_tvecs)[[2, 1, 0]]
            for name, transformed in transformed_world_coords_object.items()
        }

        for difference in base_to_objects.values():
            difference[1] = -difference[1]

        for name, difference in base_to_objects.items():
            print(f"Difference for {name}: {difference}")

        annotated_image = self.color_image.copy()

        annotated_image = cv2.drawFrameAxes(
            annotated_image, self.intrinsic_matrix, self.distortion_coeffs, new_rvecs, world_coords, 0.1
        )
        annotated_image = cv2.drawFrameAxes(
            annotated_image, self.intrinsic_matrix, self.distortion_coeffs, new_rvecs, new_tvecs, 0.1
        )

        for coord in world_coords_object.values():
            annotated_image = cv2.drawFrameAxes(
                annotated_image, self.intrinsic_matrix, self.distortion_coeffs, new_rvecs, coord, 0.1
            )

        cv2.imwrite("image_with_axes.png", annotated_image)
        np.save("differences.npy", base_to_objects)

        return base_to_objects, annotated_image

    def get_depth(self, depth_image: np.ndarray, depth_scale: float, centroid: tuple) -> float:
        """Get the depth value at the given centroid from the depth image.

        Args:
            depth_image (np.ndarray): The depth image.
            depth_scale (float): The depth scale factor for the RealSense camera.
            centroid (tuple): The (u, v) coordinates of the centroid.

        Returns:
            float: The depth value at the centroid.

        Example:
            >>> estimator = ArucoMarkerBasedObjectPoseEstimation(color_image, depth_image, intrinsic_matrix)
            >>> estimator.get_depth(depth_image, 0.001, (320, 240))
        """
        u, v = centroid
        depth = depth_image[int(v), int(u)] * depth_scale
        return depth[0] if isinstance(depth, np.ndarray) else depth

    def pixel_to_3dpoint_realsense(self, centroid: tuple, depth: float, realsense_intrinsics: object) -> np.ndarray:
        """Convert a 2D pixel coordinate to a 3D point using the depth and camera intrinsics.

        Args:
            centroid (tuple): The (u, v) coordinates of the pixel.
            depth (float): The depth value at the pixel.
            realsense_intrinsics (object): Camera intrinsics.

        Returns:
            np.ndarray: The 3D coordinates of the point.

        Example:
            >>> estimator = ArucoMarkerBasedObjectPoseEstimation(color_image, depth_image, intrinsic_matrix)
            >>> estimator.pixel_to_3dpoint_realsense((320, 240), 1.5, realsense_intrinsics)
        """
        u, v = centroid
        points = rs.rs2_deproject_pixel_to_point(realsense_intrinsics, [u, v], depth)
        return np.array(points)

    def pixel_to_3dpoint_generic(
        self, centroid: tuple, depth: float, fx: float, fy: float, cx: float, cy: float
    ) -> np.ndarray:
        """Convert a 2D pixel coordinate to a 3D point using the depth and camera intrinsics for a different device.

        Args:
            centroid (tuple): The (u, v) coordinates of the pixel.
            depth (float): The depth value at the pixel.
            fx (float): Focal length of the camera in x direction.
            fy (float): Focal length of the camera in y direction.
            cx (float): Principal point x coordinate.
            cy (float): Principal point y coordinate.

        Returns:
            np.ndarray: The 3D coordinates of the point.

        Example:
            >>> estimator = ArucoMarkerBasedObjectPoseEstimation(color_image, depth_image, intrinsic_matrix)
            >>> estimator.pixel_to_3dpoint_generic((320, 240), 1.5, 618.0, 618.0, 320.0, 240.0)
        """
        u, v = centroid
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        return np.array([x, y, z])


if __name__ == "__main__":
    import doctest

    doctest.testmod()
