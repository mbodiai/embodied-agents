import logging

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    logging.warning("pyrealsense2 is not installed.")
import base64
import json


class RealsenseCamera:
    """A class to handle capturing images from an Intel RealSense camera and encoding camera intrinsics.

    Attributes:
        width (int): Width of the image frames.
        height (int): Height of the image frames.
        fps (int): Frames per second for the video stream.
        pipeline (rs.pipeline): RealSense pipeline for streaming.
        config (rs.config): Configuration for the RealSense pipeline.
        profile (rs.pipeline_profile): Pipeline profile containing stream settings.
        depth_sensor (rs.sensor): Depth sensor of the RealSense camera.
        depth_scale (float): Depth scale factor for the RealSense camera.
        align (rs.align): Object to align depth frames to color frames.
    """

    def __init__(self, width=1280, height=720, fps=30) -> None:
        """Initialize the RealSense camera with the given dimensions and frame rate.

        Args:
            width (int): Width of the image frames.
            height (int): Height of the image frames.
            fps (int): Frames per second for the video stream.
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.profile = self.pipeline.start(self.config)
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        self.align = rs.align(rs.stream.color)

    def capture_realsense_images(self) -> tuple[np.ndarray, np.ndarray, rs.intrinsics, np.ndarray]:
        """Capture color and depth images from the RealSense camera along with intrinsics.

        Returns:
            tuple: color_image (np.ndarray), depth_image (np.ndarray),
                   intrinsics (rs.intrinsics), intrinsics_matrix (np.ndarray)
        """
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not aligned_depth_frame or not color_frame:
                    continue

                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
                intrinsics.model = rs.distortion.inverse_brown_conrady

                intrinsics_matrix = np.array(
                    [[intrinsics.fx, 0, intrinsics.ppx], [0, intrinsics.fy, intrinsics.ppy], [0, 0, 1]]
                )

                return color_image, depth_image, intrinsics, intrinsics_matrix

        finally:
            self.pipeline.stop()

    @staticmethod
    def serialize_intrinsics(intrinsics: rs.intrinsics) -> dict:
        """Serialize camera intrinsics to a dictionary.

        Args:
            intrinsics (rs.intrinsics): The intrinsics object to serialize.

        Returns:
            dict: Serialized intrinsics as a dictionary.
        """
        intrinsics_dict = {
            "width": intrinsics.width,
            "height": intrinsics.height,
            "ppx": intrinsics.ppx,
            "ppy": intrinsics.ppy,
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "model": intrinsics.model,
            "coeffs": list(intrinsics.coeffs),
        }

        for key, value in intrinsics_dict.items():
            if isinstance(value, np.ndarray):
                intrinsics_dict[key] = value.tolist()
            elif isinstance(value, (bytes, bytearray)):
                intrinsics_dict[key] = value.decode()
            elif isinstance(value, object) and not isinstance(value, (int, float, str, list, dict, bool, type(None))):
                intrinsics_dict[key] = str(value)

        return intrinsics_dict

    @staticmethod
    def intrinsics_to_base64(intrinsics: rs.intrinsics) -> str:
        """Convert camera intrinsics to a base64 string.

        Args:
            intrinsics (rs.intrinsics): The intrinsics object to encode.

        Returns:
            str: Base64 encoded string of the intrinsics.
        """
        intrinsics_dict = RealsenseCamera.serialize_intrinsics(intrinsics)
        intrinsics_json = json.dumps(intrinsics_dict)
        intrinsics_base64 = base64.b64encode(intrinsics_json.encode("utf-8")).decode("utf-8")
        return intrinsics_base64

    @staticmethod
    def base64_to_intrinsics(base64_str: str) -> rs.intrinsics:
        """Convert a base64 encoded string to an rs.intrinsics object.

        Args:
            base64_str (str): Base64 encoded string representing camera intrinsics.

        Returns:
            rs.intrinsics: An rs.intrinsics object with the decoded intrinsics data.
        """
        distortion_mapping = {
            "distortion.none": rs.distortion.none,
            "distortion.modified_brown_conrady": rs.distortion.modified_brown_conrady,
            "distortion.inverse_brown_conrady": rs.distortion.inverse_brown_conrady,
            "distortion.ftheta": rs.distortion.ftheta,
            "distortion.brown_conrady": rs.distortion.brown_conrady,
            "distortion.kannala_brandt4": rs.distortion.kannala_brandt4,
        }

        intrinsics_json = base64.b64decode(base64_str).decode("utf-8")
        intrinsics_dict = json.loads(intrinsics_json)

        intrinsics = rs.intrinsics()
        intrinsics.width = intrinsics_dict["width"]
        intrinsics.height = intrinsics_dict["height"]
        intrinsics.ppx = intrinsics_dict["ppx"]
        intrinsics.ppy = intrinsics_dict["ppy"]
        intrinsics.fx = intrinsics_dict["fx"]
        intrinsics.fy = intrinsics_dict["fy"]
        intrinsics.model = distortion_mapping[intrinsics_dict["model"]]
        intrinsics.coeffs = intrinsics_dict["coeffs"]

        return intrinsics

    @staticmethod
    def matrix_and_distortion_to_intrinsics(
        image_height: int, image_width: int, matrix: np.ndarray, coeffs: np.ndarray
    ) -> rs.intrinsics:
        """Convert a 3x3 intrinsic matrix and a 1x5 distortion coefficients array to an rs.intrinsics object.

        Args:
            image_height (int): The height of the image.
            image_width (int): The width of the image.
            matrix (np.ndarray): A 3x3 intrinsic matrix.
            coeffs (np.ndarray): A 1x5 array of distortion coefficients.

        Returns:
            rs.intrinsics: An rs.intrinsics object with the given intrinsics data.

        Example:
            >>> matrix = np.array([[525.0, 0.0, 319.5], [0.0, 525.0, 239.5], [0.0, 0.0, 1.0]])
            >>> coeffs = np.array([0.1, 0.01, 0.001, 0.0001, 0.00001])
            >>> intrinsics = RealsenseCamera.matrix_and_distortion_to_intrinsics(480, 640, matrix, coeffs)
            >>> expected = rs.intrinsics()
            >>> expected.width = 640
            >>> expected.height = 480
            >>> expected.ppx = 319.5
            >>> expected.ppy = 239.5
            >>> expected.fx = 525.0
            >>> expected.fy = 525.0
            >>> expected.model = rs.distortion.none
            >>> expected.coeffs = [0.1, 0.01, 0.001, 0.0001, 0.00001]
            >>> assert (
            ...     intrinsics.width == expected.width
            ...     and intrinsics.height == expected.height
            ...     and intrinsics.ppx == expected.ppx
            ...     and intrinsics.ppy == expected.ppy
            ...     and intrinsics.fx == expected.fx
            ...     and intrinsics.fy == expected.fy
            ...     and intrinsics.model == expected.model
            ...     and np.allclose(intrinsics.coeffs, expected.coeffs)
            ... )
        """
        assert matrix.shape == (3, 3), "Input matrix must be 3x3"
        assert coeffs.shape == (5,), "Distortion coefficients must be a 1x5 array"

        intrinsics = rs.intrinsics()
        intrinsics.width = image_width
        intrinsics.height = image_height
        intrinsics.ppx = matrix[0, 2]
        intrinsics.ppy = matrix[1, 2]
        intrinsics.fx = matrix[0, 0]
        intrinsics.fy = matrix[1, 1]
        intrinsics.model = rs.distortion.none
        intrinsics.coeffs = coeffs.tolist()

        return intrinsics

    @staticmethod
    def pixel_to_3dpoint_realsense(centroid: tuple, depth: float, realsense_intrinsics: object) -> np.ndarray:
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


if __name__ == "__main__":
    import doctest

    doctest.testmod()
