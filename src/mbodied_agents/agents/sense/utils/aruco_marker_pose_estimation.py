import cv2
import numpy as np
from cv2 import aruco


class ArucoMarkerPoseEstimation:
    """Class for detecting ArUco markers and estimating their pose."""

    @staticmethod
    def detect_markers(
        frame: np.ndarray, intrinsic_matrix: np.ndarray, distortion_coeffs: np.ndarray, marker_size: float
    ) -> tuple:
        """Detect ArUco markers in the given frame.

        Args:
            frame (np.ndarray): The image frame to detect markers in.
            intrinsic_matrix (np.ndarray): Camera intrinsic matrix.
            distortion_coeffs (np.ndarray): Distortion coefficients.
            marker_size (float): The size of the ArUco marker in meters.

        Returns:
            tuple: Corners, centroids, rvecs, tvecs.

        Example:
            >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
            >>> intrinsic_matrix = np.eye(3)
            >>> distortion_coeffs = np.zeros(5)
            >>> marker_size = 0.1
            >>> estimator = ArucoMarkerPoseEstimation()
            >>> estimator.detect_markers(frame, intrinsic_matrix, distortion_coeffs, marker_size)
            ([], [], None, None)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters()
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if corners:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
            for corner in corners:
                cv2.cornerSubPix(gray, corner, winSize=(3, 3), zeroZone=(-1, -1), criteria=criteria)

        mtx = intrinsic_matrix
        dist = distortion_coeffs
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, mtx, dist)

        centroids = [np.mean(corner[0], axis=0) for corner in corners]

        return corners, centroids, rvecs, tvecs


if __name__ == "__main__":
    import doctest

    doctest.testmod()
