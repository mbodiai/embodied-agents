import cv2
import numpy as np
from scipy.optimize import least_squares


def estimate_intrinsic_parameters(
    unscaled_depth_map: np.ndarray,
    image: np.ndarray,
    seg_map: np.ndarray = None,
) -> dict:
    """Estimate intrinsic camera parameters given an unscaled depth map, image, and optionally a semantic segmentation map.

    Args:
        unscaled_depth_map (np.ndarray): Unscaled depth map.
        image (np.ndarray): Image corresponding to the depth map.
        seg_map (np.ndarray, optional): Semantic segmentation map. Defaults to None.

    Returns:
        dict: Estimated intrinsic parameters including focal lengths and principal point.

    Example:
        >>> import cv2
        >>> import numpy as np
        >>> from mbodied.agents.sense.utils.estimate_intrinsics import estimate_intrinsic_parameters
        >>> unscaled_depth_map = np.random.rand(480, 640)
        >>> image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        >>> estimate_intrinsic_parameters(unscaled_depth_map, image)
        {'fx': 1.0, 'fy': 1.0, 'cx': 320.0, 'cy': 240.0}
    """
    # Extract feature points from the image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)

    if seg_map is not None:
        # Filter feature points using the segmentation map
        filtered_keypoints = []
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if seg_map[y, x] != 0:  # Use only points within the relevant segmented areas
                filtered_keypoints.append(kp)
        keypoints = filtered_keypoints

    if not keypoints:
        raise ValueError("No valid feature points found.")

    # Get the 2D coordinates of the filtered feature points
    points_2d = np.array([kp.pt for kp in keypoints], dtype=np.float32)

    # Get the 3D coordinates of the filtered feature points using the depth map
    h, w = unscaled_depth_map.shape
    fx, fy = 1.0, 1.0  # Initial guess for focal lengths
    cx, cy = w / 2, h / 2  # Initial guess for principal point

    points_3d = []
    for pt in points_2d:
        x, y = int(pt[0]), int(pt[1])
        z = unscaled_depth_map[y, x]
        if z > 0:  # Valid depth
            X = (x - cx) * z / fx
            Y = (y - cy) * z / fy
            points_3d.append([X, Y, z])

    if len(points_3d) == 0:
        raise ValueError("No valid 3D points found.")

    points_3d = np.array(points_3d, dtype=np.float32)
    points_2d = points_2d[: points_3d.shape[0]]  # Match the number of points

    # Wrap points in lists for cv2.calibrateCamera
    object_points = [points_3d]
    image_points = [points_2d]

    # Create initial guess for the intrinsic matrix
    initial_intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    # Use OpenCV's calibration function with RANSAC
    _, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        (w, h),
        initial_intrinsic_matrix,
        None,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_RATIONAL_MODEL,
    )

    # Bundle adjustment refinement
    def reprojection_error(params, points_3d, points_2d):
        fx, fy, cx, cy = params
        projected_points = []
        for p in points_3d:
            x, y, z = p
            u = fx * x / z + cx
            v = fy * y / z + cy
            projected_points.append([u, v])
        projected_points = np.array(projected_points)
        return (projected_points - points_2d).ravel()

    initial_params = [camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]]
    res = least_squares(reprojection_error, initial_params, args=(points_3d, points_2d))

    refined_params = res.x
    return {
        "fx": refined_params[0],
        "fy": refined_params[1],
        "cx": refined_params[2],
        "cy": refined_params[3],
    }


if __name__ == "__main__":
    import doctest

    doctest.testmod()
