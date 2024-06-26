import math

import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import LineString, Point, Polygon


# Rotation matrices and transformations
def rotation_matrix(deg: float) -> np.ndarray:
    """Generate a 2x2 rotation matrix for a given angle in degrees."""
    theta = np.radians(deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def rotation_to_transformation_matrix(R: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a transformation matrix."""
    T = np.eye(4)
    T[:3, :3] = R
    return T


def pose_to_transformation_matrix(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Convert a pose (position and rotation) to a transformation matrix."""
    position_flat = position.ravel()
    return np.block([[rotation, position_flat[:, np.newaxis]], [np.zeros(3), 1]])


def transformation_matrix_to_pose(T: np.ndarray) -> tuple:
    """Extract position and rotation matrix from a transformation matrix."""
    position = T[:3, 3]
    rotation = T[:3, :3]
    return position, rotation


def transformation_matrix_to_position(T: np.ndarray) -> np.ndarray:
    """Extract position from a transformation matrix."""
    return T[:3, 3]


def transformation_matrix_to_rotation(T: np.ndarray) -> np.ndarray:
    """Extract rotation matrix from a transformation matrix."""
    return T[:3, :3]


def rpy_to_rotation_matrix(rpy_rad: np.ndarray) -> np.ndarray:
    """Convert roll, pitch, yaw angles (in radians) to a rotation matrix."""
    roll, pitch, yaw = rpy_rad
    c_roll, s_roll = np.cos(roll), np.sin(roll)
    c_pitch, s_pitch = np.cos(pitch), np.sin(pitch)
    c_yaw, s_yaw = np.cos(yaw), np.sin(yaw)

    R = np.array(
        [
            [c_yaw * c_pitch, -s_yaw * c_roll + c_yaw * s_pitch * s_roll, s_yaw * s_roll + c_yaw * s_pitch * c_roll],
            [s_yaw * c_pitch, c_yaw * c_roll + s_yaw * s_pitch * s_roll, -c_yaw * s_roll + s_yaw * s_pitch * c_roll],
            [-s_pitch, c_pitch * s_roll, c_pitch * c_roll],
        ],
    )
    assert R.shape == (3, 3)
    return R


def rotation_matrix_to_rpy(R: np.ndarray, unit: str = "rad") -> np.ndarray:
    """Convert a rotation matrix to roll, pitch, yaw angles (in radians or degrees)."""
    roll = math.atan2(R[2, 1], R[2, 2])
    pitch = math.atan2(-R[2, 0], math.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    yaw = math.atan2(R[1, 0], R[0, 0])

    if unit == "rad":
        return np.array([roll, pitch, yaw])
    elif unit == "deg":
        return np.array([roll, pitch, yaw]) * 180 / np.pi
    else:
        raise ValueError(f"Unknown unit: {unit}")


def rotation_matrix_to_angular_velocity(R: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to an angular velocity vector."""
    el = np.array([[R[2, 1] - R[1, 2]], [R[0, 2] - R[2, 0]], [R[1, 0] - R[0, 1]]])
    norm_el = np.linalg.norm(el)

    if norm_el > 1e-10:
        w = np.arctan2(norm_el, np.trace(R) - 1) / norm_el * el
    elif R[0, 0] > 0 and R[1, 1] > 0 and R[2, 2] > 0:
        w = np.array([[0, 0, 0]]).T
    else:
        w = np.pi / 2 * np.array([[R[0, 0] + 1], [R[1, 1] + 1], [R[2, 2] + 1]])

    return w.flatten()


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a quaternion."""
    R = np.asarray(R, dtype=np.float64)
    Qxx, Qyx, Qzx = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    Qxy, Qyy, Qzy = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    Qxz, Qyz, Qzz = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    K = np.zeros(R.shape[:-2] + (4, 4), dtype=np.float64)
    K[..., 0, 0] = Qxx - Qyy - Qzz
    K[..., 1, 0] = Qyx + Qxy
    K[..., 1, 1] = Qyy - Qxx - Qzz
    K[..., 2, 0] = Qzx + Qxz
    K[..., 2, 1] = Qzy + Qyz
    K[..., 2, 2] = Qzz - Qxx - Qyy
    K[..., 3, 0] = Qyz - Qzy
    K[..., 3, 1] = Qzx - Qxz
    K[..., 3, 2] = Qxy - Qyx
    K[..., 3, 3] = Qxx + Qyy + Qzz
    K /= 3.0

    q = np.empty(K.shape[:-2] + (4,))
    it = np.nditer(q[..., 0], flags=["multi_index"])
    while not it.finished:
        vals, vecs = np.linalg.eigh(K[it.multi_index])
        q[it.multi_index] = vecs[[3, 0, 1, 2], np.argmax(vals)]
        if q[it.multi_index][0] < 0:
            q[it.multi_index] *= -1
        it.iternext()

    return q


def skew_symmetric_matrix(vector: np.ndarray) -> np.ndarray:
    """Generate a skew-symmetric matrix from a vector."""
    return np.array([[0, -vector[2], vector[1]], [vector[2], 0, -vector[0]], [-vector[1], vector[0], 0]])


def rodrigues_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Compute the rotation matrix from an angular velocity vector."""
    axis_norm = np.linalg.norm(axis)
    if abs(axis_norm - 1) > 1e-6:
        raise ValueError("Norm of axis should be 1.0")

    axis = axis / axis_norm
    angle_rad = angle_rad * axis_norm
    axis_skew = skew_symmetric_matrix(axis)

    return np.eye(3) + axis_skew * np.sin(angle_rad) + axis_skew @ axis_skew * (1 - np.cos(angle_rad))


def unit_vector(vector: np.ndarray) -> np.ndarray:
    """Return the unit vector of the input vector."""
    return vector / np.linalg.norm(vector)


def get_rotation_matrix_from_two_points(p_from: np.ndarray, p_to: np.ndarray) -> np.ndarray:
    """Generate a rotation matrix that aligns one point with another."""
    p_a = np.array([0, 0, 1])
    if np.linalg.norm(p_to - p_from) < 1e-8:
        return np.eye(3)

    p_b = (p_to - p_from) / np.linalg.norm(p_to - p_from)
    v = np.cross(p_a, p_b)
    S = skew_symmetric_matrix(v)
    if np.linalg.norm(v) == 0:
        return np.eye(3)
    else:
        return np.eye(3) + S + S @ S * (1 - np.dot(p_a, p_b)) / (np.linalg.norm(v) ** 2)


# Utility functions
def trim_scale(x: np.ndarray, threshold: float) -> np.ndarray:
    """Scale down the input array if its maximum absolute value exceeds the threshold."""
    x_copy = np.copy(x)
    x_abs_max = np.abs(x_copy).max()
    if x_abs_max > threshold:
        x_copy = x_copy * threshold / x_abs_max
    return x_copy


def soft_squash(x: np.ndarray, x_min: float = -1, x_max: float = 1, margin: float = 0.1) -> np.ndarray:
    """Softly squash the values of an array within a specified range with margins."""

    def threshold_function(z, margin=0.0):
        return margin * (np.exp(2 / margin * z) - 1) / (np.exp(2 / margin * z) + 1)

    x_copy = np.copy(x)
    upper_idx = np.where(x_copy > (x_max - margin))
    x_copy[upper_idx] = threshold_function(x_copy[upper_idx] - (x_max - margin), margin=margin) + (x_max - margin)

    lower_idx = np.where(x_copy < (x_min + margin))
    x_copy[lower_idx] = threshold_function(x_copy[lower_idx] - (x_min + margin), margin=margin) + (x_min + margin)

    return x_copy


def soft_squash_multidim(x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray, margin: float = 0.1) -> np.ndarray:
    """Apply soft squashing to a multi-dimensional array."""
    x_squashed = np.copy(x)
    dim = x.shape[1]
    for d_idx in range(dim):
        x_squashed[:, d_idx] = soft_squash(x[:, d_idx], x_min[d_idx], x_max[d_idx], margin)
    return x_squashed


def squared_exponential_kernel(X1: np.ndarray, X2: np.ndarray, hyp: dict) -> np.ndarray:
    """Compute the squared exponential (SE) kernel between two sets of points."""
    return hyp["g"] * np.exp(-cdist(X1, X2, "sqeuclidean") / (2 * hyp["l"] ** 2))


def leveraged_squared_exponential_kernel(
    X1: np.ndarray,
    X2: np.ndarray,
    L1: np.ndarray,
    L2: np.ndarray,
    hyp: dict,
) -> np.ndarray:
    """Compute the leveraged SE kernel between two sets of points."""
    K = hyp["g"] * np.exp(-cdist(X1, X2, "sqeuclidean") / (2 * hyp["l"] ** 2))
    L = np.cos(np.pi / 2.0 * cdist(L1, L2, "cityblock"))
    return np.multiply(K, L)


def is_point_in_polygon(point: np.ndarray, polygon: Polygon) -> bool:
    """Check if a point is inside a given polygon."""
    point_geom = Point(point) if isinstance(point, np.ndarray) else point
    return polygon.contains(point_geom)


def is_point_feasible(point: np.ndarray, obstacles: list) -> bool:
    """Check if a point is feasible (not inside any obstacles)."""
    return not any(is_point_in_polygon(point, obs) for obs in obstacles)


def is_line_connectable(p1: np.ndarray, p2: np.ndarray, obstacles: list) -> bool:
    """Check if a line between two points is connectable (does not intersect any obstacles)."""
    line = LineString([p1, p2])
    return not any(line.intersects(obs) for obs in obstacles)


def interpolate_constant_velocity_trajectory(
    traj_anchor: np.ndarray,
    velocity: float = 1.0,
    hz: int = 100,
    order: int = np.inf,
) -> tuple:
    """Interpolate a trajectory to achieve constant velocity."""
    num_points = traj_anchor.shape[0]
    dims = traj_anchor.shape[1]

    distances = np.zeros(num_points)
    for i in range(1, num_points):
        distances[i] = np.linalg.norm(traj_anchor[i] - traj_anchor[i - 1], ord=order)

    times_anchor = np.cumsum(distances / velocity)
    interp_len = int(times_anchor[-1] * hz)
    times_interp = np.linspace(0, times_anchor[-1], interp_len)
    traj_interp = np.zeros((interp_len, dims))

    for d in range(dims):
        traj_interp[:, d] = np.interp(times_interp, times_anchor, traj_anchor[:, d])

    return times_interp, traj_interp


def depth_image_to_pointcloud(depth_img: np.ndarray, cam_matrix: np.ndarray) -> np.ndarray:
    """Convert a scaled depth image to a point cloud."""
    fx, fy = cam_matrix[0, 0], cam_matrix[1, 1]
    cx, cy = cam_matrix[0, 2], cam_matrix[1, 2]

    height, width = depth_img.shape
    indices = np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)

    z = depth_img
    x = (indices[..., 1] - cx) * z / fx
    y = (indices[..., 0] - cy) * z / fy

    return np.stack([z, -x, -y], axis=-1)


def compute_view_params(
    camera_pos: np.ndarray,
    target_pos: np.ndarray,
    up_vector: np.ndarray = np.array([0, 0, 1]),
) -> tuple:
    """Compute view parameters (azimuth, distance, elevation, lookat) for a camera."""
    cam_to_target = target_pos - camera_pos
    distance = np.linalg.norm(cam_to_target)

    azimuth = np.rad2deg(np.arctan2(cam_to_target[1], cam_to_target[0]))
    elevation = np.rad2deg(np.arcsin(cam_to_target[2] / distance))
    lookat = target_pos

    zaxis = cam_to_target / distance
    xaxis = np.cross(up_vector, zaxis)
    np.cross(zaxis, xaxis)

    return azimuth, distance, elevation, lookat


def sample_points_in_3d(
    n_sample: int,
    x_range: list,
    y_range: list,
    z_range: list,
    min_dist: float,
    xy_margin: float = 0.0,
) -> np.ndarray:
    """Sample points in 3D space ensuring a minimum distance between them."""
    xyzs = np.zeros((n_sample, 3))
    iter_tick = 0

    for i in range(n_sample):
        while True:
            x_rand = np.random.uniform(x_range[0] + xy_margin, x_range[1] - xy_margin)
            y_rand = np.random.uniform(y_range[0] + xy_margin, y_range[1] - xy_margin)
            z_rand = np.random.uniform(z_range[0], z_range[1])
            xyz = np.array([x_rand, y_rand, z_rand])

            if i == 0 or cdist(xyz.reshape(1, -1), xyzs[:i]).min() > min_dist:
                break

            iter_tick += 1
            if iter_tick > 1000:
                print("[sample_points_in_3d] Exceeded 1000 iterations.")
                break

        xyzs[i] = xyz

    return xyzs


def quintic_trajectory(
    start_pos: np.ndarray,
    start_vel: np.ndarray,
    start_acc: np.ndarray,
    end_pos: np.ndarray,
    end_vel: np.ndarray,
    end_acc: np.ndarray,
    duration: float,
    num_points: int,
    max_velocity: float,
    max_acceleration: float,
) -> tuple:
    """Generate a quintic trajectory with velocity and acceleration constraints."""
    t = np.linspace(0, duration, num_points)
    joint_coeffs = []

    for i in range(6):
        A = np.array(
            [
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 2, 0, 0],
                [duration**5, duration**4, duration**3, duration**2, duration, 1],
                [5 * duration**4, 4 * duration**3, 3 * duration**2, 2 * duration, 1, 0],
                [20 * duration**3, 12 * duration**2, 6 * duration, 2, 0, 0],
            ],
        )
        b = np.array([start_pos[i], start_vel[i], start_acc[i], end_pos[i], end_vel[i], end_acc[i]])
        x = np.linalg.solve(A, b)
        joint_coeffs.append(x)

    positions = np.zeros((num_points, 6))
    velocities = np.zeros((num_points, 6))
    accelerations = np.zeros((num_points, 6))
    jerks = np.zeros((num_points, 6))

    for i in range(num_points):
        for j in range(6):
            positions[i, j] = np.polyval(joint_coeffs[j], t[i])
            velocities[i, j] = np.polyval(np.polyder(joint_coeffs[j]), t[i])
            accelerations[i, j] = np.polyval(np.polyder(np.polyder(joint_coeffs[j])), t[i])
            jerks[i, j] = np.polyval(np.polyder(np.polyder(np.polyder(joint_coeffs[j]))), t[i])

    velocities = np.clip(velocities, -max_velocity, max_velocity)
    accelerations = np.clip(accelerations, -max_acceleration, max_acceleration)

    return positions, velocities, accelerations, jerks


def passthrough_filter(pcd: np.ndarray, axis: int, interval: list) -> np.ndarray:
    """Filter a point cloud along a specified axis within a given interval."""
    mask = (pcd[:, axis] > interval[0]) & (pcd[:, axis] < interval[1])
    return pcd[mask]


def remove_duplicates(pointcloud: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Remove duplicate points from a point cloud within a given threshold."""
    filtered_pointcloud = []
    for point in pointcloud:
        if all(np.linalg.norm(point - existing_point) > threshold for existing_point in filtered_pointcloud):
            filtered_pointcloud.append(point)
    return np.array(filtered_pointcloud)


def remove_duplicates_with_reference(
    pointcloud: np.ndarray,
    reference_point: np.ndarray,
    threshold: float = 0.05,
) -> np.ndarray:
    """Remove duplicate points close to a specific reference point within a given threshold."""
    return np.array([point for point in pointcloud if np.linalg.norm(point - reference_point) < threshold])


def downsample_pointcloud(pointcloud: np.ndarray, grid_size: float) -> np.ndarray:
    """Downsample a point cloud based on a specified grid size."""
    min_vals = pointcloud.min(axis=0)
    grid_pointcloud = np.floor((pointcloud - min_vals) / grid_size).astype(int)
    unique_pointcloud = {
        tuple(pos): original_pos for pos, original_pos in zip(grid_pointcloud, pointcloud, strict=False)
    }
    return np.array(list(unique_pointcloud.values()))
