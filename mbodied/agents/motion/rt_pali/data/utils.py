def scale_pose_data(pose_data, statistics):
    """Scales the pose data based on provided statistics.

    Parameters:
        pose_data (dict): The pose data with keys ['terminated','x', 'y', 'z', 'roll', 'pitch', 'yaw', 'grasp'].
        statistics (dict): The statistics containing 'min' and 'max' values for normalization.

    Returns:
        dict: Scaled pose data.
    """
    scaled_data = {}
    for key, value in pose_data.items():
        if key == 'terminated' or key == 'grasp':
            scaled_data[key] = value
        else:
            stats = statistics[key]
            scaled_data[key] = (value - stats['min']) / \
                (stats['max'] - stats['min'])
    return scaled_data


def descale_pose_data(scaled_data, statistics):
    """Descale the normalized pose data back to original scale based on provided statistics.

    Parameters:
        scaled_data (dict): The scaled pose data with keys ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'grasp'].
        statistics (dict): The statistics containing 'min' and 'max' values for the original scale.

    Returns:
        dict: Descaled (original) pose data.
    """
    descaled_data = {}
    for key, value in scaled_data.items():
        stats = statistics[key]
        if key == 'grasp' or key == 'terminated':
            descaled_data[key] = round(value)
        else:
            descaled_data[key] = value * \
                (stats['max'] - stats['min']) + stats['min']
    return descaled_data