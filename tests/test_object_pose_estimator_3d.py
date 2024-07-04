import pytest
from unittest.mock import Mock, patch
from mbodied.types.sense.vision import Image
from mbodied.types.sense.camera_params import IntrinsicParameters, DistortionParameters
from mbodied.types.geometry import Pose6D
from mbodied.types.sense.world import SceneData
from gradio_client import Client, handle_file
from mbodied.agents.sense.object_pose_estimator_3d import ObjectPoseEstimator3D


@pytest.fixture
def mock_client():
    with patch('gradio_client.Client') as mock_client_class, \
         patch('gradio_client.utils.handle_file') as mock_handle_file:
        
        mock_client_instance = mock_client_class.return_value
        mock_client_instance.predict.return_value = ("annotated_image_path", {
            "Object1": [[0], [0], [0]],
            "Object2": [[1], [1], [1]]
        })
        
        def mock_handle_file_side_effect(filepath_or_url):
            if filepath_or_url is None or filepath_or_url == "existing/path/to/rgb_image.png" or filepath_or_url == "existing/path/to/depth_image.png":
                return {
                    "path": filepath_or_url,
                    "meta": {"_type": "gradio.FileData"},
                    "orig_name": filepath_or_url.split("/")[-1] if filepath_or_url else 'image.png'
                }
            else:
                raise ValueError(f"File {filepath_or_url} does not exist on local filesystem and is not a valid URL.")

        mock_handle_file.side_effect = mock_handle_file_side_effect

        yield mock_client_instance

@pytest.fixture
def estimator(mock_client):
    return ObjectPoseEstimator3D()

def test_act_with_image_array(estimator, mock_client):
    # Mock image data
    rgb_image = Mock(spec=Image)
    rgb_image.path = None
    rgb_image.save = Mock(side_effect=lambda path: setattr(rgb_image, 'path', path))

    depth_image = Mock(spec=Image)
    depth_image.path = None
    depth_image.save = Mock(side_effect=lambda path: setattr(depth_image, 'path', path))

    # Other required parameters
    camera_intrinsics = IntrinsicParameters(focal_length_x=1, focal_length_y=1, optical_center_x=1, optical_center_y=1)
    distortion_coeffs = DistortionParameters(k1=0, k2=0, p1=0, p2=0, k3=0)
    aruco_pose_world_frame = Pose6D(x=0, y=0, z=0, roll=0, pitch=0, yaw=0)
    object_names = ["Object1", "Object2"]

    # Call the act method
    estimator.act(rgb_image, depth_image, camera_intrinsics, distortion_coeffs, aruco_pose_world_frame, object_names)

    # Assert that save was called on both images
    rgb_image.save.assert_called_once_with("resources/color_image.png")
    depth_image.save.assert_called_once_with("resources/depth_image.png")

    # Assert that the client's predict method was called with file paths
    args, kwargs = mock_client.predict.call_args
    assert kwargs['image']['path'].startswith('resources/color_image.png')
    assert kwargs['depth']['path'].startswith('resources/depth_image.png')

def test_act_with_image_path(estimator, mock_client):
    # Mock image data
    rgb_image = Mock(spec=Image)
    rgb_image.path = "existing/path/to/rgb_image.png"
    rgb_image.save = Mock()

    depth_image = Mock(spec=Image)
    depth_image.path = "existing/path/to/depth_image.png"
    depth_image.save = Mock()

    # Other required parameters
    camera_intrinsics = IntrinsicParameters(focal_length_x=1, focal_length_y=1, optical_center_x=1, optical_center_y=1)
    distortion_coeffs = DistortionParameters(k1=0, k2=0, p1=0, p2=0, k3=0)
    aruco_pose_world_frame = Pose6D(x=0, y=0, z=0, roll=0, pitch=0, yaw=0)
    object_names = ["Object1", "Object2"]

    # Call the act method
    estimator.act(rgb_image, depth_image, camera_intrinsics, distortion_coeffs, aruco_pose_world_frame, object_names)

    # Assert that save was not called on either image
    rgb_image.save.assert_not_called()
    depth_image.save.assert_not_called()

    # Assert that the client's predict method was called with file paths
    args, kwargs = mock_client.predict.call_args
    assert kwargs['image']['path'].startswith('existing/path/to/rgb_image.png')
    assert kwargs['depth']['path'].startswith('existing/path/to/depth_image.png')
