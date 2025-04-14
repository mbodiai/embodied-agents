import pytest
from unittest.mock import patch, MagicMock
from mbodied.types.motion.control import HandControl, JointControl, Pose6D
from mbodied.types.sense.vision import Image
from mbodied.agents.motion.openvla_agent import OpenVlaAgent
from mbodied.agents.backends.gradio_backend import GradioBackend
from mbodied.agents.auto.auto_agent import AutoAgent, get_agent
import numpy as np
import tempfile
import base64
from io import BytesIO
from PIL import Image as PILImage


def get_dummy_base64_image():
    """Create a minimal valid base64 image for testing."""
    img = PILImage.new("RGB", (10, 10), color="red")
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


@pytest.fixture
def mock_openvla_gradio_backend():
    with patch.object(GradioBackend, "__init__", lambda x, model_src=None, **kwargs: None):
        with patch.object(GradioBackend, "predict", return_value="[1 2 3 0 0 0 0]"):
            yield GradioBackend(endpoint="http://1.2.3.4:1234")


@pytest.fixture
def auto_openvla_agent(mock_openvla_gradio_backend):
    agent = AutoAgent(task="motion-openvla", model_src="http://1.2.3.4:1234/")
    agent.actor = mock_openvla_gradio_backend
    return agent


def test_auto_openvla_agent_act(auto_openvla_agent):
    mock_image = MagicMock(spec=Image)
    mock_image.base64 = "base64encodedimage"
    instruction = "move hand forward"
    unnorm_key = "bridge_orig"

    result = auto_openvla_agent.act(instruction, mock_image, unnorm_key)

    expected_action = HandControl(pose=Pose6D(x=1, y=2, z=3, roll=0, pitch=0, yaw=0), grasp=JointControl(value=0))

    assert result == expected_action


@pytest.fixture
def auto_openvla_agent_get_method(mock_openvla_gradio_backend):
    agent = get_agent(task="motion-openvla", model_src="http://1.2.3.4:1234/")
    agent.actor = mock_openvla_gradio_backend
    return agent


def test_auto_openvla_agent_act(auto_openvla_agent_get_method):
    mock_image = MagicMock(spec=Image)
    mock_image.base64 = "base64encodedimage"
    instruction = "move hand forward"
    unnorm_key = "bridge_orig"

    result = auto_openvla_agent_get_method.act(instruction, mock_image, unnorm_key)

    expected_action = HandControl(pose=Pose6D(x=1, y=2, z=3, roll=0, pitch=0, yaw=0), grasp=JointControl(value=0))

    assert result == expected_action


@pytest.fixture
def mock_depth_gradio_backend():
    # Create a temp path that would be a valid location for a numpy file
    temp_path = tempfile.mktemp(suffix=".npy")
    dummy_image_b64 = get_dummy_base64_image()

    with patch.object(GradioBackend, "__init__", lambda x, model_src=None, **kwargs: None):
        # Return a valid base64 image string and a path
        with patch.object(GradioBackend, "predict", return_value=(dummy_image_b64, temp_path)):
            # Intercept np.load calls to avoid actual file system access
            with patch("numpy.load", return_value=np.zeros((224, 224))):
                yield GradioBackend(endpoint="http://1.2.3.4:1234")


@pytest.fixture
def depth_agent(mock_depth_gradio_backend):
    agent = AutoAgent(task="sense-depth-estimation", model_src="http://1.2.3.4:1234/")
    agent.actor = mock_depth_gradio_backend
    return agent


def test_auto_depth_agent_act(depth_agent):
    mock_image = MagicMock(spec=Image)
    mock_image.base64 = "base64encodedimage"

    result, depth_array = depth_agent.act(mock_image)

    assert isinstance(result, Image)
    assert isinstance(depth_array, np.ndarray)
