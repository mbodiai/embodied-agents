import pytest
from unittest.mock import patch, MagicMock
from mbodied.types.motion.control import HandControl, JointControl, Pose6D
from mbodied.types.sense.vision import Image
from mbodied.agents.motion.openvla_agent import OpenVlaAgent
from mbodied.agents.backends.gradio_backend import GradioBackend
from mbodied.agents.auto.auto_agent import AutoAgent, get_agent


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
    with patch.object(GradioBackend, "__init__", lambda x, model_src=None, **kwargs: None):
        with patch.object(GradioBackend, "predict", return_value=Image(size=(224, 224))):
            yield GradioBackend(endpoint="http://1.2.3.4:1234")


@pytest.fixture
def depth_agent(mock_depth_gradio_backend):
    agent = AutoAgent(task="sense-depth-estimation", model_src="http://1.2.3.4:1234/")
    agent.actor = mock_openvla_gradio_backend
    return agent


def test_auto_depth_agent_act(depth_agent):
    mock_image = MagicMock(spec=Image)
    mock_image.base64 = "base64encodedimage"

    result = depth_agent.act(mock_image)

    assert isinstance(result, Image)
