import pytest
from unittest.mock import patch, MagicMock
from mbodied.types.motion.control import HandControl, JointControl, Pose6D
from mbodied.types.sense.image import Image
from mbodied.agents.motion.openvla_agent import OpenVlaAgent
from mbodied.agents.backends.gradio_backend import GradioBackend


@pytest.fixture
def mock_gradio_backend():
    with patch.object(GradioBackend, "__init__", lambda x, model_src=None, **kwargs: None):
        with patch.object(GradioBackend, "predict", return_value="[1 2 3 0 0 0 0]"):
            yield GradioBackend(endpoint="http://1.2.3.4:1234")


@pytest.fixture
def openvla_agent(mock_gradio_backend):
    agent = OpenVlaAgent(model_src="http://1.2.3.4:1234/")
    agent.actor = mock_gradio_backend
    return agent


def test_openvla_agent_initialization(openvla_agent):
    assert isinstance(openvla_agent, OpenVlaAgent)
    assert openvla_agent.actor is not None


@pytest.fixture
def openvla_agent_2(mock_gradio_backend):
    agent = OpenVlaAgent(model_src="gradio", model_kwargs={"endpoint": "http://1.2.3.4:1234/"})
    agent.actor = mock_gradio_backend
    return agent


def test_openvla_agent_initialization_2(openvla_agent_2):
    assert isinstance(openvla_agent_2, OpenVlaAgent)
    assert openvla_agent_2.actor is not None


def test_openvla_agent_act(openvla_agent):
    mock_image = MagicMock(spec=Image)
    mock_image.base64 = "base64encodedimage"

    instruction = "move hand forward"
    unnorm_key = "bridge_orig"

    result = openvla_agent.act(instruction, mock_image, unnorm_key)

    expected_action = HandControl(pose=Pose6D(x=1, y=2, z=3, roll=0, pitch=0, yaw=0), grasp=JointControl(value=0))

    assert result == expected_action


@pytest.mark.network
def test_real_openvla_agent_act(openvla_agent):
    openvla_agent = OpenVlaAgent(model_src="https://api.mbodi.ai/community-models/")
    image = Image("resources/xarm.jpeg")
    response = openvla_agent.act("move forward", image)
    assert isinstance(response, HandControl)
