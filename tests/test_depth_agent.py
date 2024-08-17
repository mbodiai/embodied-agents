import pytest
from unittest.mock import patch, MagicMock
from mbodied.types.sense.image import Image
from mbodied.agents.sense.depth_estimation_agent import DepthEstimationAgent
from mbodied.agents.sense.object_detection_agent import ObjectDetectionAgent
from mbodied.agents.backends.gradio_backend import GradioBackend


@pytest.fixture
def mock_gradio_backend():
    with patch.object(GradioBackend, "__init__", lambda x, model_src=None, **kwargs: None):
        with patch.object(GradioBackend, "predict", return_value=Image(size=(224, 224))):
            yield GradioBackend(endpoint="http://1.2.3.4:1234")


@pytest.fixture
def depth_agent(mock_gradio_backend):
    agent = DepthEstimationAgent(model_src="http://1.2.3.4:1234/")
    agent.actor = mock_gradio_backend
    return agent


def test_depth_agent_initialization(depth_agent):
    assert isinstance(depth_agent, DepthEstimationAgent)
    assert depth_agent.actor is not None


def test_depth_agent_act(depth_agent):
    mock_image = MagicMock(spec=Image)
    mock_image.base64 = "base64encodedimage"

    result = depth_agent.act(mock_image)

    assert isinstance(result, Image)


@pytest.mark.network
def test_real_depth_agent_act():
    # Make real network call.
    agent = DepthEstimationAgent(model_src="https://api.mbodi.ai/sense/")
    result = agent.act(image=Image("resources/xarm.jpeg", size=(224, 224)))
    assert isinstance(result, Image)
