import pytest
from unittest.mock import patch, MagicMock
from mbodied.types.sense.vision import Image
from mbodied.agents.sense.depth_estimation_agent import DepthEstimationAgent
from mbodied.agents.sense.object_detection_agent import ObjectDetectionAgent
from mbodied.agents.backends.gradio_backend import GradioBackend
from mbodied.types.sense.world import World


@pytest.fixture
def mock_gradio_backend():
    with patch.object(GradioBackend, "__init__", lambda x, model_src=None, **kwargs: None):
        with patch.object(GradioBackend, "predict", return_value=(Image((224, 224)), World().dict())):
            yield GradioBackend(endpoint="http://1.2.3.4:1234")


@pytest.fixture
def depth_agent(mock_gradio_backend):
    agent = DepthEstimationAgent(model_src="http://1.2.3.4:1234/")
    agent.actor = mock_gradio_backend
    return agent


@pytest.fixture
def yolo_agent(mock_gradio_backend):
    agent = ObjectDetectionAgent(model_src="http://1.2.3.4:1234/")
    agent.actor = mock_gradio_backend
    return agent


def test_object_detection_agent_initialization(yolo_agent):
    assert isinstance(yolo_agent, ObjectDetectionAgent)
    assert yolo_agent.actor is not None


def test_object_detection_agent_act(yolo_agent):
    mock_image = MagicMock(spec=Image)
    mock_image.base64 = "base64encodedimage"

    result = yolo_agent.act(mock_image, "person")
    assert isinstance(result, World)

    result = yolo_agent.act(mock_image, ["person", "car"])
    assert isinstance(result, World)

@pytest.mark.network
def test_real_object_detection_agent_act():
    agent = ObjectDetectionAgent(model_src="https://api.mbodi.ai/sense/")
    result = agent.act(image=Image("resources/bridge_example.jpeg", size=(224, 224)), objects=["spoon", "bowl"])
    assert isinstance(result, World)
