import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import tempfile
import base64
from io import BytesIO
from PIL import Image as PILImage
from mbodied.types.sense.vision import Image
from mbodied.agents.sense.depth_estimation_agent import DepthEstimationAgent
from mbodied.agents.backends.gradio_backend import GradioBackend


def get_dummy_base64_image():
    """Create a minimal valid base64 image for testing."""
    img = PILImage.new("RGB", (10, 10), color="red")
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


@pytest.fixture
def mock_gradio_backend():
    # Create a temp file that would be a valid location for a numpy file
    temp_file = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    temp_path = temp_file.name
    temp_file.close()
    dummy_image_b64 = get_dummy_base64_image()

    with patch.object(GradioBackend, "__init__", lambda x, model_src=None, **kwargs: None):
        # Return a valid base64 image string and a path
        with patch.object(GradioBackend, "predict", return_value=(dummy_image_b64, temp_path)):
            # Intercept np.load calls to avoid actual file system access
            with patch("numpy.load", return_value=np.zeros((224, 224))):
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

    result_img, depth_array = depth_agent.act(mock_image)

    assert isinstance(result_img, Image)
    assert isinstance(depth_array, np.ndarray)


@pytest.mark.network
def test_real_depth_agent_act():
    # Make real network call.
    agent = DepthEstimationAgent(model_src="https://api.mbodi.ai/sense/")
    result_img, depth_array = agent.act(image=Image("resources/xarm.jpeg", size=(224, 224)))
    assert isinstance(result_img, Image)
    assert isinstance(depth_array, np.ndarray)
