import pytest
import numpy as np
import tempfile
import base64
from io import BytesIO
from PIL import Image as PILImage
from unittest.mock import patch, MagicMock
from mbodied.types.sense.world import BBox2D, PixelCoords
from mbodied.types.sense.vision import Image
from mbodied.agents.sense.segmentation_agent import SegmentationAgent


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

    with patch(
        "mbodied.agents.backends.gradio_backend.GradioBackend.__init__", lambda x, model_src=None, **kwargs: None
    ):
        with patch(
            "mbodied.agents.backends.gradio_backend.GradioBackend.predict",
            return_value=(dummy_image_b64, temp_path),
        ):
            with patch("numpy.load", return_value=np.zeros((224, 224, 1))):
                from mbodied.agents.backends.gradio_backend import GradioBackend

                yield GradioBackend(endpoint="http://1.2.3.4:1234")


@pytest.fixture
def segmentation_agent(mock_gradio_backend):
    agent = SegmentationAgent(model_src="http://1.2.3.4:1234/")
    agent.actor = mock_gradio_backend
    return agent


def test_segmentation_agent_initialization(segmentation_agent):
    assert isinstance(segmentation_agent, SegmentationAgent)
    assert segmentation_agent.actor is not None


def test_segmentation_agent_act_with_coordinates(segmentation_agent):
    mock_image = MagicMock(spec=Image)
    mock_image.base64 = "base64encodedimage"

    pixel_coords = PixelCoords(u=800, v=100)
    result_image, masks = segmentation_agent.act(mock_image, pixel_coords)
    assert isinstance(result_image, Image)
    assert isinstance(masks, np.ndarray)


def test_segmentation_agent_act_with_bounding_boxes(segmentation_agent):
    mock_image = MagicMock(spec=Image)
    mock_image.base64 = "base64encodedimage"

    bboxes = [BBox2D(x1=225, y1=196, x2=408, y2=355), BBox2D(x1=378, y1=179, x2=494, y2=236)]
    result_image, masks = segmentation_agent.act(mock_image, bboxes)
    assert isinstance(result_image, Image)
    assert isinstance(masks, np.ndarray)


@pytest.mark.network
def test_real_segmentation_agent_act_with_coordinates():
    agent = SegmentationAgent(model_src="https://api.mbodi.ai/sense/")
    image = Image("resources/bridge_example.jpeg")
    pixel_coords = PixelCoords(u=800, v=100)
    result_image, masks = agent.act(image=image, input_data=pixel_coords)
    assert isinstance(result_image, Image)
    assert isinstance(masks, np.ndarray)


@pytest.mark.network
def test_real_segmentation_agent_act_with_bounding_boxes():
    agent = SegmentationAgent(model_src="https://api.mbodi.ai/sense/")
    image = Image("resources/bridge_example.jpeg")
    bboxes = [BBox2D(x1=225, y1=196, x2=408, y2=355), BBox2D(x1=378, y1=179, x2=494, y2=236)]
    result_image, masks = agent.act(image=image, input_data=bboxes)
    assert isinstance(result_image, Image)
    assert isinstance(masks, np.ndarray)
