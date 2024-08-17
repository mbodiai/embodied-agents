from importlib.resources import files
from pathlib import Path
import numpy as np
from PIL import Image as PILImage
import pytest
from mbodied.types.sense.depth import Depth
from mbodied.types.sense.image import Image


@pytest.fixture
def image_path():
    return "resources/color_image.png"


@pytest.fixture
def depth_path():
    return "resources/depth_image.png"


def test_depth_initialization():
    depth = Depth(mode="I", points=None, array=None)
    assert depth.mode == "I"
    assert depth.points is None
    assert depth.array is None


def test_depth_from_pil():
    pil_image = PILImage.new("RGB", (100, 100))
    depth = Depth.from_pil(pil_image, mode="I")
    assert depth.mode == "I"
    assert depth.points is None
    assert depth.array is not None
    assert isinstance(depth.array, np.ndarray)
    assert depth.array.dtype == np.uint16


def test_rgb(depth_path, image_path):
    depth = Depth(
        path=depth_path,
        mode="I",
        encoding="png",
        size=(1280, 720),
        rgb=Image(path=image_path, mode="RGB", encoding="png"),
    )

    print("depth.rgb:", depth.rgb.mode)
    assert depth.rgb is not None
    assert depth.rgb.path is not None
    assert depth.rgb.mode == "RGB"
    assert depth.rgb.encoding == "png"


def test_load_from_path(depth_path):
    depth = Depth(path=depth_path, mode="I", encoding="png")
    assert depth.mode == "I"
    assert depth.points is None
    assert depth.array is not None
    assert isinstance(depth.array, np.ndarray)


def test_depth_pil_computed_field():
    depth_array = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
    depth = Depth(array=depth_array)
    pil_image = depth.pil

    assert pil_image.size == (100, 100)


def test_depth_rgb_computed_field():
    depth_array = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
    depth = Depth(array=depth_array)
    rgb_image = depth.rgb

    assert rgb_image.mode == "RGB"
    assert isinstance(rgb_image.array, np.ndarray)
    assert rgb_image.array.shape == (100, 100, 3)  # Check for RGB shape


def test_depth_base64_computed_field():
    depth_array = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
    depth = Depth(array=depth_array)
    base64_str = depth.base64

    assert isinstance(base64_str, str)


def test_depth_url_computed_field():
    depth_array = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
    depth = Depth(array=depth_array)
    url_str = depth.url

    assert isinstance(url_str, str)
    assert url_str.startswith("data:image/png;base64,")
