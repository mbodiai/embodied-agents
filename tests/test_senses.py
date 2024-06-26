# Copyright 2024 mbodi ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import io
import os
import numpy as np
import pytest
import cv2
from PIL import Image as PILImage
from mbodied.types.sense.vision import Image
import tempfile


@pytest.fixture
def temp_file():
    """Create a temporary file and provide its path to tests, remove it after the test."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        yield tmp_file.name
    os.remove(tmp_file.name)


def test_create_image_with_array():
    array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    img = Image(array=array)
    assert img.size == (640, 480)
    assert img.base64 != ""


def test_create_image_with_path(temp_file):
    test_image_path = temp_file
    PILImage.new("RGB", (640, 480)).save(test_image_path, format="PNG")
    img = Image(test_image_path)
    assert img.size is not None
    assert img.base64 != ""
    assert img.array is not None


def test_create_image_with_base64():
    buffer = io.BytesIO()
    PILImage.new("RGB", (10, 10)).save(buffer, format="PNG")
    encoded_str = base64.b64encode(buffer.getvalue()).decode()
    img = Image(encoded_str, encoding="png")
    assert img.size is not None
    # Decode both base64 strings to images and compare
    original_image_data = io.BytesIO(base64.b64decode(encoded_str))
    original_image = PILImage.open(original_image_data)
    test_image_data = io.BytesIO(base64.b64decode(img.base64))
    test_image = PILImage.open(test_image_data)
    # Compare images pixel by pixel
    assert list(original_image.getdata()) == list(test_image.getdata())


def test_base64_encode():
    array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    img = Image(array=array)
    encoded_base64 = img.base64
    assert encoded_base64 != ""
    assert isinstance(encoded_base64, str)


def test_repr():
    array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    img = Image(array=array)
    dumped_data = img.__repr__()
    assert "base64" in dumped_data
    assert "array" not in dumped_data


def test_resize():
    array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    img = Image(array=array, size=(224, 224))
    assert img.size == (224, 224)
    assert img.array.shape == (224, 224, 3)


def test_space():
    array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    img = Image(array=array)
    assert img.space().sample().shape == (640, 480, 3)


def test_encode_decode_array():
    array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    img = Image(array, encoding="png")
    encoded_base64 = img.base64
    decoded_array_bgr = cv2.imdecode(np.frombuffer(base64.b64decode(encoded_base64), np.uint8), cv2.IMREAD_COLOR)
    decoded_array = cv2.cvtColor(decoded_array_bgr, cv2.COLOR_BGR2RGB)
    assert decoded_array.shape == (480, 640, 3)
    assert np.array_equal(decoded_array, array)


def test_png_tojpeg(temp_file):
    array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    img = Image(array, encoding="png")
    img.save(temp_file, encoding="png")

    with open(temp_file, "rb") as file:
        decoded_bytes = file.read()
    decoded_image = PILImage.open(io.BytesIO(decoded_bytes))
    decoded_array = np.array(decoded_image)
    assert np.array_equal(decoded_array, array)


def test_image_save(temp_file):
    # Create a random image
    image_file_path = temp_file
    array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    img = Image(array, encoding="png")
    img.save(image_file_path, "PNG", quality=100)
    # Reload the image to check if saved correctly
    reloaded_img = Image(image_file_path)
    assert np.array_equal(reloaded_img.array, array)


def test_image_model_dump_load():
    array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    img = Image(array, encoding="png")
    json = img.model_dump_json()
    reconstructed_img = Image.model_validate_json(json)
    assert np.array_equal(reconstructed_img.array, array)


def test_image_model_dump_load_with_base64():
    array = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    img = Image(array, encoding="png")
    print(img.array[10][10])
    json = img.model_dump_json(round_trip=True)
    reconstructed_img = Image.model_validate_json(json)
    print(reconstructed_img.array[10][10])
    assert np.array_equal(reconstructed_img.array, array)


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
