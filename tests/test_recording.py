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

from pathlib import Path
import h5py
import numpy as np
import os
from gymnasium import spaces
from gymnasium import spaces
from mbodied.data.recording import Recorder, create_dataset_for_space_dict
from mbodied.types.sample import Sample
from tempfile import TemporaryDirectory
from mbodied.types.sense.vision import Image
from PIL import Image as PILImage
import pytest
import logging

logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def tempdir():
    with TemporaryDirectory("test") as temp_dir:
        yield temp_dir


@pytest.fixture
def recorder(tempdir):
    name = "test_recorder"
    observation_space = spaces.Dict({"image": Image(size=(224, 224)).space(), "instruction": spaces.Discrete(10)})
    action_space = spaces.Dict(
        {
            "gripper_position": spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
            "gripper_action": spaces.Discrete(2),
        }
    )
    return Recorder(
        name=name,
        observation_space=observation_space,
        action_space=action_space,
        out_dir=tempdir,
        image_keys_to_save=["image"],
    )


def test_create_dataset_for_space_dict(recorder):
    observation_space = spaces.Dict({"image": Image(size=(224, 224)).space(), "instruction": spaces.Discrete(10)})
    with recorder.file as f:
        group = f.create_group("test_group")
        create_dataset_for_space_dict(observation_space, group)
        assert "image" in group
        assert "instruction" in group


def test_init(tempdir, recorder):
    assert os.path.exists(recorder.filename)
    assert os.path.exists(recorder.frames_dir)


def test_record_timestep(recorder):
    with recorder.file as f:
        group = f.create_group("test_group")
        group.create_dataset("test_dataset", (1,), maxshape=(None,))
        timestep = Sample(**{"test_dataset": 1})
        recorder.record_timestep(group, timestep, 0)
        assert group["test_dataset"][0] == 1


def test_record(tempdir, recorder):
    observation = {"image": np.ones((224, 224, 3), dtype=np.uint8), "instruction": 0}
    action = {"gripper_position": np.zeros((3,), dtype=np.float32), "gripper_action": 1}
    recorder.record(observation, action)

    with h5py.File(recorder.filename, "r") as f:
        assert np.array_equal(f["observation/image"][0], observation["image"])
        assert f["observation/instruction"][0] == observation["instruction"]
        assert np.array_equal(f["action/gripper_position"][0], action["gripper_position"])
        assert f["action/gripper_action"][0] == action["gripper_action"]


def test_record_image_class(tempdir, recorder):
    observation = {"image": Image(np.ones((224, 224, 3), dtype=np.uint8)), "instruction": 0}
    action = {"gripper_position": np.zeros((3,), dtype=np.float32), "gripper_action": 1}
    recorder.record(observation, action)

    with h5py.File(recorder.filename, "r") as f:
        # Assuming Image class has a method to return numpy array
        assert np.array_equal(f["observation/image"][0], observation["image"].array)
        assert f["observation/instruction"][0] == observation["instruction"]
        assert np.array_equal(f["action/gripper_position"][0], action["gripper_position"])
        assert f["action/gripper_action"][0] == action["gripper_action"]


def test_automatic_recording(tempdir):
    r = Recorder("test", out_dir=tempdir)
    observation = {"image": Image(np.zeros((224, 244, 3), dtype=np.uint8)), "instruction": "Hello, there."}
    action = {"gripper_position": 0, "gripper_action": np.array([0, 1, 0])}

    # Assuming Image class has a method to return numpy array
    r.record(observation, action)

    with h5py.File(r.filename, "r") as f:
        # Assuming Image class has a method to return numpy array
        assert np.array_equal(f["observation/image"][0].astype(np.uint8), observation["image"].array)
        assert f["observation/instruction"][0].decode() == observation["instruction"]
        assert np.array_equal(f["action/gripper_position"][0], action["gripper_position"])
        assert np.array_equal(f["action/gripper_action"][0], action["gripper_action"])


def test_close(recorder):
    recorder.close()
    with pytest.raises((KeyError, ValueError)):
        _ = recorder.file["observation"]


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
