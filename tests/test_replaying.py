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

import os
import pytest
import numpy as np
import h5py
from pathlib import Path
from mbodied.data.replaying import Replayer, parse_slice
from mbodied.data.recording import Recorder
from mbodied.types.motion.control import LocobotActionOrAnswer as ActionOrAnswer
from gymnasium import spaces
import sys
import logging


@pytest.fixture
def mock_hdf5_file(tmpdir):
    # Create a temporary directory for the HDF5 file and output images
    filepath = Path(tmpdir) / "test.hdf5"
    frames_path = Path(tmpdir) / "test_frames"

    # Create an HDF5 file for testing
    with h5py.File(filepath, "w") as f:
        # Create datasets for 'observation' and 'action' groups
        obs_group = f.create_group("observation")
        act_group = f.create_group("action")

        # Populate with some dummy data
        img_data = np.random.randint(0, 255, (10, 224, 224, 3), dtype=np.uint8)
        instruction_data = np.random.randint(0, 10, (10,), dtype=np.int32)
        position_data = np.random.rand(10, 3).astype(np.float32)
        action_data = np.random.randint(0, 2, (10,), dtype=np.int32)

        obs_group.create_dataset("image", data=img_data)
        obs_group.create_dataset("instruction", data=instruction_data)
        act_group.create_dataset("gripper_position", data=position_data)
        act_group.create_dataset("gripper_action", data=action_data)
        f.attrs["size"] = 10  # Store the number of samples
    yield filepath, frames_path


def test_replayer_iteration(mock_hdf5_file):
    filepath, _ = mock_hdf5_file
    replayer = Replayer(path=str(filepath))

    count = 0
    for observation, action in replayer:
        print("loop")
        assert isinstance(observation, dict)
        assert isinstance(action, dict)
        assert "image" in observation
        assert "instruction" in observation
        assert "gripper_position" in action
        assert "gripper_action" in action
        count += 1

    assert count == 10  # Ensure we iterated over all samples
    replayer.close()


def test_image_saving(mock_hdf5_file):
    filepath, _ = mock_hdf5_file
    replayer = Replayer(path=str(filepath), image_keys_to_save=["observation/image"])
    frames_path = replayer.get_frames_path()

    for _ in replayer:
        pass

    replayer.close()

    # Check if images were saved
    saved_images = list(Path(frames_path).glob("image_*.png"))
    assert len(saved_images) == 10  # We expect 10 images to be saved


@pytest.fixture
def spaces_setup():
    observation_space = spaces.Dict(
        {
            "image": spaces.Box(low=0, high=255, shape=(100, 100, 3), dtype="uint8"),
            "instruction": spaces.Text(1000),
            "system_prompt": spaces.Text(1000),
        }
    )
    action_space = ActionOrAnswer().space()
    return observation_space, action_space


@pytest.fixture
def recorder_and_path(spaces_setup, tmp_path):
    observation_space, action_space = spaces_setup
    recorder_path = tmp_path / "recorder.h5"
    recorder = Recorder(
        "recorder", out_dir=str(tmp_path), observation_space=observation_space, action_space=action_space
    )
    yield recorder, recorder_path
    recorder.close()
    if os.path.exists(recorder_path):
        os.remove(recorder_path)


def test_record_and_replay_basic(recorder_and_path):
    recorder, recorder_path = recorder_and_path
    dummy_observation = {
        "image": recorder.root_spaces[0]["image"].sample(),
        "instruction": "What do you see?",
        "system_prompt": "Please respond.",
    }
    # Create actions as dictionaries directly
    dummy_action = ActionOrAnswer.default_sample()
    recorder.record(dummy_observation, dummy_action)

    replayer = Replayer(str(recorder_path))
    for observation, action in replayer:
        assert observation["instruction"] == dummy_observation["instruction"]
        assert ActionOrAnswer(action).dict() == dummy_action.dict()


logging.basicConfig(level=logging.DEBUG)


def test_record_multiple_entries(recorder_and_path):
    recorder, recorder_path = recorder_and_path
    entries = [
        (
            {
                "image": recorder.root_spaces[0]["image"].sample(),
                "instruction": f"Entry {i}",
                "system_prompt": "Please respond.",
            },
            ActionOrAnswer.default_sample(),
        )
        for i in range(5)
    ]
    logging.basicConfig(level=logging.DEBUG, force=True)
    for obs, act in entries:
        recorder.record(obs, act)

    replayer = Replayer(str(recorder_path))
    from pprint import pprint

    pprint(replayer.get_structure())
    replayed_entries = list(replayer)
    assert len(replayed_entries) == len(entries)
    for (replayed_obs, replayed_act), (original_obs, original_act) in zip(replayed_entries, entries):
        assert replayed_obs["instruction"] == original_obs["instruction"]
        assert ActionOrAnswer(replayed_act).dict() == original_act.dict()


def test_empty_record(recorder_and_path):
    recorder, recorder_path = recorder_and_path
    replayer = Replayer(str(recorder_path))
    assert list(replayer) == []  # Expecting no entries


def test_parse_slice():
    assert parse_slice("1") == 1
    assert parse_slice("1:5:2") == slice(1, 5, 2)
    assert parse_slice(":5:2") == slice(None, 5, 2)
    assert parse_slice("1::2") == slice(1, None, 2)
    assert parse_slice("1:5") == slice(1, 5, None)
    assert parse_slice("::") == slice(None, None, None)


# Run the test
if __name__ == "__main__":
    # parse for debug flag
    if "--debug" in sys.argv:
        sys.argv.remove("--debug")
    logging.basicConfig(level=logging.DEBUG)
    pytest.main([__file__, "-vv"])
