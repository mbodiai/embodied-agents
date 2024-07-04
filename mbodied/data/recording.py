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

"""Module for recording data to an h5 file."""

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from gymnasium import spaces
from h5py import string_dtype

from mbodied.types.sample import Sample
from mbodied.types.sense.vision import Image


def add_space_metadata(space, group) -> None:
    group.attrs["space_type"] = space.__class__.__name__

    if isinstance(space, spaces.Box):
        if isinstance(space.low, float | int):
            low = space.low
            high = space.high
        else:
            low = np.ravel(space.low)[0]
            high = np.ravel(space.high)[0]
        group.attrs["low"] = low
        group.attrs["high"] = high
        group.attrs["shape"] = space.shape
    elif isinstance(space, spaces.Discrete):
        group.attrs["n"] = space.n
        group.attrs["string_values"] = [v for _, v in space.__dict__.items() if isinstance(v, str)]
    elif isinstance(space, spaces.MultiDiscrete):
        group.attrs["nvec"] = space.nvec
    elif isinstance(space, spaces.MultiBinary):
        group.attrs["n"] = space.n
    elif isinstance(space, spaces.Tuple):
        group.attrs["tuple_length"] = len(space.spaces)
    elif isinstance(space, spaces.Text):
        group.attrs["max_length"] = space.max_length

    if isinstance(space, np.ndarray):
        schema = Sample.from_space(space).model_json_schema()
    else:
        schema = str(Sample.from_space(space).model_json_schema())
    group.attrs["json_schema"] = schema


def create_dataset_for_space_dict(space_dict: spaces.Dict, group: h5py.Group) -> None:
    if not isinstance(space_dict, spaces.Dict):
        raise ValueError("space_dict must be a Dict at the root level")
    add_space_metadata(space_dict, group)
    logging.debug("data group keys: %s", str(space_dict.keys()))
    for key, space in space_dict.items():
        logging.debug(' key: "%s", value: %s', key, space)
        if isinstance(space, spaces.Dict):
            subgroup = group.create_group(key)
            create_dataset_for_space_dict(space, subgroup)
        else:
            shape = space.shape if hasattr(space, "shape") and space.shape is not None else ()
            dtype = space.dtype if space.dtype is not None and space.dtype != str else string_dtype()
            logging.debug(f"creating dataset: {key, shape, dtype}")
            group.create_dataset(key, (1, *shape), dtype=dtype, maxshape=(None, *shape))
        add_space_metadata(space, group[key])


def copy_and_delete_old(filename) -> None:
    if Path.exists(filename):
        stem = str(Path(filename).parent / Path(filename).stem)
        new_filename = stem + datetime.now().strftime("%Y%m%d%H%M%S") + ".h5"
        shutil.copyfile(filename, new_filename)
        Path.unlink(filename)


class Recorder:
    """Records a dataset to an h5 file. Saves images defined to folder with _frames appended to the name stem.

    Example:
      ```
      # Define the observation and action spaces
      observation_space = spaces.Dict({
          'image': spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
          'instruction': spaces.Discrete(10)
      })
      action_space = spaces.Dict({
          'gripper_position': spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
          'gripper_action': spaces.Discrete(2)
      }).

      # Create a recorder instance
      recorder = Recorder(name='test_recorder', observation_space=observation_space, action_space=action_space)

      # Generate some sample data
      num_steps = 10
      for i in range(num_steps):
          observation = {
              'image': np.ones((224, 224, 3), dtype=np.uint8),
              'instruction': i
          }
          action = {
              'gripper_position': np.zeros((3,), dtype=np.float32),
              'gripper_action': 1
          }
          recorder.record(observation, action)

      # Save the statistics
      recorder.save_stats()

      # Close the recorder
      recorder.close()

      # Assert that the HDF5 file and directories are created
      assert os.path.exists('test_recorder.h5')
      assert os.path.exists('test_recorder_frames')
      ```
    """

    def __init__(
        self,
        name: str,
        observation_space: spaces.Dict | str | None = None,
        action_space: spaces.Dict | str | None = None,
        supervision_space: spaces.Dict | str | None = None,
        out_dir: str = "saved_datasets",
        image_keys_to_save: list = None,
    ):
        """Initialize the Recorder.

        Args:
          name (str): Name of the file.
          observation_space (spaces.Dict): Observation space.
          action_space (spaces.Dict): Action space.
          out_dir (str, optional): Directory of the output file. Defaults to 'saved_datasets'.
          num_steps (int, optional): Number of steps. Defaults to 10.
          image_keys_to_save (list, optional): List of image keys to save. Defaults to ['image'].
        """
        print("\nInitializing dataset recorder...")

        if image_keys_to_save is None:
            image_keys_to_save = ["image"]
        self.out_dir = out_dir
        self.frames_dir = Path(out_dir) / (Path(name).stem + "_frames")
        self.frames_dir.mkdir(exist_ok=True, parents=True)

        filename = Path(out_dir) / Path(name).with_suffix(".h5")
        Path(out_dir).mkdir(exist_ok=True, parents=True)
        if Path.exists(filename):
            copy_and_delete_old(filename)
        self.file = h5py.File(filename, "a")

        self.name = name
        self.filename = filename

        self.observation_space = observation_space
        self.action_space = action_space
        self.supervision_space = supervision_space
        self.root_keys, self.root_spaces = self.configure_root_spaces(
            observation=observation_space,
            action=action_space,
            supervision=supervision_space,
        )

        self.image_keys_to_save = image_keys_to_save
        self.index = 0
        print("Recording dataset to", self.filename)

    def configure_root_spaces(self, **spaces: spaces.Dict):
        """Configure the root spaces.

        Args:
          observation_space (spaces.Dict): Observation space.
          action_space (spaces.Dict): Action space.
          supervision_space (spaces.Dict): Supervision space.
        """
        root_keys = []
        root_spaces = []
        for name, space in spaces.items():
            if space is None:
                continue

            root_keys.append(name)
            root_spaces.append(space)
            group = self.file.create_group(name)
            logging.debug("creating group %s", name)
            create_dataset_for_space_dict(space, group)
        return root_keys, root_spaces

    def record_timestep(self, group: h5py.Group, sample: Any, index: int) -> None:
        """Record a timestep.

        Args:
          group (h5py.Group): Group to record to.
          sample (Any): Sample to record.
          index (int): Index to record at.
        """
        if isinstance(group, h5py.Dataset):
            if index >= group.shape[0]:
                group.resize((2 * index, *group.shape[1:]))
            if hasattr(sample, "value"):
                sample = sample.value
            group[index] = sample
            return
        logging.debug("group keys: %s", str(group.keys()))
        if not hasattr(sample, "dict"):
            sample = Sample(sample)
        for key, value in sample:
            if value is None:
                continue
            if hasattr(value, "array"):
                dataset = group[key]
                if index >= dataset.shape[0]:
                    dataset.resize((2 * index, *dataset.shape[1:]))
                dataset[index] = value.array
                if key in self.image_keys_to_save and hasattr(value, "save"):
                    value.save(self.frames_dir / f"{self.index}.png")
                continue
            logging.debug(" key: %s, value: %s", key, value)

            if key not in group:
                logging.warning("key %s not in group %s. Skipping key", key, group)
                continue
            if isinstance(value, dict | Sample):
                subgroup = group[key]
                self.record_timestep(subgroup, value, index)
                continue

            if group[key].attrs.get("tuple_length") is not None:
                value = Sample.pack_from(value).model_dump_json(round_trip=True)  # noqa: PLW2901

            dataset = group[key]
            if index >= dataset.shape[0]:
                dataset.resize((2 * index, *dataset.shape[1:]))
            dataset[index] = value

    def record(self, observation: Any | None = None, action: Any | None = None, supervision: Any | None = None) -> None:
        """Record a timestep.

        Args:
          observation (Any): Observation to record.
          action (Any): Action to record.
          supervision (Any): Supervision to record.
        """

        def recursive_setarray(sample):
            if not hasattr(sample, "dict"):
                sample = Sample(sample)
            for key, value in sample:
                if isinstance(value, Image):
                    setattr(sample, key, value.array)
                elif isinstance(value, dict | Sample):
                    setattr(sample, key, recursive_setarray(value))
            return sample

        if observation:
            if not hasattr(observation, "dict"):
                observation = Sample(observation)
                observation = recursive_setarray(observation)  # Bug hacky fix for Image recording.
            if "observation" not in self.file:
                logging.warning("Recorder: observation not in file, creating new group")
                new_root_keys, new_root_spaces = self.configure_root_spaces(observation=observation.space())
                self.root_keys += new_root_keys
                self.root_spaces += new_root_spaces
            self.record_timestep(self.file["observation"], observation, self.index)
        if action:
            if not hasattr(action, "dict"):
                action = Sample(action)
                action = recursive_setarray(action)  # Bug hacky fix for Image recording.
            if "action" not in self.file:
                logging.warning("Recorder: action not in file, creating new group")
                new_root_keys, new_root_spaces = self.configure_root_spaces(action=action.space())
                self.root_keys += new_root_keys
                self.root_spaces += new_root_spaces
            self.record_timestep(self.file["action"], action, self.index)
        if supervision:
            if not hasattr(supervision, "dict"):
                supervision = Sample(supervision)
                supervision = recursive_setarray(supervision)  # Bug hacky fix for Image recording.
            if "supervision" not in self.file:
                logging.warning("Recorder: supervision not in file, creating new group")
                new_root_keys, new_root_spaces = self.configure_root_spaces(supervision=supervision.space())
                self.root_keys += new_root_keys
                self.root_spaces += new_root_spaces
            self.record_timestep(self.file["supervision"], supervision, self.index)

        self.index += 1
        self.file.attrs["size"] = self.index

    def close(self) -> None:
        """Closes the Recorder and send the data if train_config is set."""
        self.file.close()
