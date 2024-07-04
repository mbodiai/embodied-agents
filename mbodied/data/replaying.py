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

import logging
import os
from pathlib import Path
from typing import Any, Callable, List, Tuple

import click
import h5py
import numpy as np
from datasets import Dataset, DatasetInfo, Features, Image, Value
from h5py import string_dtype
from huggingface_hub import login
from PIL import Image as PILImage

from mbodied.data.utils import infer_features
from mbodied.types.sample import Sample
from mbodied.types.sense.vision import Image as MbImage


class Replayer:
    """Replays datasets recorded by Recorder.

    This class provides methods to read, process, and analyze HDF5 files recorded by the Recorder class.

    Example:
        replayer = Replayer("data.h5")
        for sample in replayer:
            observation, action = sample
            ...
    """

    def __init__(
        self,
        path: str,
        file_keys: List[str] = None,
        image_keys_to_save: List[str] = None,
    ) -> None:
        """Initialize the Replayer class with the given parameters.

        Args:
            path (str): Path to the HDF5 file.
            file_keys (List[str], optional): List of keys in the file. Defaults to None.
            image_keys_to_save (List[str], optional): List of image keys to save. Defaults to None.
        """
        self.path = path
        self.frames_path = f"{Path(self.path).parent}/{Path(self.path).stem}_frames"
        self.file = h5py.File(path, "r")

        self.file_keys = file_keys or list(self.file.keys())
        self._reorder_keys()
        self.image_keys_to_save = image_keys_to_save or []
        if self.image_keys_to_save:
            Path(self.frames_path).mkdir(parents=True, exist_ok=True)
        self.size = self._get_size()

    def _reorder_keys(self) -> None:
        """Reorder keys to have 'observation' first, 'action' last, and 'supervision' last but one."""
        if "observation" in self.file_keys:
            self.file_keys.remove("observation")
            self.file_keys = ["observation"] + self.file_keys
        if "action" in self.file_keys:
            self.file_keys.remove("action")
            self.file_keys = self.file_keys + ["action"]
        if "supervision" in self.file_keys:
            self.file_keys.remove("supervision")
            self.file_keys = self.file_keys + ["supervision"]

    def _get_size(self) -> int:
        """Get the size attribute from the HDF5 file or infer it from the first key."""
        size = self.file.attrs.get("size")
        if size is None:
            logging.warning(
                "No 'size' attribute found in the HDF5 file. The number of samples will be inferred from the first key.",
            )
            try:
                key = self.file_keys[0]
                while isinstance(self.file[key], h5py.Group):
                    key = list(self.file[key].keys())[0]
                size = len(self.file[key])
            except Exception:
                logging.warning("No size attribute and empty dataset found. Setting size to 0.")
                size = 0
        return size

    def get_frames_path(self) -> str | None:
        """Get the path to the frames directory."""
        return self.frames_path

    def recursive_do(self, do: Callable, key="", prefix="", **kwargs) -> Any:
        """Recursively perform a function on each key in the HDF5 file.

        Args:
            do (Callable): Function to perform.
            key (str, optional): Key in the HDF5 file. Defaults to ''.
            prefix (str, optional): Prefix for the key. Defaults to ''.
            **kwargs: Additional arguments to pass to the function.

        Returns:
            Any: Result of the function.
        """
        full_key = f"{prefix}/{key}".strip("/")
        if full_key and isinstance(self.file[full_key], h5py.Dataset):
            return do(full_key, **kwargs)

        keys = self.file_keys if not full_key else self.file[full_key].keys()

        result = {}
        for sub_key in keys:
            result[sub_key] = self.recursive_do(do, key=sub_key, prefix=full_key, **kwargs)
        return result

    def get_unique_items(self, key: str) -> List[str]:
        """Get unique items for a given key.

        Args:
            key (str): Key in the HDF5 file.

        Returns:
            List[str]: List of unique items.
        """
        if self.file[key].dtype == string_dtype():
            return list({item.decode() for item in self.file[key][: self.size]})
        return list(set(self.file[key][: self.size]))

    def read_sample(self, index: int) -> Tuple[dict, ...]:
        """Read a sample from the HDF5 file at a given index.

        Args:
            index (int): Index of the sample.

        Returns:
            Tuple[dict, ...]: Tuple of dictionaries containing the sample data.
        """

        def read_do(full_key: str, index: int) -> Any:
            if index >= self.file[full_key].shape[0]:
                logging.warning(f"Index {index} is out of bounds for key {full_key}. Skipping...")
                return None
            value = self.file[full_key][index]
            if full_key in self.image_keys_to_save and len(value.shape) == 3:
                image_path = f"{self.frames_path}/{Path(full_key).name}_{index}.png"
                PILImage.fromarray(value).save(image_path)
            if self.file[full_key].dtype == string_dtype():
                if self.file[full_key].attrs.get("tuple_length") is not None:
                    return Sample.model_validate_json(value.decode()).unpack(to_dicts=True)
                try:
                    value = value.decode()
                except Exception as e:
                    logging.error(f"Error decoding string for key {full_key}: {e}")
                    value = value[0].decode()
            return value

        return tuple(self.recursive_do(read_do, key=key, index=index) for key in self.file_keys)

    def get_structure(self, key="", prefix="") -> dict:
        """Get the structure of the HDF5 file.

        Args:
            key (str, optional): Key in the HDF5 file. Defaults to ''.
            prefix (str, optional): Prefix for the key. Defaults to ''.

        Returns:
            dict: Structure of the HDF5 file.
        """

        def structure_do(full_key: str) -> tuple:
            return self.file[full_key][: self.size].shape, self.file[full_key][: self.size].dtype

        return self.recursive_do(structure_do, key, prefix)

    def pack_one(self, index: int) -> Sample:
        """Pack a single sample into a Sample object.

        Args:
            index (int): Index of the sample.

        Returns:
            Sample: Sample object.
        """
        return Sample(**{key: self.read_sample(index)[i] for i, key in enumerate(self.file_keys)})

    def pack(self) -> Sample:
        """Pack all samples into a Sample object with attributes being lists of samples.

        Returns:
            Sample: Sample object containing all samples.
        """
        return Sample.pack_from([self.pack_one(i) for i in range(self.size)])

    def sample(self, index: int | slice | None = None, n: int = 1) -> Sample:
        """Get a sample from the HDF5 file.

        Args:
            index (Optional[Union[int, slice]], optional): Index or slice of the sample. Defaults to None.
            n (int, optional): Number of samples to get. Defaults to 1.

        Returns:
            Sample: Sample object.
        """
        if index is None:
            index = np.random.randint(0, self.size, n)
        if isinstance(index, int):
            return self.pack_one(index)
        if isinstance(index, slice):
            return [self.read_sample(i) for i in range(index.start, index.stop, index.step)]
        return None

    def get_stats(self, key="", prefix="") -> dict:
        """Get statistics for a given key in the HDF5 file.

        Args:
            key (str, optional): Key in the HDF5 file. Defaults to ''.
            prefix (str, optional): Prefix for the key. Defaults to ''.

        Returns:
            dict: Statistics for the given key.
        """

        def stats_do(full_key: str) -> dict:
            if self.file[full_key].dtype == h5py.string_dtype():
                return {
                    "unique": len({item.decode() for item in self.file[full_key][: self.size]}),
                    "shape": self.file[full_key][: self.size].shape,
                    "dtype": self.file[full_key][: self.size].dtype,
                }
            data = self.file[full_key][: self.size]
            return {
                "min": np.min(data, axis=0),
                "max": np.max(data, axis=0),
                "mean": np.mean(data, axis=0),
                "std": np.std(data, axis=0),
                "median": np.median(data, axis=0),
                "shape": self.file[full_key][: self.size].shape,
                "dtype": self.file[full_key][: self.size].dtype,
            }

        return {key: self.recursive_do(stats_do, key, prefix)}

    def __iter__(self):
        """Iterate over the HDF5 file."""
        self.index = -1
        return self

    def __next__(self) -> Tuple[dict, ...]:
        """Get the next sample from the HDF5 file.

        Returns:
            Tuple[dict, ...]: Tuple of dictionaries containing the sample data.
        """
        if self.index >= self.size - 1:
            raise StopIteration
        self.index += 1
        return self.read_sample(self.index)

    def close(self) -> None:
        """Close the HDF5 file."""
        self.file.close()


def clean_folder(folder: str, image_keys_to_save: List[str]) -> None:
    """Clean the folder by iterating through the files and asking for deletion.

    Args:
        folder (str): Path to the folder.
        image_keys_to_save (List[str]): List of image keys to save.
    """
    for f in os.listdir(folder):
        r = Replayer(f"{folder}/{f}", image_keys_to_save=image_keys_to_save)
        for _i, sample in enumerate(r):
            obs, act = sample
        should_delete = input(f"Delete {f}? (y/n): ")
        if should_delete.lower() == "y":
            Path(f"{folder}/{f}").rmdir()


class FolderReplayer:
    def __init__(self, path: str) -> None:
        """Initialize the FolderReplayer class with the given path.

        Args:
            path (str): Path to the folder containing HDF5 files.
        """
        self.path = path

    def __iter__(self):
        """Iterate through the HDF5 files in the folder."""
        for f in os.listdir(self.path):
            if f.endswith(".h5"):
                r = Replayer(f"{self.path}/{f}")
                for _i, sample in enumerate(r):
                    observation = sample[0]
                    action = sample[1]
                    image = np.asarray(observation["image"])
                    instruction = observation["instruction"]
                    yield {"observation": {"image": image, "instruction": instruction}, "action": action}


def to_dataset(folder: str, name: str, description: str = None, **kwargs) -> None:
    """Convert the folder of HDF5 files to a Hugging Face dataset.

    Args:
        folder (str): Path to the folder containing HDF5 files.
        name (str): Name of the dataset.
        description (str, optional): Description of the dataset. Defaults to None.
        **kwargs: Additional arguments to pass to the Dataset.push_to_hub method.
    """
    r = FolderReplayer(folder)
    data = list(r.__iter__())

    def list_of_dicts_to_dict(data: List[dict]) -> dict:
        if not data:
            return {}
        columnar_data = {key: [] for key in data[0]}
        for item in data:
            for key, value in item.items():
                columnar_data[key].append(value)
        return columnar_data

    features = Features(
        {
            "observation": {"image": Image(), "instruction": Value("string")},
            "action": infer_features(data[0]["action"]),
        },
    )

    data = list_of_dicts_to_dict(data)
    info = DatasetInfo(
        description=description,
        license="Apache-2.0",
        citation="None",
        size_in_bytes=8000000,
        features=features,
    )

    ds = Dataset.from_dict(data, info=info)
    ds = ds.with_format("pandas")
    login(os.getenv("HF_TOKEN"))
    ds.push_to_hub(name, **kwargs)


def parse_slice(s: str) -> int | slice:
    """Parse a string to an integer or slice.

    Args:
        s (str): String to parse.

    Returns:
        Union[int, slice]: Integer or slice.

    Example:
        >>> lst = [0, 1, 2, 3, 4, 5]
        >>> lst[parse_slice("1")]
        1
        >>> lst[parse_slice("1:5:2")]
        [1, 3]
    """
    if ":" in s or s == "all":
        parts = s.strip().split(":")
        start = int(parts[0]) if parts[0] else None
        stop = int(parts[1]) if len(parts) > 1 and parts[1] else None
        step = int(parts[2]) if len(parts) > 2 and parts[2] else None
        return slice(start, stop, step)
    return int(s)


@click.command("replay")
@click.argument("path", type=click.Path(exists=True))
@click.option("-ik", "--image_keys_to_save", multiple=True, help="Specify image keys to save during processing.")
@click.option("-s", "--stats", is_flag=True, help="Enable statistics generation for specified keys.")
@click.option("-u", "--unique", is_flag=True, help="Enable extraction of unique items from specified keys.")
@click.option("-k", "--keys", multiple=True, help="Specify keys to process for statistics or unique item extraction.")
@click.option(
    "--show-slice",
    "-v",
    default="all",
    type=str,
    help="Display and print samples at the specified index or range. Use 'all' to show all samples.",
)
@click.option("--clean", "-c", is_flag=True, help="Clean the specified folder by removing specified image keys.")
@click.option("--upload-name", "-up", default=None, help="Upload the dataset after processing.")
def replay(
    path: str,
    image_keys_to_save: List[str],
    stats: bool,
    unique: bool,
    keys: List[str],
    show: str,
    clean: bool,
    upload_name: str,
) -> None:
    """Replay command for processing HDF5 files.

    This command provides various options to manipulate and analyze HDF5 files including cleaning, uploading,
    showing samples, and generating statistics or extracting unique items based on provided keys.
    """
    if clean:
        clean_folder(path, image_keys_to_save)
        return

    replayer = Replayer(path, image_keys_to_save=image_keys_to_save)
    if stats:
        if keys:
            for _key in keys:
                pass
        else:
            logging.info(replayer.get_stats())

    if unique:
        if keys:
            for _key in keys:
                pass
        else:
            raise ValueError("Keys must be provided to get unique items.")

    if show:
        slice_ = parse_slice(show)
        for _sample in replayer[slice_]:
            showing = {}
            for key in replayer.file_keys:
                if key in replayer.image_keys_to_save:
                    img = MbImage(_sample[key])
                    img.save(f"{replayer.get_frames_path()}/{key}.png")
                    showing[key] = img
                else:
                    showing[key] = _sample[key]
            logging.info(showing)
    replayer.close()

    if upload_name:
        to_dataset(path, name=upload_name)
        return


if __name__ == "__main__":
    replay()
