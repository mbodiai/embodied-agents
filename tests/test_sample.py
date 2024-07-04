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

import pytest
from gymnasium import spaces
import numpy as np
import h5py
import os
import tempfile

from mbodied.data.recording import Recorder
from mbodied.types.sample import Sample


def test_from_dict():
    d = {"key1": 1, "key2": [2, 3, 4], "key3": {"nested_key": 5}}
    sample = Sample(**d)
    assert sample.key1 == 1
    assert np.array_equal(sample.key2, [2, 3, 4])
    # Assuming key3 is a dict, not a Sample
    assert sample.key3["nested_key"] == 5


def test_from_space():
    space = spaces.Dict(
        {
            "key1": spaces.Discrete(3),
            "key2": spaces.Box(low=0, high=1, shape=(2,)),
            "key3": spaces.Dict({"nested_key": spaces.Discrete(2)}),
        }
    )
    sample = Sample.from_space(space)
    assert isinstance(sample.key1, int | np.int64)
    assert isinstance(sample.key2, np.ndarray)
    assert (sample.key2 >= 0).all() and (sample.key2 <= 1).all()
    assert isinstance(sample.key3["nested_key"], int | np.int64)


def test_to_dict():
    sample = Sample(key1=1, key2=[2, 3, 4], key3={"nested_key": 5})
    d = sample.dict()  # Adjusted to use the .dict() method
    assert d == {"key1": 1, "key2": [2, 3, 4], "key3": {"nested_key": 5}}


def test_serialize_nonstandard_types():
    data = {"array": np.array([1, 2, 3])}
    sample = Sample(**data)
    assert np.array_equal(sample.array, [1, 2, 3]), "Numpy arrays should work fine"


def test_structured_flatten():
    nested_data = {"key1": 1, "key2": [2, 3, 4], "key3": {"nested_key": 5}}
    sample = Sample(**nested_data)
    flat_list = sample.flatten("list")
    assert sorted(flat_list) == [1, 2, 3, 4, 5], "Flat list should contain all values from the structure"


def test_unpack_as_dict():
    # Scenario with asdict=True
    data = {"key1": [1, 2, 3], "key2": ["a", "b", "c"], "key3": [[1, 2], [3, 4], [5, 6]]}
    sample = Sample(**data)
    unpacked = sample.unpack(to_dicts=True)

    expected = [
        {"key1": 1, "key2": "a", "key3": [1, 2]},
        {"key1": 2, "key2": "b", "key3": [3, 4]},
        {"key1": 3, "key2": "c", "key3": [5, 6]},
    ]

    assert unpacked == expected, "Unrolled items as dict do not match expected values"


def test_unpack_as_sample_instances():
    # Scenario with asdict=False
    data = {"key1": [4, 5, 6], "key2": ["d", "e", "f"], "key3": [[7, 8], [9, 10], [11, 12]]}
    sample = Sample(**data)
    unpacked = sample.unpack(to_dicts=False)

    # Validate each unrolled item
    for idx, item in enumerate(unpacked):
        assert isinstance(item, Sample), "Unrolled item is not a Sample instance"
        assert item.key1 == data["key1"][idx], f"Unrolled item key1 does not match expected value for index {idx}"
        assert item.key2 == data["key2"][idx], f"Unrolled item key2 does not match expected value for index {idx}"
        assert item.key3 == data["key3"][idx], f"Unrolled item key3 does not match expected value for index {idx}"


@pytest.fixture
def sample_instance():
    # Fixture to create a sample instance with a variety of attribute types
    return Sample(
        int_attr=5,
        float_attr=3.14,
        list_attr=[1, 2, 3],
        dict_attr={"nested": 10},
        numpy_attr=np.array([1, 2, 3]),
        list_of_dicts_attr=[{"a": 1}, {"a": 2}],
        list_of_samples_attr=[Sample(x=1), Sample(x=2)],
        sample_attr=Sample(x=10),
    )


@pytest.fixture
def tmp_path():
    try:
        # Fixture to create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    finally:
        if os.path.exists(tmpdir):
            os.rmdir(tmpdir)


def test_space_for_list_attribute(sample_instance: Sample):
    space = sample_instance.space()
    assert isinstance(space.spaces["list_attr"], spaces.Box), "List attribute should correspond to a Box space"


def test_space_for_dict_attribute(sample_instance: Sample):
    space = sample_instance.space()
    assert isinstance(space.spaces["dict_attr"], spaces.Dict), "Dict attribute should correspond to a Dict space"


def test_space(sample_instance: Sample):
    space = sample_instance.space()
    expected_keys = {
        "int_attr",
        "float_attr",
        "list_attr",
        "dict_attr",
        "numpy_attr",
        "list_of_dicts_attr",
        "list_of_samples_attr",
        "sample_attr",
    }
    assert set(space.spaces.keys()) == expected_keys, "Space should include all attributes of the sample instance"


def test_serialize_deserialize():
    sample = Sample(key1=1, key2=[2, 3, 4], key3={"nested_key": 5})
    serialized = sample.model_dump_json()
    deserialized = Sample.model_validate_json(serialized)
    assert sample == deserialized, "Deserialized sample should match the original sample"


def test_recorder_record_and_save(sample_instance: Sample, tmp_path: str):
    recorder = Recorder(
        name="test_recorder",
        observation_space=sample_instance.space(),
        supervision_space=sample_instance.space(),
        out_dir=str(tmp_path),
    )
    recorder.record(observation=sample_instance, supervision=sample_instance)

    recorder.close()

    with h5py.File(f"{tmp_path}/test_recorder.h5", "r") as f:
        assert "observation" in f, "Observation group should be present in the HDF5 file"
        assert "supervision" in f, "Action group should be present in the HDF5 file"

    with h5py.File(f"{tmp_path}/test_recorder.h5", "r") as f:
        assert "observation" in f, "Observation group should be present in the HDF5 file"
        assert "supervision" in f, "Action group should be present in the HDF5 file"


def test_unflatten_dict():
    sample = Sample(x=1, y=2, z={"a": 3, "b": 4}, extra_field=5)
    schema = sample.schema()
    flat_dict = sample.flatten(output_type="dict")
    unflattened_sample = Sample.unflatten(flat_dict, schema)
    assert unflattened_sample == sample


def test_unflatten_list():
    sample = Sample(x=1, y=2, z={"a": 3, "b": 4}, extra_field=5)
    flat_list = sample.flatten(output_type="list")
    unflattened_sample = Sample.unflatten(flat_list, sample.schema())
    assert unflattened_sample.x == 1
    assert unflattened_sample.y == 2
    assert unflattened_sample.z == {"a": 3, "b": 4}
    assert unflattened_sample.extra_field == 5


def test_unflatten_numeric_only():
    class AnotherSample(Sample):
        a: int
        b: str = "default"

    class DerivedSample(Sample):
        x: int
        y: str = "default"
        z: AnotherSample
        another_number: float

    sample = DerivedSample(x=1, y="hello", z=AnotherSample(**{"a": 3, "b": "world"}), another_number=5)

    flat_list = sample.flatten(output_type="list")
    unflattened_sample = DerivedSample.unflatten(flat_list, sample.schema())
    assert unflattened_sample.x == 1
    assert hasattr(unflattened_sample, "y")
    assert unflattened_sample.z.a == 3
    assert unflattened_sample.z.b == "world"


def test_unflatten_numpy_array():
    sample = Sample(x=1, y=2, z={"a": 3, "b": 4}, extra_field=5)
    flat_array = sample.flatten(output_type="np")
    unflattened_sample = Sample.unflatten(flat_array, sample.schema())
    assert unflattened_sample.x == 1
    assert unflattened_sample.y == 2
    assert unflattened_sample.z == {"a": 3, "b": 4}
    assert unflattened_sample.extra_field == 5


def test_unflatten_torch_tensor():
    sample = Sample(x=1, y=2, z={"a": 3, "b": 4}, extra_field=5)
    flat_tensor = sample.flatten(output_type="pt")
    unflattened_sample = Sample.unflatten(flat_tensor, sample.schema())
    assert unflattened_sample.x == 1
    assert unflattened_sample.y == 2
    assert unflattened_sample.z == {"a": 3, "b": 4}
    assert unflattened_sample.extra_field == 5


# Sample unit test for the schema method
def test_schema():
    sample_instance = Sample(x=1, y=2, z={"a": 3, "b": 4}, extra_field=5)
    schema = sample_instance.schema(include_descriptions=True)
    expected_schema = {
        "description": "A base model class for serializing, recording, and manipulating arbitray data.",
        "properties": {
            "x": {"type": "integer"},
            "y": {"type": "integer"},
            "z": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
            "extra_field": {"type": "integer"},
        },
        "title": "Sample",
        "type": "object",
    }
    assert schema == expected_schema


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
