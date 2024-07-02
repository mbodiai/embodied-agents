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


import numpy as np
from datasets import Features, Image, Value


def to_features(indict, image_keys=None, exclude_keys=None, prefix="") -> Features:
    """Convert a dictionary to a Datasets Features object.

    Args:
        indict (dict): The dictionary to convert.
        image_keys (dict): A dictionary of keys that should be treated as images.
        exclude_keys (set): A set of full-path-keys to exclude.
        prefix (str): A prefix to add to the keys.
    """
    if exclude_keys is None:
        exclude_keys = set()

    if image_keys is None:
        image_keys = {}

    if isinstance(indict, str):
        return Value("string")
    if isinstance(indict, int):
        return Value("int32")
    if isinstance(indict, float):
        return Value("float32")
    if isinstance(indict, np.int32):
        return Value("int32")
    if isinstance(indict, np.float32):
        return Value("float32")

    if isinstance(indict, list | tuple | np.ndarray):
        if len(indict) == 0:
            raise ValueError("Cannot infer schema from empty list")
        return [to_features(indict[0])]

    if isinstance(indict, dict):
        out_dict = {}
        for key, value in indict.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if full_key in image_keys and full_key not in exclude_keys:
                out_dict[key] = Image(decode=True)
            elif full_key not in exclude_keys:
                out_dict[key] = to_features(value, image_keys, exclude_keys, full_key)
        return out_dict

    raise ValueError(f"Cannot infer schema from {indict}")


def infer_features(example) -> Features:
    """Infer Hugging Face Datasets Features from an example."""
    return Features(to_features(example))
