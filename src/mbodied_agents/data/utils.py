# Copyright 2024 Mbodi AI
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

import datasets
from datasets import Features, Sequence
import numpy as np


def dict_generator(indict, image_keys=None):
    if image_keys is None:
        image_keys = {}
    if isinstance(indict, str):
        return datasets.Value("string")
    if isinstance(indict, int):
        return datasets.Value("int32")
    if isinstance(indict, float):
        return datasets.Value("float32")
    if isinstance(indict, list | tuple | np.ndarray):
        if len(indict) == 0:
            return Sequence(datasets.Value("float32"))
        return Sequence(dict_generator(indict[0]))
    if isinstance(indict, dict):
        out_dict = {}
        for key, value in indict.items():
            if key in image_keys:
                out_dict[key] = datasets.Image(image_keys[key])
            if key not in ["entities", "answer_url"]:
                out_dict[key] = dict_generator(value)
        return out_dict
    return None


def infer_features(example):
    return Features(dict_generator(example))