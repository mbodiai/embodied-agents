from typing import Dict, Union

import numpy as np
import torch
from gym import spaces


# This function will turn the space into the batched space and return a batched action sample.
# The output format is compatible with OpenAI gym's Vectorized Environments.
def batched_space_sampler(space: spaces.Dict, batch_size: int):
    batched_sample : Dict[str, np.ndarray] = {}
    samples = [space.sample() for _ in range(batch_size)] # 辞書のリスト
    for key in samples[0].keys():
        value_list = []
        for i in range(batch_size):
            value_list.append(samples[i][key])
        value_list = np.stack(value_list, axis=0)
        batched_sample[key] = value_list
    return batched_sample

# This function turn all dict values into tensor.
def np_to_tensor(sample_dict: Dict[str, Union[int,np.ndarray]], device=None) -> Dict[str, torch.Tensor]:
    new_dict = {}
    for key, value in sample_dict.items():
        value = torch.tensor(value, device=device)
        new_dict[key] = value

    return new_dict