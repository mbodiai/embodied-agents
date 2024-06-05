# Subject to the terms and conditions of the Apache License, Version 2.0 that the original code follows,
# I have retained the following copyright notice written on it.

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# You can find the original code from here[https://github.com/google-research/robotics_transformer].


from typing import Dict, Union

import numpy as np
import torch
from gym import spaces

"""
Please define action space using gym.spaces.Dict.

As an example, if an action is:
done = [0, 1] # this is one hot vector.
xyz = [0.9, 0.8, -0.3]
rpy = [-0.1, 0.2, .6]
grasp = 0.9

then action space will look like
action_space = gym.spaces.Dict(
    {
        'done': gym.spaces.Discrete(2),
        'xyz': gym.spaces.Box(low= -1.0, high= 1.0, shape=(3,), dtype=np.float32),
        'rpy': gym.spaces.Box(low= -np.pi / 2  , high= np.pi / 2, shape=(3,), dtype=np.float32),
        'grasp': gym.spaces.Box(low= 0  , high= 1.0, shape=(1,), dtype=np.float32)
    }
)
or 
action_space = gym.spaces.Dict(
            OrderedDict([
                ('done', gym.spaces.Discrete(2)), 
                ('xyz', gym.spaces.Box(low= -1.0, high= 1.0, shape=(3,), dtype=np.float32)),
                ('rpy', gym.spaces.Box(low= -np.pi / 2., high= np.pi / 2., shape=(3,), dtype=np.float32)),
                ('grasp', gym.spaces.Box(low= 0  , high= 1.0, shape=(1,), dtype=np.float32))
                ])
        )
Please use OrderedDict if you want gym.spaces.Dict to keep order of actions.

This action_space is just information about each action.
These information are very convenient when interpreting, examining, cliping, and processing the action
because the action is dictionary with key names which are the same as the action_space.

action value will be look like
action = {
    'done': 1,
    'xyz': [0.9, 0.8, -0.3],
    'rpy': [-0.1, 0.2, .6],
    'grasp': [0.9]
}
Note that values are int and numpy 1-d arrays.
"""


class RT1ActionTokenizer():
    def __init__(self,
                 action_space: spaces.Dict,
                 vocab_size: int,
                 device: Union[str, torch.device] = None):
        """Instantiates an RT1ActionTokenizer.

        Args:
        action_space: A dictionary of OpenAI gym spaces of the expected actions.
        vocab_size: Number of buckets to discretize action to.
        """
        self.device = device
        self._action_space = action_space
        self._vocab_size = vocab_size
        self._action_order = list(action_space.keys())  # Order of tokenizing

        self._tokens_per_action = 0
        # Calculate the number of tokens per action
        for action in self._action_order:
            # If this action is spaces.Discrete, this is one token.
            if isinstance(self._action_space[action], spaces.Discrete):
                self._tokens_per_action += 1

            # If a action is spaces.Box, get the number of tokens using shape.
            elif isinstance(self._action_space[action], spaces.Box):
                action_shape = self._action_space[action].shape
                if len(action_shape) != 1:
                    raise ValueError(
                        f'Only action shapes with single dimension supported, got {action_shape}')
                self._tokens_per_action += action_shape[0]
            else:
                raise ValueError(
                    'We assume action_space is defined by either gym.spaces.Discrete or gym.spaces.Box')

    @property
    def tokens_per_action(self) -> int:
        return self._tokens_per_action

    def tokenize(self, action: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Tokenizes an action."""
        action_tokens = []
        # Perform tokenizing in order of self._action_order
        # device = action[self._action_order[-1]].device
        # If a is Discrete, the size of a is () or (batch,), which means the former is scalar of Tensor type and the later is 1-d Tensor.
        for k in self._action_order:
            a = action[k]
            a_space = self._action_space[k]
            if isinstance(a_space, spaces.Discrete):
                assert torch.all(
                    a < self._vocab_size), "Discrete action should be smaller than vocab size."
                token = a  # Discrete action is already token. The size is () or (batch,)
                # The size is (1,) or (batch, 1). Discrete action will be one token.
                token = a.unsqueeze(-1)
            # if a is Box, size of a is (action_size) or (batch, action_size).
            else:
                low = torch.tensor(a_space.low, device=a.device)
                high = torch.tensor(a_space.high, device=a.device)
                a = torch.clamp(a, low, high)
                # Normalize the action.
                token = (a - low) / (high - low)
                # Bucket and discretize the action to vocab_size.
                token = token * (self._vocab_size - 1)
                # token = token.to(torch.int32) # The size is (action_size) or (batch, action_size).
                # print('\n\n\n\n token device: ', self.device)
            # if this action has action_size, this action will be action_size tokens.
            action_tokens.append(token.to(self.device).long())
        # Contatenate all actions. The size will be (tokens_per_action) or (batch,  tokens_per_action)
        action_tokens = torch.concat(action_tokens, dim=-1)
        return action_tokens

    # The size of action_tokens is (tokens_per_action) or  (batch, tokens_per_action)
    def detokenize(self, action_tokens: torch.Tensor, action_mean=None, action_std=None) -> Dict[str, torch.Tensor]:
        """Detokenizes an action."""
        action = {}
        token_index = 0
        # action_tokens is in self._action_order order
        # So we will detokenize in self._action_order order
        if action_mean is None:
            a_mean = np.zeros(self._tokens_per_action)
        if action_std is None:
            a_std = np.ones(self._tokens_per_action)

        action_mean = iter(action_mean)
        action_std = iter(action_std)

        a_mean = next(action_mean)
        a_std = next(action_std)
        for k in self._action_order:
            # Discrete actions are already assumed to be tokens.
            space = self._action_space[k]
            if isinstance(space, spaces.Discrete):
                # The size of action_tokens[k] : (1,) or (batch,), which means the former is scalar of Tensor type and the later is 1-d Tensor.
                action[k] = action_tokens[..., token_index]
                # A poor model may output tokens outside the allowed range, in that case
                # set them to a default value, the 0 token in this case.
                action[k] = torch.where(
                    action[k] > space.n, torch.zeros_like(action[k]), action[k])
                token_index += 1
            else:
                actions = []
                action_dim = space.shape[0]
                for j in range(action_dim):
                    # The size of a: (1,) or (batch, 1)
                    a = action_tokens[..., token_index:token_index + 1]
                    # a = a.to(torch.float32) # Change int32 to float32
                    # de-normalize
                    a = a / (self._vocab_size - 1)
                    a = a * (space.high[j] - space.low[j]) + space.low[j]
                    a = a / a_std + a_mean
                    a_mean = next(action_mean)
                    a_std = next(action_std)
                    # The size of actions: (1,) or (batch, 1)
                    actions.append(a.to(self.device))
                    token_index += 1
                # size: (action_dim) or (batch, action_dim)
                action[k] = torch.concat(actions, dim=-1)
        return action
