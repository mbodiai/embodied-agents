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

"""PyTorch implementation of Token Learner(Ryoo et al 2021)."""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from einops import rearrange


class TokenLearnerModule(nn.Module):
    def __init__(
        self,
        inputs_channels: int,
        num_tokens: int,
        bottleneck_dim: int = 64,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.layerNorm = nn.LayerNorm(inputs_channels)

        self.conv1 = nn.Conv2d(
            in_channels=inputs_channels,
            out_channels=bottleneck_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.gelu1 = nn.GELU(approximate="tanh")

        if dropout_rate > 0:
            self.dropout1 = nn.Dropout2d(dropout_rate)
        else:
            self.dropout1 = nn.Identity()

        self.conv2 = nn.Conv2d(
            in_channels=bottleneck_dim,
            out_channels=num_tokens,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        if dropout_rate > 0:
            self.dropout2 = nn.Dropout2d(dropout_rate)
        else:
            self.dropout2 = nn.Identity()

    def forward(self, inputs: torch.Tensor):

        inputs = rearrange(inputs, 'batch (h w) c -> batch h w c', h=14, w=14)

        # layer norm
        x = self.layerNorm(inputs)
        x = x.permute(0, 3, 1, 2)

        # create weights map
        x = self.gelu1(self.conv1(x))
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.dropout2(x)  # (bs, num_tokens, h, w)

        x = x.view(x.shape[0], x.shape[1], -1)  # (bs, num_tokens, h*w)
        weights_maps = F.softmax(x, dim=-1)

        # create tokens
        bs, h, w, c = inputs.shape
        inputs = inputs.view(bs, h * w, c)

        tokens = torch.bmm(weights_maps, inputs)
        # weighs_maps: [bs, n_token, h*w]
        # inputs: [bs, h * w, c]
        # tokens: [bs, n_token, c]

        # Above computation is equivalent to below explanation.
        # weights_maps has n_tokens channels. each channels is a weight map with the size of (h, w).
        # inputs has c channels, each channels size is (h, w).
        # compute the element-wise product of one weight map of weights_maps and one of channels of inputs
        # That result in (H, W). After sum this, we get a scalar.
        # Iterate this operation to all other channels of inputs using the same weight map, we get c scalar.
        # reshape (1, 1, c), then we get a learned token, as shown in Fig. 1 in tokenlearner paper.
        # We do the computation using all other weight map, then we get all tokens.
        return tokens
