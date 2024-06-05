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
"""Encoder based on Efficientnet."""

# film_efficientnet_encoder receive [bs, 3, 300, 300] and returns [bs, 1536, 10, 10]
# Here we use 1x1 conv. [bs, 1536, 10, 10] -> [bs, 512, 10, 10]
# then apply FiLM.

from typing import Optional

import torch
import torch.nn as nn
from film_efficientnet.film import film_conditioned
from film_efficientnet.film_efficientnet_encoder import EfficientNetB3


@film_conditioned(context_key="context")
class EfficientNetEncoder(nn.Module):

  def __init__(
      self,
      token_embedding_size: int = 512,
      language_embedding_size: int = 512,
      weights: Optional[str] = "imagenet",
      early_film: bool = True,
      include_top: bool = True,
      pooling: bool = True,
  ):
    super().__init__()

    self.conv1x1 = nn.Conv2d(
        in_channels=
        1536,  # If we use EfficientNetB3 and input image has 3 channels, in_channels is 1536.
        out_channels=token_embedding_size,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=False,
    )
    self.net = EfficientNetB3(
        weights=weights,
        include_top=include_top,
        include_film=early_film,
        text_embed_dim=language_embedding_size,
    )

    self.early_film = early_film
    self._pooling = pooling

  def forward(self, image: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    features = self.net(image, context if self.early_film else None)
    features = self.conv1x1(features)
    features = self.film_layer(features, context)

    if not self._pooling:
      return features

    # Global average pool.
    return torch.mean(features, dim=(2, 3))
