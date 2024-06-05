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

"""A FiLM Efficientnet contextual image tokenizer used in Robotics Transformer 1.
"""

from typing import Optional

import torch
import torch.nn as nn
from film_efficientnet.efficient_net import EfficientNetB5
from tokenizers.token_learner import TokenLearnerModule


class RT1ImageTokenizer(nn.Module):
    def __init__(
        self,
        embedding_output_dim: int = 512,
        language_embedding_size: int = 512,
        use_token_learner: bool = False,
        num_tokens: int = 8,
    ):
        super().__init__()
        self._tokenizer = EfficientNetB5(
            language_embedding_size,
        )

        self._use_token_learner = use_token_learner
        if self._use_token_learner:
            self._num_tokens = num_tokens
            self._token_learner = TokenLearnerModule(
                inputs_channels=embedding_output_dim, num_tokens=self._num_tokens
            )

    @property
    def tokens_per_context_image(self) -> int:
        if self._use_token_learner:
            num_tokens = self._num_tokens
        else:
            num_tokens = 100
        return num_tokens

    # Note that context is the same value along with time axis.
    # This means (b, 0, embedding_dim) == (b, 1, embedding_dim) == (b, 2, embedding_dim) ...
    def forward(
        self, image: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Gets image tokens.

        Args:
        image: Images of shape (b, t, 3, h, w) to tokenize.
        context: An optional context vector (e.g., a natural language embedding).
            Expected to have shape (b, t, embedding_dim).
        training: Whether or not we are in training mode.

        Returns:
        tokens: has shape (batch, t, num_tokens_per_timestep, embedding_dim)
        """
        b, t, c, h, w = image.shape

        # Fold the time axis into the batch axis.
        image = image.view(b * t, c, h, w)
        if context is not None:
            context = context.reshape(b * t, -1)

        tokens = self._tokenizer(image, context=context)  # [b * t, 512 , 10, 10]

        if self._use_token_learner:
            tokens = self._token_learner(tokens)  # [b * t, num_token, 512]
            # Unflatten the time axis, which was previously flattened into the batch.
            tokens = tokens.view(b, t, tokens.shape[1], -1)
            return tokens  # [b, t, num_token, 512]
        else:
            # Unflatten the time axis, which was previously flattened into the batch.
            tokens = tokens.view(b, t, 512, -1)  # [b, t, 512 , 10 * 10]
            # If you don't use token learner, the number of token is 100.
            tokens = tokens.transpose(2, 3)  # [b, t, 10 * 10, 512]
            return tokens
