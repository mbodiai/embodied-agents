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


from functools import partial

import torch
import torch.nn as nn
from einops import rearrange
from torchvision.models.efficientnet import MBConv


def film_conditioned(cls):
    """Decorator to add FiLM conditioning to a forward method. Adds argument context
    to forward method and applies FiLM conditioning to the output of the forward.

    Args:
    context_key (str, optional): The argument name for the context on which to 
    condition the forward pass. Defaults to "context".
    """
    init_func = cls.__init__

    def init_with_film_layers(*args, **kwargs):
        context_dim = kwargs.get("context_dim", None)
        if context_dim is None:
            return init_func(*args, **kwargs)
        self = init_func(*args, **kwargs)
        film_layers = []
        for layer in self.features:
            for sublayer in layer:
                if isinstance(sublayer, MBConv):
                    film_layers.append(
                        FilmLayer(sublayer.out_channels, context_dim))

        # Don't add a film layer to the last layer
        self.film_layers = nn.ModuleList(film_layers[:-1])
        return self

    cls.__init__ = init_with_film_layers

    forward_func = cls.forward

    def forward_with_film_layers(*args, **kwargs):
        if 'conditioning_funcs' in kwargs or kwargs.get('context', None) is None:
            return forward_func(*args, **kwargs)

        self = args[0]
        context = kwargs['context']
        conditioners = iter([partial(film_layer, context=context)
                            for film_layer in self.film_layers])
        return forward_func(*args, **kwargs, conditioning_funcs=conditioners)

    cls.forward = forward_with_film_layers
    return cls


class FilmLayer(nn.Module):
    """Layer to conditionally modulate the input tensor with the context tensor."""

    def __init__(
        self,
        num_channels: int,
        context_dim: int = 512,
    ):
        super().__init__()
        self.beta = nn.Linear(context_dim, num_channels, bias=False)
        self.gamma = nn.Linear(context_dim, num_channels, bias=False)

        nn.init.constant_(self.beta.weight, 0)
        nn.init.constant_(self.gamma.weight, 0)

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        context = context.to(x.device)
        beta = self.beta(context)
        gamma = self.gamma(context)

        beta = rearrange(beta, 'b c -> b c 1 1')
        gamma = rearrange(gamma, 'b c -> b c 1 1')

        # Initialize to identity op.
        result = (1 + gamma) * x + beta

        return result
