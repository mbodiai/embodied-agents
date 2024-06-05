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

import copy
import math
import os

import torch
import torch.nn as nn
from film_efficientnet.film import FilmLayer
from torchvision.ops import StochasticDepth
from torchvision.ops.misc import Conv2dNormActivation

# This is based on Table 1 in a EfficientNet paper.
DEFAULT_BLOCKS_ARGS = [
    {
        "kernel_size": 3,
        "repeats": 1,
        "in_size": 32,
        "out_size": 16,
        "expand_ratio": 1,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 2,
        "in_size": 16,
        "out_size": 24,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 2,
        "in_size": 24,
        "out_size": 40,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 3,
        "in_size": 40,
        "out_size": 80,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 3,
        "in_size": 80,
        "out_size": 112,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 4,
        "in_size": 112,
        "out_size": 192,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 1,
        "in_size": 192,
        "out_size": 320,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
]

CONV_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {"scale": 2.0, "mode": "fan_out", "distribution": "truncated_normal"},
}

DENSE_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {"scale": 1.0 / 3.0, "mode": "fan_out", "distribution": "uniform"},
}


# Multiply the number of filters by width_coefficient.
# Usually, divisor = 8. This means new_filters is a multiple of 8.
# Filters means channels.
# We round by the 8, not by the 10.
def round_filters(filters, divisor, width_coefficient):
    """Round number of filters based on depth multiplier."""
    filters *= width_coefficient

    # int(filters + divisor / 2) // divisor * divisor is an operation like rounding off.
    # Usual rounding off is operated at 5(=10 / 2) interval. But this is operated at 4(=8 / 2) interval.
    # If filters = 35, new_filter = 32. If filters = 36, new_filter = 40.
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


# Multiply the number of repeats by depth_coefficient.
def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier."""
    return int(math.ceil(depth_coefficient * repeats))


class SeModule(nn.Module):
    def __init__(self, expand_size, block_in_size, se_ratio=0.25):
        super(SeModule, self).__init__()

        se_size = max(1, int(block_in_size * se_ratio))

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            expand_size, se_size, kernel_size=1, stride=1, padding=0
        )  # Note that we use bias=True here.
        self.silu0 = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(
            se_size, expand_size, kernel_size=1, stride=1, padding=0
        )  # Note that we use bias=True here.
        self.act = nn.Sigmoid()

    def forward(self, inputs):
        x = self.avgpool(inputs)
        x = self.fc1(x)
        x = self.silu0(x)
        x = self.fc2(x)
        x = self.act(x)
        return inputs * x


# Inverted Residual + Squeeze-and-Excitation
class MBConvBlock(nn.Module):
    """expand + depthwise + pointwise"""

    def __init__(
        self,
        kernel_size,
        in_size,
        out_size,
        expand_ratio,
        id_skip,
        strides,
        se_ratio,
        drop_rate,
    ):
        super(MBConvBlock, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.expand_ratio = expand_ratio
        self.strides = strides
        self.se_ratio = se_ratio
        self.id_skip = id_skip
        self.drop_rate = drop_rate

        expand_size = in_size * expand_ratio

        layers = []

        if not (1 <= strides <= 2):
            raise ValueError("illegal stride value")

        # expansion phase (1x1 conv)
        if expand_ratio != 1:
            layers.append(
                Conv2dNormActivation(
                    in_size,
                    expand_size,
                    kernel_size=1,
                    stride=1,
                    norm_layer=nn.BatchNorm2d,
                    activation_layer=nn.SiLU,
                )
            )

        # depthwise conv
        layers.append(
            Conv2dNormActivation(
                expand_size,
                expand_size,
                kernel_size=kernel_size,
                stride=strides,
                groups=expand_size,
                norm_layer=nn.BatchNorm2d,
                activation_layer=nn.SiLU,
            )
        )

        # Squeeze-and-Excitation module
        if 0 < se_ratio <= 1:
            layers.append(SeModule(expand_size, in_size, se_ratio))

        # output phase (1x1 conv)
        layers.append(
            Conv2dNormActivation(
                expand_size,
                out_size,
                kernel_size=1,
                stride=1,
                norm_layer=nn.BatchNorm2d,
                activation_layer=None,  # Note that there is no activation here
            )
        )

        self.block = nn.Sequential(*layers)

        # Dropout
        if drop_rate > 0:
            self.dropout = StochasticDepth(drop_rate, "row")

    def forward(self, inputs):
        x = self.block(inputs)

        # Dropout and skip connection
        if self.id_skip and self.strides == 1 and self.in_size == self.out_size:
            if self.drop_rate > 0:
                x = self.dropout(x)
            x = inputs + x

        return x


class EfficientNet(nn.Module):
    def __init__(
        self,
        width_coefficient,
        depth_coefficient,
        dropout_rate=0.2,
        drop_connect_rate=0.2,
        depth_divisor=8,
        blocks_args="default",
        include_top=True,
        classes=1000,
        include_film=False,
        text_embed_dim=512,
    ):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.include_top = include_top
        self.include_film = include_film

        # input dimensions
        in_channels = 3

        # get defualt setting of MBConv blocks
        if blocks_args == "default":
            blocks_args = DEFAULT_BLOCKS_ARGS

        # stem
        out_channels = round_filters(32, depth_divisor, width_coefficient)
        self.convNormAct0 = Conv2dNormActivation(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.SiLU,
        )

        # Build blocks
        blocks_args = copy.deepcopy(blocks_args)
        # Detach blocks_args from original DEFAULT_BLOCKS_ARGS. This is necessary to keep DEFAULT_BLOCKS_ARGS as it is when you add change to blocks_args.
        # Therefore, you can make multiple models.
        blocks = []
        films = []
        b = 0
        total_repeats = float(
            sum(
                round_repeats(args["repeats"], depth_coefficient)
                for args in blocks_args
            )
        )  # sum of all of repeat
        for args in blocks_args:  # args is dictionary
            assert args["repeats"] > 0
            # Update block input and output filters based on depth multiplier.
            args["in_size"] = round_filters(
                args["in_size"], depth_divisor, width_coefficient
            )
            args["out_size"] = round_filters(
                args["out_size"], depth_divisor, width_coefficient
            )

            # We delete repeats in args so that we could write MBConv(**args).
            for j in range(round_repeats(args.pop("repeats"), depth_coefficient)):
                if j == 0:
                    # The first block
                    blocks.append(
                        MBConvBlock(
                            **args,
                            drop_rate=drop_connect_rate
                            * b
                            / total_repeats,  # increase drop_connect_rate linearlly
                        )
                    )
                    args["strides"] = 1
                    args["in_size"] = args["out_size"]
                else:
                    blocks.append(
                        MBConvBlock(
                            **args,
                            drop_rate=drop_connect_rate * b / total_repeats,
                        )
                    )

                if include_film:
                    films.append(
                        FilmLayer(
                            num_channels=args["out_size"],
                            context_dim=text_embed_dim,
                        )
                    )
                b += 1

        self.blocks = nn.ModuleList(blocks)
        if include_film:
            self.films = nn.ModuleList(films)

        # Build top
        in_channels = args["out_size"]
        out_channels = 1280
        out_channels = round_filters(out_channels, depth_divisor, width_coefficient)
        self.convNormAct1 = Conv2dNormActivation(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.SiLU,
        )

        # If we include top, we do gloval average pooling and dropout. Then, we add fully connected layer.
        if include_top:
            self.glovalAvePool = nn.AdaptiveAvgPool2d(1)
            if dropout_rate > 0:
                self.dropout = nn.Dropout(dropout_rate)
            # Fully connected layer
            self.fc = nn.Linear(out_channels, classes)


    def forward(self, x: torch.Tensor, context=None):
        # stem
        outputs = self.convNormAct0(x)

        # Blocks(26 MBconv)
        if self.include_film:
            for block, film in zip(self.blocks, self.films):
                outputs = block(outputs)  # MBConv
                outputs = film(outputs, context)  # FiLM

        else:
            for block in self.blocks:
                outputs = block(outputs)

        # top
        outputs = self.convNormAct1(outputs)

        if self.include_top:
            outputs = self.glovalAvePool(outputs)
            if self.dropout_rate > 0:
                outputs = self.dropout(outputs)
            outputs = torch.flatten(outputs, 1)
            outputs = self.fc(outputs)
            return outputs
        else:
            return outputs


# If you use FiLM, this function allow us to load pretrained weight from naive efficientnet.
def maybe_restore_with_film(
    *args, weights="imagenet", include_top=False, include_film=True, **kwargs
):
    assert (
        weights is None or weights == "imagenet"
    ), "Set weights to either None or 'imagenet'."
    # Create model without FiLM
    n1 = EfficientNet(*args, include_top=include_top, include_film=False, **kwargs)

    # Load weights.
    if weights == "imagenet":
        if include_top:
            weights_path = os.path.join(
                os.path.dirname(__file__), "efficientnet_checkpoints/efficientnetb3.pth"
            )
        else:
            weights_path = os.path.join(
                os.path.dirname(__file__),
                "efficientnet_checkpoints/efficientnetb3_notop.pth",
            )
        # This EfficientNet differs from the official pytorch implementation only in parameter names.
        # So we load the pytorch weights in using this function.
        n1 = load_official_pytorch_param(n1, weights_path)

    # If you don't use FiLM, that's all.
    if not include_film:
        return n1

    # Create model with FiLM if you use FiLM. And we load pretrained weights from n1.
    n2 = EfficientNet(*args, include_top=include_top, include_film=True, **kwargs)
    if weights is None:
        return n2

    n1_state_dict = n1.state_dict().copy()
    n2_state_dict = n2.state_dict().copy()

    for name, param in n2_state_dict.items():
        if name in n1_state_dict:
            n2_state_dict[name] = n1_state_dict[name]

    n2.load_state_dict(n2_state_dict)
    return n2


# This function helps load official pytorch efficientnet's weights.
def load_official_pytorch_param(model: nn.Module, weights_path):
    # load weights
    official_state_dict = torch.load(weights_path)

    film_eff_state_dict = model.state_dict().copy()
    keys_official_list = list(official_state_dict)
    keys_film_eff_list = list(film_eff_state_dict)

    for key_official, key_film_eff in zip(keys_official_list, keys_film_eff_list):
        film_eff_state_dict[key_film_eff] = official_state_dict[key_official]
        # print(str(key_official) + "->" + str(key_film_eff))

    # load new weights
    model.load_state_dict(film_eff_state_dict)
    print('Loaded pretrained EfficientNet!')
    return model


# EfficientNetB3 is tranined on 300x300 image.
def EfficientNetB3(
    weights="imagenet", include_top=True, classes=1000, include_film=True, **kwargs
):
    return maybe_restore_with_film(
        1.2,
        1.4,
        0.3,
        weights=weights,
        include_top=include_top,
        classes=classes,
        include_film=include_film,
        **kwargs,
    )
