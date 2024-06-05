from collections import OrderedDict

import numpy as np
import torch.nn as nn
from gym import spaces

from .tokenizers.action_tokenizer import RT1ActionTokenizer as ActionTokenizer
from .transformer_network import TransformerNetwork


class RT1Module(nn.Module):
    def __init__(self, config):
        super(RT1Module, self).__init__()
        self.config = config
        self.action_tokenizer = ActionTokenizer()
        self.configure_model()

    def configure_model(self) -> None:
        if hasattr(self, 'model'):
            return
        observation_space = spaces.Dict({
            'image_primary': spaces.Box(low=0.0, high=1.0, shape=(3, 224, 224), dtype=np.float32),
            'natural_language_embedding': spaces.Box(low=-np.inf, high=np.inf, shape=[512], dtype=np.float32)
        })
        action_space_dict = OrderedDict([
            ("xyz", spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)),
            ("rpy", spaces.Box(low=-np.pi, high=np.pi, shape=(3,), dtype=np.float32)),
            ("grasp", spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)),
        ])
        action_space = spaces.Dict(action_space_dict)

        # TODO: Get action mean and std
        action_mean = None
        action_std = None
        
        self.model = TransformerNetwork(
            observation_history_length=self.config['observation_history_size'],
            future_prediction_length=self.config['future_action_window_size'],
            token_embedding_dim=self.config['token_embedding_dim'],
            causal_attention=self.config['causal_attention'],
            num_layers=self.config['num_layers'],
            layer_size=self.config['layer_size'],
            observation_space=observation_space,
            action_space=action_space,
            image_keys=['image_primary'],
            context_key='natural_language_embedding',
            action_mean=action_mean,
            action_std=action_std,
        )

    def forward(self, x):
        return self.model(x)
