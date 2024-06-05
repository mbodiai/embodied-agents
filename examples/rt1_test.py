import torch
from mbodied_agents.agents.models.rt1.rt1_module import RT1Module

config = {
    'observation_history_size': 10,
    'future_action_window_size': 10,
    'token_embedding_dim': 512,
    'causal_attention': True,
    'num_layers': 4,
    'layer_size': 256,
}

# model = RT1Module(config)

# Create a dummy input
dummy_input = {
    'image_primary': torch.rand(1, 3, 224, 224),  # Batch size of 1
    'natural_language_embedding': torch.rand(1, 512),
}

# Pass the dummy input through the model
# output = model(dummy_input)