import numpy as np
import torch
from gym import spaces
from mbodied_agents.agents.motion.rt1.rt1_agent import RT1Agent

# Define the observation and action spaces
observation_space = spaces.Dict({
    'image': spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
    'instruction': spaces.Discrete(10)
})
action_space = spaces.Dict({
    'gripper_position': spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
    'gripper_action': spaces.Discrete(2)
})

config = {
    "num_layers": 8,
    "layer_size": 128,
    "observation_history_size": 6,
    "future_prediction": 6,
    "token_embedding_dim": 512,
    "causal_attention": True,
}

rt1_agent = RT1Agent(config)

for name, param in rt1_agent.model.named_parameters():
    print(f"{name}: {param.device}")

rt1_agent.act(image=torch.rand(224, 224, 3),
              instruction_emb=torch.rand(1, 512))

# Pass the dummy input through the model
# output = model(dummy_input)
