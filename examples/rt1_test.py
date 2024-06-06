import torch
from mbodied_agents.agents.motion.rt1.rt1_agent import RT1Agent

config = {
    "num_layers": 8,
    "layer_size": 128,
    "observation_history_size": 6,
    "future_prediction": 6,
    "token_embedding_dim": 512,
    "causal_attention": True,
}

rt1_agent = RT1Agent(config)

actions = rt1_agent.act(image=torch.rand(224, 224, 3),
              instruction_emb=torch.rand(1, 512))
