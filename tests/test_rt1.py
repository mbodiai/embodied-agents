import torch
import pytest
from mbodied_agents.agents.motion.rt1.rt1_agent import RT1Agent
from mbodied_agents.types.controls import HandControl

@pytest.fixture
def rt1_agent_config():
    return {
        "num_layers": 8,
        "layer_size": 128,
        "observation_history_size": 6,
        "future_prediction": 6,
        "token_embedding_dim": 512,
        "causal_attention": True,
    }

@pytest.fixture
def rt1_agent(rt1_agent_config):
    return RT1Agent(rt1_agent_config)

def test_rt1_agent_act(rt1_agent):
    # Create dummy inputs
    image = torch.rand(224, 224, 3)  # Assuming input shape is (224, 224, 3)
    instruction_emb = torch.rand(1, 512)  # Assuming embedding shape is (1, 512)
    
    # Call the act method
    actions = rt1_agent.act(image=image, instruction_emb=instruction_emb)
    
    # Verify the actions output
    assert isinstance(actions, list), "Actions should be a dictionary."
    

# Run the test
if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
