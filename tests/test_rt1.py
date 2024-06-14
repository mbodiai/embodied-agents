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

import torch
import pytest
from mbodied_agents.agents.motion.rt1.rt1_agent import RT1Agent
from mbodied_agents.types.controls import Motion

@pytest.fixture
def rt1_agent_config():
    return {
        "num_layers": 8,
        "layer_size": 128,
        "observation_history_size": 6,
        "future_prediction": 6,
        "token_embedding_dim": 768,
        "causal_attention": True,
    }

@pytest.fixture
def rt1_agent(rt1_agent_config):
    return RT1Agent(rt1_agent_config)

def test_rt1_agent_act(rt1_agent):
    # Create dummy inputs
    image = torch.rand(224, 224, 3)  # Assuming input shape is (224, 224, 3)
    instruction = "Pick up the ball"  # Assuming embedding shape is (1, 512)
    
    # Call the act method
    actions = rt1_agent.act(image=image, instruction=instruction)
    
    # Verify the actions output
    assert isinstance(actions, list) and all(isinstance(action, Motion) for action in actions), "Actions should be a list."
    

# Run the test
if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
