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

import numpy as np
import pytest
from mbodied_agents.agents.sense.dino.dino_object_detection_agent import DinoObjectDetectionAgent

@pytest.fixture
def detection_agent_config():
    return {
        # Add any necessary configuration parameters for DinoObjectDetectionAgent if required
    }

@pytest.fixture
def detection_agent(detection_agent_config):
    return DinoObjectDetectionAgent(**detection_agent_config)

def test_detection_agent_act(detection_agent):
    # Create a dummy input image
    image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    
    # Define the labels you are interested in detecting
    labels = ["person", "car"]
    
    # Call the act method
    detections = detection_agent.act(image=image, labels=labels)
    
    # Verify the output
    assert isinstance(detections, list) and all(isinstance(d, dict) for d in detections), "Detections should be a list of dictionaries."

# Run the test
if __name__ == "__main__":
    pytest.main([__file__, "-vv"])