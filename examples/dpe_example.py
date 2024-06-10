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
from mbodied_agents.agents.sense.dino.dino_object_detection_agent import DinoObjectDetectionAgent


def main():
    # Initialize the instance of DinoPoseEstimator
    object_detection_agent = DinoObjectDetectionAgent()

    # Load an image
    image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    # Define the labels you are interested in detecting
    labels = ["person", "car"]

    # Use the detect_and_segment function
    image_array, detections = object_detection_agent.act(
        image=image,
        labels=labels,
    )

    # Output results
    print(f"Processed Image Array Shape: {image_array.shape}")
    print("Detections:")
    print(detections)
        
        
if __name__ == "__main__":
    main()
