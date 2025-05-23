# Copyright 2024 mbodi ai
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

from mbodied.agents.sense.sensory_agent import SensoryAgent
from mbodied.types.sense.vision import Image


class DepthEstimationAgent(SensoryAgent):
    """A depth estimation agent that uses a remote depth estimation server to estimate depth from an image.

    Examples:
    >>> agent = DepthEstimationAgent(model_src="https://api.mbodi.ai/sense/")
    >>> result = agent.act(image=Image("resources/xarm.jpeg", size=(224, 224)))
    """

    def __init__(
        self,
        model_src="https://api.mbodi.ai/sense/",
        model_kwargs=None,
        **kwargs,
    ):
        super().__init__(
            model_src=model_src,
            model_kwargs=model_kwargs,
            **kwargs,
        )

    def act(self, image: Image, *args, api_name: str = "/depth", **kwargs) -> Image:
        """Act based on the prompt and image using the remote depth estimation server.

        Args:
            image (Image): The image to act on.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Image: The depth image generated by the agent
        """
        if self.actor is None:
            raise ValueError("Remote actor for agent not initialized.")
        response, depth_file = self.actor.predict(image.base64, *args, api_name=api_name, **kwargs)
        return Image(response), np.load(depth_file)


# Example usage:
if __name__ == "__main__":
    agent = DepthEstimationAgent(model_src="https://api.mbodi.ai/sense/")
    result, depth_array = agent.act(image=Image("resources/bridge_example.jpeg"))
    result.pil.show()
    print("Depth array shape", depth_array.shape)
