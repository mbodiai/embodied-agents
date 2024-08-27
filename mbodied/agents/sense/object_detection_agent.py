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

from mbodied.agents.sense.sensory_agent import SensoryAgent
from mbodied.types.sense.vision import Image
from mbodied.types.sense.world import World


class ObjectDetectionAgent(SensoryAgent):
    """A object detection agent that uses a remote object detection, i.e. YOLOWorld, Grounding Dino, to detect objects in an image."""

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

    def act(
        self,
        image: Image,
        objects: list[str] | str,
        model_type: str = "YOLOWorld",
        *args,
        api_name: str = "/detect",
        **kwargs,
    ) -> World:
        """Act based on the prompt and image using the remote object detection server.

        Args:
            image (Image): The image to act on.
            objects (list[str] | str): The objects to detect in the image.
            model_type (str): The model type to use for the object detection. Options: "YOLOWorld", "Grounding DINO".
            *args: Variable length argument list.
            api_name (str): The name of the API endpoint to use.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            World: The world data with the detected objects.
        """
        if self.actor is None:
            raise ValueError("Remote actor for agent not initialized.")

        if isinstance(objects, list):
            objects = ",".join(objects)
        annotated_img, json_dict = self.actor.predict(
            image.base64, objects, model_type=model_type, *args, api_name=api_name, **kwargs
        )
        return World.model_validate(json_dict)


# Example usage:
if __name__ == "__main__":
    agent = ObjectDetectionAgent(model_src="https://api.mbodi.ai/sense/")
    result = agent.act(
        image=Image("resources/bridge_example.jpeg", size=(224, 224)),
        objects=["spoon", "bowl"],
        model_type="Grounding DINO",
    )
    result.annotated.pil.show()
