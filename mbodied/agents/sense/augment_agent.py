from mbodied.agents.sense.sensory_agent import SensoryAgent
from mbodied.types.sense.vision import Image


class AugmentAgent(SensoryAgent):
    """Augment agent to generate augmented images."""

    def __init__(
        self,
        model_src=None,
        model_kwargs=None,
        **kwargs,
    ):
        super().__init__(
            model_src=model_src,
            model_kwargs=model_kwargs,
            **kwargs,
        )

    def act(self, instruction: str, image: Image) -> Image:
        """Act based on the instruction and image using the remote server."""
        if self.actor is None:
            raise ValueError("Remote actor for Augment agent not initialized.")
        response = self.actor.predict(image.base64, instruction)
        return Image(response)


# Example usage:
if __name__ == "__main__":
    augment_agent = AugmentAgent(model_src="https://api.mbodi.ai/augment/")
    result = augment_agent.act(instruction="change lighting", image=Image("resources/xarm.jpeg", size=(224, 224)))
    print(result)
    result.pil.show()
