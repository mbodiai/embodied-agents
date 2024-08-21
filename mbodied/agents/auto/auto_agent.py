from typing import Any

from transformers import pipeline

from mbodied.agents import Agent
from mbodied.agents.language import LanguageAgent
from mbodied.agents.motion.openvla_agent import OpenVlaAgent
from mbodied.agents.sense.depth_estimation_agent import DepthEstimationAgent
from mbodied.agents.sense.object_detection_agent import ObjectDetectionAgent
from mbodied.agents.sense.segmentation_agent import SegmentationAgent
from mbodied.types.sense.vision import Image


class AutoAgent(Agent):
    """AutoAgent that dynamically selects and initializes the correct agent based on the task and model.

    Example Usage:
    # AutoAgent as LanguageAgent:
    auto_agent = AutoAgent(task="language", model_src="openai")
    response = auto_agent.act("What is the capital of France?")

    # AutoAgent as MotorAgent:
    auto_agent = AutoAgent(task="motion-openvla", model_src="https://api.mbodi.ai/community-models/")
    action = auto_agent.act("move hand forward", Image(size=(224, 224)))

    # AutoAgent as SenseAgent:
    auto_agent = AutoAgent(task="sense-depth-estimation", model_src="https://api.mbodi.ai/sense/")
    depth = auto_agent.act(image=Image("resources/bridge_example.jpeg", size=(224, 224)))
    """

    TASK_TO_AGENT_MAP = {
        "language": LanguageAgent,
        "motion-openvla": OpenVlaAgent,
        "sense-object-detection": ObjectDetectionAgent,
        "sense-image-segmentation": SegmentationAgent,
        "sense-depth-estimation": DepthEstimationAgent,
    }

    def __init__(self, task: str, model_src: str, model_kwargs: dict = None, **kwargs):
        """Initialize the AutoAgent with the specified task and model.

        Automatically selects the appropriate agent based on the task.

        Args:
            task (str): The task to perform.
                - "language": Language understanding and generation.
                    = initialize as LanguageAgent
                - "motion-openvla": Motion generation using OpenVLA.
                    = initialize as OpenVlaAgent
                - "sense-object-detection": Object detection.
                    = initialize as ObjectDetectionAgent
                - "sense-image-segmentation": Image segmentation.
                    = initialize as SegmentationAgent
                - "sense-depth-estimation": Depth estimation.
                    = initialize as DepthEstimationAgent

            model_src (str): The model source to use for the task.
            model_kwargs (dict): Additional keyword arguments to pass to the model.
            **kwargs: Additional keyword arguments to pass to the agent.
        """
        if task not in self.TASK_TO_AGENT_MAP:
            raise ValueError(f"Task '{task}' is not supported. Supported tasks: {list(self.TASK_TO_AGENT_MAP.keys())}")

        if model_kwargs is None:
            model_kwargs = {}
        self.agent = self.TASK_TO_AGENT_MAP[task](model_src=model_src, model_kwargs=model_kwargs, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the agent if not found in AutoAgent."""
        try:
            return getattr(self.agent, name)
        except AttributeError as err:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'") from err

    def act(self, *args, **kwargs) -> Any:
        """Invoke the agent's act method."""
        return self.agent.act(*args, **kwargs)


def get_agent(task: str, model_src: str, model_kwargs: dict = None, **kwargs) -> Agent:
    """Initialize the AutoAgent with the specified task and model.

    This is an alternative to using the AutoAgent class directly. It returns the corresponding agent instance directly.

    Usage:
    # Get LanguageAgent instance
    language_agent = get_agent(task="language", model_src="openai")
    response = language_agent.act("What is the capital of France?")

    # Get OpenVlaAgent instance
    openvla_agent = get_agent(task="motion-openvla", model_src="https://api.mbodi.ai/community-models/")
    action = openvla_agent.act("move hand forward", Image(size=(224, 224)))

    # Get DepthEstimationAgent instance
    depth_agent = get_agent(task="sense-depth-estimation", model_src="https://api.mbodi.ai/sense/")
    depth = depth_agent.act(image=Image("resources/bridge_example.jpeg", size=(224, 224)))
    """
    if task not in AutoAgent.TASK_TO_AGENT_MAP:
        raise ValueError(f"Task '{task}' is not supported. Supported tasks: {list(AutoAgent.TASK_TO_AGENT_MAP.keys())}")

    if model_kwargs is None:
        model_kwargs = {}
    return AutoAgent.TASK_TO_AGENT_MAP[task](model_src=model_src, model_kwargs=model_kwargs, **kwargs)


# Example usage
if __name__ == "__main__":
    auto_agent = AutoAgent(task="language", model_src="openai")
    response = auto_agent.act("What is the capital of France?")
    print(response)

    stream = auto_agent.act_and_stream("What is the capital of France?")
    for chunk in stream:
        print(chunk)

    auto_agent = AutoAgent(task="motion-openvla", model_src="https://api.mbodi.ai/community-models/")
    action = auto_agent.act("move hand forward", Image(size=(224, 224)))
    print(action)

    auto_agent = AutoAgent(
        task="motion-openvla", model_src="gradio", model_kwargs={"endpoint": "https://api.mbodi.ai/community-models/"}
    )
    action = auto_agent.act("move hand forward", Image(size=(224, 224)), "bridge_orig")
    print(action)

    auto_agent = AutoAgent(task="sense-depth-estimation", model_src="https://api.mbodi.ai/sense/")
    image = Image("resources/bridge_example.jpeg", size=(224, 224))
    result = auto_agent.act(image=image)
    result.pil.show()

    auto_agent = get_agent(task="language", model_src="openai")
    response = auto_agent.act("What is the capital of France?")
    print(response)
