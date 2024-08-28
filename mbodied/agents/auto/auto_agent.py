from functools import wraps
from typing import Any, Dict, Literal, Type

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

    TASK_TO_AGENT_MAP: Dict[
        Literal[
            "language", "motion-openvla", "sense-object-detection", "sense-image-segmentation", "sense-depth-estimation"
        ],
        Type[Agent],
    ] = {
        "language": LanguageAgent,
        "motion-openvla": OpenVlaAgent,
        "sense-object-detection": ObjectDetectionAgent,
        "sense-image-segmentation": SegmentationAgent,
        "sense-depth-estimation": DepthEstimationAgent,
    }

    def __init__(
        self, task: str | None = None, model_src: str | None = None, model_kwargs: Dict | None = None, **kwargs
    ):
        """Initialize the AutoAgent with the specified task and model."""
        if model_kwargs is None:
            model_kwargs = {}
        self.task = task
        self.model_src = model_src
        self.model_kwargs = model_kwargs
        self.kwargs = kwargs
        self._initialize_agent()

    def _initialize_agent(self):
        """Initialize the appropriate agent based on the task."""
        if self.task not in self.TASK_TO_AGENT_MAP:
            if self.model_src is None:
                self.model_src = "openai"
            self.agent = LanguageAgent(model_src=self.model_src, model_kwargs=self.model_kwargs, **self.kwargs)
        else:
            self.agent = self.TASK_TO_AGENT_MAP[self.task](
                model_src=self.model_src, model_kwargs=self.model_kwargs, **self.kwargs
            )

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the agent if not found in AutoAgent."""
        try:
            attr = getattr(self.agent, name)
            if callable(attr):

                @wraps(attr)
                def wrapper(*args, **kwargs):
                    return attr(*args, **kwargs)

                return wrapper
            return attr
        except AttributeError as err:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'") from err

    def act(self, *args, **kwargs) -> Any:
        """Invoke the agent's act method without reinitializing the agent."""
        return self.agent.act(*args, **kwargs)

    @staticmethod
    def available_tasks() -> None:
        """Print available tasks that can be used with AutoAgent."""
        print("Available tasks:")  # noqa: T201
        for task in AutoAgent.TASK_TO_AGENT_MAP:
            print(f"- {task}")  # noqa: T201


def get_agent(
    task: Literal[
        "language", "motion-openvla", "sense-object-detection", "sense-image-segmentation", "sense-depth-estimation"
    ],
    model_src: str,
    model_kwargs: Dict | None = None,
    **kwargs,
) -> Agent:
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
        raise ValueError(
            f"Task '{task}' is not supported. Supported tasks: {list(AutoAgent.TASK_TO_AGENT_MAP.keys())}. "
            "Use AutoAgent.available_tasks() to view all available tasks."
        )

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
