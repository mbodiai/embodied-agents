from typing import Any, Dict, Literal, Protocol, Type, overload

from rich.console import Console
from typing_extensions import TypedDict, cast

from mbodied.agents import Agent
from mbodied.agents.agent import ModelSource
from mbodied.agents.language import LanguageAgent
from mbodied.agents.motion.openvla_agent import OpenVlaAgent
from mbodied.agents.sense.depth_estimation_agent import DepthEstimationAgent
from mbodied.agents.sense.object_detection_agent import ObjectDetectionAgent
from mbodied.agents.sense.segmentation_agent import SegmentationAgent
from mbodied.agents.sense.sensor_agent import SensorAgent
from mbodied.types.sense.vision import Image

TaskTypes = Literal["language","motion","sense","detection","segmentation","depth"]
class TaskAgentMap(TypedDict):
    language: Type[LanguageAgent]
    motion: Type[OpenVlaAgent]
    detection: Type[ObjectDetectionAgent]
    segmentation: Type[SegmentationAgent]
    depth: Type[DepthEstimationAgent]
    sense: Type[SensorAgent]

console = Console()
class CustomTypedDict(Protocol):
    @overload
    @classmethod
    def __getitem__(cls, key: Literal["language"]) -> Type[LanguageAgent]: ...
    @overload
    @classmethod
    def __getitem__(cls, key: Literal["motion"]) -> Type[OpenVlaAgent]: ...
    @overload
    @classmethod
    def __getitem__(cls, key: Literal["detection"]) -> Type[ObjectDetectionAgent]: ...
    @overload
    @classmethod
    def __getitem__(cls, key: Literal["segmentation"]) -> Type[SegmentationAgent]: ...
    @overload
    @classmethod
    def __getitem__(cls, key: Literal["depth"]) -> Type[DepthEstimationAgent]: ...
    
    
class AutoAgent(Agent):
    """AutoAgent that dynamically selects and initializes the correct agent based on the task and model.

    Example Usage:
    ```python
    # AutoAgent as LanguageAgent:
    auto_agent = AutoAgent(task="language", model_src="openai")
    response = auto_agent.act("What is the capital of France?")

    # AutoAgent as MotorAgent:
    auto_agent = AutoAgent(task="motion-openvla", model_src="https://api.mbodi.ai/community-models/")
    action = auto_agent.act("move hand forward", Image(size=(224, 224)))

    # AutoAgent as SenseAgent:
    auto_agent = AutoAgent(task="sense-depth-estimation", model_src="https://api.mbodi.ai/sense/")
    depth = auto_agent.act(image=Image("resources/bridge_example.jpeg", size=(224, 224)))
    ```
    """

    TASK_TO_AGENT_MAP: TaskAgentMap = {
        "language": LanguageAgent,
        "motion": OpenVlaAgent,
        "depth": DepthEstimationAgent,
        "detection": ObjectDetectionAgent,
        "segmentation": SegmentationAgent,
        "sense":SensorAgent,
    }
    @overload
    def __init__(   
        self, input: Any, model_src: ModelSource="openai", model_kwargs: Dict | None = None, **kwargs
    ): ...
    @overload
    def __init__(
        self, task: TaskTypes = "language", model_src: ModelSource="openai", model_kwargs: Dict | None = None, **kwargs
    ):...
    def __init__(
        self, *args, task: TaskTypes = "language", model_src: ModelSource = "openai", 
        model_kwargs: Dict | None = None, **kwargs):
        """Initialize the AutoAgent with the specified task and model."""
        self.console = Console()
        if task not in self.TASK_TO_AGENT_MAP:
            self.task = "language"
            self._pending_instruction = task
        else:
            self._pending_instruction = None
            self.task = task
        
        # Initialize core properties
        self.model_src = model_src
        self.model_kwargs = model_kwargs or {}
        self.kwargs = kwargs
        
        # Initialize agent
        self.agent = self.TASK_TO_AGENT_MAP[self.task](
            model_src=self.model_src,
            model_kwargs=self.model_kwargs,
            **self.kwargs
        )

    def __getattr__(self, name: str) -> Any:
        if hasattr(self.agent, name):
            return getattr(self.agent, name)
        return super().__getattribute__(name)

    def act(self, *args, **kwargs) -> Any:
        """Execute agent action with pending instruction if available."""
        if self.task == "language" and self._pending_instruction:
            return self.agent.act(self._pending_instruction, *args, **kwargs)
        return self.agent.act(*args, **kwargs)


    @staticmethod
    def available_tasks() -> None:
        """Print available tasks that can be used with AutoAgent."""
        console.print("Available tasks:")  # noqa: T201
        for task in AutoAgent.TASK_TO_AGENT_MAP:
            console.print(f"- {task}")  # noqa: T201


def get_agent(
    task: TaskTypes,
    model_src: ModelSource="openai",
    model_kwargs: Dict | None = None,
    **kwargs,
) -> Agent:
    """Initialize the AutoAgent with the specified task and model.

    This is an alternative to using the AutoAgent class directly. It returns the corresponding agent instance directly.

    Usage:
    ```python
    # Get LanguageAgent instance
    language_agent = get_agent(task="language", model_src="openai")
    response = language_agent.act("What is the capital of France?")

    # Get OpenVlaAgent instance
    openvla_agent = get_agent(task="motion-openvla", model_src="https://api.mbodi.ai/community-models/")
    action = openvla_agent.act("move hand forward", Image(size=(224, 224)))

    # Get DepthEstimationAgent instance
    depth_agent = get_agent(task="sense-depth-estimation", model_src="https://api.mbodi.ai/sense/")
    depth = depth_agent.act(image=Image("resources/bridge_example.jpeg", size=(224, 224)))
    ```
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

    auto_agent = AutoAgent(task="motion", model_src="https://api.mbodi.ai/community-models/")
    action = auto_agent.act("move hand forward", Image(size=(224, 224)))
    print(action)

    auto_agent = AutoAgent(
        task="motion", model_src="gradio", model_kwargs={"endpoint": "https://api.mbodi.ai/community-models/"}
    )
    action = auto_agent.act("move hand forward", Image(size=(224, 224)), "bridge_orig")
    print(action)

    auto_agent = AutoAgent(task="depth", model_src="https://api.mbodi.ai/sense/")
    image = Image("resources/bridge_example.jpeg", size=(224, 224))
    result = auto_agent.act(image=image)
    result.pil.show()

    auto_agent = get_agent(task="language", model_src="openai")
    response = auto_agent.act("What is the capital of France?")
    print(response)
