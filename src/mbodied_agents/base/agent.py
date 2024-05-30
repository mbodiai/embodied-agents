from abc import ABC, abstractmethod

from mbodied_agents.base.sample import Sample
from mbodied_agents.data.recording import Recorder


class Agent(ABC):
    """Abstract base class for agents.

    This class provides a template for creating agents that can 
    optionally record their actions and observations.

    Attributes:
        recorder (Recorder): The recorder to record observations and actions.
        kwargs (dict): Additional arguments to pass to the recorder.
    """

    def __init__(self, recorder="omit", **kwargs):
        """Initialize the agent.

        Args:
            recorder (str): The recorder to record the observations and actions.
            **kwargs: Additional arguments to pass to the recorder.
        """
        if recorder is None or recorder == 'omit':
            self.recorder = None
        elif recorder == 'default':
            self.recorder = Recorder('base_agent', out_dir='outs', **kwargs)
        else:
            self.recorder = Recorder(recorder, out_dir='outs', **kwargs)
        self.kwargs = kwargs

    def __call__(self, observation, verify=False, **kwargs) -> Sample:
        """Act based on the observation.

        Record the observation and action.

        Args:
            observation: The current observation based on which the agent acts.
            verify (bool): Whether to verify the action before sending.
            **kwargs: Additional arguments to pass to the `act` method.

        Returns:
            Sample: The action sample created by the agent.
        """
        action = self.act(observation, **kwargs)
        if self.recorder is not None:
            self.recorder.record(observation, action)
        if verify:
            is_verified = input("Cancel Action? (y/n)")
        if not verify or "y" not in is_verified:
            self.recorder.record(observation, action)
        return action

    @abstractmethod
    def act(self, observation, **kwargs) -> Sample:
        """Abstract method to define the action based on the observation.

        Args:
            observation: The current observation based on which the agent acts.
            **kwargs: Additional arguments to customize the action.

        Returns:
            Sample: The action sample created by the agent.
        """
        pass
