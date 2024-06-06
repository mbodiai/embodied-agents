from abc import ABC, abstractmethod
from typing import List

from mbodied_agents.types.controls import Motion


class MotionAgent(ABC):
    """Abstract base class for a Motion Agent.

    Subclasses must implement the `act` method, which generates a list of
    Motion objects based on given parameters.
    """

    @abstractmethod
    def act(self, **kwargs) -> List['Motion']:
        """Generate a list of Motion objects based on given parameters.

        Args:
            **kwargs: Arbitrary keyword arguments that will be used to determine the Motion objects.

        Returns:
            List[Motion]: A list of Motion objects based on the provided arguments.
        """
        pass
