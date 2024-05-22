from abc import abstractmethod
from typing import List

from mbodied.base.agent import Agent
from mbodied.base.sample import Sample


class LanguageAction(Sample):
    """A sample representing a language action.

    Attributes:
        actions: A list of actions in string format.
    """
    actions: List[str]


class LanguageAgent(Agent):
    """An agent that outputs a list of strings as an action."""

    @abstractmethod
    def act(self, **kwargs) -> LanguageAction:
        """Generates an action as a list of strings.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            LanguageAction: The generated language action.
        """
        pass
