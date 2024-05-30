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

from abc import abstractmethod
from typing import List

from mbodied_agents.base.agent import Agent
from mbodied_agents.base.sample import Sample


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
