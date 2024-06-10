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
from mbodied_agents.types.world import SceneObject


class WorldAgent(Agent):
    """Represents an agent that acts in the context of a world.

    Methods:
        act(**kwargs) -> SceneObject:
            Abstract method to perform actions and return the resulting world/scene state.
    """
    
    @abstractmethod
    def act(self, **kwargs) -> List['SceneObject']:
        """Perform actions in the world context and return the resulting scene object state.

        Args:
            **kwargs: Additional parameters for the action.

        Returns:
            SceneObject: The resulting state of the world after the action.
        """
        pass
