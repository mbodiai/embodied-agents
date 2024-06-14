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
from mbodied_agents.types.controls import Motion


class MotorAgent(Agent):
    @abstractmethod
    def act(self, **kwargs) -> List['Motion']:
        """Generate a list of Motion objects based on given parameters.

        Args:
            **kwargs: Arbitrary keyword arguments that will be used to determine the Motion objects.

        Returns:
            List[Motion]: A list of Motion objects based on the provided arguments.
        """
        pass
