# Copyright 2024 mbodi ai
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

from mbodied.agents import Agent
from mbodied.types.motion import Motion


class MotorAgent(Agent):
    """Abstract base class for motor agents.

    Subclassed from Agent, thus possessing the ability to make remote calls, etc.
    """

    @abstractmethod
    def act(self, **kwargs) -> Motion:
        """Generate a Motion based on given parameters.

        Args:
            **kwargs: Arbitrary keyword arguments for motor agent to act on.

        Returns:
            Motion: A Motion object based on the provided arguments.
        """
        pass
