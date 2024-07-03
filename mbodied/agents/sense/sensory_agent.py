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

from mbodied.agents import Agent
from mbodied.types.sense.sensor_reading import SensorReading


class SensoryAgent(Agent):
    """Abstract base class for sensory agents.

    This class provides a template for creating agents that can sense the environment.

    Attributes:
        kwargs (dict): Additional arguments to pass to the recorder.
    """

    def __init__(self, **kwargs):
        """Initialize the agent.

        Args:
            **kwargs: Additional arguments to pass to the recorder.
        """
        super().__init__(**kwargs)

    def act(self, **kwargs) -> SensorReading:
        """Abstract method to define the sensing mechanism of the agent.

        Args:
            **kwargs: Additional arguments to pass to the `sense` method.

        Returns:
            Sample: The sensory sample created by the agent.
        """
        raise NotImplementedError

    def sense(self, **kwargs) -> SensorReading:
        """Generate a SensorReading based on given parameters.

        Args:
            **kwargs: Arbitrary keyword arguments for sensory agent to sense on.

        Returns:
            SensorReading: A SensorReading object based on the provided arguments.
        """
        return self.act(**kwargs)
