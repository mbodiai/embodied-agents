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

import asyncio
from abc import ABC, abstractmethod

from mbodied.types.sample import Sample


class Robot(ABC):
    """Abstract base class for robot hardware interfaces.

    This class provides a template for creating robot hardware interfaces that can
    control robots or other hardware devices.

    Methods:
        do (required): Executes motion on the robot.
        fetch: Fetches data from the hardware.
        capture: Captures continuous data from the hardware i.e. image.
        get_robot_state: Gets the current pose of the hardware, optional for robot recorder.
        calculate_action: Calculates the action between two robot states, optional for robot recorder.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """Initializes the robot hardware interface.

        Args:
            kwargs: Additional arguments to pass to the robot hardware interface.
        """
        raise NotImplementedError

    @abstractmethod
    def do(self, *args, **kwargs) -> None:  # noqa
        """Executes motion.

        Args:
            args: Arguments to pass to the robot hardware interface.
            kwargs: Additional arguments to pass to the robot hardware interface.
        """
        raise NotImplementedError

    async def async_do(self, *args, **kwargs) -> None:
        """Asynchronously executes motion.

        Args:
            args: Arguments to pass to the robot hardware interface.
            kwargs: Additional arguments to pass to the robot hardware interface.
        """
        return await asyncio.to_thread(self.do, *args, **kwargs)

    def fetch(self, *args, **kwargs) -> None:
        """Fetches data from the hardware.

        Args:
            args: Arguments to pass to the robot hardware interface.
            kwargs: Additional arguments to pass to the robot hardware interface.
        """
        raise NotImplementedError

    def capture(self, *args, **kwargs) -> None:
        """Captures continuous data from the hardware.

        Args:
            args: Arguments to pass to the robot hardware interface.
            kwargs: Additional arguments to pass to the robot hardware interface.
        """
        raise NotImplementedError

    def get_robot_state(self) -> Sample:
        """(Optional for robot recorder): Gets the current state (pose) of the robot.

        This will be used by the robot recorder to record the current state of the robot.
        """
        raise NotImplementedError

    def calculate_action(self, old_state: Sample, new_state: Sample) -> Sample:
        """(Optional for robot recorder): Calculates the the action between two robot states.

        For example, substract old from new hand position and use absolute value for grasp, etc.

        Args:
            old_state: The old state (pose) of the robot.
            new_state: The new state (pose) of the robot.
        """
        raise NotImplementedError
