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

# 1. Measure time to first action
# 2. We should be able to set horizon, prompts per agent, 
import time
from pathlib import Path

from mbodied.robots import Robot
from mbodied.types.motion.control import HandControl
from mbodied.types.sense.vision import Image


class SimRobot(Robot):
    """A simulated robot interface for testing and validating purposes.

    This class simulates the interface between the robot arm and the control system.
    do() simulates the execution of HandControl motions that got executed in execution_time.

    Attributes:
        home_pos: The home position of the robot arm.
        current_pos: The current position of the robot arm.
    """

    def __init__(self, execution_time: float = 1.0):
        """Initializes the SimRobot and sets up the robot arm.

        Args:
            execution_time: The time it takes to execute a motion.

        position: [x, y, z, r, p, y, grasp]
        """
        self.execution_time = execution_time
        self.home_pos = [0, 0, 0, 0, 0, 0, 0]
        self.current_pos = self.home_pos

    def do(self, motion: HandControl | list[HandControl]) -> list[float]:
        """Executes HandControl motions and returns the new position of the robot arm.

        This simulates the execution of each motion for self.execution_time. It divides the motion into 10 steps.

        Args:
            motion: The HandControl motion to be executed.
        """
        if not isinstance(motion, list):
            motion = [motion]
        for m in motion:
            print("Executing motion:", m)  # noqa: T201

            # Number of steps to divide the motion into
            steps = 10
            sleep_duration = self.execution_time / steps
            step_motion = [value / steps for value in m.flatten()]
            for _ in range(steps):
                self.current_pos = [round(x + y, 5) for x, y in zip(self.current_pos, step_motion, strict=False)]
                time.sleep(sleep_duration)

            print("New position:", self.current_pos)  # noqa: T201

        return self.current_pos

    def capture(self, **_) -> Image:
        """Captures an image."""
        resource = Path("resources") / "xarm.jpeg"
        return Image(resource, size=(224, 224))

    def get_observation(self) -> Image:
        """Alias of capture for recording."""
        return self.capture()

    def get_state(self) -> HandControl:
        """Gets the current pose of the robot arm.

        Returns:
            list[float]: A list of the current pose values [x, y, z, r, p, y, grasp].
        """
        return HandControl.unflatten(self.current_pos)

    def prepare_action(self, old_pose: HandControl, new_pose: HandControl) -> HandControl:
        """Calculates the action between two poses."""
        # Calculate the difference between the old and new poses. Use absolute value for grasp.
        old = list(old_pose.flatten())
        new = list(new_pose.flatten())
        result = [(new[i] - old[i]) for i in range(len(new) - 1)] + [new[-1]]
        return HandControl.unflatten(result)
