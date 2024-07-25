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
from pathlib import Path

from mbodied.hardware.interface import HardwareInterface
from mbodied.types.motion.control import HandControl
from mbodied.types.sense.vision import Image


class SimInterface(HardwareInterface):
    """A simulated interface for testing and validating purposes.

    This class simulates the interface between the robot arm and the control system.

    Attributes:
        home_pos: The home position of the robot arm.
        current_pos: The current position of the robot arm.
    """

    def __init__(self):
        """Initializes the SimInterface and sets up the robot arm.

        position: [x, y, z, r, p, y, grasp]
        """
        self.home_pos = [0, 0, 0, 0, 0, 0, 0]
        self.current_pos = self.home_pos

    def do(self, motion: HandControl | list[HandControl]) -> list[float]:
        """Executes a given HandControl motion and returns the new position of the robot arm.

        Args:
            motion: The HandControl motion or list of HandControl to be executed.
        """
        if not isinstance(motion, list):
            motion = [motion]
        for m in motion:
            print("Executing motion:", motion)  # noqa: T201
            self.current_pos[0] = round(self.current_pos[0] + m.pose.x, 5)
            self.current_pos[1] = round(self.current_pos[1] + m.pose.y, 5)
            self.current_pos[2] = round(self.current_pos[2] + m.pose.z, 5)
            self.current_pos[3] = round(self.current_pos[3] + m.pose.roll, 5)
            self.current_pos[4] = round(self.current_pos[4] + m.pose.pitch, 5)
            self.current_pos[5] = round(self.current_pos[5] + m.pose.yaw, 5)
            self.current_pos[6] = round(m.grasp.value, 5)
            print("New position:", self.current_pos)  # noqa: T201

        return self.current_pos

    def get_pose(self) -> list[float]:
        """Gets the current pose of the robot arm.

        Returns:
            list[float]: A list of the current pose values [x, y, z, r, p, y, grasp].
        """
        return self.current_pos

    def capture(self, **_) -> Image:
        """Captures an image."""
        resource = Path("resources") / "xarm.jpeg"
        return Image(resource, size=(224, 224))
