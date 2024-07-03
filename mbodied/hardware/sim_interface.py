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

    def do(self, motion: HandControl) -> list[float]:
        """Executes a given HandControl motion and returns the new position of the robot arm.

        Args:
            motion: The HandControl motion to be executed.
        """
        print("Executing motion:", motion)  # noqa: T201
        self.current_pos[0] += motion.pose.x
        self.current_pos[1] += motion.pose.y
        self.current_pos[2] += motion.pose.z
        self.current_pos[3] += motion.pose.roll
        self.current_pos[4] += motion.pose.pitch
        self.current_pos[5] += motion.pose.yaw
        self.current_pos[6] = motion.grasp.value
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
        return Image(size=(224,224))
