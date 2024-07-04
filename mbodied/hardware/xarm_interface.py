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

import math

from gymnasium import spaces
from xarm.wrapper import XArmAPI

from mbodied.hardware.recording_interface import RecordingHardwareInterface
from mbodied.types.motion.control import HandControl
from mbodied.types.sense.vision import Image


class XarmInterface(RecordingHardwareInterface):
    """Control the xArm robot arm with SDK.

    Usage:
        xarm = XarmInterface()
        xarm.do(HandControl(...))

    Attributes:
        ip: The IP address of the xArm robot.
        arm: The XArmAPI instance for controlling the robot.
        home_pos: The home position of the robot arm.
    """

    def __init__(self, ip: str = "192.168.1.228", record=False):
        """Initializes the XarmInterface and sets up the robot arm.

        Args:
            ip: The IP address of the xArm robot.
            record: Whether to record the state of the robot arm to dataset.
        """
        if record:
            recorder_kwargs = {
                "name": "xarm_record",
                "observation_space": spaces.Dict(
                    {"image": Image(size=(224, 224)).space(), "instruction": spaces.Text(1000)},
                ),
                "action_space": HandControl().space(),
                "out_dir": "xarm_dataset",
            }
            super().__init__(**recorder_kwargs)
        self.ip = ip

        self.arm = XArmAPI(self.ip)
        self.arm.motion_enable(True)
        self.arm.clean_error()
        self.arm.set_mode(0)
        self.arm.set_state(0)

        self.arm.set_gripper_mode(0)
        self.arm.set_gripper_enable(True)
        self.arm.set_gripper_speed(1000)
        self.arm.set_gripper_mode(0)

        self.home_pos = [300, 0, 325, -180, 0, 0]
        self.arm.set_position(*self.home_pos, wait=True, speed=200)
        self.arm.set_gripper_position(800, wait=True)

    def do(self, motion: HandControl) -> None:
        """Executes a given HandControl motion.

        HandControl is in meters and radians, so we convert it to mm and degrees here.

        Args:
            motion: The HandControl motion to be executed.
        """
        current_pos = self.arm.get_position()[1]
        current_pos[0] += motion.pose.x * 1000
        current_pos[1] += motion.pose.y * 1000
        current_pos[2] += motion.pose.z * 1000
        current_pos[3] += math.degrees(motion.pose.roll)
        current_pos[4] += math.degrees(motion.pose.pitch)
        current_pos[5] += math.degrees(motion.pose.yaw)

        self.arm.set_position(*current_pos, wait=False, speed=200)
        self.arm.set_gripper_position(motion.grasp.value * 800, wait=False)

    def set_pose(self, motion: HandControl) -> None:
        """Sets the robot arm to a given absolute HandControl pose."""
        current_pos = self.arm.get_position()[1]
        current_pos[0] = motion.pose.x * 1000
        current_pos[1] = motion.pose.y * 1000
        current_pos[2] = motion.pose.z * 1000
        current_pos[3] = math.degrees(motion.pose.roll)
        current_pos[4] = math.degrees(motion.pose.pitch)
        current_pos[5] = math.degrees(motion.pose.yaw)
        self.arm.set_position(*current_pos, wait=True, speed=200)
        self.arm.set_gripper_position(motion.grasp.value * 800, wait=True)

    def get_pose(self) -> HandControl:
        """Gets the current pose (absolute HandControl) of the robot arm.

        Returns:
            A list of the current pose values [x, y, z, r, p, y].
        """
        current_pos = self.arm.get_position()[1]
        current_pos[0] = round(current_pos[0], 2) / 1000
        current_pos[1] = round(current_pos[1], 2) / 1000
        current_pos[2] = round(current_pos[2], 2) / 1000
        current_pos[3] = round(math.radians(current_pos[3]), 6)
        current_pos[4] = round(math.radians(current_pos[4]), 6)
        current_pos[5] = round(math.radians(current_pos[5]), 6)

        hand_control = current_pos.copy()
        hand_control.append(self.arm.get_gripper_position()[1] / 800)
        return HandControl.unflatten(hand_control)

    def capture(self) -> Image:
        """Captures an image from the robot camera. Placeholder for real camera input."""
        return Image(size=(224, 224))
