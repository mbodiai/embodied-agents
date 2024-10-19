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

import logging
import math
from typing import Any

from gymnasium import spaces

try:
    from xarm.wrapper import XArmAPI
except ImportError:
    logging.warning("XarmAPI not found. Please install the xArm-Python-SDK package.")
    xarm = Any
    class XArmAPI:
        def __init__(self, *args, **kwargs):
            pass
    xarm.wrapper = XArmAPI


from mbodied.robots import Robot
from mbodied.types.motion.control import HandControl
from mbodied.types.sense.vision import Image


class XarmRobot(Robot):
    """Control the xArm robot arm with SDK.

    Usage:
    ```python
    xarm = XarmRobot()
    xarm.do(HandControl(...))
    ```

    Attributes:
        ip: The IP address of the xArm robot.
        arm: The XArmAPI instance for controlling the robot.
        home_pos: The home position of the robot arm.
    """

    def __init__(self, ip: str = "192.168.1.228", use_realsense: bool = False):
        """Initializes the XarmRobot and sets up the robot arm.

        Args:
            ip: The IP address of the xArm robot.
            use_realsense: Whether to use a RealSense camera for capturing images
        """
        self.ip = ip

        self.arm = XArmAPI(self.ip, is_radian=True)
        self.arm.motion_enable(True)
        self.arm.clean_error()
        self.arm.set_mode(0)
        self.arm.set_state(0)

        self.arm.set_gripper_mode(0)
        self.arm.set_gripper_enable(True)
        self.arm.set_gripper_speed(2000)

        self.home_pos = [300, 0, 325, math.radians(180), 0, 0]
        self.arm_speed = 200
        self.arm.set_position(*self.home_pos, wait=True, speed=self.arm_speed)
        self.arm.set_gripper_position(800, wait=True)
        self.arm.set_collision_sensitivity(3)
        self.arm.set_self_collision_detection(True)

        self.use_realsense = False
        if use_realsense:
            self.use_realsense = True
            from mbodied.hardware.realsense_camera import RealsenseCamera

            self.realsense_camera = RealsenseCamera(width=640, height=480, fps=30)

    def do(self, motion: HandControl | list[HandControl]) -> None:
        """Executes HandControl(s).

        HandControl is in meters and radians.

        Args:
            motion: The HandControl motion(s) to be executed.
        """
        if not isinstance(motion, list):
            motion = [motion]
        for m in motion:
            current_pos = self.arm.get_position()[1]
            current_pos[0] += m.pose.x * 1000
            current_pos[1] += m.pose.y * 1000
            current_pos[2] += m.pose.z * 1000
            current_pos[3] += m.pose.roll
            current_pos[4] += m.pose.pitch
            current_pos[5] += m.pose.yaw
            self.arm.set_position(*current_pos, wait=True, speed=self.arm_speed)
            self.arm.set_gripper_position(0 if m.grasp.value <= 0.5 else 800, wait=True)

    def get_state(self) -> HandControl:
        """Gets the current pose (absolute HandControl) of the robot arm.

        Returns:
            The current pose of the robot arm.
        """
        current_pos = self.arm.get_position()[1]
        current_pos[0] = round(current_pos[0] / 1000, 5)
        current_pos[1] = round(current_pos[1] / 1000, 5)
        current_pos[2] = round(current_pos[2] / 1000, 5)
        current_pos[3] = round(current_pos[3], 3)
        current_pos[4] = round(current_pos[4], 3)
        current_pos[5] = round(current_pos[5], 3)

        hand_control = current_pos.copy()
        if self.arm.get_gripper_position()[1] >= 750:
            hand_control.append(1)
        else:
            hand_control.append(0)
        return HandControl.unflatten(hand_control)

    def prepare_action(self, old_pose: HandControl, new_pose: HandControl) -> HandControl:
        """Calculates the action between two poses.

        Args:
            old_pose: The old pose(state) of the hardware.
            new_pose: The new pose(state) of the hardware.

        Returns:
            The action to be taken between the old and new poses.
        """
        # Calculate the difference between the old and new poses. Use absolute value for grasp.
        old = list(old_pose.flatten())
        new = list(new_pose.flatten())
        result = [(new[i] - old[i]) for i in range(len(new) - 1)] + [new[-1]]
        return HandControl.unflatten(result)

    def capture(self) -> Image:
        """Captures an image from the robot camera."""
        if self.use_realsense:
            rgb_image, _, _ = self.realsense_camera.capture_realsense_images()
            return Image(rgb_image, size=(224, 224))
        return Image(size=(224, 224))

    def get_observation(self) -> Image:
        """Captures an image for recording."""
        return self.capture()
