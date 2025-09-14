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
import numpy as np


try:
    from xarm.wrapper import XArmAPI
except ImportError:
    logging.warning("XarmAPI not found. Please install the xArm-Python-SDK package.")
  

from mbodied.robots import Robot
from embdata.motion.control import AbsoluteHandControl, HandControl
from embdata.coordinate import Pose6D
from embdata.array import ArrayLike, sz

DEFAULT_HOME_POSE = Pose6D(translation=[300, 0, 325], orientation=[math.radians(180), 0, 0])
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

    def __init__(self, ip: str = "192.168.1.228", arm_speed: int = 200, gripper_speed: int = 2000, home_pose: ArrayLike[sz[6],float]|Pose6D=DEFAULT_HOME_POSE):
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
        self.arm.set_gripper_speed(gripper_speed)

        self.home_pos = home_pose[:]
        self.arm_speed = arm_speed
        self.arm.set_position(*self.home_pos, is_radian=True, wait=True, speed=self.arm_speed)
        self.arm.set_gripper_position(800, wait=True)
        self.arm.set_collision_sensitivity(3)
        self.arm.set_self_collision_detection(True)

    def clip_scale_grasp(self, grasp: float) -> float:
            n = grasp if 0.0 <= grasp <= 1.0 else ((grasp + 1.0) * 0.5 if -1.0 <= grasp <= 1.0 else grasp)
            return max(0, min(800, int(round(n * 800.0))))
        

    def do(self, motion:  list[HandControl]|HandControl) -> None:
        """Execute relative pose(s) via SE(3) using translation/orientation vectors.

        Accepts a single `Pose6D` or `HandControl`, or a list of them. If a
        `HandControl` is provided, its `pose` is used directly.
        """
        items = motion if isinstance(motion, list) else [motion]
        current_pose = self.get_state().pose
        for item in items:
            is_relative = item.info["fields"]["pose"]["motion_type"] == "relative"
            target_pose = current_pose * item.pose if is_relative else item.pose
            t_mm = target_pose.translation * 1000.0
            self.arm.set_position(*[*t_mm, *target_pose.orientation], wait=True, speed=self.arm_speed)
         
            self.arm.set_gripper_position(self.clip_scale_grasp(item.grasp), wait=True)
            current_pose = target_pose

    def get_state(self) -> HandControl:
        """Gets the current pose (absolute HandControl) of the robot arm.

        Returns:
            The current pose of the robot arm.
        """
        pos = self.arm.get_position(is_radian=True)[1]
        gripper_pos = self.arm.get_gripper_position()
        grasp = gripper_pos[1] if  isinstance(gripper_pos, tuple) and gripper_pos[0] == 0 else 0
        return AbsoluteHandControl(
            pose=Pose6D(
                translation=np.array(pos[:3]) / 1000.0,
                orientation=np.array(pos[3:6]),
            ),
            grasp=grasp or 0,
        )


    def capture(self) -> HandControl:
        """Captures an image from the robot camera."""
        return self.get_state()

    def get_observation(self) -> HandControl:
        """Captures an image for recording."""
        return self.capture()
