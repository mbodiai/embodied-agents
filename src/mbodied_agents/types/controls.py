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

from typing import Sequence

import numpy as np
from gym import spaces
from mbodied_agents.base.motion import AbsoluteMotionField, Motion, MotionField, RelativeMotionField
from pydantic import Field

"""Motions to control a robot.

This module defines the motions to control a robot as pydantic models.
Examples:
    LocationAngle: A 2D+1 space representing x, y, and theta.
    Pose: A 6D space representing x, y, z, roll, pitch, and yaw.
    JointControl: A joint value, typically an angle.
    FullJointControl: Full joint control.
    HandControl: A 7D space representing x, y, z, roll, pitch, yaw, and oppenness of the hand.
    HeadControl: Head control. Tilt and pan.
    MobileSingleArmControl: Control for a robot that can move in 2D space with a single arm.
"""


class Pose3D(Motion):
    """Action for a 2D+1 space representing x, y, and theta."""

    x: float = MotionField(
        default_factory=lambda: 0.0,
        bounds=[-1, 1],
        description="Movement of X position in 2D space. +x is forward; -x is backward.",
    )
    y: float = MotionField(
        default_factory=lambda: 0.0,
        bounds=[-1, 1],
        description="Movement of Y position in 2D space. +y is left; -y is right.",
    )
    theta: float = MotionField(
        default_factory=lambda: 0.0,
        bounds=[-3.14, 3.14],
        description="Rotation (radians) about the z axis. Positive is counter-clockwise.",
    )


class LocationAngle(Pose3D):
    """Alias for Pose3D. A 2D+1 space representing x, y, and theta."""

    pass


class Pose6D(Motion):
    """Movement for a 6D space representing x, y, z, roll, pitch, and yaw."""

    x: float = MotionField(
        default_factory=lambda: 0.0,
        bounds=None,
        description="Movement of X position in 3D space. +x is forward; -x is backward.",
    )
    y: float = MotionField(
        default_factory=lambda: 0.0,
        bounds=None,
        description="Movement of Y position in 3D space. +y is left; -y is right.",
    )
    z: float = MotionField(
        default_factory=lambda: 0.0,
        bounds=None,
        description="Movement of Z position in 3D space. +z is up; -z is down.",
    )
    roll: float = MotionField(
        default_factory=lambda: 0.0,
        bounds=None,
        description="Roll about the X-axis in radians. Positive roll is clockwise.",
    )
    pitch: float = MotionField(
        default_factory=lambda: 0.0,
        bounds=None,
        description="Pitch about the Y-axis in radians. Positive pitch is down.",
    )
    yaw: float = MotionField(
        default_factory=lambda: 0.0,
        bounds=None,
        description="Yaw about the Z-axis in radians. Positive yaw is left.",
    )


class Pose(Pose6D):
    """Alias for Pose6D. A movement for a 6D space representing x, y, z, roll, pitch, and yaw."""

    pass


class AbsolutePose(Pose):
    """Absolute pose of the robot in 3D space."""

    x: float = AbsoluteMotionField(
        default_factory=lambda: 0.0,
        bounds=[-1, 1],
        description="X position in 3D space. +x is forward, -x is backward.",
    )
    y: float = AbsoluteMotionField(
        default_factory=lambda: 0.0,
        bounds=[-1, 1],
        description="Y position in 3D space. +y is left, -y is right.",
    )
    z: float = AbsoluteMotionField(
        default_factory=lambda: 0.0,
        bounds=[-1, 1],
        description="Z position in 3D space. +z is up, -z is down.",
    )
    roll: float = AbsoluteMotionField(
        default_factory=lambda: 0.0,
        bounds=[-3.14, 3.14],
        description="Roll in 3D space. Positive roll is clockwise.",
    )
    pitch: float = AbsoluteMotionField(
        default_factory=lambda: 0.0,
        bounds=[-3.14, 3.14],
        description="Pitch in 3D space. Positive pitch is down.",
    )
    yaw: float = AbsoluteMotionField(
        default_factory=lambda: 0.0,
        bounds=[-3.14, 3.14],
        description="Yaw in 3D space. Positive yaw is left.",
    )


class RelativePose(Pose6D):
    """Relative pose displacement."""

    x: float = RelativeMotionField(
        default_factory=lambda: 0.0,
        bounds=[-1, 1],
        description="X position delta in 3D space. +x is forward; -x is backward.",
    )
    y: float = RelativeMotionField(
        default_factory=lambda: 0.0,
        bounds=[-1, 1],
        description="Y position delta in 3D space. +y is left; -y is right.",
    )
    z: float = RelativeMotionField(
        default_factory=lambda: 0.0,
        bounds=[-1, 1],
        description="Z position delta in 3D space. +z is up; -z is down.",
    )
    roll: float = RelativeMotionField(
        default_factory=lambda: 0.0,
        bounds=[-3.14, 3.14],
        description="Roll delta in radians. Positive roll is clockwise.",
    )
    pitch: float = RelativeMotionField(
        default_factory=lambda: 0.0,
        bounds=[-3.14, 3.14],
        description="Pitch delta in radians. Positive pitch is down.",
    )
    yaw: float = RelativeMotionField(
        default_factory=lambda: 0.0,
        bounds=[-3.14, 3.14],
        description="Yaw delta in radians. Positive yaw is left.",
    )


class JointControl(Motion):
    """Motion for joint control."""

    value: float = MotionField(default_factory=lambda: 0.0,
                               bounds=[-3.14, 3.14], description="Joint value in radians.")

    def space(self):  # noqa: ANN201
        return spaces.Box(low=-3.14, high=3.14, shape=(), dtype=np.float32)


class FullJointControl(Motion):
    """Full joint control."""

    joints: Sequence[JointControl] | list[float] = MotionField(
        default_factory=list,
        description="List of joint values in radians.",
    )
    names: Sequence[str] | list[float] | None = MotionField(
        default=None, description="List of joint names.")

    def space(self):  # noqa: ANN201
        space = dict(super().space())

        for i, joint in enumerate(self.joints):
            name = self.names[i] if self.names else f"joint_{i}"
            space[name] = joint.space()
        return spaces.Dict(space)


class HandControl(Motion):
    """Action for a 7D space representing x, y, z, roll, pitch, yaw, and oppenness of the hand."""

    pose: Pose6D = MotionField(default_factory=Pose,
                               description="Pose of the robot hand.")
    grasp: JointControl = MotionField(
        default_factory=JointControl,
        description="Openness of the robot hand. 0 is closed, 1 is open.",
    )


class HeadControl(Motion):
    tilt: JointControl = MotionField(
        default_factory=lambda: JointControl(),
        description="Tilt of the robot head in radians (down is negative)."
    )
    pan: JointControl = MotionField(
        default_factory=lambda: JointControl(),
        description="Pan of the robot head in radians (left is negative)."
    )


class MobileSingleHandControl(Motion):
    """Control for a robot that can move its base in 2D space with a 6D EEF control + grasp."""

    # Location of the robot on the ground.
    base: LocationAngle | None = MotionField(
        default_factory=LocationAngle,
        description="Location of the robot on the ground.",
    )
    hand: HandControl | None = MotionField(
        default_factory=HandControl, description="Control for the robot hand.")
    head: HeadControl | None = MotionField(
        default=None, description="Control for the robot head.")


class MobileSingleArmControl(Motion):
    """Control for a robot that can move in 2D space with a single arm."""

    base: LocationAngle | None = MotionField(
        default_factory=LocationAngle,
        description="Location of the robot on the ground.",
    )
    arm: FullJointControl | None = MotionField(
        default_factory=FullJointControl,
        description="Control for the robot arm.",
    )
    head: HeadControl | None = MotionField(
        default=None, description="Control for the robot head.")


class MobileBimanualArmControl(Motion):
    """Control for a robot that can move in 2D space with two arms."""

    base: LocationAngle | None = MotionField(
        default_factory=LocationAngle,
        description="Location of the robot on the ground.",
    )
    left_arm: FullJointControl | None = MotionField(
        default_factory=FullJointControl,
        description="Control for the left robot arm.",
    )
    right_arm: FullJointControl | None = MotionField(
        default_factory=FullJointControl,
        description="Control for the right robot arm.",
    )
    head: HeadControl | None = MotionField(
        default=None, description="Control for the robot head.")


class HumanoidControl(Motion):
    """Control for a humanoid robot."""

    left_arm: FullJointControl | None = MotionField(
        default_factory=FullJointControl,
        description="Control for the left robot arm.",
    )
    right_arm: FullJointControl | None = MotionField(
        default_factory=FullJointControl,
        description="Control for the right robot arm.",
    )
    left_leg: FullJointControl | None = MotionField(
        default_factory=FullJointControl,
        description="Control for the left robot leg.",
    )
    right_leg: FullJointControl | None = MotionField(
        default_factory=FullJointControl,
        description="Control for the right robot leg.",
    )
    head: HeadControl | None = MotionField(
        default=None, description="Control for the robot head.")


class LocobotActionOrAnswer(MobileSingleHandControl):
    answer: str | None = Field(
        default="",
        description="Short, one sentence answer to any question a user might have asked. 20 words max.",
    )
    sleep: bool | None = Field(
        default=False,
        description="Whether the robot should go to sleep after executing the motion.",
    )
    home: bool | None = Field(
        default=False,
        description="Whether the robot should go to home after executing the motion.",
    )
