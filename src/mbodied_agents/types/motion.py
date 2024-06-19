from typing import Sequence

import numpy as np
from gymnasium import spaces
from pydantic import Field

from mbodied_agents.base.motion import (  # noqa: F401
    AbsoluteMotionField,
    Motion,
    MotionField,
    MotionType,
    RelativeMotionField,
)
from mbodied_agents.types.ndarray import NumpyArray

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

XYZ_DESCRIPTION = "Position (meters) in 3D space. x is forward, y is left, and z is up."
RPY_DESCRIPTION = (
    "Rotation (radians) in 3D space. Positive roll is clockwise, positive pitch is down, and positive yaw is left."
)

XY_DESCRIPTION = "Position (meters) in 2D space. x is forward, y is left."
THETA_DESCRIPTION = "Rotation (radians) about the z axis. Positive is counter-clockwise."


class Pose3D(Motion):
    """Action for a 2D+1 space representing x, y, and theta."""

    xy: NumpyArray = MotionField(
        default_factory=lambda: np.array([0, 0]),
        bounds=[-1, 1],
        shape=(2,),
        description=XY_DESCRIPTION,
    )
    theta: float = MotionField(default_factory=lambda: 0.0, bounds=[-3.14, 3.14], description=THETA_DESCRIPTION)


class LocationAngle(Pose3D):
    """Alias for Pose3D. A 2D+1 space representing x, y, and theta."""

    pass


class Pose6D(Motion):
    """Movement for a 6D space representing x, y, z, roll, pitch, and yaw."""

    xyz: NumpyArray = MotionField(
        default_factory=lambda: np.array([0, 0, 0]),
        bounds=[-1, 1],
        shape=(3,),
        description=XYZ_DESCRIPTION,
    )
    rpy: NumpyArray = MotionField(
        default_factory=lambda: np.array([0, 0, 0]),
        bounds=[-3.14, 3.14],
        shape=(3,),
        description=RPY_DESCRIPTION,
    )


class Pose(Pose6D):
    """Alias for Pose6D. A movement for a 6D space representing x, y, z, roll, pitch, and yaw."""

    pass


class AbsolutePose(Pose):
    """Absolute pose of the robot in 3D space."""

    xyz: NumpyArray = AbsoluteMotionField(
        default_factory=lambda: np.array([0, 0, 0]),
        bounds=[-1, 1],
        shape=(3,),
        description="Location in 3D space. x is forward, y is left, and z is up.",
    )
    rpy: NumpyArray = AbsoluteMotionField(
        default_factory=lambda: np.array([0, 0, 0]),
        bounds=[-3.14, 3.14],
        shape=(3,),
        description="Rotation in 3D space. Positive roll is clockwise, positive pitch is down, and positive yaw is left.",  # noqa: E501
    )


class RelativePose(Pose6D):
    """Relative pose displacement."""

    xyz: NumpyArray = RelativeMotionField(
        default_factory=lambda: np.array([0, 0, 0]),
        bounds=[-1, 1],
        shape=(3,),
        description=XYZ_DESCRIPTION,
    )
    rpy: NumpyArray = RelativeMotionField(
        default_factory=lambda: np.array([0, 0, 0]),
        bounds=[-3.14, 3.14],
        shape=(3,),
        description=RPY_DESCRIPTION,
    )


class JointControl(Motion):
    """Motion for joint control."""

    value: float = MotionField(default_factory=lambda: 0.0, bounds=[-3.14, 3.14], description="Joint value in radians.")

    def space(self):  # noqa: ANN201
        return spaces.Box(low=-3.14, high=3.14, shape=(), dtype=np.float32)


class FullJointControl(Motion):
    """Full joint control."""

    joints: Sequence[JointControl] | list[float] = MotionField(
        default_factory=list,
        description="List of joint values in radians.",
    )
    names: Sequence[str] | list[float] | None = MotionField(default=None, description="List of joint names.")

    def space(self):  # noqa: ANN201
        space = dict(super().space())

        for i, joint in enumerate(self.joints):
            name = self.names[i] if self.names else f"joint_{i}"
            space[name] = joint.space()
        return spaces.Dict(space)


class HandControl(Motion):
    """Action for a 7D space representing x, y, z, roll, pitch, yaw, and oppenness of the hand."""

    pose: Pose = MotionField(default_factory=Pose, description="Pose of the robot hand.")
    grasp: JointControl = MotionField(
        default_factory=JointControl,
        description="Openness of the robot hand. -1 is closed, 1 is open.",
    )


class HeadControl(Motion):
    """Action for head control. Tilt and pan."""

    tilt: JointControl = MotionField(
        default_factory=JointControl,
        bounds=[-3.14, 3.14],
        description="Tilt of the robot head in radians (down is negative).",
    )
    pan: JointControl = MotionField(
        default_factory=JointControl,
        bounds=[-3.14, 3.14],
        description="Pan of the robot head in radians (left is negative).",
    )


class MobileSingleHandControl(Motion):
    """Control for a robot that can move its base in 2D space with a 6D EEF control + grasp."""

    # Location of the robot on the ground.
    base: LocationAngle | None = MotionField(
        default_factory=LocationAngle,
        description="Location of the robot on the ground.",
    )
    hand: HandControl | None = MotionField(default_factory=HandControl, description="Control for the robot hand.")
    head: HeadControl | None = MotionField(default=None, description="Control for the robot head.")


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
    head: HeadControl | None = MotionField(default=None, description="Control for the robot head.")


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
    head: HeadControl | None = MotionField(default=None, description="Control for the robot head.")


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
    head: HeadControl | None = MotionField(default=None, description="Control for the robot head.")


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
