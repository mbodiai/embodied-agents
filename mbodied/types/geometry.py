from mbodied.types.motion import AbsoluteMotionField, Motion, MotionField, RelativeMotionField


class Pose3D(Motion):
    """Action for a 2D+1 space representing x, y, and theta."""

    x: float = MotionField(
        0.0,
        description="X position in 2D space. +x is forward; -x is backward.",
    )
    y: float = MotionField(
        0.0,
        description="Y position in 2D space. +y is left; -y is right.",
    )
    theta: float = MotionField(
        default_factory=lambda: 0.0,
        description="Rotation (radians) about the z axis. Positive is counter-clockwise.",
    )


class LocationAngle(Pose3D):
    """Alias for Pose3D. A 2D+1 space representing x, y, and theta."""

    pass


class Pose6D(Motion):
    """Movement for a 6D space representing x, y, z, roll, pitch, and yaw."""

    x: float = MotionField(
        default=0,
        description="X position in 3D space. +x is forward; -x is backward.",
    )
    y: float = MotionField(
        0,
        description="Y position in 3D space. +y is left; -y is right.",
    )
    z: float = MotionField(
        0,
        description="Z position in 3D space. +z is up; -z is down.",
    )
    roll: float = MotionField(
        0,
        description="Roll about the X-axis in radians. Positive roll is clockwise.",
    )
    pitch: float = MotionField(
        0,
        description="Pitch about the Y-axis in radians. Positive pitch is down.",
    )
    yaw: float = MotionField(
        0,
        description="Yaw about the Z-axis in radians. Positive yaw is left.",
    )


class Pose(Pose6D):
    """Alias for Pose6D. A movement for a 6D space representing x, y, z, roll, pitch, and yaw."""

    pass


class AbsolutePose(Pose):
    """Absolute pose of the robot in 3D space."""

    x: float = AbsoluteMotionField(
        0,
        description="Absolute X position in 3D space. +x is forward; -x is backward.",
    )
    y: float = AbsoluteMotionField(
        0,
        description="Absolute Y position in 3D space. +y is left; -y is right.",
    )
    z: float = AbsoluteMotionField(
        0,
        description="Absolute Z position in 3D space. +z is up; -z is down.",
    )
    roll: float = AbsoluteMotionField(
        0,
        description="Absolute roll about the X-axis in radians. Positive roll is clockwise.",
    )
    pitch: float = AbsoluteMotionField(
        0,
        description="Absolute pitch about the Y-axis in radians. Positive pitch is down.",
    )
    yaw: float = AbsoluteMotionField(
        0,
        description="Absolute yaw about the Z-axis in radians. Positive yaw is left.",
    )


class RelativePose(Pose):
    """Relative pose of the robot in 3D space."""

    x: float = RelativeMotionField(
        0,
        description="Relative X position in 3D space. +x is forward; -x is backward.",
    )
    y: float = RelativeMotionField(
        0,
        description="Relative Y position in 3D space. +y is left; -y is right.",
    )
    z: float = RelativeMotionField(
        0,
        description="Relative Z position in 3D space. +z is up; -z is down.",
    )
    roll: float = RelativeMotionField(
        0,
        description="Relative roll about the X-axis in radians. Positive roll is clockwise.",
    )
    pitch: float = RelativeMotionField(
        0,
        description="Relative pitch about the Y-axis in radians. Positive pitch is down.",
    )
    yaw: float = RelativeMotionField(
        0,
        description="Relative yaw about the Z-axis in radians. Positive yaw is left.",
    )
