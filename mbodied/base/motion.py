from typing import Any

from mbodied.base.sample import Sample
from pydantic import ConfigDict, Field
from pydantic_core import PydanticUndefined
from typing_extensions import Literal

MotionType = Literal[
    "UNSPECIFIED",
    "OTHER",
    "ABSOLUTE",
    "RELATIVE",
    "VELOCITY",
    "TORQUE",
]


def MotionField(  # noqa: N802
    default: Any = PydanticUndefined,  # noqa: N805
    bounds: list[float] | None = None,  # noqa: N802, D417
    shape: tuple[int] | None = None,
    description: str | None = None,
    motion_type: MotionType = "UNSPECIFIED",
    **kwargs,
) -> Any:
    """Field for a motion.

    Args:
        default: Default value for the field.
        bounds: Bounds of the motion.
        shape: Shape of the motion.
        description: Description of the motion.
        motion_type: Type of the motion. Can be "UNSPECIFIED", "OTHER", "ABSOLUTE", "RELATIVE", "VELOCITY", "TORQUE".
    """
    if description is None:
        description = ""
    if shape is not None and len(shape) > 1:
        description += f" Shape: {shape}"
    if bounds is not None:
        description += f" Bounds: {bounds}"
    if motion_type != "unspecified":
        description += f" Motion type: {motion_type}"

    return Field(
        default=default,
        description=description,
        json_schema_extra={"bounds": bounds, "motion_type": motion_type, "shape": shape},
        **kwargs,
    )  # type: ignore


def AbsoluteMotionField(
    default: Any = PydanticUndefined,
    bounds: list[float] | None = None,
    shape: tuple[int] | None = None,
    description: str | None = None,
    **kwargs,
) -> Any:
    """Field for an absolute motion.

    This field is used to define the shape and bounds of an absolute motion.

    Args:
        bounds: Bounds of the motion.
        shape: Shape of the motion.
        description: Description of the motion.
    """
    return MotionField(
        default=default,
        bounds=bounds,
        shape=shape,
        description=description,
        motion_type="ABSOLUTE",
        **kwargs,
    )


def RelativeMotionField(
    default: Any = PydanticUndefined,
    bounds: list[float] | None = None,
    shape: tuple[int] | None = None,
    description: str | None = None,
    **kwargs,
) -> Any:
    """Field for a relative motion.

    This field is used to define the shape and bounds of a relative motion.

    Args:
        bounds: Bounds of the motion.
        shape: Shape of the motion.
        description: Description of the motion.
    """
    return MotionField(
        default=default,
        bounds=bounds,
        shape=shape,
        description=description,
        motion_type="RELATIVE",
        **kwargs,
    )


def VelocityMotionField(
    default: Any = PydanticUndefined,
    bounds: list[float] | None = None,
    shape: tuple[int] | None = None,
    description: str | None = None,
    **kwargs,
) -> Any:
    """Field for a velocity motion.

    This field is used to define the shape and bounds of a velocity motion.

    Args:
        bounds: Bounds of the motion.
        shape: Shape of the motion.
        description: Description of the motion.
    """
    return MotionField(
        default=default,
        bounds=bounds,
        shape=shape,
        description=description,
        motion_type="VELOCITY",
        **kwargs,
    )


def TorqueMotionField(
    default: Any = PydanticUndefined,
    bounds: list[float] | None = None,
    shape: tuple[int] | None = None,
    description: str | None = None,
    **kwargs,
) -> Any:
    """Field for a torque motion.

    This field is used to define the shape and bounds of a torque motion.

    Args:
        bounds: Bounds of the motion.
        shape: Shape of the motion.
        description: Description of the motion.
    """
    return MotionField(
        default=default,
        bounds=bounds,
        shape=shape,
        description=description,
        motion_type="TORQUE",
        **kwargs,
    )


def OtherMotionField(
    default: Any = PydanticUndefined,
    bounds: list[float] | None = None,
    shape: tuple[int] | None = None,
    description: str | None = None,
    **kwargs,
) -> Any:
    """Field for an other motion.

    This field is used to define the shape and bounds of an other motion.

    Args:
        bounds: Bounds of the motion.
        shape: Shape of the motion.
        description: Description of the motion.
    """
    return MotionField(
        default=default,
        bounds=bounds,
        shape=shape,
        description=description,
        motion_type="OTHER",
        **kwargs,
    )


class Motion(Sample):
    """Base class for a motion."""

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
