"""This module contains the base class for a motion.

There are four basic motion types that are supported:
- Absolute motion: The desired absolute coordinates of a limb or joint in the chosen reference frame. 
- Relative motion: The displacement from the current position of a limb or joint (frame-independent).
- Velocity motion: The desired absolute velocity of a limb or joint (frame-independent).
- Torque motion: The desired torque of a limb or joint (frame-independent).

The bounds is a list of two floats representing the lower and upper bounds of the motion.
The shape is a tuple of integers representing the shape of the motion. 
The reference_frame is a string representing the reference frame for the coordinates (only applies to absolute motions).

To create a new Pydantic model for a motion, inherit from the Motion class and define pydantic fields with the MotionField,
function as you would with any other Pydantic field. 

Example:
    from mbodied_agents.motion import Motion, AbsoluteMotionField, MotionField, MotionType, VelocityMotionField
    from mbodied_agents.data.sample import Sample
    
    class Twist(Motion):
        x: float = VelocityMotionField(default=0.0, bounds=[-1.0, 1.0])
        y: float = VelocityMotionField(default=0.0, bounds=[-1.0, 1.0])
        z: float = VelocityMotionField(default=0.0, bounds=[-1.0, 1.0])
        roll: float = VelocityMotionField(default=0.0, bounds=['-pi', 'pi'])
        pitch: float = VelocityMotionField(default=0.0, bounds=['-pi', 'pi'])
        yaw: float = VelocityMotionField(default=0.0, bounds=['-pi', 'pi'])
        

This automatically generates a Pydantic model with the specified fields and the additional properties of a motion.
It is vectorizable, serializable, and validated according to its type. Furthermore, convience methods from
the class allow for direct conversion to numpy, pytorch, and gym spaces.
See the Sample class documentation for more information: https://mbodi-ai-mbodied-agents.readthedocs-hosted.com/en/latest/
See the Pydantic documentation for more information on how to define Pydantic models: https://pydantic-docs.helpmanual.io/
"""
from typing import Any

from pydantic import ConfigDict, Field
from pydantic_core import PydanticUndefined
from typing_extensions import Literal

from mbodied.types.sample import Sample

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
