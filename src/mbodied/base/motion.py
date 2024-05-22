import logging
import numpy as np
from pydantic import ConfigDict, Field
from typing_extensions import Literal, get_origin

from mbodied.types.ndarray import NumpyArray
from mbodied.base.sample import Sample

MotionSubtype = Literal[
    "unspecified",
    "sample",
    "absolute",
    "relative",
    "velocity",
    "torque",
]


def MotionField(
    *arg,
    bounds: NumpyArray | None = None,
    shape: tuple[int] | None = None,
    description: str | None = None,
    motion_type: MotionSubtype = "unspecified",
    **kwargs,
) -> Field:
    """Field for a motion.

    This field is used to define the shape and bounds of a motion.

    Args:
        bounds (Optional[NumpyArray]): Bounds of the motion.
        shape (Optional[Tuple[int]]): Shape of the motion.
        description (Optional[str]): Description of the motion.
        motion_type (MotionSubtype): Type of the motion.

    Returns:
        Field: Configured Pydantic Field for motion.
    """
    ge = bounds[0] if bounds is not None else None
    le = bounds[1] if bounds is not None else None

    if shape is not None and len(shape) > 1:
        raise ValueError("Only 1D supported currently.")

    min_length = shape[0] if shape and len(shape) > 1 else None
    max_length = shape[0] if shape and len(shape) > 1 else None

    if description is None:
        description = ""
    if shape is not None:
        description += f" Shape: {shape}"
    if bounds is not None:
        description += f" Bounds: {bounds}"
    if motion_type != "unspecified":
        description += f" Motion type: {motion_type}"

    logging.debug("motion field:", min_length,
                  max_length, description, ge, le, kwargs)
    return Field(min_length=min_length, max_length=max_length, description=description, ge=ge, le=le, **kwargs)


def AbsoluteMotionField(
    bounds: NumpyArray | None = None, shape: tuple[int] | None = None, description: str | None = None, **kwargs,
) -> Field:
    """Field for absolute motion.

    Args:
        bounds (Optional[NumpyArray]): Bounds of the motion.
        shape (Optional[Tuple[int]]): Shape of the motion.
        description (Optional[str]): Description of the motion.
        **kwargs: Additional keyword arguments.

    Returns:
        Field: Configured Pydantic Field for absolute motion.
    """
    return MotionField(bounds=bounds, shape=shape, description=description, motion_type="absolute", **kwargs)


def RelativeMotionField(
    bounds: NumpyArray | None = None, shape: tuple[int] | None = None, description: str | None = None, **kwargs,
) -> Field:
    """Field for relative motion.

    Args:
        bounds (Optional[NumpyArray]): Bounds of the motion.
        shape (Optional[Tuple[int]]): Shape of the motion.
        description (Optional[str]): Description of the motion.
        **kwargs: Additional keyword arguments.

    Returns:
        Field: Configured Pydantic Field for relative motion.
    """
    return MotionField(bounds=bounds, shape=shape, description=description, motion_type="relative", **kwargs)


def VelocityMotionField(
    bounds: NumpyArray | None = None, shape: tuple[int] | None = None, description: str | None = None, **kwargs,
) -> Field:
    """Field for velocity motion.

    Args:
        bounds (Optional[NumpyArray]): Bounds of the motion.
        shape (Optional[Tuple[int]]): Shape of the motion.
        description (Optional[str]): Description of the motion.
        **kwargs: Additional keyword arguments.

    Returns:
        Field: Configured Pydantic Field for velocity motion.
    """
    return MotionField(bounds=bounds, shape=shape, description=description, motion_type="velocity", **kwargs)


def TorqueMotionField(
    bounds: NumpyArray | None = None, shape: tuple[int] | None = None, description: str | None = None, **kwargs,
) -> Field:
    """Field for torque motion.

    Args:
        bounds (Optional[NumpyArray]): Bounds of the motion.
        shape (Optional[Tuple[int]]): Shape of the motion.
        description (Optional[str]): Description of the motion.
        **kwargs: Additional keyword arguments.

    Returns:
        Field: Configured Pydantic Field for torque motion.
    """
    return MotionField(bounds=bounds, shape=shape, description=description, motion_type="torque", **kwargs)


def CustomMotionField(
    bounds: NumpyArray | None = None, shape: tuple[int] | None = None, description: str | None = None, **kwargs,
) -> Field:
    """Field for custom motion.

    Args:
        bounds (Optional[NumpyArray]): Bounds of the motion.
        shape (Optional[Tuple[int]]): Shape of the motion.
        description (Optional[str]): Description of the motion.
        **kwargs: Additional keyword arguments.

    Returns:
        Field: Configured Pydantic Field for custom motion.
    """
    return MotionField(bounds=bounds, shape=shape, description=description, motion_type="sample", **kwargs)


class Motion(Sample):
    """Base class for a motion."""

    model_config: ConfigDict = ConfigDict(
        arbitrary_types_allowed=True, extra="forbid")

    def get_metadata(self) -> None:
        """Retrieve metadata for the motion."""
        pass

    def flatten(self, *args, with_metadata=False, **kwargs) -> NumpyArray:
        """Flatten the motion data.

        Args:
            with_metadata (bool): Whether to include metadata in the flattened data.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            NumpyArray: Flattened motion data.

        Raises:
            NotImplementedError: If metadata inclusion is requested but not supported.
        """
        flattened = np.array(super().flatten(*args, **kwargs))
        if with_metadata and isinstance(flattened, (list, np.ndarray)):
            list(get_origin(MotionSubtype)).index("absolute")
            raise NotImplementedError("Metadata not supported yet.")
        return flattened
