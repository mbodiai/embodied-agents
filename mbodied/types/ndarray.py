import sys
from functools import partial, singledispatch
from pathlib import Path

import numpy as np
import numpy.typing as npt
from pydantic import (
    BaseModel,
    FilePath,
    GetJsonSchemaHandler,
    PositiveInt,
    validate_call,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticCustomError, core_schema
from pydantic_numpy.helper.annotation import (
    MultiArrayNumpyFile,
)
from pydantic_numpy.helper.validation import (
    validate_multi_array_numpy_file,
    validate_numpy_array_file,
)
from typing_extensions import (
    Annotated,
    Any,
    Callable,
    ClassVar,
    List,
    Sequence,
    Tuple,
    TypedDict,
)

SupportedDTypes = (
    type[np.generic]
    | type[np.number]
    | type[np.bool_]
    | type[np.int64]
    | type[np.dtypes.Int64DType]
    | type[np.uint64]
    | type[np.dtypes.UInt64DType]
    | type[np.float64]
    | type[np.timedelta64]
    | type[np.datetime64]
)


class NumpyDataDict(TypedDict):
    data: List
    data_type: SupportedDTypes
    shape: Tuple[int, ...]


if sys.version_info < (3, 11):

    def array_validator(array: np.ndarray, shape: Tuple[int, ...] | None, dtype: SupportedDTypes | None) -> npt.NDArray:
        if shape is not None:
            expected_ndim = len(shape)
            actual_ndim = array.ndim
            if actual_ndim != expected_ndim:
                details = f"Array has {actual_ndim} dimensions, expected {expected_ndim}"
                msg = "ShapeError"
                raise PydanticCustomError(msg, details)
            for i, (expected, actual) in enumerate(zip(shape, array.shape, strict=False)):
                if expected != -1 and expected is not None and expected != actual:
                    details = f"Dimension {i} has size {actual}, expected {expected}"
                    msg = "ShapeError"
                    raise PydanticCustomError(msg, details)

        if (
            dtype
            and array.dtype.type != dtype
            and issubclass(dtype, np.integer)
            and issubclass(array.dtype.type, np.floating)
        ):
            array = np.round(array).astype(dtype, copy=False)
        if dtype and issubclass(dtype, np.dtypes.UInt64DType | np.dtypes.Int64DType):
            dtype = np.int64
            array = array.astype(dtype, copy=True)
        return array
else:

    @singledispatch
    def array_validator(
        array: np.ndarray, shape: Tuple[int, ...] | None, dtype: SupportedDTypes | None, labels: List[str] | None
    ) -> npt.NDArray:
        if shape is not None:
            expected_ndim = len(shape)
            actual_ndim = array.ndim
            if actual_ndim != expected_ndim:
                details = f"Array has {actual_ndim} dimensions, expected {expected_ndim}"
                msg = "ShapeError"
                raise PydanticCustomError(msg, details)
            for i, (expected, actual) in enumerate(zip(shape, array.shape, strict=False)):
                if expected != -1 and expected is not None and expected != actual:
                    details = f"Dimension {i} has size {actual}, expected {expected}"
                    msg = "ShapeError"
                    raise PydanticCustomError(msg, details)

        if (
            dtype
            and array.dtype.type != dtype
            and issubclass(dtype, np.integer)
            and issubclass(array.dtype.type, np.floating)
        ):
            array = np.round(array).astype(dtype, copy=False)
        if dtype and issubclass(dtype, np.dtypes.UInt64DType | np.dtypes.Int64DType):
            dtype = np.int64
            array = array.astype(dtype, copy=True)
        return array

    @array_validator.register
    def list_tuple_validator(
        array: list | tuple,
        shape: Tuple[int, ...] | None,
        dtype: SupportedDTypes | None,
    ) -> npt.NDArray:
        return array_validator.dispatch(np.ndarray)(np.asarray(array), shape, dtype)

    @array_validator.register
    def dict_validator(
        array: dict, shape: Tuple[int, ...] | None, dtype: SupportedDTypes | None, labels: List[str] | None
    ) -> npt.NDArray:
        array = np.array(array["data"]).astype(array["data_type"]).reshape(array["shape"])
        return array_validator.dispatch(np.ndarray)(array, shape, dtype, labels)


def create_array_validator(
    shape: Tuple[int, ...] | None,
    dtype: SupportedDTypes | None,
    labels: List[str] | None,
) -> Callable[[Any], npt.NDArray]:
    """Creates a validator function for NumPy arrays with a specified shape and data type."""
    return partial(array_validator, shape=shape, dtype=dtype, labels=labels)


@validate_call
def _deserialize_numpy_array_from_data_dict(data_dict: NumpyDataDict) -> np.ndarray:
    return np.array(data_dict["data"]).astype(data_dict["data_type"]).reshape(data_dict["shape"])


_common_numpy_array_validator = core_schema.union_schema(
    [
        core_schema.chain_schema(
            [
                core_schema.is_instance_schema(Path),
                core_schema.no_info_plain_validator_function(validate_numpy_array_file),
            ],
        ),
        core_schema.chain_schema(
            [
                core_schema.is_instance_schema(MultiArrayNumpyFile),
                core_schema.no_info_plain_validator_function(validate_multi_array_numpy_file),
            ],
        ),
        core_schema.is_instance_schema(np.ndarray),
        core_schema.chain_schema(
            [
                core_schema.is_instance_schema(Sequence),
                core_schema.no_info_plain_validator_function(lambda v: np.asarray(v)),
            ],
        ),
        core_schema.chain_schema(
            [
                core_schema.is_instance_schema(dict),
                core_schema.no_info_plain_validator_function(_deserialize_numpy_array_from_data_dict),
            ],
        ),
    ],
)


def get_numpy_json_schema(
    _field_core_schema: core_schema.CoreSchema,
    _handler: GetJsonSchemaHandler,
    shape: List[PositiveInt] | None = None,
    data_type: SupportedDTypes | None = None,
    labels: List[str] | None = None,
) -> JsonSchemaValue:
    """Generates a JSON schema for a NumPy array field within a Pydantic model.

    This function constructs a JSON schema definition compatible with Pydantic models
    that are intended to validate NumPy array inputs. It supports specifying the data type
    and dimensions of the NumPy array, which are used to construct a schema that ensures
    input data matches the expected structure and type.

    Parameters
    ----------
    _field_core_schema : core_schema.CoreSchema
        The core schema component of the Pydantic model, used for building basic schema structures.
    _handler : GetJsonSchemaHandler
        A handler function or object responsible for converting Python types to JSON schema components.
    shape : Optional[List[PositiveInt]], optional
        The expected shape of the NumPy array. If specified, the schema will enforce that the input
    data_type : Optional[SupportedDTypes], optional
        The expected data type of the NumPy array elements. If specified, the schema will enforce
        that the input array's data type is compatible with this. If `None`, any data type is allowed,
        by default None.

    Returns:
    -------
    JsonSchemaValue
        A dictionary representing the JSON schema for a NumPy array field within a Pydantic model.
        This schema includes details about the expected array dimensions and data type.
    """
    array_shape = shape if shape else "Any"
    if data_type:
        array_data_type = data_type.__name__
        item_schema = core_schema.list_schema(
            items_schema=core_schema.any_schema(metadata=f"Must be compatible with numpy.dtype: {array_data_type}"),
        )
    else:
        array_data_type = "Any"
        item_schema = core_schema.list_schema(items_schema=core_schema.any_schema())

    if shape:
        data_schema = core_schema.list_schema(items_schema=item_schema, min_length=shape[0], max_length=shape[0])
    else:
        data_schema = item_schema

    return {
        "title": "Numpy Array",
        "type": f"np.ndarray[{array_shape}, np.dtype[{array_data_type}]]",
        "required": ["data_type", "data"],
        "properties": {
            "data_type": {"title": "dtype", "default": array_data_type, "type": "string"},
            "shape": {"title": "shape", "default": array_shape, "type": "array"},
            "data": data_schema,
        },
    }


def array_to_data_dict_serializer(array: npt.ArrayLike) -> NumpyDataDict:
    array = np.array(array)

    if issubclass(array.dtype.type, np.timedelta64) or issubclass(array.dtype.type, np.datetime64):
        data = array.astype(int).tolist()
    else:
        data = array.astype(float).tolist()
    dtype = str(array.dtype) if hasattr(array, "dtype") else "float"
    return NumpyDataDict(data=data, data_type=dtype, shape=array.shape)


class NumpyArray:
    """Pydantic validation for shape and dtype. Specify shape with a tuple of integers, "*" or `Any` for any size.

    If the last dimension is a type (e.g. np.uint8), it will validate the dtype as well.

    Examples:
        from typing import Any
        NumpyArray[1, 2, 3] will validate a 3D array with shape (1, 2, 3).
        NumpyArray[Any, Any, Any] will validate a 3D array with any shape.
        NumpyArray[3, 224, 224, np.uint8] will validate an array with shape (3, 224, 224) and dtype np.uint8.

    Lazy loading and caching by default.

    Usage:
    >>> from pydantic import BaseModel
    >>> from embdata.ndarray import NumpyArray
    >>> class MyModel(BaseModel):
    ...     uint8_array: NumpyArray[np.uint8]
    ...     must_have_exact_shape: NumpyArray[1, 2, 3]
    ...     must_be_3d: NumpyArray["*", "*", "*"]  # NumpyArray[Any, Any, Any] also works.
    ...     must_be_1d: NumpyArray["*",]  # NumpyArray[Any,] also works.
    """

    shape: ClassVar[Tuple[PositiveInt, ...] | None] = None
    dtype: ClassVar[SupportedDTypes | None] = None
    labels: ClassVar[Tuple[str, ...] | None] = None

    def __repr__(self) -> str:
        class_params = str(*self.shape) if self.shape is not None else "*"
        dtype = f", {self.dtype.__name__}" if self.dtype is not None else ", Any"
        if self.labels:
            class_params = ",".join([f"{l}={s}" for l, s in zip(self.labels, self.shape, strict=False)])

        return f"NumpyArray[{class_params}{dtype}]"

    def __str__(self) -> str:
        return repr(self)

    @classmethod
    def __class_getitem__(cls, params=None) -> Any:
        _shape = None
        _dtype = None
        _labels = None
        if params is None or params in ("*", Any, (Any,)):
            params = ("*",)
        if not isinstance(params, tuple):
            params = (params,)
        if len(params) == 1:
            if isinstance(params[0], type):
                _dtype = params[0]
        else:
            *_shape, _dtype = params
            _shape = tuple(s if s not in ("*", Any) else -1 for s in _shape)

        _labels = []
        if isinstance(_dtype, str | int):
            _shape += (_dtype,)
            _dtype = Any
        _shape = _shape or ()
        for s in _shape:
            if isinstance(s, str):
                if s.isnumeric():
                    _labels.append(int(s))
                elif s in ("*", Any):
                    _labels.append(-1)
                elif "=" in s:
                    s = s.split("=")[1]  # noqa: PLW2901
                    if not s.isnumeric():
                        msg = f"Invalid shape parameter: {s}"
                        raise ValueError(msg)
                    _labels.append(int(s))
                else:
                    msg = f"Invalid shape parameter: {s}"
                    raise ValueError(msg)
        if _dtype is int:
            _dtype: SupportedDTypes | None = np.int64
        elif _dtype is float:
            _dtype = np.float64
        elif _dtype is not None and _dtype not in ("*", Any) and isinstance(_dtype, type):
            _dtype = np.dtype(_dtype).type

        if _shape == ():
            _shape = None

        class ParameterizedNumpyArray(cls):
            shape = _shape
            dtype = _dtype
            labels = _labels or None

            __repr__ = cls.__repr__
            __str__ = cls.__str__
            __doc__ = cls.__doc__

        return Annotated[np.ndarray | FilePath | MultiArrayNumpyFile, ParameterizedNumpyArray]

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: Callable[[Any], core_schema.CoreSchema],
    ) -> core_schema.CoreSchema:
        np_array_validator = create_array_validator(cls.shape, cls.dtype, cls.labels)
        np_array_schema = core_schema.no_info_plain_validator_function(np_array_validator)

        return core_schema.json_or_python_schema(
            python_schema=core_schema.chain_schema(
                [
                    core_schema.union_schema(
                        [
                            core_schema.is_instance_schema(np.ndarray),
                            core_schema.is_instance_schema(list),
                            core_schema.is_instance_schema(tuple),
                            core_schema.is_instance_schema(dict),
                        ],
                    ),
                    _common_numpy_array_validator,
                    np_array_schema,
                ],
            ),
            json_schema=core_schema.chain_schema(
                [
                    core_schema.union_schema(
                        [
                            core_schema.list_schema(),
                            core_schema.dict_schema(),
                        ],
                    ),
                    np_array_schema,
                ],
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                array_to_data_dict_serializer,
                when_used="json-unless-none",
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        field_core_schema: core_schema.CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        return get_numpy_json_schema(field_core_schema, handler, cls.shape, cls.dtype, cls.labels)


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
