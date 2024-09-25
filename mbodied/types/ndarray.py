from __future__ import annotations

import pickle as pickle_pkg
import sys
from dataclasses import dataclass
from functools import lru_cache, partial, singledispatch
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    Iterable,
    Optional,
    TypeVarTuple,
    get_type_hints,
)

import compress_pickle
import numpy as np
import numpy.typing as npt
from numpy.lib.npyio import NpzFile
from numpy.typing import NDArray
from packaging.version import Version
from pydantic import (
    BaseModel,
    DirectoryPath,
    FilePath,
    GetJsonSchemaHandler,
    PositiveInt,
    computed_field,
    validate_call,
)
from pydantic.json_schema import JsonSchemaValue, core_schema
from pydantic_core import PydanticCustomError
from ruamel import yaml
from typing_extensions import (
    Annotated,
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

try:
    from numpy._core._exceptions import UFuncTypeError
except ImportError:
    from numpy.core._exceptions import UFuncTypeError # type: ignore # noqa


if TYPE_CHECKING:
    from pydantic.json_schema import JsonSchemaValue
    from pydantic.types import DirectoryPath

class PydanticNumpyMultiArrayNumpyFileOnFilePathError(Exception):
    pass


def validate_numpy_array_file(v: FilePath) -> npt.NDArray:
    """Validate file path to numpy file by loading and return the respective numpy array."""
    result = np.load(v)

    if isinstance(result, NpzFile):
        files = result.files
        if len(files) > 1:
            msg = (
                f"The provided file path is a multi array NpzFile, which is not supported; "
                f"convert to single array NpzFiles.\n"
                f"Path to multi array file: {result}\n"
                f"Array keys: {', '.join(result.files)}\n"
                f"Use embdata.ndarray.{MultiArrayNumpyFile.__name__} instead of a PathLike alone"
            )
            raise PydanticNumpyMultiArrayNumpyFileOnFilePathError(msg)
        result = result[files[0]]

    return result


def validate_multi_array_numpy_file(v: MultiArrayNumpyFile) -> npt.NDArray:
    """Validation function for loading numpy array from a name mapping numpy file.

    Parameters
    ----------
    v: MultiArrayNumpyFile
        MultiArrayNumpyFile to load

    Returns:
    -------
    NDArray from MultiArrayNumpyFile
    """
    return v.load()


def np_general_all_close(arr_a: npt.NDArray, arr_b: npt.NDArray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    """Data type agnostic function to define if two numpy array have elements that are close.

    Parameters
    ----------
    arr_a: npt.NDArray
    arr_b: npt.NDArray
    rtol: float
        See np.allclose
    atol: float
        See np.allclose

    Returns:
    -------
    Bool
    """
    return _np_general_all_close(arr_a, arr_b, rtol, atol)


if Version(np.version.version) < Version("1.25.0"):

    def _np_general_all_close(arr_a: npt.NDArray, arr_b: npt.NDArray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
        try:
            return np.allclose(arr_a, arr_b, rtol=rtol, atol=atol, equal_nan=True)
        except UFuncTypeError:
            return np.allclose(arr_a.astype(np.float64), arr_b.astype(np.float64), rtol=rtol, atol=atol, equal_nan=True)
        except TypeError:
            return bool(np.all(arr_a == arr_b))

else:
    from numpy.exceptions import DTypePromotionError

    def _np_general_all_close(arr_a: npt.NDArray, arr_b: npt.NDArray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
        try:
            return np.allclose(arr_a, arr_b, rtol=rtol, atol=atol, equal_nan=True)
        except UFuncTypeError:
            return np.allclose(arr_a.astype(np.float64), arr_b.astype(np.float64), rtol=rtol, atol=atol, equal_nan=True)
        except DTypePromotionError:
            return bool(np.all(arr_a == arr_b))

yaml = yaml.YAML()


@dataclass(frozen=True)
class MultiArrayNumpyFile:
    path: FilePath
    key: str
    cached_load: bool = False

    def load(self) -> npt.NDArray:
        """Load the NDArray stored in the given path within the given key.

        Returns:
        -------
        NDArray
        """
        loaded = _cached_np_array_load(self.path) if self.cached_load else np.load(self.path)
        try:
            return loaded[self.key]
        except IndexError as e:
            msg = f"The given path points to an uncompressed numpy file, which only has one array in it: {self.path}"
            raise AttributeError(msg) from e


class NumpyModel(BaseModel):
    _dump_compression: ClassVar[str] = "lz4"
    _dump_numpy_savez_file_name: ClassVar[str] = "arrays.npz"
    _dump_non_array_file_stem: ClassVar[str] = "object_info"

    _directory_suffix: ClassVar[str] = ".pdnp"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BaseModel):
            return NotImplemented  # delegate to the other item in the comparison

        self_type = self.__pydantic_generic_metadata__["origin"] or self.__class__
        other_type = other.__pydantic_generic_metadata__["origin"] or other.__class__

        if not (
            self_type == other_type
            and getattr(self, "__pydantic_private__", None) == getattr(other, "__pydantic_private__", None)
            and self.__pydantic_extra__ == other.__pydantic_extra__
        ):
            return False

        if isinstance(other, NumpyModel):
            self_ndarray_field_to_array, self_other_field_to_value = self._dump_numpy_split_dict()
            other_ndarray_field_to_array, other_other_field_to_value = other._dump_numpy_split_dict()

            return self_other_field_to_value == other_other_field_to_value and _compare_np_array_dicts(
                self_ndarray_field_to_array, other_ndarray_field_to_array,
            )

        # Self is NumpyModel, other is not; likely unequal; checking anyway.
        return super().__eq__(other)

    @classmethod
    @validate_call
    def model_directory_path(cls, output_directory: DirectoryPath, object_id: str) -> DirectoryPath:
        return output_directory / f"{object_id}.{cls.__name__}{cls._directory_suffix}"

    @classmethod
    @validate_call
    def load(
        cls,
        output_directory: DirectoryPath,
        object_id: str,
        *,
        pre_load_modifier: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
    ):
        """Load NumpyModel instance.

        Parameters
        ----------
        output_directory: DirectoryPath
            The root directory where all model instances of interest are stored
        object_id: String
            The ID of the model instance
        pre_load_modifier: Callable[[dict[str, Any]], dict[str, Any]] | None
            Optional function that modifies the loaded arrays

        Returns:
        -------
        NumpyModel instance
        """
        object_directory_path = cls.model_directory_path(output_directory, object_id)

        npz_file = np.load(object_directory_path / cls._dump_numpy_savez_file_name)

        other_path: FilePath
        if (other_path := object_directory_path / cls._dump_compressed_pickle_file_name).exists():  # pyright: ignore
            other_field_to_value = compress_pickle.load(other_path)
        elif (other_path := object_directory_path / cls._dump_pickle_file_name).exists():  # pyright: ignore
            with open(other_path, "rb") as in_pickle:
                other_field_to_value = pickle_pkg.load(in_pickle)
        elif (other_path := object_directory_path / cls._dump_non_array_yaml_name).exists():  # pyright: ignore
            with open(other_path) as in_yaml:
                other_field_to_value = yaml.load(in_yaml)
        else:
            other_field_to_value = {}

        field_to_value = {**npz_file, **other_field_to_value}
        if pre_load_modifier:
            field_to_value = pre_load_modifier(field_to_value)

        return cls(**field_to_value)

    @validate_call
    def dump(
        self, output_directory: Path, object_id: str, *, compress: bool = True, pickle: bool = False,
    ) -> DirectoryPath:
        assert "arbitrary_types_allowed" not in self.model_config or (
            self.model_config["arbitrary_types_allowed"] and pickle
        ), "Arbitrary types are only supported in pickle mode"

        dump_directory_path = self.model_directory_path(output_directory, object_id)
        dump_directory_path.mkdir(parents=True, exist_ok=True)

        ndarray_field_to_array, other_field_to_value = self._dump_numpy_split_dict()

        if ndarray_field_to_array:
            (np.savez_compressed if compress else np.savez)(
                dump_directory_path / self._dump_numpy_savez_file_name, **ndarray_field_to_array,
            )

        if other_field_to_value:
            if pickle:
                if compress:
                    compress_pickle.dump(
                        other_field_to_value,
                        dump_directory_path / self._dump_compressed_pickle_file_name,  # pyright: ignore
                        compression=self._dump_compression,
                    )
                else:
                    with open(dump_directory_path / self._dump_pickle_file_name, "wb") as out_pickle:  # pyright: ignore
                        pickle_pkg.dump(other_field_to_value, out_pickle)

            else:
                with open(dump_directory_path / self._dump_non_array_yaml_name, "w") as out_yaml:  # pyright: ignore
                    yaml.dump(other_field_to_value, out_yaml)

        return dump_directory_path

    def _dump_numpy_split_dict(self) -> tuple[dict, dict]:
        ndarray_field_to_array = {}
        other_field_to_value = {}

        for k, v in self.model_dump().items():
            if isinstance(v, np.ndarray):
                ndarray_field_to_array[k] = v
            elif v:
                other_field_to_value[k] = v

        return ndarray_field_to_array, other_field_to_value

    @classmethod  # type: ignore[misc]
    @computed_field(return_type=str)
    @property
    def _dump_compressed_pickle_file_name(cls) -> str:
        return f"{cls._dump_non_array_file_stem}.pickle.{cls._dump_compression}"

    @classmethod  # type: ignore[misc]
    @computed_field(return_type=str)
    @property
    def _dump_pickle_file_name(cls) -> str:
        return f"{cls._dump_non_array_file_stem}.pickle"

    @classmethod  # type: ignore[misc]
    @computed_field(return_type=str)
    @property
    def _dump_non_array_yaml_name(cls) -> str:
        return f"{cls._dump_non_array_file_stem}.yaml"


def model_agnostic_load(
    output_directory: DirectoryPath,
    object_id: str,
    models: Iterable[type[NumpyModel]],
    not_found_error: bool = False,
    **load_kwargs,
) -> Optional[NumpyModel]:
    """Provided an Iterable containing possible models, and the directory where they have been dumped.

     Load the first
    instance of model that matches the provided object ID.

    Parameters
    ----------
    output_directory: DirectoryPath
        The root directory where all model instances of interest are stored
    object_id: String
        The ID of the model instance
    models: Iterable[type[NumpyModel]]
        All NumpyModel instances of interest, note that they should have differing names
    not_found_error: bool
        If True, throw error when the respective model instance was not found
    load_kwargs
        Key-word arguments to pass to the load function

    Returns:
    -------
    NumpyModel instance if found
    """
    for model in models:
        if model.model_directory_path(output_directory, object_id).exists():
            return model.load(output_directory, object_id, **load_kwargs)

    if not_found_error:
        msg = (
            f"Could not find NumpyModel with {object_id} in {output_directory}."
            f"Tried from following classes:\n{', '.join(model.__name__ for model in models)}"
        )
        raise FileNotFoundError(
            msg,
        )

    return None


@lru_cache
def _cached_np_array_load(path: FilePath):
    """Store the loaded numpy object within LRU cache in case we need it several times.

    Parameters
    ----------
    path: FilePath
        Path to the numpy file

    Returns:
    -------
    Same as np.load
    """
    return np.load(path)


def _compare_np_array_dicts(
    dict_a: dict[str, npt.NDArray], dict_b: dict[str, npt.NDArray], rtol: float = 1e-05, atol: float = 1e-08,
) -> bool:
    """Compare two dictionaries containing numpy arrays as values.

    Parameters:
    dict_a, dict_b: dictionaries to compare. They should have same keys.
    rtol, atol: relative and absolute tolerances for np.isclose()

    Returns:
    Boolean value for each key, True if corresponding arrays are close, else False.
    """
    keys1 = frozenset(dict_a.keys())
    keys2 = frozenset(dict_b.keys())

    if keys1 != keys2:
        return False

    for key in keys1:
        arr_a = dict_a[key]
        arr_b = dict_b[key]

        if arr_a.shape != arr_b.shape or not np_general_all_close(arr_a, arr_b, rtol, atol):
            return False

    return True



# def generate_pydantic_signature(
#     init: Callable[..., None], fields: dict[str, FieldInfo], config_wrapper: ConfigWrapper, is_dataclass: bool = False,
# ) -> Signature:
#     """Generate signature for a pydantic BaseModel or dataclass.

#     Args:
#         init: The class init.
#         fields: The model fields.
#         config_wrapper: The config wrapper instance.
#         is_dataclass: Whether the model is a dataclass.

#     Returns:
#         The dataclass/BaseModel subclass signature.
#     """
#     merged_params = _generate_signature_parameters(init, fields, config_wrapper)

#     if is_dataclass:
#         merged_params = {k: _process_param_defaults(v) for k, v in merged_params.items()}

#     return Signature(parameters=list(merged_params.values()), return_annotation=None)

class NumpyDataDict(TypedDict):
    data: List
    data_type: SupportedDTypes | str
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
        array: np.ndarray,
        shape: Tuple[int, ...] | None,
        dtype: SupportedDTypes | None,
        labels: List[str] | None,
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
        array: dict,
        shape: Tuple[int, ...] | None,
        dtype: SupportedDTypes | None,
        labels: List[str] | None,
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

Ts = TypeVarTuple("Ts")

class NumpyArray(Generic[*Ts], NDArray[Any]):
    """Pydantic validation for shape and dtype. Specify shape with a tuple of integers, "*" or `Any` for any size.

    If the last dimension is a type (e.g. np.uint8), it will validate the dtype as well.

    Examples:
        - NumpyArray[1, 2, 3] will validate a 3D array with shape (1, 2, 3).
        - NumpyArray[Any, "*", Any] will validate a 3D array with any shape.
        - NumpyArray[3, 224, 224, np.uint8] will validate an array with shape (3, 224, 224) and dtype np.uint8.

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
        if isinstance(_dtype, int) or _dtype == "*":
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

            __str__ = cls.__str__
            __doc__ = cls.__doc__

            def __repr__(self):
                if self.shape is None and self.dtype is None:
                    return "NumpyArray"
                if self.shape is Any and self.dtype and self.dtype is not Any:
                    return f"NumpyArray[Any, {self.dtype.__name__}]"
                return f"NumpyArray[{', '.join(str(s) for s in self.shape)}" + (f", {self.dtype.__name__}" if self.dtype else "") + "]"

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

    # doctest.testmod(verbose=True)
    from pydantic import ConfigDict
    from pydantic.fields import Field
    class MyModel(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        uint8_array: NumpyArray[np.uint8] | None = Field(default=None, description="A 3D array with shape (3, 224, 224) and dtype np.uint8.")
        must_have_exact_shape: NumpyArray[1, 2, 3]
        must_be_3d: NumpyArray[Any, Any, Any]
        must_be_1d: NumpyArray[Any, Any, Any]

    my_failing_model = MyModel(
        uint8_array=[1, 2, 3, 4],
        must_have_exact_shape=[[[1]], [[2]]],
        must_be_3d=[[[1, 2, 3], [4, 5, 6]]],
        must_be_1d=[[[1, 2, 3]]],
    )
    m: MyModel = MyModel()
    hints = get_type_hints(MyModel)
