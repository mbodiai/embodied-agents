from typing import Any, Type, TypeAlias

import numpy as np
import pytest
from pydantic import ConfigDict, ValidationError
from pydantic_core import PydanticUndefined
from pydantic import BaseModel
from pydantic.fields import Field
from typing_extensions import Annotated, Dict, Unpack

from mbodied.types.ndarray import NumpyArray
from mbodied.types.sample import Sample


class ModelWithArrays(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    any_array: NumpyArray[Any]
    flexible_array: NumpyArray = Field(
        default_factory=lambda: np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    )
    int_vector: NumpyArray[3, int] = Field(default_factory=lambda: np.array([1, 2, 3]))
    float_matrix: NumpyArray[2, 2, np.float64] = Field(
        default_factory=lambda: np.array([[1.0, 2.0], [3.0, 4.0]])
    )
    any_3d_array: NumpyArray["*", "*", "*", Any]  # type: ignore
    any_float_array: NumpyArray[float] = Field(description="Any float array")
    array: NumpyArray = Field(default_factory=lambda: np.array([1.0, 2.0, 3.0]))
    rotation_matrix: NumpyArray[3, 3, float] = Field(
        default_factory=lambda: np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
    )


@pytest.fixture()
def nested_model():
    class OtherModel(Sample):
        name: str = "OtherModel"
        any_array: NumpyArray[Any, float] = Field(
            default_factory=lambda: np.array([1.0, 2.0, 3.0])
        )
        coordinate: NumpyArray = Field(
            default_factory=lambda: np.array([1.0, 2.0, 3.0])
        )

    class NestedSample(Sample):
        any_array: NumpyArray[Any] = Field(
            default_factory=lambda: np.array([1, 2, 3, 4])
        )
        array: NumpyArray = Field(default_factory=lambda: np.array([1.0, 2.0, 3.0]))
        model: ModelWithArrays
        flexible_array: NumpyArray = Field(
            default_factory=lambda: np.array([[1, 2], [3, 4]])
        )
        int_vector: NumpyArray[3, int]
        float_matrix: NumpyArray[2, 2, np.float64] = Field(
            default_factory=lambda: np.array([[1.0, 2.0], [3.0, 4.0]])
        )
        any_3d_array: NumpyArray["*", "*", "*", Any]
        any_float_array: NumpyArray[float] = Field(description="Any float array")
        rotation_matrix: NumpyArray[3, 3, float] = Field(
            default_factory=lambda: np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            )
        )
        coordinate: NumpyArray[Any, float] = Field(
            default_factory=lambda: np.array([1.0, 2.0, 3.0])
        )
        nested: OtherModel

    return NestedSample(
        array=np.array([[1.0, 2.0], [3.0, 4.0]]),
        model=ModelWithArrays(
            any_array=np.array([1, 2, 3, 4]),
            flexible_array=np.array([[1, 2], [3, 4]]),
            int_vector=np.array([1, 2, 3]),
            any_3d_array=np.zeros((2, 3, 4)),
            any_float_array=np.array([1.0, 2.0, 3.0]),
        ),
        coordinate=np.array([1.0, 2.0, 3.0]),
        any_array=np.array([1, 2, 3, 4]),
        flexible_array=np.array([[1, 2], [3, 4]]),
        int_vector=np.array([1, 2, 3]),
        float_matrix=np.array([[1.0, 2.0], [3.0, 4.0]]),
        any_3d_array=np.zeros((2, 3, 4)),
        any_float_array=np.array([1.0, 2.0, 3.0]),
        rotation_matrix=np.eye(3),
        nested=OtherModel(
            name="OtherModel",
            any_array=np.array([1, 2, 3, 4]),
            # flexible_array=np.array([[1, 2], [3, 4]]),
            # int_vector=np.array([1, 2, 3]),
            # any_3d_array=np.zeros((2, 3, 4)),
            # any_float_array=np.array([1.0, 2.0, 3.0]),
        ),
    )


def test_basic_once():
    class TestModelWithArrays(Sample):
        float_matrix: NumpyArray[2, 2, np.float64]

    assert TestModelWithArrays(
        float_matrix=np.array([[1.0, 2.0], [3.0, 4.0]]).tolist()
    ).float_matrix.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_basic_serialize():
    class TestModelWithArrays(Sample):
        float_matrix: NumpyArray[2, 2, np.float64]

    instance = TestModelWithArrays(
        float_matrix=np.array([[1.0, 2.0], [3.0, 4.0]]).tolist()
    )
    assert instance.float_matrix.tolist() == [
        [1.0, 2.0],
        [3.0, 4.0],
    ]
    assert np.array_equal(
        instance.model_dump()["float_matrix"],
        np.array([[1.0, 2.0], [3.0, 4.0]]).tolist(),
    )
    json = instance.model_dump_json()
    assert (
        json
        == '{"float_matrix":{"data":[[1.0,2.0],[3.0,4.0]],"data_type":"float64","shape":[2,2]}}'
    )
    assert np.array_equal(
        TestModelWithArrays.model_validate_json(json).float_matrix.tolist(),
        np.array([[1.0, 2.0], [3.0, 4.0]]).tolist(),
    )


def assert_models_equal(model1: ModelWithArrays, model2: ModelWithArrays):
    assert np.array_equal(model1.any_array, model2.any_array)
    assert np.array_equal(model1.flexible_array, model2.flexible_array)
    assert np.array_equal(model1.int_vector, model2.int_vector)
    assert np.array_equal(model1.float_matrix, model2.float_matrix)
    assert np.array_equal(model1.any_3d_array, model2.any_3d_array)
    assert np.array_equal(model1.any_float_array, model2.any_float_array)


def test_model_with_arrays():
    model = ModelWithArrays(
        any_array=np.array([1, 2, 3, 4]),
        flexible_array=np.array([[1, 2], [3, 4]]),
        int_vector=np.array([1, 2, 3]),
        float_matrix=np.array([[1.0, 2.0], [3.0, 4.0]]),
        any_3d_array=np.zeros((2, 3, 4)),
        any_float_array=np.array([1.0, 2.0, 3.0]),
        array=np.array([1.0, 2.0, 3.0]),
        rotation_matrix=np.eye(3),
    )

    assert isinstance(model.any_array, np.ndarray)
    assert isinstance(model.flexible_array, np.ndarray)
    assert isinstance(model.int_vector, np.ndarray)
    assert isinstance(model.float_matrix, np.ndarray)
    assert isinstance(model.any_3d_array, np.ndarray)
    assert isinstance(model.any_float_array, np.ndarray)
    assert isinstance(model.array, np.ndarray)
    assert isinstance(model.rotation_matrix, np.ndarray)


def test_serialization_deserialization_nested(nested_model):
    # Test serialization with model_dump
    serialized = nested_model.model_dump()
    assert isinstance(serialized, Dict)

    # Test serialization with model_dump_json
    serialized_json = nested_model.model.model_dump_json()
    assert isinstance(serialized_json, str)
    # Test serialization with model_dump_json
    serialized_json = nested_model.model_dump_json()
    assert isinstance(serialized_json, str)

    # Test deserialization with model_validate
    deserialized = nested_model.model_validate(serialized)
    assert isinstance(deserialized, type(nested_model))

    # Test deserialization with model_validate_json
    deserialized_json = nested_model.model_validate_json(serialized_json)
    assert isinstance(deserialized_json, type(nested_model))

    # Compare original and deserialized models
    assert_models_equal(nested_model, deserialized)
    assert_models_equal(nested_model, deserialized_json)


def test_serialization_deserialization():
    model = ModelWithArrays(
        any_array=np.array([1, 2, 3, 4]),
        flexible_array=np.array([[1, 2], [3, 4]]),
        int_vector=np.array([1, 2, 3]),
        float_matrix=np.array([[1.0, 2.0], [3.0, 4.0]]),
        any_3d_array=np.zeros((2, 3, 4)),
        any_float_array=np.array([1.0, 2.0, 3.0]),
        array=np.array([1.0, 2.0, 3.0]),
        rotation_matrix=np.eye(3),
    )

    # Test serialization with model_dump
    serialized = model.model_dump()
    assert isinstance(serialized, dict)

    # Test serialization with model_dump_json
    serialized_json = model.model_dump_json()
    assert isinstance(serialized_json, str)

    # Test deserialization with model_validate
    deserialized = ModelWithArrays.model_validate(serialized)
    assert isinstance(deserialized, ModelWithArrays)

    # Test deserialization with model_validate_json
    deserialized_json = ModelWithArrays.model_validate_json(serialized_json)
    assert isinstance(deserialized_json, ModelWithArrays)

    # Compare original and deserialized models
    assert_models_equal(model, deserialized)
    assert_models_equal(model, deserialized_json)


def test_validation_errors():
    with pytest.raises(ValidationError):
        ModelWithArrays(
            any_array=np.array([1, 2, 3, 4]),  # This is fine
            flexible_array=np.array([1, 2, 3, 4]),  # This is fine
            int_vector=np.array([1, 2]),  # Wrong shape
            float_matrix=np.array([[1, 2], [3, 4]]),  # Wrong dtype
            any_3d_array=np.zeros((2, 3)),  # Wrong number of dimensions
            any_float_array=np.array([1, 2, 3]),  # Wrong dtype
        )


def test_edge_cases():
    # Test with empty arrays
    model = ModelWithArrays(
        any_array=np.array([]),
        flexible_array=np.array([[]]),
        int_vector=np.array([0, 0, 0]),
        float_matrix=np.array([[0.0, 0.0], [0.0, 0.0]]),
        any_3d_array=np.array([[[]]]),
        any_float_array=np.array([]),
        array=np.array([]),
    )
    assert model.any_array.size == 0
    assert model.flexible_array.size == 0
    assert np.all(model.int_vector == 0)
    assert np.all(model.float_matrix == 0.0)
    assert model.any_3d_array.size == 0
    assert model.any_float_array.size == 0

    # Test with extreme values
    model = ModelWithArrays(
        any_array=np.array([np.inf, -np.inf, np.nan], dtype=object),
        flexible_array=np.array([[np.finfo(np.float64).max, np.finfo(np.float64).min]]),
        int_vector=np.array([np.iinfo(np.int64).max, 0, np.iinfo(np.int64).min]),
        float_matrix=np.array([[np.inf, -np.inf], [np.nan, 0.0]]),
        any_3d_array=np.array([[[np.inf, -np.inf, np.nan]]]),
        any_float_array=np.array([np.finfo(np.float64).max, np.finfo(np.float64).min]),
    )
    assert np.any(np.isinf(model.any_array.astype(float)))
    assert np.any(np.isnan(model.any_array.astype(float)))
    assert np.all(
        model.int_vector
        == np.array([np.iinfo(np.int64).max, 0, np.iinfo(np.int64).min])
    )
    assert np.isinf(model.float_matrix).any()
    assert np.isnan(model.float_matrix).any()


def test_type_conversion():
    # Test passing lists instead of numpy arrays
    model = ModelWithArrays(
        any_array=[1, 2, 3, 4],
        flexible_array=[[1, 2], [3, 4]],
        int_vector=[1, 2, 3],
        float_matrix=[[1.0, 2.0], [3.0, 4.0]],
        any_3d_array=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        any_float_array=[1.0, 2.0, 3.0],
    )
    assert isinstance(model.any_array, np.ndarray)
    assert isinstance(model.flexible_array, np.ndarray)
    assert isinstance(model.int_vector, np.ndarray)
    assert isinstance(model.float_matrix, np.ndarray)
    assert isinstance(model.any_3d_array, np.ndarray)
    assert isinstance(model.any_float_array, np.ndarray)


def test_wrong_shape():
    class TestModelWithArrays(BaseModel):
        any_array: NumpyArray[Any]
        flexible_array: NumpyArray = NumpyArray[...]
        int_vector: NumpyArray[3, int]
        float_matrix: NumpyArray[2, 2, np.float64]
        any_3d_array: NumpyArray["*", "*", "*", Any]
        any_float_array: NumpyArray[float]
        array: NumpyArray = Field(default_factory=lambda: np.array([1.0, 2.0, 3.0]))

    with pytest.raises((ValidationError, TypeError)):
        model = TestModelWithArrays(
            any_array=[1, 2, 3, 4],
            flexible_array=[[1, 2], [3, 4]],
            int_vector=[1, 2, 3],
            float_matrix=[[1.0, 2.0], [3.0, 4.0]],
            any_3d_array=[
                [1, 2],
                [3, 4],
            ],  # This should raise an error as it's 2D, not 3D
            any_float_array=[1.0, 2.0, 3.0],
        )

    # Test that correct shapes pass validation
    model = TestModelWithArrays(
        any_array=[1, 2, 3, 4],
        flexible_array=[[1, 2], [3, 4]],
        int_vector=[1, 2, 3],
        float_matrix=[[1.0, 2.0], [3.0, 4.0]],
        any_3d_array=[[[1, 2], [3, 4]]],  # This is now 3D
        any_float_array=[1.0, 2.0, 3.0],
    )
    assert isinstance(model.any_3d_array, np.ndarray)
    assert model.any_3d_array.ndim == 3


def test_specific_validation_errors():
    with pytest.raises(ValidationError):
        model = ModelWithArrays(
            any_array=[1, 2, 3, 4],
            flexible_array=[[1, 2], [3, 4]],
            int_vector=[1, 2],  # Wrong shape
            float_matrix=[[1.0, 2.0], [3.0, 4.0]],
            any_3d_array=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            any_float_array=[1.0, 2.0, 3.0],
        )
