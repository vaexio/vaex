"""This module contains tests regarding struct array support.

"""

import vaex
import pyarrow as pa
import pytest


@pytest.fixture
def df():
    arrays = [[1, 2, 3],
              ["a", "b", "c"],
              [4, 5, 6]]
    names = ["col1", "col2", "col3"]

    array = pa.StructArray.from_arrays(arrays=arrays, names=names)
    return vaex.from_arrays(array=array, integer=[8, 9, 10])


def test_struct_get_field(df):
    expr = df.array.struct.get_field("col1")
    assert expr.tolist() == [1, 2, 3]

    expr = df.array.struct.get_field("col2")
    assert expr.tolist() == ["a", "b", "c"]


def test_struct_get_field_getitem_notation(df):
    assert df.array[:, "col1"].tolist() == [1, 2, 3]
    assert df.array[:, "col2"].tolist() == ["a", "b", "c"]


def test_struct_get_field_invalid_field(df):
    with pytest.raises(ValueError):
        df.array.struct.get_field("doesNotExist").tolist()


def test_struct_get_field_invalid_dtype(df):
    """Ensure that struct function is only applied to correct dtype."""
    with pytest.raises(TypeError):
        df.integer.struct.get_field("col1").tolist()


def test_struct_project(df):
    expr = df.array.struct.project(["col3", "col1"])
    assert expr.tolist() == [{"col3": 4, "col1": 1},
                             {"col3": 5, "col1": 2},
                             {"col3": 6, "col1": 3}]


def test_struct_project_getitem_notation(df):
    assert df.array[:, ["col3", "col1"]].tolist() == [{"col3": 4, "col1": 1},
                                                      {"col3": 5, "col1": 2},
                                                      {"col3": 6, "col1": 3}]


def test_struct_project_invalid_field(df):
    with pytest.raises(ValueError):
        df.array.struct.project(["doesNotExist"]).tolist()


def test_struct_project_invalid_dtype(df):
    """Ensure that struct function is only applied to correct dtype."""
    with pytest.raises(TypeError):
        df.integer.struct.project(["col1"]).tolist()


def test_struct_field_names(df):
    assert df.array.struct.field_names == ["col1", "col2", "col3"]


def test_struct_field_names_invalid_dtypes(df):
    """Ensure that struct function is only applied to correct dtype."""
    with pytest.raises(TypeError):
        df.integer.struct.field_names


def test_struct_field_types(df):
    types = df.array.struct.field_types
    assert types["col1"] == vaex.datatype.DataType(pa.int64())
    assert types["col2"] == vaex.datatype.DataType(pa.string())
    assert types["col3"] == vaex.datatype.DataType(pa.int64())


def test_struct_field_types_invalid_dtypes(df):
    """Ensure that struct function is only applied to correct dtype."""
    with pytest.raises(TypeError):
        df.integer.struct.field_types


def test_struct_repr(df):
    """Ensure that `repr` works without failing and contains correct dtype information."""

    string = repr(df.array)

    assert "dtype: struct" in string
    assert "array" in string

def test_struct_correct_df_dtypes(df):
    """Ensure that `dtypes` works correctly on vaex dataframe containing a struct."""

    assert "array" in df.dtypes
    assert df.dtypes["array"].is_struct


def test_struct_correct_expression_dtype(df):
    """Ensure that `dtype` works correctly on vaex expression containing a struct."""

    assert df.array.dtype.is_struct
