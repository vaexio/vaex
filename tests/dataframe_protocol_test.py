import numpy as np
import pyarrow as pa
import pytest
from typing import Any, Optional, Tuple, Dict, Iterable, Sequence

DataFrameObject = Any
ColumnObject = Any

import vaex
from common import *
from vaex.dataframe_protocol import _from_dataframe_to_vaex, _DtypeKind, _VaexBuffer, _VaexColumn, _VaexDataFrame


xfail_memory_bug = pytest.mark.xfail(
    reason=(
        "Erroneous due to bug where memory is released prematurely - "
        "see https://github.com/vaexio/vaex/pull/2150#issuecomment-1263336551"
    )
)


def test_float_only(df_factory):
    df = df_factory(x=[1.5, 2.5, 3.5], y=[9.2, 10.5, 11.8])
    df2 = _from_dataframe_to_vaex(df.__dataframe__())

    assert df2.x.tolist() == df.x.tolist()
    assert df2.y.tolist() == df.y.tolist()
    assert df2.__dataframe__().get_column_by_name("x").null_count == 0
    assert df2.__dataframe__().get_column_by_name("y").null_count == 0

    assert_dataframe_equal(df.__dataframe__(), df)


def test_mixed_intfloat(df_factory):
    df = df_factory(x=[1, 2, 0], y=[9.2, 10.5, 11.8])
    df2 = _from_dataframe_to_vaex(df.__dataframe__())

    assert df2.x.tolist() == df.x.tolist()
    assert df2.y.tolist() == df.y.tolist()
    assert df2.__dataframe__().get_column_by_name("x").null_count == 0
    assert df2.__dataframe__().get_column_by_name("y").null_count == 0

    assert_dataframe_equal(df.__dataframe__(), df)


def test_mixed_intfloatbool(df_factory):
    df = df_factory(x=np.array([True, True, False]), y=np.array([1, 2, 0]), z=np.array([9.2, 10.5, 11.8]))
    df2 = _from_dataframe_to_vaex(df.__dataframe__())

    assert df2.x.tolist() == df.x.tolist()
    assert df2.y.tolist() == df.y.tolist()
    assert df2.z.tolist() == df.z.tolist()
    assert df2.__dataframe__().get_column_by_name("x").null_count == 0
    assert df2.__dataframe__().get_column_by_name("y").null_count == 0
    assert df2.__dataframe__().get_column_by_name("z").null_count == 0

    # Additionl tests for _VaexColumn
    assert df2.__dataframe__().get_column_by_name("x")._allow_copy == True
    assert df2.__dataframe__().get_column_by_name("x").size() == 3
    assert df2.__dataframe__().get_column_by_name("x").offset == 0

    assert df2.__dataframe__().get_column_by_name("z").dtype[0] == 2  # 2: float64
    assert df2.__dataframe__().get_column_by_name("z").dtype[1] == 64  # 64: float64
    assert df2.__dataframe__().get_column_by_name("z").dtype == (2, 64, "<f8", "=")

    with pytest.raises(TypeError):
        assert df2.__dataframe__().get_column_by_name("y").describe_categorical
    if df2['y'].dtype.is_arrow:
        assert df2.__dataframe__().get_column_by_name("y").describe_null == (3, 0)
    else:
        assert df2.__dataframe__().get_column_by_name("y").describe_null == (0, None)

    assert_dataframe_equal(df.__dataframe__(), df)


def test_mixed_missing(df_factory_arrow):
    df = df_factory_arrow(x=np.array([True, None, False, None, True]), y=np.array([None, 2, 0, 1, 2]), z=np.array([9.2, 10.5, None, 11.8, None]))

    df2 = _from_dataframe_to_vaex(df.__dataframe__())

    assert df.__dataframe__().metadata == df2.__dataframe__().metadata

    assert df["x"].tolist() == df2["x"].tolist()
    assert not df2["x"].is_masked
    assert df2.__dataframe__().get_column_by_name("x").null_count == 2
    assert df["x"].dtype == df2["x"].dtype

    assert df["y"].tolist() == df2["y"].tolist()
    assert not df2["y"].is_masked
    assert df2.__dataframe__().get_column_by_name("y").null_count == 1
    assert df["y"].dtype == df2["y"].dtype

    assert df["z"].tolist() == df2["z"].tolist()
    assert not df2["z"].is_masked
    assert df2.__dataframe__().get_column_by_name("z").null_count == 2
    assert df["z"].dtype == df2["z"].dtype

    assert_dataframe_equal(df.__dataframe__(), df)


def test_missing_from_masked(df_factory_numpy):
    df = df_factory_numpy(
        x=np.ma.array([1, 2, 3, 4, 0], mask=[0, 0, 0, 1, 1], dtype=int),
        y=np.ma.array([1.5, 2.5, 3.5, 4.5, 0], mask=[False, True, True, True, False], dtype=float),
        z=np.ma.array([True, False, True, True, True], mask=[1, 0, 0, 1, 0], dtype=bool),
    )

    df2 = _from_dataframe_to_vaex(df.__dataframe__())

    assert df.__dataframe__().metadata == df2.__dataframe__().metadata

    assert df["x"].tolist() == df2["x"].tolist()
    assert not df2["x"].is_masked
    assert df2.__dataframe__().get_column_by_name("x").null_count == 2
    assert df["x"].dtype == df2["x"].dtype

    assert df["y"].tolist() == df2["y"].tolist()
    assert not df2["y"].is_masked
    assert df2.__dataframe__().get_column_by_name("y").null_count == 3
    assert df["y"].dtype == df2["y"].dtype

    assert df["z"].tolist() == df2["z"].tolist()
    assert not df2["z"].is_masked
    assert df2.__dataframe__().get_column_by_name("z").null_count == 2
    assert df["z"].dtype == df2["z"].dtype

    assert_dataframe_equal(df.__dataframe__(), df)


@xfail_memory_bug
def test_categorical():
    df = vaex.from_arrays(year=[2012, 2013, 2015, 2019], weekday=[0, 1, 4, 6])
    df = df.categorize("year", min_value=2012, max_value=2019)
    df = df.categorize("weekday", labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

    # Some detailed testing for correctness of dtype and null handling:
    col = df.__dataframe__().get_column_by_name("year")
    assert col.dtype == (_DtypeKind.CATEGORICAL, 64, "u", "=")
    catinfo = col.describe_categorical
    assert not catinfo["is_ordered"]
    assert catinfo["is_dictionary"]
    assert catinfo["categories"]._col.tolist() == [
        2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019
    ]
    assert col.describe_null == (0, None)

    col2 = df.__dataframe__().get_column_by_name("weekday")
    catinfo2 = col2.describe_categorical
    assert not catinfo2["is_ordered"]
    assert catinfo2["is_dictionary"]
    assert catinfo2["categories"]._col.tolist() == [
        "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"
    ]
    assert col2.dtype == (_DtypeKind.CATEGORICAL, 64, "u", "=")
    assert col2.describe_null == (0, None)

    df2 = _from_dataframe_to_vaex(df.__dataframe__())
    assert df2["year"].tolist() == [2012, 2013, 2015, 2019]
    assert df2["weekday"].tolist() == ["Mon", "Tue", "Fri", "Sun"]

    assert_dataframe_equal(df.__dataframe__(), df)


@xfail_memory_bug
def test_arrow_dictionary():
    indices = pa.array([0, 1, 0, 1, 2, 0, 1, 2])
    dictionary = pa.array(["foo", "bar", "baz"])
    dict_array = pa.DictionaryArray.from_arrays(indices, dictionary)
    df = vaex.from_arrays(x=dict_array)

    # Some detailed testing for correctness of dtype and null handling:
    col = df.__dataframe__().get_column_by_name("x")
    assert col.dtype[0] == _DtypeKind.CATEGORICAL
    catinfo = col.describe_categorical
    assert not catinfo["is_ordered"]
    assert catinfo["is_dictionary"]
    assert catinfo["categories"]._col.tolist() == ["foo", "bar", "baz"]
    if df['x'].dtype.is_arrow:
        assert col.describe_null == (3, 0)
    else:
        assert col.describe_null == (0, None)
    assert col.dtype == (23, 64, "u", "=")

    df2 = _from_dataframe_to_vaex(df.__dataframe__())
    assert df2.x.tolist() == df.x.tolist()
    assert df2.__dataframe__().get_column_by_name("x").null_count == 0

    assert_dataframe_equal(df.__dataframe__(), df)


@xfail_memory_bug
def test_arrow_dictionary_missing():
    indices = pa.array([0, 1, 2, 0, 1], mask=np.array([0, 1, 1, 0, 0], dtype=bool))
    dictionary = pa.array(["aap", "noot", "mies"])
    dict_array = pa.DictionaryArray.from_arrays(indices, dictionary)
    df = vaex.from_arrays(x=dict_array)

    # Some detailed testing for correctness of dtype and null handling:
    col = df.__dataframe__().get_column_by_name("x")
    assert col.dtype[0] == _DtypeKind.CATEGORICAL
    catinfo = col.describe_categorical
    assert not catinfo["is_ordered"]
    assert catinfo["is_dictionary"]
    assert catinfo["categories"]._col.tolist() == ["aap", "noot", "mies"]

    df2 = _from_dataframe_to_vaex(df.__dataframe__())
    assert df2.x.tolist() == df.x.tolist()
    assert df2.__dataframe__().get_column_by_name("x").null_count == 2
    assert df["x"].dtype.index_type == df2["x"].dtype.index_type

    assert_dataframe_equal(df.__dataframe__(), df)


def test_string():
    df = vaex.from_dict({"A": ["a", None, "cdef", "", "g"]})
    col = df.__dataframe__().get_column_by_name("A")

    assert col._col.tolist() == df.A.tolist()
    assert col.size() == 5
    assert col.null_count == 1
    assert col.dtype[0] == _DtypeKind.STRING
    assert col.describe_null == (3,0)

    df2 = _from_dataframe_to_vaex(df.__dataframe__())
    assert df2.A.tolist() == df.A.tolist()
    assert df2.__dataframe__().get_column_by_name("A").null_count == 1
    assert df2.__dataframe__().get_column_by_name("A").describe_null == (3,0)
    assert df2.__dataframe__().get_column_by_name("A").dtype[0] == _DtypeKind.STRING

    df_sliced = df[1:]
    col = df_sliced.__dataframe__().get_column_by_name("A")
    assert col.size() == 4
    assert col.null_count == 1
    assert col.dtype[0] == _DtypeKind.STRING
    assert col.describe_null == (3,0)

    df2 = _from_dataframe_to_vaex(df_sliced.__dataframe__())
    assert df2.A.tolist() == df_sliced.A.tolist()
    assert df2.__dataframe__().get_column_by_name("A").null_count == 1
    assert df2.__dataframe__().get_column_by_name("A").describe_null == (3,0)
    assert df2.__dataframe__().get_column_by_name("A").dtype[0] == _DtypeKind.STRING


def test_no_mem_copy():
    strings = ["a", "", "cdef", "", "g"]
    # data for above string array
    dbuf = np.array([ 97,  99, 100, 101, 102, 103], dtype='uint8')
    obuf = np.array([0, 1, 1, 5, 5, 6], dtype='int64')
    length = 5
    buffers = [None, pa.py_buffer(obuf), pa.py_buffer(dbuf)]
    s = pa.Array.from_buffers(pa.large_utf8(), length, buffers)
    x = np.arange(0, 5)
    df = vaex.from_arrays(x=x, s=s)
    df2 = _from_dataframe_to_vaex(df.__dataframe__())

    # primitive data
    x[0] = 999
    assert df2.x.tolist() == [999, 1, 2, 3, 4]

    # strings
    assert df.s.tolist() == strings
    assert df2.s.tolist() == strings
    # mutate the buffer data (which actually arrow and vaex both don't support/want)
    strings[0] = "b"
    dbuf[0] += 1
    assert df.s.tolist() == strings
    assert df2.s.tolist() == strings


def test_object():
    df = vaex.from_arrays(x=np.array([None, True, False]))
    col = df.__dataframe__().get_column_by_name("x")

    assert col._col.tolist() == df.x.tolist()
    assert col.size() == 3

    with pytest.raises(ValueError):
        assert col.dtype
    with pytest.raises(ValueError):
        assert col.describe_null


def test_virtual_column():
    df = vaex.from_arrays(x=np.array([True, True, False]), y=np.array([1, 2, 0]), z=np.array([9.2, 10.5, 11.8]))
    df.add_virtual_column("r", "sqrt(y**2 + z**2)")
    df2 = _from_dataframe_to_vaex(df.__dataframe__())
    assert df2.r.tolist() == df.r.tolist()


def test_VaexBuffer():
    x = np.ndarray(shape=(5,), dtype=float, order="F")
    x_buffer = _VaexBuffer(x)

    assert x_buffer.bufsize == 5 * x.itemsize
    assert x_buffer.ptr == x.__array_interface__["data"][0]
    assert x_buffer.__dlpack_device__() == (1, None)
    assert x_buffer.__repr__() == f"VaexBuffer({{'bufsize': {5*x.itemsize}, 'ptr': {x.__array_interface__['data'][0]}, 'device': 'CPU'}})"

    with pytest.raises(NotImplementedError):
        assert x_buffer.__dlpack__()


def test_VaexDataFrame():
    df = vaex.from_arrays(x=np.array([True, True, False]), y=np.array([1, 2, 0]), z=np.array([9.2, 10.5, 11.8]))

    df2 = df.__dataframe__()

    assert df2._allow_copy == True
    assert df2.num_columns() == 3
    assert df2.num_rows() == 3
    assert df2.num_chunks() == 1

    assert df2.column_names() == ["x", "y", "z"]
    assert df2.get_column(0)._col.tolist() == df.x.tolist()
    assert df2.get_column_by_name("y")._col.tolist() == df.y.tolist()

    for col in df2.get_columns():
        assert col._col.tolist() == df[col._col.expression].tolist()

    assert df2.select_columns((0, 2))._df[:, 0].tolist() == df2.select_columns_by_name(("x", "z"))._df[:, 0].tolist()
    assert df2.select_columns((0, 2))._df[:, 1].tolist() == df2.select_columns_by_name(("x", "z"))._df[:, 1].tolist()

    assert_dataframe_equal(df2.__dataframe__(), df)


def test_chunks(df_factory):
    x = np.arange(10)
    df = df_factory(x=x)
    df2 = df.__dataframe__()
    chunk_iter = iter(df2.get_chunks(3))
    chunk = next(chunk_iter)
    assert chunk.num_rows() == 4
    chunk = next(chunk_iter)
    assert chunk.num_rows() == 4
    chunk = next(chunk_iter)
    assert chunk.num_rows() == 2
    with pytest.raises(StopIteration):
        chunk = next(chunk_iter)


def assert_buffer_equal(buffer_dtype: Tuple[_VaexBuffer, Any], vaexcol: vaex.expression.Expression):
    buf, dtype = buffer_dtype
    pytest.raises(NotImplementedError, buf.__dlpack__)
    assert buf.__dlpack_device__() == (1, None)
    assert dtype[1] == vaexcol.dtype.index_type.numpy.itemsize * 8
    if not isinstance(vaexcol.values, np.ndarray) and isinstance(vaexcol.values.type, pa.DictionaryType):
        assert dtype[2] == vaexcol.index_values().dtype.numpy.str
    else:
        assert dtype[2] == vaexcol.dtype.numpy.str


def assert_column_equal(col: _VaexColumn, vaexcol: vaex.expression.Expression):
    assert col.size() == vaexcol.df.count("*")
    assert col.offset == 0
    assert col.null_count == vaexcol.countmissing()
    assert_buffer_equal(col._get_data_buffer(), vaexcol)


def assert_dataframe_equal(dfo: DataFrameObject, df: vaex.dataframe.DataFrame):
    assert dfo.num_columns() == len(df.columns)
    assert dfo.num_rows() == len(df)
    assert dfo.column_names() == list(df.get_column_names())
    for col in df.get_column_names():
        assert_column_equal(dfo.get_column_by_name(col), df[col])


def test_smoke_get_buffers_for_numpy_column_with_duplicate_categorical_values():
    # See https://github.com/vaexio/vaex/issues/2122
    df = vaex.from_items(("x", np.array([1, 1])))
    df = df.categorize("x")
    interchange_df = df.__dataframe__()
    interchange_col = interchange_df.get_column_by_name("x")
    buffers = interchange_col.get_buffers()
    assert buffers['data'][0]._x.tolist() == [0, 0]


@pytest.mark.parametrize(
    "x",
    [
        np.array([float("nan")]),
        np.ma.MaskedArray(data=np.array([42]), mask=np.array([1])),
    ]
)
def test_null_count(df_factory, x):
    df = df_factory(x=x)
    interchange_df = df.__dataframe__()
    interchange_col = interchange_df.get_column_by_name("x")
    assert isinstance(interchange_col.null_count, int)
    assert interchange_col.null_count == 1


def test_size(df_factory):
    # See https://github.com/vaexio/vaex/issues/2093
    x = np.arange(5)
    df = df_factory(x=x)
    interchange_df = df.__dataframe__()
    interchange_col = interchange_df.get_column_by_name('x')
    size = interchange_col.size()
    assert isinstance(size, int)
    assert size == 5


def test_smoke_get_buffers_on_categorical_columns(df_factory):
    # See https://github.com/vaexio/vaex/issues/2134#issuecomment-1195731379
    x = np.array([3, 1, 1, 2, 0])
    df = df_factory(x=x)
    df = df.categorize('x')
    interchange_df = df.__dataframe__()
    interchange_col = interchange_df.get_column_by_name('x')
    interchange_col.get_buffers()


@pytest.mark.xfail()
def test_interchange_pandas_string_column():
    import pandas as pd
    data = ["foo", "bar"]
    try:
        from pandas.api.interchange import from_dataframe
    except ImportError:
        pytest.skip(f"pandas.api.interchange not found ({pd.__version__})")
    pd_df = pd.DataFrame({"x": pd.Series(data, dtype=pd.StringDtype())})
    pd_interchange_df = pd_df.__dataframe__()
    vaex_df = _from_dataframe_to_vaex(pd_interchange_df)
    assert vaex_df["x"].tolist() == data


def test_string_buffers(df_factory):
    data = ["foo", "bar"]
    x = np.array(data, dtype="U8")
    df = df_factory(x=x)
    if isinstance(df["x"].values, pa.lib.ChunkedArray):
        pytest.xfail()
    interchange_df = df.__dataframe__()
    roundtrip_df = _from_dataframe_to_vaex(interchange_df)
    assert roundtrip_df["x"].tolist() == data


@pytest.mark.parametrize(
    "labels", [[10, 11, 12, 13], ["foo", "bar", "baz", "qux"]]
)
def test_describe_categorical(df_factory, labels):
    # See https://github.com/vaexio/vaex/issues/2113
    data = [3, 1, 1, 2, 0]
    x = np.array(data)
    df = df_factory(x=x)
    df = df.categorize('x', labels=labels)
    interchange_df = df.__dataframe__()
    interchange_col = interchange_df.get_column_by_name('x')
    catinfo = interchange_col.describe_categorical
    assert isinstance(catinfo, dict)
    assert isinstance(catinfo["is_ordered"], bool)
    assert isinstance(catinfo["is_dictionary"], bool)
    assert catinfo["is_dictionary"]
    assert isinstance(catinfo["categories"], _VaexColumn)
    assert catinfo["categories"]._col.tolist() == labels


@xfail_memory_bug
@pytest.mark.parametrize(
    "labels", [[10, 11, 12, 13], ["foo", "bar", "baz", "qux"]]
)
def test_interchange_categorical_column(df_factory, labels):
    data = [3, 1, 1, 2, 0]
    x = np.array(data)
    df = df_factory(x=x)
    if isinstance(df["x"].values, pa.lib.ChunkedArray):
        pytest.xfail()
    df = df.categorize('x', labels=labels)
    interchange_df = df.__dataframe__()
    interchange_col = interchange_df.get_column_by_name('x')
    roundtrip_df = _from_dataframe_to_vaex(interchange_df)
    data_as_labels = [labels[i] for i in data]
    assert roundtrip_df["x"].values.tolist() == data_as_labels
    assert roundtrip_df.category_labels("x") == labels


@pytest.mark.parametrize("n_chunks", [None, 1])
def test_smoke_get_chunks(df_factory, n_chunks):
    if n_chunks is not None:
        pytest.xfail("get_chunks(n_chunks=...) doesn't work on already chunked columns")
    df = df_factory(x=[0])
    interchange_df = df.__dataframe__()
    interchange_col = interchange_df.get_column_by_name('x')
    if isinstance(df["x"].values, pa.lib.ChunkedArray):
        pytest.skip("get_chunks() is slow/halts with chunked arrow arrays")
    interchange_col.get_chunks(n_chunks=n_chunks)
