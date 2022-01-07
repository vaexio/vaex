import vaex
import numpy as np
import pyarrow as pa
from vaex.dataframe import DataFrameLocal


def test_shape_1d_columns():
    Nrows = 8
    x = np.arange(Nrows)
    df = vaex.from_arrays(x=x)
    assert df.shape == (Nrows, 1)
    assert df.x.shape == (Nrows,)
    df = vaex.from_arrays(x=x, y=x ** 2)
    assert df.shape == (Nrows, 2)


def test_shape_2d_columns():
    Nrows = 8
    x = np.arange(Nrows * 3).reshape((Nrows, 3))
    df = vaex.from_arrays(x=x)
    assert df.shape == (Nrows, 1)
    assert df.x.shape == (Nrows, 3)


def test_shape_category():
    s = ["aap", "noot", "mies", "mies", "aap"]
    df: DataFrameLocal = vaex.from_arrays(s=s)
    df = df.ordinal_encode('s')
    df = df._future()
    assert df.shape == (len(s), 1)
    assert df.s.shape == (len(s),)
