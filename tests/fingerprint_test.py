import vaex
import numpy as np
import pyarrow as pa


def test_dataframe(df_factory):
    df1 = df_factory(x=[1, 2], y=[4, 5])
    df1b = df_factory(x=[1, 2], y=[4, 5])
    df2 = df_factory(x=[1, 3], y=[4, 5])

    assert df1.fingerprint() == df1b.fingerprint()
    assert df1.fingerprint() != df2.fingerprint()

    assert df1.fingerprint() == df1b.fingerprint()
    df1.add_variable('q', 1)  # this changes the state
    assert df1.fingerprint() != df1b.fingerprint()
    # but if we treeshake, it does not
    assert df1.fingerprint(treeshake=True) != df1b.fingerprint()


def test_groupby(df_factory):
    df1 = df_factory(x=[1, 2], y=[4, 5])
    df2 = df_factory(x=[1, 2], y=[4, 5])

    df1g = df1.groupby('x', agg='count', sort=True)
    df2g = df2.groupby('x', agg='count', sort=True)

    assert df1g.fingerprint() == df2g.fingerprint()


def test_expression(df_factory):
    df1 = df_factory(x=[1, 2], y=[4, 5])
    df1b = df_factory(x=[1, 2], y=[4, 5])
    df2 = df_factory(x=[1, 3], y=[4, 5])
    df1['z'] = 'x + y'
    df1b['z'] = 'x + y'
    df2['z'] = 'x + y'
    assert df1.x.fingerprint() == df1b.x.fingerprint()
    assert df1.y.fingerprint() == df1b.y.fingerprint()
    assert df1.z.fingerprint() == df1b.z.fingerprint()

    assert df1.z.fingerprint() != df2.z.fingerprint()


def test_column_file():
    path = vaex.example().dataset.path
    df = vaex.open(path, nommap=True)
    x = df.dataset._columns['x']
    assert isinstance(x, vaex.file.column.ColumnFile)
    df = vaex.from_arrays(x=x)  # will trigger fingerprint
    x.fingerprint()  # just to be sure


def test_column_numpy_like():
    x = np.arange(5)
    x1 = vaex.column.ColumnNumpyLike(x)
    x2 = vaex.column.ColumnNumpyLike(x)
    x3 = vaex.column.ColumnNumpyLike(x**2)
    assert x1.fingerprint() == x2.fingerprint()
    assert x1.fingerprint() != x3.fingerprint()


def test_column_arrow_cast():
    x = np.arange(5)
    x1 = vaex.column.ColumnArrowLazyCast(x, pa.float32())
    x2 = vaex.column.ColumnArrowLazyCast(x, pa.float32())
    x3 = vaex.column.ColumnArrowLazyCast(x**2, pa.float32())
    assert x1.fingerprint() == x2.fingerprint()
    assert x1.fingerprint() != x3.fingerprint()


def test_column_indexed():
    x = np.arange(5)
    i = np.array([0, 2, 3])
    x1 = vaex.column.ColumnIndexed(x, i)
    x2 = vaex.column.ColumnIndexed(x, i)
    x3 = vaex.column.ColumnIndexed(x**2, i)
    assert x1.fingerprint() == x2.fingerprint()
    assert x1.fingerprint() != x3.fingerprint()
