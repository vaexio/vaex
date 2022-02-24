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


# these fingerprints may change over time, they may change as we change versions
# but they should at least not change per Python version, OS or after restarts

def test_dataset_arrays():
    x = np.arange(10, dtype='i4')
    y = x**2
    ds = vaex.dataset.DatasetArrays(x=x, y=y, z=x+y)
    assert dict(ds._ids) == {
        'x': '031385dd4f0d2ba1aba2aeab0ad7c99814c90c11e96e5bc7cc8bd72112556dff',
        'y': '4d48c88e587db8f3855eed9f5d5f51eea769451b7371ecf7bdee4e0258238631',
        'z': 'a4cead13bef1fd1ec5974d1a2f5ceffd243a7aa6c6b08b80e09a7454b7d04293'
    }
    assert ds.fingerprint == 'dataset-arrays-hashed-88244cf38fe91c6bf435caa6160b089b'


df_fingerprint_xy = 'dataframe-943761acaa2ff2060d21ef519c77e1b9'


def test_df():
    x = np.arange(10, dtype='i4')
    y = x**2
    df = vaex.from_arrays(x=x, y=y, z=x+y)
    assert df.fingerprint() == df_fingerprint_xy


def test_df_different_virtual_columns():
    x = np.arange(10, dtype='i4')
    y = x**2
    df1 = vaex.from_arrays(x=x, y=y, z=x+y)
    df1['z'] = df1.x + df1.z
    assert df1.fingerprint() == 'dataframe-8f2202e2b4e7845c8ace767db5a49bc4'
    df2 = vaex.from_arrays(x=x, y=y, z=x+y)
    df2['z'] = df2.x - df2.z
    assert df2.fingerprint() == 'dataframe-81043a3c5b32eaa4b18bf4a915492e23'


def test_df_with_dependencies():
    x = np.arange(10, dtype='i4')
    y = x**2
    df = vaex.from_arrays(x=x, y=y, z=x+y)
    df['q'] = df.x + df.y
    assert df.fingerprint() != df_fingerprint_xy
    assert df.fingerprint(dependencies={'x', 'y', 'z'}) == df_fingerprint_xy


def test_df_project():
    x = np.arange(10, dtype='i4')
    y = x**2
    df = vaex.from_arrays(x=x, y=y, z1=x+y, z2=x-y)
    # projecting 2 columns will drop 2 columns, which could be done in different order
    df_a = df[['x', 'y']]
    df_b = df[['x', 'y']]
    assert df_a.fingerprint() == df_b.fingerprint()
    assert df_a.fingerprint() == 'dataframe-c13a4ab588272f03855ae5627731f7e5'


def test_df_selection_references_virtual_column():
    x = np.arange(10, dtype='i4')
    y = x**2
    df = vaex.from_arrays(x=x, y=y)
    df1 = df.copy()
    df1['z'] = df1.x + df1.y
    df2 = df.copy()
    df2['z'] = df1.x - df1.y
    df1f = df1[df1.z < 5]
    df2f = df2[df2.z < 5]
    fp1 = df1f.fingerprint(dependencies=['x', 'y'])
    fp2 = df2f.fingerprint(dependencies=['x', 'y'])
    assert fp1 != fp2


def test_set():
    x = np.arange(10)
    df = vaex.from_arrays(x=x)
    assert df._set('x').fingerprint == df._set('x').fingerprint
    df2 = vaex.from_arrays(x=x)
    assert df._set('x').fingerprint == df2._set('x').fingerprint
    df3 = vaex.from_arrays(x=x, y=x**2)
    assert df.dataset.fingerprint != df3.dataset.fingerprint
    assert df._set('x').fingerprint != df3._set('x').fingerprint
    df4 = vaex.from_arrays(x=x)
    df4['y'] = df.x+1
    assert df._set('x').fingerprint == df4._set('x').fingerprint
