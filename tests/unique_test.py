from common import small_buffer

import numpy as np

import vaex


def test_unique_arrow(df_factory):
    ds = df_factory(x=vaex.string_column(['a', 'b', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'a']))
    with small_buffer(ds, 2):
        assert set(ds.unique(ds.x).to_pylist()) == {'a', 'b'}
        values, index = ds.unique(ds.x, return_inverse=True)
        assert np.array(values)[index].tolist() == ds.x.tolist()


def test_unique(df_factory):
    ds = df_factory(colors=['red', 'green', 'blue', 'green'])
    with small_buffer(ds, 2):
        assert set(ds.unique(ds.colors).to_pylist()) == {'red', 'green', 'blue'}
        values, index = ds.unique(ds.colors, return_inverse=True)
        assert np.array(values)[index].tolist() == ds.colors.tolist()

    ds = df_factory(x=['a', 'b', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'a'])
    with small_buffer(ds, 2):
        assert set(ds.unique(ds.x).to_pylist()) == {'a', 'b'}
        values, index = ds.unique(ds.x, return_inverse=True)
        assert np.array(values)[index].tolist() == ds.x.tolist()


def test_unique_f4(df_factory):
    x = np.array([np.nan, 0, 1, np.nan, 2, np.nan], dtype='f4')
    df = df_factory(x=x)
    assert list(sorted(df.x.unique()))[1:] == [np.nan, 0, 1, 2][1:]


def test_unique_nan(df_factory):
    x = [np.nan, 0, 1, np.nan, 2, np.nan]
    df = df_factory(x=x)
    assert list(sorted(df.x.unique()))[1:] == [np.nan, 0, 1, 2][1:]
    with small_buffer(df, 2):
        values, indices = df.unique(df.x, return_inverse=True)
        values = values[indices]
        mask = np.isnan(values)
        assert values[~mask].tolist() == df.x.to_numpy()[~mask].tolist()
        # assert indices.tolist() == [0, 1, 2, 0, 3, 0]


def test_unique_missing(df_factory):
    # Create test databn
    x = np.array([None, 'A', 'B', -1, 0, 2, '', '', None, None, None, np.nan, np.nan, np.nan, np.nan])
    df = df_factory(x=x)
    uniques = df.x.unique(dropnan=True).tolist()
    assert set(uniques) == set(['', 'A', 'B', -1, 0, 2, None])


def test_unique_missing_numeric(array_factory):
    df = vaex.from_arrays(x=array_factory([1, None]))
    values = df.x.unique()
    assert set(values.tolist()) == {1, None}
    # assert list(sorted(df.x.unique()))[1:] == [np.nan, 0, 1, 2][1:]


def test_unique_string_missing(df_factory):
    x = ['John', None, 'Sally', None, '0.0']
    df = df_factory(x=x)
    result = df.x.unique().to_pylist()

    assert len(result) == 4
    assert'John' in result
    assert None in result
    assert 'Sally'
