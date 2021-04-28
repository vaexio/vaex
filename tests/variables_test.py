import numpy as np


def test_variables(df):
    df.add_variable('a', 2)
    df['w'] = df['x'] + df['a']
    assert (df.x + 2).tolist() == df.w.tolist()
    assert (df.x + 2).tolist() == df[['w']].w.tolist()


def test_variable_rename(df):
    df.add_variable('a', 2)
    df['w'] = df['x'] + df['a']
    assert (df.x + 2).tolist() == df.w.tolist()
    df.rename('a', 'a2')
    # arrow df has an extra variable, since it does the virtual as_numpy
    assert set(df.w.variables()) in [{'x', 'a2'} , {'__x', 'x', 'a2'}]
    assert (df.x + 2).tolist() == df.w.tolist()


def test_numpy_array_argument(df):
    xx = -np.arange(10)
    assert len(df.variables) == 1
    # this should insert a variable (the numpy array)
    df['w'] = df.func.where(df.x > 5, xx, df.x)
    assert len(df.variables) == 2
    assert list(df.variables.values())[1] is xx
