import numbers
import pytest

import numpy as np
import pyarrow as pa

import vaex


@pytest.fixture(scope='session')
def x(array_factory1):
    return array_factory1([0, 1, 2., None])


@pytest.fixture(scope='session')
def y(array_factory2):
    return array_factory2([1, 2, None, None])


def test_add(x, y):
    df = vaex.from_arrays(x=x, y=y)
    df['z'] = df.x + df.y
    assert df['z'].tolist() == [1, 3, None, None]
    assert isinstance(df['z'].values.tolist()[0], numbers.Number)
    assert df['z'].values.tolist() == [1, 3, None, None]
    assert df['z'].dtype == 'numeric'
    # import pdb; pdb.set_trace()


def test_stay_same_type(x):
    df = vaex.from_arrays(x=x)
    assert df.x.tolist()[0] == 0
    assert isinstance(df.x.values, type(x))

    assert (df.x*2).tolist() == [0, 2, 4, None]
    assert isinstance((df.x*2).values, type(x))
    assert (df.x*2).tolist()[0] == 0

    assert (-df.x).tolist() == [0, -1, -2, None]
    assert isinstance((-df.x).values, type(x))

    # should trigger reverse add
    assert (1-df.x).tolist() == [1, 0, -1, None]
    assert isinstance((1-df.x).values, type(x))

    assert (df.x.sin()).tolist()[0] == 0
    assert (df.x.sin()).tolist()[-1] == None
    assert isinstance(df.x.sin().values, type(x))

# def test_values2():
#     df = vaex.from_arrays(x=, y=y)
#     df = vaex.from_scalars(x=0, y=1)
#     assert isinstance(df.evaluate('cos(x)+y'), type(x))