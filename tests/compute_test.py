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


@pytest.fixture(scope='session')
def s(array_factory_arrow):
    return array_factory_arrow(['a', 'b', None, 'd'])


def test_add(x, y):
    df = vaex.from_arrays(x=x, y=y)
    df['z'] = df.x + df.y
    assert df['z'].tolist() == [1, 3, None, None]
    assert isinstance(df['z'].tolist()[0], numbers.Number)
    assert df['z'].tolist() == [1, 3, None, None]
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


def test_mix_string_and_numeric(x, s):
    df = vaex.from_arrays(x=x, s=s)
    # TODO: Note that this is a seperate bug, it ignored the missing value
    assert (df.s == 'a').tolist() == [True, False, False, False]
    assert (df.x == 1).tolist() == [False, True, False, None]
    assert ((df.s == 'a') | (df.x == 1)).tolist()[0] is True
    assert ((df.s == 'a') | (df.x == 1)).tolist() == [True, True, False, None]
    assert (('a' == df.s) | (df.x == 1)).tolist() == [True, True, False, None]
    assert ((df.x == 1) | (df.s == 'a')).tolist() == [True, True, False, None]


def test_where(s):
    df = vaex.from_arrays(s=s)
    expr = df.func.where(df['s'] == 'a', 'A', df['s'])
    assert expr.tolist() == ['A', 'b', None, 'd']
    assert expr.dtype.is_string


def test_where_large():
    df = vaex.from_arrays(s=pa.array(['a', 'b', None, 'd'], type=pa.large_string()))
    assert (df['s'] + df['s']).dtype.internal == pa.large_string()
    expr = df.func.where(df['s'] == 'a', 'A', df['s'])
    assert expr.tolist() == ['A', 'b', None, 'd']
    assert expr.dtype.is_string

def test_where_str_str(x):
    df = vaex.from_arrays(x=x)
    df['s'] = (df.x==1).where('a', 'b')
    assert df.s.tolist() == ['b', 'a', 'b', None]
    assert df.s.dtype.is_string


def test_where_str_array(x, s):
    df = vaex.from_arrays(x=x, s=s)
    df['s2'] = (df.x==1).where('c', df.s)
    assert df.s2.tolist() == ['a', 'c', None, None]
    assert df.s2.dtype.is_string


def test_where_array_str(x, s):
    df = vaex.from_arrays(x=x, s=s)
    df['s2'] = (df.x==1).where(df.s, 'c')
    assert df.s2.tolist() == ['c', 'b', 'c', None]
    assert df.s2.dtype.is_string


def test_where_array_array(x, s):
    df = vaex.from_arrays(x=x, s=s)
    df['s2'] = (df.x==1).where(df.s, df.s + df.s)
    assert df.s2.tolist() == ['aa', 'b', None, None]
    assert df.s2.dtype.is_string


def test_where_primitive_masked_argument(array_factory1, array_factory2):
    x = array_factory1([1, 2, 3, 4])
    m = array_factory2([1, 2, None, 4])
    df = vaex.from_arrays(x=x, m=m)
    df['m2'] = df.func.where(df.x > 2, -10, df.m)
    assert df.m2.tolist() == [1, 2, -10, -10]

    df['m3'] = df.func.where(df.x > 2, df.m, -10)
    assert df.m3.tolist() == [-10, -10, None, 4]


def test_where_primitive_masked_condition(array_factory1, array_factory2):
    x = array_factory1([10, None, 30, 40])
    m = array_factory2([1, 2, None, 4])
    df = vaex.from_arrays(x=x, m=m)
    df['m2'] = df.func.where(df.x > 20, -10, df.m)
    assert df.m2.tolist() == [1, None, -10, -10]

    df['m3'] = df.func.where(df.x > 20, df.m, -10)
    assert df.m3.tolist() == [-10, None, None, 4]


def test_where_with_none(array_factory1, array_factory2):
    x = array_factory1([1, None, 3, 4])
    y = array_factory2([10, 20, 30, 40])
    df = vaex.from_arrays(x=x, y=y)
    df['y2'] = df.func.where(df.y > 20, None, df.x)
    assert df.y2.dtype == df.x.dtype
    assert df.y2.tolist() == [1, None, None, None]

    df['y3'] = df.func.where(df.y > 20, df.x, None)
    assert df.y3.dtype == df.x.dtype
    assert df.y3.tolist() == [None, None, 3, 4]
