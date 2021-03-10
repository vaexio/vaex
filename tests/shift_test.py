import pytest

import numpy as np
import pyarrow as pa


@pytest.mark.parametrize("virtual", [False, True])
def test_shift_basics(df_factory, virtual):
    x = [0, 1, 2, None, 4]
    y = [0, 1, None, 9, 16]
    df = df_factory(x=x, y=y)
    if virtual:
        df['x'] = df.x + 0
    dfp1 = df._shift(1, ['x'])
    dfn1 = df._shift(-1, ['x'])
    assert dfp1.x.tolist() == [None, 0, 1, 2, None]
    assert dfp1.y.tolist() == [0, 1, None, 9, 16]
    assert dfn1.x.tolist() == [1, 2, None, 4, None]
    assert dfn1.y.tolist() == [0, 1, None, 9, 16]

    assert dfp1._shift(1).x.tolist() == [None, None, 0, 1, 2]
    assert dfp1._shift(-1).x.tolist() == [0, 1, 2, None, None]
    assert dfp1._shift(-1, fill_value=99).x.tolist() == [0, 1, 2, None, 99]

    assert dfn1._shift(1).x.tolist() == [None, 1, 2, None, 4]
    assert dfn1._shift(-1).x.tolist() == [2, None, 4, None, None]
    assert dfn1._shift(-1, fill_value=99).x.tolist() == [2, None, 4, None, 99]

    assert df._shift(4).x.tolist() == [None, None, None, None, 0]
    assert df._shift(5).x.tolist() == [None, None, None, None, None]
    assert df._shift(6).x.tolist() == [None, None, None, None, None]

    assert df._shift(-4).x.tolist() == [4, None, None, None, None]
    assert df._shift(-5).x.tolist() == [None, None, None, None, None]
    assert df._shift(-6).x.tolist() == [None, None, None, None, None]


def test_shift_filtered(df_factory):
    x = [0, 99, 1, 99, 2, 99, None, 99, 4, 99]
    y = [0, 88, 1, 88, None, 88, 9, 88, 16, 88]
    assert len(x) == len(y)
    df = df0 = df_factory(x=x, y=y)
    df = df[((df.x != 99) | df.x.ismissing()).fillna(True)]
    dfp1 = df._shift(1, ['x'])
    dfn1 = df._shift(-1, ['x'])
    assert dfp1.x.tolist() == [None, 0, 1, 2, None]
    assert dfp1.y.tolist() == [0, 1, None, 9, 16]
    assert dfn1.x.tolist() == [1, 2, None, 4, None]
    assert dfn1.y.tolist() == [0, 1, None, 9, 16]

    assert dfp1._shift(1).x.tolist() == [None, None, 0, 1, 2]
    assert dfp1._shift(-1).x.tolist() == [0, 1, 2, None, None]
    assert dfp1._shift(-1, fill_value=99).x.tolist() == [0, 1, 2, None, 99]

    assert dfn1._shift(1).x.tolist() == [None, 1, 2, None, 4]
    assert dfn1._shift(-1).x.tolist() == [2, None, 4, None, None]
    assert dfn1._shift(-1, fill_value=99).x.tolist() == [2, None, 4, None, 99]

    assert df._shift(4).x.tolist() == [None, None, None, None, 0]
    assert df._shift(5).x.tolist() == [None, None, None, None, None]
    assert df._shift(6).x.tolist() == [None, None, None, None, None]

    assert df._shift(-4).x.tolist() == [4, None, None, None, None]
    assert df._shift(-5).x.tolist() == [None, None, None, None, None]
    assert df._shift(-6).x.tolist() == [None, None, None, None, None]


def test_shift_string(df_factory):
    x = np.arange(4)
    s = pa.array(['aap', None, 'noot', 'mies'])
    df = df_factory(x=x, s=s)
    assert df._shift(1).s.tolist() == [None, 'aap', None, 'noot']
    assert df._shift(-1).s.tolist() == [None, 'noot', 'mies', None]
    assert df._shift(1, ['s'], fill_value='VAEX').s.tolist() == ['VAEX', 'aap', None, 'noot']
    assert df._shift(-1, ['s'], fill_value='VAEX').s.tolist() == [None, 'noot', 'mies', 'VAEX']


def test_shift_virtual(df_factory):
    x = [0, 1, 2, None, 4]
    y = [0, 1, None, 9, 16]
    xsp1 = [None, 0, 1, 2, None]
    xsn1 = [1, 2, None, 4, None]
    df = df_factory(x=x, y=y)

    # # a is a virtual column that depends on x, but we don't shift a
    df['a'] = df.x + 0
    df['b'] = df.a
    dfs = df._shift(1, ['x'])
    assert dfs.x.tolist() == xsp1
    assert dfs.a.tolist() == x
    assert dfs.y.tolist() == y
    dfs = df._shift(-1, ['x'])
    assert dfs.x.tolist() == xsn1
    assert dfs.a.tolist() == x
    assert dfs.y.tolist() == y

    # a is a virtual column that depends on x, we shift a, but we don't shift x
    # we expect, a: __x_shifted, x: __x
    df = df_factory(x=x, y=y)
    df['a'] = df.x + 0
    dfs = df._shift(1, ['a'])
    assert dfs.x.tolist() == x
    assert dfs.a.tolist() == xsp1
    assert dfs.y.tolist() == y
    dfs = df._shift(-1, ['a'])
    assert dfs.x.tolist() == x
    assert dfs.a.tolist() == xsn1
    assert dfs.y.tolist() == y

    # same, but now we also have a reference to a, which we also do not shift
    df = df_factory(x=x, y=y)
    df['a'] = df.x + 0
    df['b'] = df.a + 0
    dfs = df._shift(1, ['a'])
    assert dfs.x.tolist() == x
    assert dfs.a.tolist() == xsp1
    assert dfs.b.tolist() == x
    assert dfs.y.tolist() == y
    dfs = df._shift(-1, ['a'])
    assert dfs.x.tolist() == x
    assert dfs.a.tolist() == xsn1
    assert dfs.b.tolist() == x
    assert dfs.y.tolist() == y
