import pytest

import numpy as np
import vaex


@pytest.mark.parametrize("use_hashmap", [False, True])
def test_isin(use_hashmap):
    x = np.array([1.01, 2.02, 3.03])
    y = np.array([1, 3, 5])
    s = np.array(['dog', 'cat', 'mouse'])
    sm = np.array(['dog', 'cat', None])
    w = np.array([2, '1.1', None])
    m = np.ma.MaskedArray(data=[np.nan, 1, 1], mask=[True, True, False])
    n = np.array([-5, np.nan, 1])
    df = vaex.from_arrays(x=x, y=y, s=s, sm=sm, w=w, m=m, n=n)

    assert df.x.isin([1, 2.02, 5, 6], use_hashmap=use_hashmap).tolist() == [False, True, False]
    assert df.y.isin([5, -1, 0], use_hashmap=use_hashmap).tolist() == [False, False, True]
    assert df.s.isin(['elephant', 'dog'], use_hashmap=use_hashmap).tolist() == [True, False, False]
    assert df.sm.isin(['cat', 'dog'], use_hashmap=use_hashmap).tolist() == [True, True, False]
    assert df.w.isin([2, None], use_hashmap=use_hashmap).tolist() == [True, False, True]
    assert df.m.isin([1, 2, 3], use_hashmap=use_hashmap).tolist() == [False, False, True]
    assert df.n.isin([2, np.nan], use_hashmap=use_hashmap).tolist() == [False, True, False]


def test_isin_object():
    df = vaex.from_arrays(x=np.array(['a', 'b', 'c'], dtype='O'),
                          y=np.array([1, 2, 3], dtype='O'))

    expr_x = df.x.isin(['a'])
    expr_y = df.y.isin([2])

    assert expr_x.tolist() == [True, False, False]
    assert expr_y.tolist() == [False, True, False]


@pytest.mark.parametrize("use_hashmap", [False, True])
def test_isin_diff_dtypes(use_hashmap):
    x = np.array([1.01, 2.02, 3.03])
    s = np.array(['dog', 'cat', 'mouse'])
    sm = np.array(['dog', 'cat', None])
    df = vaex.from_arrays(x=x, s=s, sm=sm)

    assert df.x.isin([1], use_hashmap=use_hashmap).tolist() == [False, False, False]
    assert df.x.isin([1.01], use_hashmap=use_hashmap).tolist() == [True, False, False]


@pytest.mark.parametrize("use_hashmap", [False, True])
@pytest.mark.parametrize("encoded", [False, True])
def test_isin_test_non_existing(use_hashmap, encoded):
    df = vaex.from_arrays(s=['dog', 'cat', 'mouse'])
    if encoded:
        df = df.ordinal_encode('s')._future()
    assert df.s.isin(['ape'], use_hashmap=use_hashmap).tolist() == [False, False, False]


@pytest.mark.parametrize("use_hashmap", [False, True])
@pytest.mark.parametrize("encoded", [False, True])
def test_isin_test_empty(use_hashmap, encoded):
    df = vaex.from_arrays(s=['dog', 'cat', 'mouse'])
    if encoded:
        df = df.ordinal_encode('s')._future()
    assert df.s.isin([], use_hashmap=use_hashmap).tolist() == [False, False, False]
