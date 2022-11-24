import pytest

import numpy as np
import vaex


@pytest.mark.parametrize("use_hashmap", [False, True])
def test_isin(df_factory, array_factory, use_hashmap):
    x = np.array([1.01, 2.02, 3.03])
    y = np.array([1, 3, 5])
    s = np.array(['dog', 'cat', 'mouse'])
    sm = np.array(['dog', 'cat', None])
    w = np.array([2, '1.1', None])
    m = np.ma.MaskedArray(data=[np.nan, 1, 1], mask=[True, True, False])
    n = np.array([-5, np.nan, 1])
    df = df_factory(x=x, y=y, s=s, sm=sm, w=w, m=m, n=n)

    isin_x = array_factory([1, 2.02, 5, 6])
    isin_y = array_factory([5, -1, 0])
    isin_s = array_factory(['elephant', 'dog'])
    isin_sm = array_factory(['cat', 'dog'])
    isin_w = array_factory([2, None])
    isin_m = array_factory([1, 2, 3])
    isin_n = array_factory([2, np.nan])

    assert df.x.isin(isin_x, use_hashmap=use_hashmap).tolist() == [False, True, False]
    assert df.y.isin(isin_y, use_hashmap=use_hashmap).tolist() == [False, False, True]
    assert df.s.isin(isin_s, use_hashmap=use_hashmap).tolist() == [True, False, False]
    assert df.sm.isin(isin_sm, use_hashmap=use_hashmap).tolist() == [True, True, False]
    assert df.w.isin(isin_w, use_hashmap=use_hashmap).tolist() == [True, False, True]
    assert df.m.isin(isin_m, use_hashmap=use_hashmap).tolist() == [False, False, True]
    assert df.n.isin(isin_n, use_hashmap=use_hashmap).tolist() == [False, True, False]


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


@pytest.mark.parametrize("use_hashmap", [False, True])
def test_isin_fingerprint(use_hashmap):
    df = vaex.from_arrays(s=["dog", "cat", "mouse"])
    df0 = df
    fp1 = df.fingerprint()
    assert df.s.isin(["cat"], use_hashmap=use_hashmap).tolist() == [False, True, False]
    fp2 = df.fingerprint()
    assert df.s.isin(["cat"], use_hashmap=use_hashmap).tolist() == [False, True, False]
    fp3 = df.fingerprint()
    # we needed to add a variable
    assert fp1 != fp2
    # we reuse a variable
    assert fp2 == fp3
