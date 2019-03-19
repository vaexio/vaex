import pytest
import numpy as np
from common import *

def test_value_counts():
    ds = create_base_ds()

    v_counts = ds.x.value_counts()
    assert len(v_counts) == 21

    v_counts = ds.y.value_counts()
    assert len(v_counts) == 19

    v_counts = ds.m.value_counts()
    assert len(v_counts) == 20

    v_counts = ds.n.value_counts()
    assert len(v_counts) == 20

    v_counts = ds.nm.value_counts()
    assert len(v_counts) == 19

    v_counts = ds.mi.value_counts()
    assert len(v_counts) == 20

    v_counts_name = ds['name'].value_counts()
    v_counts_name_arrow = ds.name_arrow.value_counts()
    assert np.all(v_counts_name == v_counts_name_arrow)


@pytest.mark.xfail(reason="numpy's unique does not handle mixed str and float")
def test_value_counts_object():
    ds = create_base_ds()
    v_counts = ds.obj.value_counts()
    assert len(v_counts) == 19


@pytest.mark.parametrize("dropna", [True, False])
def test_value_counts_with_pandas(ds_local, dropna):
    ds = ds_local
    df = ds.to_pandas_df()
    assert df.x.value_counts(dropna=dropna).values.tolist() == ds.x.value_counts(dropna=dropna).values.tolist()

@pytest.mark.xfail(reason="our unique does not handle masked values yet")
def test_value_counts_simple():
    x = np.array([0, 1, 1, 2, 2, 2, np.nan])
    y = np.ma.array(x, mask=[True, True, False, False, False, False, False])
    s = np.array(list(map(str, x)))
    # print(s)
    ds = vaex.from_arrays(x=x, y=y, s=s)
    df = ds.to_pandas_df()

    assert ds.x.value_counts(dropna=True, ascending=True).values.tolist() == [1, 2, 3]
    assert ds.x.value_counts(dropna=False, ascending=True).values.tolist() == [1, 1, 2, 3]

    # print(ds.s.value_counts(dropna=True, ascending=True))
    assert ds.s.value_counts(dropna=True, ascending=True).index.tolist() == ['0.0', 'nan', '1.0', '2.0']
    assert ds.s.value_counts(dropna=True, ascending=True).values.tolist() == [1, 1.0, 2, 3]

    # this doesn't handle masked values correctly
    assert ds.y.value_counts(dropna=True, ascending=True).index.tolist() == [1, 2]
    assert ds.y.value_counts(dropna=True, ascending=True).values.tolist() == [1, 3]
    # nan comparison with == never works
    # assert ds.y.value_counts(dropna=False, ascending=True).index.tolist() == [1, np.nan, None, 2]
    assert ds.y.value_counts(dropna=False, ascending=True).values.tolist() == [1, 1, 2, 3]
    # assert ds.y.value_counts(dropna=False, ascending=True).index.tolist() == ['2', 'missing', '1']

    assert df.x.value_counts(dropna=False).values.tolist() == ds.x.value_counts(dropna=False).values.tolist()
    assert df.x.value_counts(dropna=True).values.tolist() == ds.x.value_counts(dropna=True).values.tolist()

    # do we want the index to be the same?
    # assert df.y.value_counts(dropna=False).index.tolist() == ds.y.value_counts(dropna=False).index.tolist()
    # assert df.y.value_counts(dropna=False).values.tolist() == ds.y.value_counts(dropna=False).values.tolist()
    # assert df.y.value_counts(dropna=True).values.tolist() == ds.y.value_counts(dropna=True).values.tolist()