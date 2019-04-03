import pytest
import numpy as np
from common import *

def test_value_counts():
    ds = create_base_ds()

    assert len(ds.x.value_counts()) == 21
    assert len(ds.y.value_counts()) == 19

    assert len(ds.m.value_counts()) == 19
    assert len(ds.m.value_counts(dropnull=False)) == 20

    assert len(ds.n.value_counts(dropna=False)) == 20
    assert len(ds.n.value_counts(dropna=True)) == 19

    assert len(ds.nm.value_counts(dropna=True, dropnull=True)) == 21-4
    assert len(ds.nm.value_counts(dropna=True, dropnull=False)) == 21-3
    assert len(ds.nm.value_counts(dropna=False, dropnull=True)) == 21-3
    assert len(ds.nm.value_counts(dropna=False, dropnull=False)) == 21-2

    assert len(ds.mi.value_counts(dropnull=True)) == 21-2
    assert len(ds.mi.value_counts(dropnull=False)) == 21-1

    v_counts_name = ds['name'].value_counts()
    v_counts_name_arrow = ds.name_arrow.value_counts()
    assert np.all(v_counts_name == v_counts_name_arrow)


def test_value_counts_object():
    ds = create_base_ds()
    assert len(ds.obj.value_counts(dropnull=True)) == 17
    assert len(ds.obj.value_counts(dropnull=False)) == 18


@pytest.mark.parametrize("dropna", [True, False])
def test_value_counts_with_pandas(ds_local, dropna):
    ds = ds_local
    df = ds.to_pandas_df()
    assert df.x.value_counts(dropna=dropna).values.tolist() == ds.x.value_counts(dropna=dropna).values.tolist()

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
    assert set(ds.s.value_counts(dropna=True, ascending=True).index.tolist()) == {'0.0', 'nan', '1.0', '2.0'}
    assert set(ds.s.value_counts(dropna=True, ascending=True).values.tolist()) == {1, 1.0, 2, 3}

    assert set(ds.y.value_counts(dropna=True, ascending=True).index.tolist()) == {1, 2}
    assert set(ds.y.value_counts(dropna=True, ascending=True).values.tolist()) == {1, 3}
    # nan comparison with == never works
    # assert ds.y.value_counts(dropna=False, ascending=True).index.tolist() == [1, np.nan, None, 2]
    assert ds.y.value_counts(dropna=False, dropnull=True, ascending=True).values.tolist()  == [1, 1, 3]
    assert ds.y.value_counts(dropna=False, dropnull=False, ascending=True).values.tolist() == [2, 1, 1, 3]
    # assert ds.y.value_counts(dropna=False, ascending=True).index.tolist() == ['2', 'missing', '1']

    assert set(df.x.value_counts(dropna=False).values.tolist()) == set(ds.x.value_counts(dropna=False).values.tolist())
    assert set(df.x.value_counts(dropna=True).values.tolist()) == set(ds.x.value_counts(dropna=True).values.tolist())

    # do we want the index to be the same?
    # assert df.y.value_counts(dropna=False).index.tolist() == ds.y.value_counts(dropna=False).index.tolist()
    # assert df.y.value_counts(dropna=False).values.tolist() == ds.y.value_counts(dropna=False).values.tolist()
    # assert df.y.value_counts(dropna=True).values.tolist() == ds.y.value_counts(dropna=True).values.tolist()