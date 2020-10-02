import pytest
from common import *


def test_slice_expression(df):
    assert df.x[:2].tolist() == df[:2].x.tolist()
    assert df.x[2:6].tolist() == df[2:6].x.tolist()
    assert df.x[-3:].tolist() == df[-3:].x.tolist()
    # we don't support non 1 steps
    # assert df.x[::-3].tolist() == df[::-3].x.tolist()


def test_slice_against_numpy(df):
    assert df.x[:2].tolist() == df.x.to_numpy()[:2].tolist()
    assert df.x[2:6].tolist() == df.x.to_numpy()[2:6].tolist()
    assert df.x[-3:].tolist() == df.x.to_numpy()[-3:].tolist()
    # we don't support non 1 steps
    # assert df.x[::-3].tolist() == df.x.values[::-3].tolist()


@pytest.mark.xfail(reason="Not supported yet")
def test_slice_filtered_remte(ds_remote):
    df = ds_remote
    dff = df[df.x > 0]
    dfs = dff[1:]
    assert dfs.x.values[0] == 2


def test_slice(ds_local):
    ds = ds_local
    ds_sliced = ds[:]
    assert ds_sliced.length_original() == ds_sliced.length_unfiltered() >= 10
    assert ds_sliced.get_active_range() == (0, ds_sliced.length_original())# == (0, 10)
    assert ds_sliced.x.tolist() == np.arange(10.).tolist()

    # trimming with a non-zero start index
    ds_sliced = ds[5:]
    assert ds_sliced.length_original() == ds_sliced.length_unfiltered() == 5
    assert ds_sliced.get_active_range() == (0, ds_sliced.length_original()) == (0, 5)
    assert ds_sliced.x.tolist() == np.arange(5, 10.).tolist()

    # slice on slice
    ds_sliced = ds_sliced[1:4]
    assert ds_sliced.length_original() == ds_sliced.length_unfiltered() == 3
    assert ds_sliced.get_active_range() == (0, ds_sliced.length_original()) == (0, 3)
    assert ds_sliced.x.tolist() == np.arange(6, 9.).tolist()


def test_head(ds_local):
    ds = ds_local
    df = ds.head(5)
    assert len(df) == 5


def test_tail(ds_local):
    ds = ds_local
    df = ds.tail(5)
    assert len(df) == 5


def test_head_with_selection():
    df = vaex.example()
    df.select(df.x > 0, name='test')
    df.head()


def test_slice_beyond_end(df):
    df2 = df[:100]
    assert df2.x.tolist() == df.x.tolist()
    assert len(df2) == len(df)


def test_slice_negative(df):
    df2 = df[:-1]
    assert df2.x.tolist() == df.x.to_numpy()[:-1].tolist()
    assert len(df2) == len(df)-1
