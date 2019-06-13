from common import *

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


def test_slice_beyond_end(df_local):
    df = df_local
    df2 = df[:100]
    assert df2.x.tolist() == df.x.tolist()
    assert len(df[:100]) == len(df)
