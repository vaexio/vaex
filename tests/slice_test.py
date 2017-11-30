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
