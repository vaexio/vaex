from common import *

def test_trim(ds_local):
    ds = ds_local
    ds_trimmed = ds.trim()
    assert ds_trimmed.length_original() == ds_trimmed.length_unfiltered() >= 10
    assert ds_trimmed.get_active_range() == (0, ds_trimmed.length_original())# == (0, 10)
    assert ds_trimmed.evaluate('x').tolist() == np.arange(10.).tolist()

    # trimming with a non-zero start index
    ds.set_active_range(5, 10)
    ds_trimmed = ds.trim()
    assert ds_trimmed.length_original() == ds_trimmed.length_unfiltered() == 5
    assert ds_trimmed.get_active_range() == (0, ds_trimmed.length_original()) == (0, 5)
    assert ds_trimmed.evaluate('x').tolist() == np.arange(5, 10.).tolist()

    # trim on trimmed
    ds_trimmed.set_active_range(1, 4)
    ds_trimmed = ds_trimmed.trim()
    assert ds_trimmed.length_original() == ds_trimmed.length_unfiltered() == 3
    assert ds_trimmed.get_active_range() == (0, ds_trimmed.length_original()) == (0, 3)
    assert ds_trimmed.evaluate('x').tolist() == np.arange(6, 9.).tolist()
