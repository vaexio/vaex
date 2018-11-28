from common import *

def test_copy(ds_local):
    ds = ds_local
    ds['v'] = ds.x + 1

    dsc = ds.copy()
    assert 'x' in dsc.get_column_names()
    assert 'v' in dsc.get_column_names()
    assert 'v' in dsc.virtual_columns
    dsc.x.values