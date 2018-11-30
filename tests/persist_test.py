from common import *

def test_persist(ds_local):
    ds = ds_local
    ds = ds.trim().extract()
    ds['v'] = ds.x + ds.y
    assert 'v' not in ds.columns
    assert 'v' in ds.virtual_columns
    ds.persist(ds.v, overwrite=True, basepath='local.arrow')
    assert 'v' not in ds.virtual_columns
    assert 'v' in ds.columns
