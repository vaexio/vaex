from common import *

def test_materialize(ds_local):
    ds = ds_local
    print(ds)
    ds['r'] = np.sqrt(ds.x**2 + ds.y**2)
    assert 'r' in ds.virtual_columns
    assert hasattr(ds, 'r')
    ds = ds.materialize(ds.r)
    assert 'r' not in ds.virtual_columns
    assert 'r' in ds.columns
    assert hasattr(ds, 'r')
    assert ds.r.evaluate().tolist() == np.sqrt(ds.x.to_numpy()**2 + ds.y.to_numpy()**2).tolist()

