from common import *

def test_materialize(ds):
    print(ds)
    ds['r'] = np.sqrt(ds.x**2 + ds.y**2)
    assert 'r' in ds.virtual_columns
    assert hasattr(ds, 'r')
    ds = ds.materialize(ds.r)
    assert 'r' not in ds.virtual_columns
    assert 'r' in ds.columns
    assert hasattr(ds, 'r')
    assert ds.r.evaluate().tolist() == np.sqrt(ds.x.evaluate()**2 + ds.y.evaluate()**2).tolist()

