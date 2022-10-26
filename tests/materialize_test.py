from common import *

def test_materialize_virtual(ds_local):
    ds = ds_local
    print(ds)
    ds['new_r'] = np.sqrt(ds.x**2 + ds.y**2)
    assert 'new_r' in ds.virtual_columns
    assert hasattr(ds, 'new_r')
    ds = ds.materialize(ds.new_r)
    assert 'new_r' not in ds.virtual_columns
    assert 'new_r' in ds.columns
    assert hasattr(ds, 'new_r')
    assert ds.new_r.evaluate().tolist() == np.sqrt(ds.x.to_numpy()**2 + ds.y.to_numpy()**2).tolist()

def test_materialize_dataset():
    df = vaex.from_scalars(x=1)
    df = df.materialize('x')
    assert df.dataset.names == ['x']

    df = vaex.from_scalars(x=1, __y=2)
    df = df.materialize()
    assert df.dataset.names == ['x', '__y']
