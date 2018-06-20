from common import *

def test_rename(ds_filtered):
    ds = ds_filtered
    ds['r'] = ds.x
    ds['q'] = np.exp(ds.x)
    qvalues = ds.q.values.tolist()
    xvalues = ds.x.tolist()
    qexpr = ds.q.expand().expression
    ds['x'] = ds.y # now r should still point to x
    import pdb
    # pdb.set_trace()
    assert ds.r.values.tolist() == xvalues
    assert ds.q.values.tolist() == qvalues

def test_rename_state_transfer():
    ds = vaex.from_scalars(x=3, y=4)
    ds['r'] = (ds.x**2 + ds.y**2)**0.5
    ds._rename('x', 'a')
    assert ds.r.tolist() == [5]
    ds2 = vaex.from_scalars(x=3, y=4)
    ds2.state_set(ds.state_get())
    assert ds2.r.tolist() == [5]
