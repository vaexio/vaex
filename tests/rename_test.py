from common import *

def test_rename(ds_filtered):
    ds = ds_filtered
    ds['r'] = ds.x
    ds['q'] = np.exp(ds.x)
    qvalues = ds.q.values.tolist()
    xvalues = ds.x.tolist()
    qexpr = ds.q.expand().expression
    ds['x'] = ds.y # now r should still point to x
    assert ds.r.values.tolist() == xvalues
    assert ds.q.values.tolist() == qvalues


def test_reassign_virtual(ds_local):
    df = ds_local
    x = df.x.values
    df['r'] = df.x+1
    df['r'] = df.r+1
    assert df.r.tolist() == (x+2).tolist()


def test_reassign_column(ds_filtered):
    df = ds_filtered.extract()
    x = df.x.values
    df['r'] = (df.x+1).evaluate()
    df['r'] = (df.r+1).evaluate()
    assert df.r.tolist() == (x+2).tolist()


def test_rename_state_transfer():
    ds = vaex.from_scalars(x=3, y=4)
    ds['r'] = (ds.x**2 + ds.y**2)**0.5
    ds['x'] = ds.x + 1
    ds['q'] = ds.x + 10
    # ds._rename('x', 'a')
    # ds.add_column('x', np.array([6.]))
    # ds['q'] = ds.x+1
    assert ds.r.tolist() == [5]
    assert ds.q.tolist() == [14]
    ds2 = vaex.from_scalars(x=3, y=4)
    ds2.state_set(ds.state_get())
    assert ds2.r.tolist() == [5]
    assert ds2.q.tolist() == [14]
