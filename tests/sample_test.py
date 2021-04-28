from common import *
import collections

def test_sample(ds_local):
    ds = ds_local
    x = np.arange(10).tolist()
    dss = ds.sample(frac=1, random_state=42)
    assert dss.x.tolist() != x
    assert list(sorted(dss.x.tolist())) == x

    dss = ds.sample(n=1, random_state=42)
    assert len(dss) == 1

    dss = ds.sample(n=100, random_state=42, replace=True)
    assert len(dss) == 100


    dss = ds.sample(n=100, random_state=42, replace=True, weights='x')
    assert 0 not in dss.x.tolist()
    assert 'bla' not in dss.x.tolist()
