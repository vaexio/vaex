import vaex
import numpy as np

def test_open():
    path = 'tests/data/gassphere_littleendian.dat'
    path = 'tests/data/galaxy_littleendian.dat'
    ds = vaex.dataset.open(path)
    assert ds is not None
    df = vaex.open(path)
    assert df is not None
    assert not np.isnan(df.x.min())
    print(df.x.minmax())