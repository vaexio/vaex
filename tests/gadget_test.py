import vaex
import vaex.file.other
import numpy as np

def test_open():
    path = 'tests/data/gassphere_littleendian.dat'
    path = 'tests/data/galaxy_littleendian.dat'
    #ds = vaex.file.other.MemoryMappedGadget(path)
    ds = vaex.open(path)
    assert ds is not None
    assert not np.isnan(ds.x.min())
    print(ds.x.minmax())