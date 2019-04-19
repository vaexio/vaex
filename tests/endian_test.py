import vaex
import numpy as np

def test_strides():
    ar = np.zeros((10, 2)).reshape(20)
    x = ar[::2]
    x[:] = np.arange(10)
    ds = vaex.from_arrays(x=x)
    counts = ds.count(binby=ds.x, limits=[-0.5, 9.5], shape=10)
    assert counts.tolist() == np.ones(10).tolist()

def test_expr():
    ar = np.zeros((10, 2)).reshape(20)
    x = ar[::2]
    x[:] = np.arange(10)
    ds = vaex.from_arrays(x=x)
    counts = ds.count('x*2', binby='x*2', limits=[-0.5, 19.5], shape=10)
    assert counts.tolist() == np.ones(10).tolist()

def test_big_endian_binning():
    x = np.arange(10, dtype='>f8')
    y = np.zeros(10, dtype='>f8')
    ds = vaex.from_arrays(x=x, y=y)
    counts = ds.count(binby=[ds.x, ds.y], limits=[[-0.5, 9.5], [-0.5, 0.5]], shape=[10,1])
    assert counts.ravel().tolist() == np.ones(10).tolist()