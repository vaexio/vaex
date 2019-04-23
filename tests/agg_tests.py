import vaex
import numpy as np

# def test_count_multiple_selections():


def test_count_1d():
    x = np.array([-1, -2, 0.5, 1.5, 4.5, 5], dtype='f8')
    df = vaex.from_arrays(x=x)

    bins = 5
    binner = df._binner_scalar('x', [0, 5], bins)
    grid = vaex.superagg.Grid([binner])
    agg = vaex.agg.count()
    grid = df._agg(agg, grid)
    assert grid.tolist() == [0, 2, 1, 1, 0, 0, 1, 1]


def test_count_1d_ordinal():
    x = np.array([-1, -2, 0, 1, 4, 5], dtype='i8')
    df = vaex.from_arrays(x=x)

    bins = 5
    binner = df._binner_ordinal('x', 5)
    grid = vaex.superagg.Grid([binner])
    agg = vaex.agg.count()
    grid = df._agg(agg, grid)
    assert grid.tolist() == [0, 2, 1, 1, 0, 0, 1, 1]


def test_big_endian_binning():
    x = np.arange(10, dtype='>f8')
    y = np.zeros(10, dtype='>f8')
    ds = vaex.from_arrays(x=x, y=y)
    counts = ds.count(binby=[ds.x, ds.y], limits=[[-0.5, 9.5], [-0.5, 0.5]], shape=[10, 1])
    assert counts.ravel().tolist() == np.ones(10).tolist()


def test_big_endian_binning_non_contiguous():
    x = np.arange(20, dtype='>f8')[::2]
    x[:] = np.arange(10, dtype='>f8')
    y = np.arange(20, dtype='>f8')[::2]
    y[:] = np.arange(10, dtype='>f8')
    ds = vaex.from_arrays(x=x, y=y)
    counts = ds.count(binby=[ds.x, ds.y], limits=[[-0.5, 9.5], [-0.5, 9.5]], shape=[10, 10])
    assert np.diagonal(counts).tolist() == np.ones(10).tolist()


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
    counts = ds.count(binby=[ds.x, ds.y], limits=[[-0.5, 9.5], [-0.5, 0.5]], shape=[10, 1])
    assert counts.ravel().tolist() == np.ones(10).tolist()
