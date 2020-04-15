from common import *

import pytest


def test_count_1d():
    ds = vaex.example()
    binned_values = ds.count(binby=ds.x, limits=[-50, 50], shape=16)
    assert len(binned_values) == 16
    binned_values = ds.count(binby=ds.x, limits='minmax', shape=16)
    assert len(binned_values) == 16
    binned_values = ds.count(binby=ds.x, limits='95%', shape=16)
    assert len(binned_values) == 16


def test_count_2d():
    ds = vaex.example()
    binned_values = ds.count(binby=[ds.x, ds.y], shape=32, limits=['minmax', '95%'])
    assert list(binned_values.shape) == [32, 32]
    binned_values = ds.count(binby=[ds.x, ds.y], shape=32, limits=None)
    assert list(binned_values.shape) == [32, 32]
    binned_values = ds.count(binby=[ds.x, ds.y], shape=32, limits=[[-50, 50], [-50, 50]])
    assert list(binned_values.shape) == [32, 32]


@pytest.mark.parametrize('limits', ['minmax', '68.2%', '99.7%', '100%'])
def test_count_1d_verify_against_numpy(ds_local, limits):
    df = ds_local

    expression = 'x'
    selection = df.y > 10
    shape = 4

    # bin with vaex
    vaex_counts = df.count(binby=[expression], selection=selection, shape=shape, limits=limits)

    # bin with numpy
    xmin, xmax = df.limits(expression=expression, value=limits, selection=selection)  # to have the same range as df.count
    x_values = df[selection][expression].values
    numpy_counts, numpy_edges = np.histogram(x_values, bins=shape, range=(xmin, xmax))

    assert vaex_counts[:-1].tolist() == numpy_counts[:-1].tolist()

# def test_count_edges():
#     ds = vaex.from_arrays(x=[-2, -1, 0, 1, 2, 3, np.nan])
#     # 1 missing, 2 to the left, 1 in the range, two to the right
#     assert ds.count(ds.x, binby=ds.x, limits=[0.5, 1.5], shape=1, edges=False).tolist() == [1]
#     assert ds.count(binby=ds.x, limits=[0.5, 1.5], shape=1, edges=True).tolist() == [1, 3, 1, 2]
#     # if the value itself is nan, it should not count it
#     assert ds.count(ds.x, binby=ds.x, limits=[0.5, 1.5], shape=1, edges=True).tolist() == [0, 3, 1, 2]

#     x = np.array([-2, -1, 0, 1, 2, 3, 4])
#     x = np.ma.array(x, mask=x==4)
#     ds = vaex.from_arrays(x=x)
#     # same, but now with a masked value
#     assert ds.count(binby=ds.x, limits=[0.5, 1.5], shape=1, edges=True).tolist() == [1, 3, 1, 2]
#     # assert ds.count(ds.x, limits=[0.5, 1.5], shape=2, edges=True).tolist() == [1, 3, 1, 2]
