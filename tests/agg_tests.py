import vaex
import numpy as np

# def test_count_multiple_selections():

def test_count_1d():
    x = np.array([-1, -2, 0.5, 1.5, 4.5, 5], dtype='f8')
    df = vaex.from_arrays(x=x)

    bins = 5
    binner = df._binner_scalar('x', [0, 5], bins)
    grid = vaex.superstats.Grid([binner])
    agg = vaex.agg.count()
    grid = df._agg(agg, grid)
    assert grid.tolist() == [0, 2, 1, 1, 0, 0, 1, 1]


def test_count_1d_ordinal():
    x = np.array([-1, -2, 0, 1, 4, 5], dtype='i8')
    df = vaex.from_arrays(x=x)

    bins = 5
    binner = df._binner_ordinal('x', 5)
    grid = vaex.superstats.Grid([binner])
    agg = vaex.agg.count()
    grid = df._agg(agg, grid)
    assert grid.tolist() == [0, 2, 1, 1, 0, 0, 1, 1]
