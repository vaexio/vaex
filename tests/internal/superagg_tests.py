import vaex.superagg
import numpy as np
import sys


def test_ref_count():
    x = np.array([-1, -2, 0.5, 1.5, 4.5, 5], dtype='f8')
    bins = 5
    binner = vaex.superagg.BinnerScalar_float64('x', 0, 5, bins)
    start_count_binner = sys.getrefcount(binner)
    grid = vaex.superagg.Grid([binner])
    assert sys.getrefcount(binner) == start_count_binner + 1
    start_count_grid = sys.getrefcount(grid)

    agg = vaex.superagg.AggCount_float64(grid)
    assert sys.getrefcount(binner) == start_count_grid + 1
    del agg
    assert sys.getrefcount(grid) == start_count_grid
    assert sys.getrefcount(binner) == start_count_binner + 1
    del grid
    assert sys.getrefcount(binner) == start_count_binner

def test_count_1d_scalar():
    x = np.array([-1, -2, 0.5, 1.5, 4.5, 5], dtype='f8')
    bins = 5
    binner = vaex.superagg.BinnerScalar_float64('x', 0, 5, bins)
    binner.set_data(x)
    grid = vaex.superagg.Grid([binner])
    agg = vaex.superagg.AggCount_float64(grid)
    agg_data = np.asarray(agg)
    grid.bin([agg])
    assert agg_data.tolist() == [0, 2, 1, 1, 0, 0, 1, 1]

def test_count_1d_strings():
    x = np.array([-1, -2, 0.5, 1.5, 4.5, 5], dtype='f8')
    y = x.astype(str).astype('O')
    y[2] = None
    y = vaex.column._to_string_sequence(y)
    bins = 5
    binner = vaex.superagg.BinnerScalar_float64('x', 0, 5, bins)
    binner.set_data(x)

    grid = vaex.superagg.Grid([binner])
    agg = vaex.superagg.AggCount_string(grid)
    agg.set_data(y, 0)
    agg_data = np.asarray(agg)
    grid.bin([agg])
    assert agg_data.tolist() == [0, 2, 0, 1, 0, 0, 1, 1]

def test_count_1d_scalar_int64():
    x = np.array([-1, -2, 0.5, 1.5, 4.5, 5], dtype='i8')
    bins = 5
    binner = vaex.superagg.BinnerScalar_int64('x', 0, 5, bins)
    binner.set_data(x)
    grid = vaex.superagg.Grid([binner])
    agg = vaex.superagg.AggCount_float64(grid)
    agg_data = np.asarray(agg)
    grid.bin([agg])
    assert agg_data.tolist() == [0, 2, 1, 1, 0, 0, 1, 1]

def test_count_1d_ordinal():
    x = np.array([-1, -2, 0, 1, 4, 6, 10], dtype='i8')
    ordinal_count = 5
    binner = vaex.superagg.BinnerOrdinal_int64('x', ordinal_count, 0)
    binner.set_data(x)
    grid = vaex.superagg.Grid([binner])
    agg = vaex.superagg.AggCount_int64(grid)
    agg_data = np.asarray(agg)
    grid.bin([agg])
    assert agg_data.tolist() == [0, 2, 1, 1, 0, 0, 1, 2]

def test_count_2d_ordinal():
    x = np.array([-1, -2, 0, 1, 4, 6, 10], dtype='i8')
    ordinal_count = 5
    binner1 = vaex.superagg.BinnerOrdinal_int64('x', ordinal_count, 0)
    binner2 = vaex.superagg.BinnerOrdinal_int64('x', ordinal_count, 0)
    binner1.set_data(x)
    binner2.set_data(x)
    grid = vaex.superagg.Grid([binner1, binner2])
    agg = vaex.superagg.AggCount_int64(grid)
    agg_data = np.asarray(agg)
    grid.bin([agg])
    diagonal = [agg_data[k,k] for k in range(agg_data.shape[0])]
    assert diagonal == [0, 2, 1, 1, 0, 0, 1, 2]

def test_min_max_1d_ordinal():
    x = np.array([-1, -1, 0, 0, 4, 6, 10], dtype='i8')
    y = np.array([-1,  2, 4, 1, 9, 6, 10], dtype='i8')
    ordinal_count = 5
    binner = vaex.superagg.BinnerOrdinal_int64('x', ordinal_count, 0)
    binner.set_data(x)
    grid = vaex.superagg.Grid([binner])
    agg = vaex.superagg.AggMax_int64(grid)
    agg_data = np.asarray(agg)
    agg_data -= 100
    agg.set_data(y, 0)
    grid.bin([agg])
    assert agg_data.tolist() == [-100, 2, 4, -100, -100, -100, 9, 10]
    
    grid = vaex.superagg.Grid([binner])
    agg = vaex.superagg.AggMin_int64(grid)
    agg_data = np.asarray(agg)
    agg_data += 100
    agg.set_data(y, 0)
    grid.bin([agg])
    assert agg_data.tolist() == [100, -1, 1, 100, 100, 100, 9, 6]

def test_sum_1d_ordinal():
    x = np.array([-1, -1, 0, 0, 4, 6, 10], dtype='i8')
    y = np.array([-1,  2, 4, 1, 9, 6, 10], dtype='i8')
    ordinal_count = 5
    binner = vaex.superagg.BinnerOrdinal_int64('x', ordinal_count, 0)
    binner.set_data(x)
    grid = vaex.superagg.Grid([binner])
    agg = vaex.superagg.AggSum_int64(grid)
    agg_data = np.asarray(agg)
    agg.set_data(y, 0)
    grid.bin([agg])
    assert agg_data.tolist() == [0, 1, 5, 0, 0, 0, 9, 16]

def test_count_1d_object():
    x = np.array([-1, -1, 0, 0, 2, 6, 10], dtype='i8')
    y = np.array([ 1,  1, 1, None, 1, '1', np.nan], dtype='O')
    ordinal_count = 5
    binner = vaex.superagg.BinnerOrdinal_int64('x', ordinal_count, 0)
    binner.set_data(x)
    grid = vaex.superagg.Grid([binner])
    agg = vaex.superagg.AggCount_object(grid)
    agg_data = np.asarray(agg)
    agg.set_data(y, 0)
    grid.bin([agg])
    assert agg_data.tolist() == [0, 2, 1, 0, 1, 0, 0, 1]


def test_sum_1d_bool():
    x = np.array([True, True, False], dtype='?')
    y = np.array([1, 9, 3], dtype='float64')
    ordinal_count = 2
    binner = vaex.superagg.BinnerOrdinal_bool('x', ordinal_count, 0)
    binner.set_data(x)
    grid = vaex.superagg.Grid([binner])
    agg = vaex.superagg.AggSum_float64(grid)
    agg_data = np.asarray(agg)
    agg.set_data(y, 0)
    grid.bin([agg])
    assert agg_data.tolist() == [0, 0, 3, 10, 0]


def test_to_bins():
    x = np.array([True, True, False, True, True, True, True], dtype='?')
    xi = [3, 3, 2, 3, 3, 3, 3]
    y = np.array([-1, -1, 0, 0, 2, 5, 10], dtype='i8')
    yi = [1, 1, 2, 2, 4, 7, 8]
    ordinal_count_x = 2
    ordinal_count_y = 6
    binner1 = vaex.superagg.BinnerOrdinal_bool('x', ordinal_count_x, 0)
    binner1.set_data(x)
    binner2 = vaex.superagg.BinnerOrdinal_int64('y', ordinal_count_y, 0)
    binner2.set_data(y)
    binners = [binner1, binner2]
    indices = vaex.superagg.binners_to_1d(len(x), binners)
    assert indices.tolist() == [xi_ + yi_ * 5 for xi_, yi_ in zip(xi, yi)]
