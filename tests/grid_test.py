from common import *

def test_binner(ds_local):
    df = ds_local
    binner_x = df._binner('x', limits=[0, 10], shape=2)
    assert binner_x is df._binner(df.x, limits=[0, 10], shape=2)
    assert binner_x is not df._binner(df.x, limits=[0, 11], shape=2)
    assert binner_x is not df._binner(df.x, limits=[0, 10], shape=3)
    assert binner_x is not df._binner(df.y, limits=[0, 10], shape=3)

def test_grid(ds_local):
    df = ds_local
    binner_x = df._binner('x', limits=[0, 10], shape=2)
    binner_y = df._binner('y', limits=[0, 10], shape=2)
    grid_x  = df._grid([binner_x])
    grid_y  = df._grid([binner_y])
    grid_xy = df._grid([binner_x, binner_y])
    assert grid_x is df._grid([binner_x])
    assert grid_x is not df._grid([binner_y])

    assert grid_y is df._grid([binner_y])
    assert grid_y is not df._grid([binner_x])

    assert grid_xy is not df._grid([binner_x])
    assert grid_xy is df._grid([binner_x, binner_y])


    gridxy = df._create_grid(binby=['x', 'y'], limits=None, shape=2)
    assert gridxy is df._create_grid(binby=['x', 'y'], limits=None, shape=2)
    assert gridxy is not df._create_grid(binby=['x', 'y'], limits=None, shape=3)