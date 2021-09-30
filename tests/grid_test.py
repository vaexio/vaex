from common import *

def test_binner(ds_local):
    df = ds_local
    binner_x = df._binner('x', limits=[0, 10], shape=2)
    assert binner_x == df._binner(df.x, limits=[0, 10], shape=2)
    assert binner_x != df._binner(df.x, limits=[0, 11], shape=2)
    assert binner_x != df._binner(df.x, limits=[0, 10], shape=3)
    assert binner_x != df._binner(df.y, limits=[0, 10], shape=3)

    binnersxy = df._create_binners(binby=['x', 'y'], limits=None, shape=2)
    assert binnersxy == df._create_binners(binby=['x', 'y'], limits=None, shape=2)
    assert binnersxy != df._create_binners(binby=['x', 'y'], limits=None, shape=3)
