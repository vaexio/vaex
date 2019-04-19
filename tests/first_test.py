from common import *
import collections

def test_first(ds_filtered):
    ds = ds_filtered
    # assert ds.first(ds.y, ds.x) == 0
    with small_buffer(ds, 3):
        assert ds.first(ds.y, ds.x).tolist() == 0
        assert ds.first(ds.y, ds.x, binby=[ds.x], limits=[0, 10], shape=2).tolist() == [0, 5**2]
        assert ds.first(ds.y, -ds.x, binby=[ds.x], limits=[0, 10], shape=2).tolist() == [4**2, 9**2]
        assert ds.first(ds.y, -ds.x, binby=[ds.x, ds.x+5], limits=[[0, 10], [5, 15]], shape=[2, 1]).tolist() == [[4**2], [9**2]]
        assert ds.first([ds.y, ds.y], ds.x).tolist() == [0, 0]
