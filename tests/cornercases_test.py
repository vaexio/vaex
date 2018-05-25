from common import *


def test_filter_and_active_range():
    x = np.arange(20)
    dsf = vaex.from_arrays(x=x)
    ds = dsf[dsf.x > 5]
    assert ds.count(ds.x) == 20-6
    ds.set_active_range(10, 20)
    assert ds.count(ds.x) == 10

    ds = dsf[dsf.x < 10]
    ds.set_active_range(13, 20)
    assert ds.count(ds.x) == 0
