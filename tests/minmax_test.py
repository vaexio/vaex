import vaex

import numpy as np


def test_minmax():
    x = np.arange(1, 10, 1)
    df = vaex.from_arrays(x=x)
    assert df.x.min() == 1
    assert df.x.max() == 9


def test_minmax_selected():
    x = np.arange(1, 10, 1)
    df = vaex.from_arrays(x=x)
    assert df[(df.x > 3) & (df.x < 7)]['x'].min() == (4)
    assert df[(df.x > 3) & (df.x < 7)]['x'].max() == (6)
