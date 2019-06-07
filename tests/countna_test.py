import vaex
import numpy as np


def test_countna():
    x = np.array([5, '', 1, 4, None, 6, np.nan, np.nan, 10, '', 0, 0, -13.5])
    df = vaex.from_arrays(x=x)

    assert df.x.countna() == 2
