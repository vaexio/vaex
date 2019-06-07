import vaex
import numpy as np


def test_isna():
    x = np.array([5, '', 1, 4, None, 6, np.nan, np.nan, 10, '', 0, 0, -13.5])
    df = vaex.from_arrays(x=x)

    assert df.x.isna().tolist() == [False, False, False, False, False, False, True, True, False, False, False, False, False]


def test_notna():
    x = np.array([5, '', 1, 4, None, 6, np.nan, np.nan, 10, '', 0, 0, -13.5])
    df = vaex.from_arrays(x=x)

    assert df.x.notna().tolist() == [True,  True,  True, True, True,  True, False, False, True, True, True, True, True]
