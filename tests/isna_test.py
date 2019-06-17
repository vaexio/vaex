import vaex
import numpy as np


def test_isna():
    x = np.array([5, '', 1, 4, None, 6, np.nan, np.nan, 10, '', 0, 0, -13.5])
    y_data = np.array([np.nan, 2, None, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    y_mask = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1])
    y = np.ma.MaskedArray(data=y_data, mask=y_mask)
    df = vaex.from_arrays(x=x, y=y)
    pandas_df = df.to_pandas_df()

    assert df.x.isna().tolist() == pandas_df.x.isna().tolist()
    assert df.y.isna().tolist() == pandas_df.y.isna().tolist()


def test_notna():
    x = np.array([5, '', 1, 4, None, 6, np.nan, np.nan, 10, '', 0, 0, -13.5])
    y_data = np.array([np.nan, 2, None, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    y_mask = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1])
    y = np.ma.MaskedArray(data=y_data, mask=y_mask)
    df = vaex.from_arrays(x=x, y=y)
    pandas_df = df.to_pandas_df()

    assert df.x.notna().tolist() == pandas_df.x.notna().tolist()
    assert df.y.notna().tolist() == pandas_df.y.notna().tolist()
