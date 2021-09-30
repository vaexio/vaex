import vaex
import numpy as np


def test_countna(df_factory):
    x = np.array([5, '', 1, 4, None, 6, np.nan, np.nan, 10, '', 0, 0, -13.5])
    y_data = np.array([np.nan, 2, None, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    y_mask = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1])
    y = np.ma.MaskedArray(data=y_data, mask=y_mask)
    df = df_factory(x=x, y=y)
    pandas_df = df.to_pandas_df(array_type='numpy')

    assert df.x.countna() == pandas_df.x.isna().sum()
    assert df.y.countna() == pandas_df.y.isna().sum()
    assert df.x.countnan() == 2
    assert df.y.countmissing() == 6
