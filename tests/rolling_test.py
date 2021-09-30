import numpy as np


def test_rolling_sum(df_factory):
    x = [0, 1, 2, 3, 4.0]
    df = df_factory(x=x)
    dfp = df.to_pandas_df(array_type='numpy')
    df = df.rolling(2, fill_value=np.nan).sum()
    dfp = dfp.rolling(2).sum()
    result = df['x'].tolist()
    expected = dfp['x'].tolist()
    assert result[1:] == expected[1:]
    assert np.isnan(result[0])
    assert np.isnan(expected[0])


def test_rolling_array(df_factory):
    x = [0, 1, 2, 3, 4]
    xm1 = [1, 2, 3, 4, None]
    y = [0, 1, None, 9, 16]
    df = df_factory(x=x, y=y)
    df = df_factory(x=x, y=y)
    df = df.rolling(2, column='x', edge="left").array()
    assert df.x.tolist() == [[0, 1], [1, 2], [2, 3], [3, 4], [4, None]]

    df = df_factory(x=x, y=y)
    df = df.rolling(2, column='x', edge="right").array()
    assert df.x.tolist() == [[None, 0], [0, 1], [1, 2], [2, 3], [3, 4]]
