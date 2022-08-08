from common import *


def test_index(df_factory):
    df = df_factory(x=[1, 2, 3], y=[4.5, 5.5, 6.5])
    assert df[0] == [1, 4.5]
    assert df[-1] == [3, 6.5]

    dff = df[df.x > 1]
    assert dff[0] == [2, 5.5]
    assert dff[-1] == [3, 6.5]
