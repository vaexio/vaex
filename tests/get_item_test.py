import vaex
import numpy as np


def test_get_item_type():
    x = np.array([1, 2, 3], dtype=np.int16)
    y = np.array([3, 2, 1], dtype=np.int64)
    s = ['a', 'ab', 'abc']
    b = np.array([True, False, True], dtype=np.bool)
    f = np.array([1.2, 2.3, 5.5], dtype=np.float32)
    g = np.array([1.5, 2.5, 5.5], dtype=np.float)

    df = vaex.from_arrays(x=x, y=y, f=f, s=s, b=b, g=g)

    df_sel = df[np.int16]
    assert df_sel.column_names == ['x']

    df_sel = df[np.float32]
    assert df_sel.column_names == ['f']

    df_sel = df[int]
    assert df_sel.column_names == ['x', 'y']

    df_sel = df[str]
    assert df_sel.column_names == ['s']

    df_sel = df[bool]
    assert df_sel.column_names == ['b']

    df_sel = df[float]
    assert df_sel.column_names == ['f', 'g']

    df_sel = df[float, int]
    assert df_sel.column_names == ['f', 'g', 'x', 'y']

    df_sel = df[str, bool]
    assert df_sel.column_names == ['s', 'b']

    df_sel = df[str, float]
    assert df_sel.column_names == ['s', 'f', 'g']
