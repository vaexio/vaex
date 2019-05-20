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
    assert set(df_sel.column_names) == set(['x'])

    df_sel = df[np.float32]
    assert set(df_sel.column_names) == set(['f'])

    df_sel = df[int]
    assert set(df_sel.column_names) == set(['x', 'y'])

    df_sel = df[str]
    assert set(df_sel.column_names) == set(['s'])

    df_sel = df[bool]
    assert set(df_sel.column_names) == set(['b'])

    df_sel = df[float]
    assert set(df_sel.column_names) == set(['f', 'g'])

    df_sel = df[float, int]
    assert set(df_sel.column_names) == set(['f', 'g', 'x', 'y'])

    df_sel = df[str, bool]
    assert set(df_sel.column_names) == set(['s', 'b'])

    df_sel = df[str, float]
    assert set(df_sel.column_names) == set(['s', 'f', 'g'])
