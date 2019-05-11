import vaex


def test_get_item_type():
    x = [1, 2, 3]
    y = [3, 2, 1]
    s = ['a', 'ab', 'abc']
    b = [True, False, True]
    f = [1.2, 2.3, 5.5]
    df = vaex.from_arrays(x=x, y=y, f=f, s=s, b=b)

    df_sel = df[int]
    assert set(df_sel.column_names) == set(['x', 'y'])

    df_sel = df[str]
    assert set(df_sel.column_names) == set(['s'])

    df_sel = df[bool]
    assert set(df_sel.column_names) == set(['b'])

    df_sel = df[float]
    assert set(df_sel.column_names) == set(['f'])

    df_sel = df[float, int]
    assert set(df_sel.column_names) == set(['f', 'x', 'y'])

    df_sel = df[str, bool]
    assert set(df_sel.column_names) == set(['s', 'b'])

    df_sel = df[str, float]
    assert set(df_sel.column_names) == set(['s', 'f'])
