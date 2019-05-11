import vaex


def test_get_item_type():
    x = [1, 2, 3]
    y = [3, 2, 1]
    s = ['a', 'ab', 'abc']
    b = [True, False, True]
    f = [1.2, 2.3, 5.5]
    df = vaex.from_arrays(x=x, y=y, f=f, s=s, b=b)

    df_sel = df[int]
    assert df_sel.column_names == ['x', 'y']

    df_sel = df[str]
    assert df_sel.column_names == ['s']

    df_sel = df[bool]
    assert df_sel.column_names == ['b']

    df_sel = df[float]
    assert df_sel.column_names == ['f']

    df_sel = df[float, int]
    assert df_sel.column_names == ['f', 'x', 'y']

    df_sel = df[str, bool]
    assert df_sel.column_names == ['s', 'b']

    df_sel = df[str, float]
    assert df_sel.column_names == ['s', 'f']
