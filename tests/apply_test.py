import vaex


def test_apply():
    ds = vaex.from_arrays(x=[1, 2, 3])
    assert ds.x.apply(lambda x: x + 1).values.tolist() == [2, 3, 4]


def test_apply_select():
    x = [1, 2, 3, 4, 5]
    ds = vaex.from_arrays(x=x)
    ds['x_ind'] = ds.x.apply(lambda w: w > 3)
    ds.select('x_ind')
    assert 2 == ds.selected_length()
