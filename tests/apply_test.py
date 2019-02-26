import vaex


def test_apply():
    ds = vaex.from_arrays(x=[1, 2, 3])
    assert ds.x.apply(lambda x: x + 1).values.tolist() == [2, 3, 4]
