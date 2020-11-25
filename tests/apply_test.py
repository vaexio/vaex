import vaex


def test_apply():
    df = vaex.from_arrays(x=[1, 2, 3])
    assert df.x.apply(lambda x: x + 1).values.tolist() == [2, 3, 4]


def test_apply_vectorized():
    df = vaex.from_arrays(x=[1, 2, 3])
    assert df.x.apply(lambda x: x + 1, vectorize=True).values.tolist() == [2, 3, 4]

def test_apply_select():
    # tests if select supports lambda/functions
    x = [1, 2, 3, 4, 5]
    df = vaex.from_arrays(x=x)
    df['x_ind'] = df.x.apply(lambda w: w > 3)
    df.state_get()
    df.select('x_ind')
    assert 2 == df.selected_length()
