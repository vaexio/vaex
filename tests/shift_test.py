import vaex


def test_shift():
    df = vaex.from_dict({'x': [0, 1, 2, 3, 4],
                         'y': [0.1, 10.2, 20.3, 30.4, 40.5],
                         'z': ['aa', 'bb', 'cc', 'dd', 'ee']})
    df['r'] = df.x + df.y

    df_shifted = df.shift(columns=['r', 'x', 'z'], periods=-3, cyclic=False)
    assert df_shifted.column_count() == 3
    assert df_shifted.x.tolist() == [3, 4, None, None, None]
    assert df_shifted.r.tolist() == [33.4, 44.5, None, None, None]
    assert df_shifted.z.tolist() == ['dd', 'ee', None, None, None]

    # Same as above but specify fill_value
    df_shifted = df.shift(columns=['r', 'x', 'z'], periods=-3, cyclic=False, fill_value='15')
    assert df_shifted.column_count() == 3
    assert df_shifted.x.tolist() == [3, 4, 15, 15, 15]
    assert df_shifted.r.tolist() == [33.4, 44.5, 15, 15, 15]
    assert df_shifted.z.tolist() == ['dd', 'ee', '15', '15', '15']

    # NOTE: This block fails most likely due to a concat issue
    df_shifted = df.shift(columns=['r', 'x', 'z'], periods=3, cyclic=False)
    assert df_shifted.column_count() == 3
    assert df_shifted.x.tolist() == [None, None, None, 0, 1]
    assert df_shifted.r.tolist() == [None, None, None, 0.1, 11.2]
    assert df_shifted.z.tolist() == [None, None, None, 'aa', 'bb']

    df_shifted = df.shift(columns=['r', 'x', 'z'], periods=-3, cyclic=True)
    assert df_shifted.column_count() == 3
    assert df_shifted.x.tolist() == [3, 4, 0, 1, 2]
    assert df_shifted.r.tolist() == [33.4, 44.5, 0.1, 11.2, 22.3]
    assert df_shifted.z.tolist() == ['dd', 'ee', 'aa', 'bb', 'cc']

    df_shifted = df.shift(columns=['r', 'x', 'z'], periods=3, cyclic=True)
    assert df_shifted.column_count() == 3
    assert df_shifted.x.tolist() == [2, 3, 4, 0, 1]
    assert df_shifted.r.tolist() == [22.3, 33.4, 44.5, 0.1, 11.2]
    assert df_shifted.z.tolist() == ['cc', 'dd', 'ee', 'aa', 'bb']
