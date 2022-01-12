import vaex

def test_from_records():
    df = vaex.from_records([
        {'a': 1, 'b': 2},
        {'a': 11, 'c': 13},
        {'b': 32},
    ])
    assert df.a.tolist() == [1, 11, None]
    assert df.b.tolist() == [2, None, 32]
    assert df.c.tolist() == [None, 13, None]

    df = vaex.from_records([
        {'a': 1, 'b': 2},
        {'a': 11, 'c': 13},
        {'b': 32},
    ], defaults={'a': 111, 'b': 222, 'c': 333})
    assert df.a.tolist() == [1, 11, 111]
    assert df.b.tolist() == [2, 222, 32]
    assert df.c.tolist() == [333, 13, 333]

    df = vaex.from_records([
        {'a': [1, 1], 'b': 2},
        {'a': [11, 12], 'c': 13},
        {'a': [13, 14], 'b': 32},
    ], array_type="numpy")
    assert df.a.tolist() == [[1, 1], [11, 12], [13, 14]]
    assert df.a.shape == (3, 2)
