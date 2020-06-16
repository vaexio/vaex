import vaex


def test_cardinality():
    df = vaex.from_dict({'x': [1.1, 1.1, 2.2, 2.2],
                         'y': ['a', 'a', 'b', 'b'],
                         'z': [1, 2, 3, 4],
                         'w': [1, 1, 2, 3]})
    assert df.x.cardinality() == 0.5
    assert df.y.cardinality() == 0.5
    assert df.z.cardinality() == 1
    assert df.w.cardinality() == 0.75
