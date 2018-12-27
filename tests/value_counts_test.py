import vaex


def test_value_counts():
    ds = vaex.from_arrays(**{'A':['a','b','b','c','c','c']})
    counts = ds['A'].value_counts()
    assert counts.get_column_names()==['value', 'count']
    assert counts[0] == ['c', 3]
    assert counts[1] == ['b', 2]
    assert counts[2] == ['a', 1]

    counts = ds['A'].value_counts(ascending=True)
    assert counts.get_column_names() == ['value', 'count']
    assert counts[2] == ['c', 3]
    assert counts[1] == ['b', 2]
    assert counts[0] == ['a', 1]
