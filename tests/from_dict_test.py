import vaex


def test_from_dict():
    data = {'A': [1, 2, 3], 'B': ['a', 'b', 'c']}
    ds = vaex.from_dict(data)
    assert 'A' in ds.get_column_names()
    assert ds['A'].tolist()[0] == 1
    assert ds['B'].tolist()[2] == 'c'
