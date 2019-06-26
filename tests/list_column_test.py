import os

import vaex


def test_list_column():
    df = vaex.from_dict(
        {'list': [[1, 2], [2, 3], [4, 5]],
         'strings': ['hey you', 'this is', 'a use case'], 'A': [1, 2, 3],
         'B': [2, 4, 6]})
    df['strings'] = df['strings'].str.split(' ')
    df['C'] = df.pack_columns(['A', 'B'])
    assert df['list'].dtype == list
    assert df['strings'].dtype == list
    assert df['C'].dtype == list
    assert len(df['C'][0]) == 2
    df.export_hdf5('.test.hdf5')
    os.remove('.test.hdf5')
