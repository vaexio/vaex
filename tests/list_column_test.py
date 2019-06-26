import vaex
import os


def test_list_column():
    df = vaex.from_dict(
        {'list': [[1, 2], [2, 3], [4, 5]], 'strings': ['hey you', 'this is', 'a use case']})
    assert df['list'].dtype == list
    assert df['strings'].dtype == list
    df['strings'] = df['strings'].str.split(' ')
    df.export_hdf5('.test.hdf5')
    os.remove('.test.hdf5')
