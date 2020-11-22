import vaex
from pathlib import Path


DATA_PATH = Path(__file__).parent / 'data'


def test_hdf5_with_alias(tmpdir):
    df = vaex.from_dict({'X-1': [1], '#': [2]})
    path = DATA_PATH / 'with_alias.hdf5'
    df = vaex.open(str(path))
    assert df['X-1'].tolist() == [1]
    assert df['#'].tolist() == [2]
