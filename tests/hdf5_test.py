from pathlib import Path

import fsspec.implementations.memory
from dask.utils import tmpdir

import vaex
from vaex.dataframe import DataFrameLocal


DATA_PATH = Path(__file__).parent / 'data'


def test_hdf5_with_alias(tmpdir):
    df = vaex.from_dict({'X-1': [1], '#': [2]})
    path = DATA_PATH / 'with_alias.hdf5'
    df = vaex.open(str(path))
    assert df['X-1'].tolist() == [1]
    assert df['#'].tolist() == [2]


def test_categorical(tmpdir, df_factory):
    path = tmpdir / "with_cats.hdf5"
    s = ["aap", "noot", "mies", "mies", "aap", None]
    df: DataFrameLocal = df_factory(s=s)
    df = df.ordinal_encode("s")
    df = df._future()
    assert df.s.tolist(()) == s
    df.export(path)
    df_result = vaex.open(path)
    assert df_result.s.tolist() == s

    # make sure we also support cloud storage etc
    fs = fsspec.implementations.memory.MemoryFileSystem()
    fs.put(str(path), "with_cats.hdf5")
    df = vaex.open('with_cats.hdf5', fs=fs)
    assert df_result.s.tolist() == s
