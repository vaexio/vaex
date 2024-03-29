import glob
import os
from pathlib import Path
import sys
import io

import fsspec.implementations.memory

import pyarrow as pa
import pyarrow.csv
import pytest
import vaex

path = Path(__file__).parent
data_path = path / 'data'
csv_path = os.path.join(path, 'data', 'small3.csv')

if sys.platform.startswith("win"):
    pytest.skip("skipping windows, since it fails with access violation", allow_module_level=True)


def test_from_csv():
    # can read with default options
    df = vaex.from_csv(csv_path, copy_index=True)
    _assert_csv_content(df, with_index=True)

    # can read an empty CSV
    df = vaex.from_csv(os.path.join(path, 'data', 'empty.csv'))
    assert len(df) == 0

    # can read csv with no header
    df = vaex.from_csv(os.path.join(path, 'data', 'noheader.csv'), header=None)
    assert len(df) == 5
    assert df.get_column_names() == ['0', '1', '2']

    # can read as chunks iterator
    df_iterator = vaex.from_csv(csv_path, chunk_size=1)
    df1 = next(df_iterator)
    assert len(df1) == 1
    df2, df3 = next(df_iterator), next(df_iterator)
    with pytest.raises(StopIteration):
        next(df_iterator)
    _assert_csv_content(vaex.concat([df1, df2, df3]))


def test_diffent_extension():
    df = vaex.from_csv(data_path / 'small2.nocsv')
    assert df.x.tolist() == [1, 3]
    df = vaex.from_csv(data_path / 'small2.nocsv', convert=True)
    assert df.x.tolist() == [1, 3]


def test_from_csv_converting_in_chunks():
    # can read several chunks with converting, intermediate files are deleted
    df = vaex.from_csv(csv_path, chunk_size=1, convert=True)
    _assert_csv_content(df)
    for filename in ['small3.csv_chunk0.hdf5', 'small3.csv_chunk1.hdf5', 'small3.csv_chunk2.hdf5']:
        assert not os.path.exists(os.path.join(path, 'data', filename))
    assert os.path.exists(os.path.join(path, 'data', 'small3.csv.hdf5'))
    _cleanup_generated_files(df)

    # fails to convert if filename cannot be derived
    with pytest.raises(ValueError, match='Cannot convert.*'):
        with io.StringIO() as f:
            vaex.from_csv(f, convert=True)
    # f.name reveals the path
    with open(csv_path) as f:
        vaex.from_csv(f, convert=True)
    with open(csv_path) as f:
        converted_path = os.path.join(path, 'data', 'small3.my.csv.hdf5')
        df = vaex.from_csv(f, convert=converted_path)
    _assert_csv_content(df)
    assert os.path.exists(converted_path)
    _cleanup_generated_files(df)

    # reuses converted HDF5 file
    vaex.from_csv(csv_path, convert=True)
    assert os.path.exists(os.path.join(path, 'data', 'small3.csv.hdf5'))
    try:
        with pytest.raises(FileNotFoundError):
            os.rename(csv_path, csv_path + '_')
            df = vaex.from_csv(csv_path, convert=True)
            _assert_csv_content(df)
            _cleanup_generated_files(df)
    finally:
        os.rename(csv_path + '_', csv_path)


def _assert_csv_content(csv_df, with_index=False):
    assert len(csv_df) == 3
    if with_index:
        assert csv_df.get_column_names() == ['x', 'y', 'index']
    else:
        assert csv_df.get_column_names() == ['x', 'y']
    assert csv_df.x.tolist() == [5, 7, 9]
    assert csv_df.y.tolist() == [6, 9, 10]


def _cleanup_generated_files(*dfs):
    for df in dfs:
        df.close()
    for hdf5_file in glob.glob(os.path.join(path, 'data', 'small*.hdf5')):
        os.remove(hdf5_file)


@pytest.mark.parametrize("use_fs_layer", [True, False])
def test_arrow_lazy_reading(use_fs_layer):
    path_to_csv_file = os.path.join(path, "data", "difficult_schema.csv")
    if use_fs_layer:
        path_to_csv_file_local = path_to_csv_file
        path_to_csv_file = "difficult_schema.csv"
        fs = fsspec.implementations.memory.MemoryFileSystem()
        fs.upload(path_to_csv_file_local, path_to_csv_file)
        fs = pyarrow.fs.FSSpecHandler(fs)
    else:
        fs = None

    df = vaex.from_csv_arrow(path_to_csv_file, lazy=True, chunk_size="3b", newline_readahead="7b", schema_infer_fraction=1.0, fs=fs)

    assert df.shape == (3, 4)

    assert df.x.dtype == int
    assert df.x.sum() == 1+2+3

    assert df.y.dtype == int
    assert df.y.sum() == 2
    assert df.y.tolist() == [None, 2, None]

    assert df.z.dtype == str
    assert df.z.tolist() == ['1', 'la', '']

    assert df.q.dtype == int
    assert df.q.tolist() == [None, None, None]

def test_arrow_non_unicode():
    path_to_csv_file = os.path.join(path, "data", "non-unicode.csv")
    read_options = pa.csv.ReadOptions(use_threads=False, encoding="ISO-8859-1")
    df = vaex.from_csv_arrow(path_to_csv_file, lazy=True, schema_infer_fraction=1.0, read_options=read_options, chunk_size="10b", newline_readahead="15b")
    df.fingerprint()  # coverage

    assert df.shape == (10, 2)
    assert df.Tailnr.dtype == str
    assert df.Tailnr.tolist()[4:6] == ["N331A", "N636Aï¿½"]
