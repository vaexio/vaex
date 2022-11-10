from common import *
import os
from pathlib import Path
import tempfile
import pyarrow as pa
import pandas as pd
import platform
import hashlib


DATA_PATH = Path(__file__).parent


@pytest.mark.skipif(platform.system().lower() == 'windows', reason="access violation?")
@pytest.mark.parametrize("filename", ["test.hdf5", "test.arrow", "test.parquet", "test.csv"])
def test_export_empty_string(tmpdir, filename):
    path = str(tmpdir.join(filename))
    s = np.array(["", ""])
    df = vaex.from_arrays(s=s)
    df.export(path)
    df = vaex.open(path)
    repr(df)


def test_export_non_identifier(tmpdir):
    df = vaex.from_dict({'#': ['foo']})
    df.export_hdf5(tmpdir / 'test.hdf5')
    df2 = vaex.open(tmpdir / 'test.hdf5')
    assert df2['#'].tolist() == ['foo']


def test_export_many_name(df_local, tmpdir):
    df = df_local[['x']]
    df.export_many(tmpdir / 'chunk.parquet', chunk_size=3)
    assert (tmpdir / 'chunk-00001.parquet').exists()
    assert (tmpdir / 'chunk-00002.parquet').exists()
    assert (tmpdir / 'chunk-00003.parquet').exists()
    assert not (tmpdir / 'chunk-00004.parquet').exists()


def test_export_large_string_parquet(tmpdir):
    s = pa.array(['aap', 'noot', 'mies'], type=pa.large_string())
    df = vaex.from_arrays(s=s)
    df.export_parquet(tmpdir / 'chunk.parquet')


def test_export_many(df_local, tmpdir):
    df = df_local
    df = df.drop('datetime')
    if 'timedelta' in df:
        df = df.drop('timedelta')
    if 'obj' in df:
        df = df.drop(['obj'])
    df.export_many(tmpdir / 'chunk_{i:05}.parquet', chunk_size=3)
    df_copy = vaex.open(str(tmpdir / 'chunk_*.parquet'))
    assert df_copy.x.tolist() == df.x.tolist()


@pytest.mark.parametrize("backend", ["arrow", "pandas"])
def test_export_many_csv(df_local, tmpdir, backend):
    df = df_local
    df = df.drop('datetime')
    if 'timedelta' in df:
        df = df.drop('timedelta')
    if 'obj' in df:
        df = df.drop(['obj'])
    df.export_many(tmpdir / 'chunk_{i:05}.csv', chunk_size=3, backend=backend)
    df_copy = vaex.open(str(tmpdir / 'chunk_*.csv'))
    assert df_copy.x.tolist() == df.x.tolist()


def test_export_basic(ds_local, tmpdir):
    ds = ds_local
    # TODO: we eventually want to support dtype=object, but not for hdf5
    if 'obj' in ds:  # df_arrow does not have it
        ds = ds.drop(ds.obj)
    path = str(tmpdir.join('test.hdf5'))
    ds.export_hdf5(path)
    ds = ds.sample(5)
    path = str(tmpdir.join('sample.hdf5'))
    ds.export_hdf5(path)

    if 'timedelta' in ds:  # df_arrow does not have it
        ds = ds.drop(ds.timedelta)

    if platform.system().lower() != 'windows':
        path = str(tmpdir.join('sample.parquet'))
        ds['datetime'] = ds.datetime.astype('datetime64[ms]')
        ds.export(path)
        df = vaex.open(path)


def test_export_hdf5_2d(tmpdir):
    x = np.arange((2*3*4)).reshape((2, 3, 4))
    df = vaex.from_arrays(x=x)
    path = str(tmpdir.join('test.hdf5'))
    df.export_hdf5(path)
    df2 = vaex.open(path)
    assert df.x.tolist() == df2.x.tolist()
    assert df2.x.shape == (2, 3, 4)


def test_export_hdf5_2d_masked(tmpdir):
    x = np.arange((2*3*4)).reshape((2, 3, 4))
    mask = x == 5
    x = np.ma.array(x, mask=mask)
    df = vaex.from_arrays(x=x)
    path = str(tmpdir.join('test.hdf5'))
    df.export_hdf5(path)
    df2 = vaex.open(path)
    assert df.x.tolist() == df2.x.tolist()
    assert df2.x.shape == (2, 3, 4)



def test_export_open_hdf5(ds_local):
    ds = ds_local
    ds = ds.drop(ds.obj)
    filename = tempfile.mktemp(suffix='.hdf5')
    ds.export(filename)
    ds_opened = vaex.open(filename)
    assert list(ds) == list(ds_opened)

def test_export_concat_missing_cols_hdf5(tmpdir):
    df1 = vaex.from_arrays(x=[1, 2, 3], s=['x1', 'x2', 'x3'], y=[10, 20, 30])
    df2 = vaex.from_arrays(x=[4, 5, 6])

    df = vaex.concat([df1, df2])

    filename = tempfile.mktemp(suffix='.hdf5')
    df.export_hdf5(filename)

    df_opened = vaex.open(filename)
    assert df_opened.x.tolist() == [1, 2, 3, 4, 5, 6]
    assert df_opened.y.tolist() == [10, 20, 30, None, None, None]
    assert df_opened.s.tolist() == ['x1', 'x2', 'x3', None, None, None]

def test_export_open_csv(ds_local, tmpdir):
    df = ds_local
    path = str(tmpdir.join('test.csv'))
    df.export_csv(path, chunk_size=3)
    df_opened = vaex.read_csv(path)
    assert list(df) == list(df_opened)
    assert df.shape == df_opened.shape

def test_export_open_hdf5(ds_local):
    ds = ds_local
    if 'obj' in ds:  # df_arrow does not have it
        ds = ds.drop(ds.obj)
    if 'timedelta' in ds:  # df_arrow does not have it
        ds = ds.drop(ds.timedelta)
    ds = ds.drop(ds.z)
    filename = tempfile.mktemp(suffix='.arrow')
    ds.export(filename)
    ds_opened = vaex.open(filename)
    assert list(ds) == list(ds_opened)


def test_export_string_mask(tmpdir):
    df = vaex.from_arrays(s=vaex.string_column(['aap', None, 'mies']))

    path = str(tmpdir.join('test.hdf5'))
    df.export(path)
    df_hdf5 = vaex.open(path)
    assert df.s.tolist() == df_hdf5.s.tolist()

    path = str(tmpdir.join('test.arrow'))
    df.export(path)
    df_arrow = vaex.open(path)
    assert df.s.tolist() == df_arrow.s.tolist()

def test_export_unicode_column_name_hdf5(tmpdir):
    # prepare many columns for multithreaded export
    src_dict = {"あ": [1, 2, 3], "a": [1, 2, 3], "b": [1, 2, 3],
            "c": [1, 2, 3], "d": [1, 2, 3], "a1": [1, 2, 3],
            "b2": [1, 2, 3], "c3": [1, 2, 3], "d4": [1, 2, 3]}
    path = str(tmpdir.join('test.hdf5'))
    df = vaex.from_dict(src_dict)
    df.export(path)
    df_open = vaex.open(path)
    assert df_open["あ"].tolist() == [1, 2, 3]

# N = 2**32+2
# @pytest.mark.skipif(not os.environ.get('VAEX_EXPORT_BIG', False),
#                     reason="only runs when the env var VAEX_EXPORT_BIG is defined")
# def test_export_big(tmpdir):
#   path = str(tmpdir.join('test.hdf5'))
#   s = np.zeros(N, dtype='U1')
#   s[:] = 'x'
#   s[-1] = 'y'
#   ds = vaex.from_arrays(s=s)
#   ds.export_hdf5(path)
#   df = ds.open(path)
#   assert df[0:2].s.tolist() == ['x', 'x']
#   assert df[-3:-1].s.tolist() == ['y', 'y']

@pytest.mark.parametrize("dtypes", [{}, {'name': np.object, 'age': 'Int64', 'weight': np.float}])
def test_multi_file_naive_read_convert_export(tmpdir, dtypes):
    current_dir = os.path.dirname(__file__)

    path1 = '/data/sample_1.csv'
    path2 = '/data/sample_2.csv'

    pdf1 = pd.read_csv(current_dir + path1, dtype=dtypes)
    pdf2 = pd.read_csv(current_dir + path2, dtype=dtypes)

    vdf1 = vaex.from_pandas(pdf1)
    vdf2 = vaex.from_pandas(pdf2)

    magic_value = 1234
    # Verify the Int64 type from pandas is read in vaex correctly
    if 'age' in dtypes:
        assert vdf1.age.fillmissing(magic_value).tolist() == pdf1.age.fillna(magic_value).tolist()

    output_path1 = str(tmpdir.join('sample_1.hdf5'))
    output_path2 = str(tmpdir.join('sample_2.hdf5'))

    vdf1.export_hdf5(output_path1)
    vdf2.export_hdf5(output_path2)

    # Concat and output the final single hdf5 file
    df = vaex.open_many([output_path1, output_path2])
    output_path_final = str(tmpdir.join('sample_final.hdf5'))
    df.export_hdf5(output_path_final)

    df_verify = vaex.open(output_path_final)
    assert len(df) == len(df_verify)
    assert df['name'].tolist() == df_verify['name'].tolist()
    assert df['age'].fillna(magic_value).tolist() == df_verify['age'].fillna(magic_value).tolist()

def test_export_csv(df_local, tmpdir):
    df = df_local
    path = str(tmpdir.join('test.csv'))
    df.export_csv(path, index=False)

    assert '123456' in vaex.open(path)


@pytest.mark.parametrize("dtypes", [{}, {'name': np.object, 'age': 'Int64', 'weight': np.float}])
def test_export_generates_same_hdf5_shasum(tmpdir, dtypes):
    current_dir = os.path.dirname(__file__)

    path1 = '/data/sample_1.csv'

    pdf1 = pd.read_csv(current_dir + path1, dtype=dtypes)
    vdf1 = vaex.from_pandas(pdf1)
    output_path1 = str(tmpdir.join('sample_1.hdf5'))
    vdf1.export_hdf5(output_path1)

    shasum1 = hashlib.sha1()
    with open(output_path1, 'rb') as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            shasum1.update(data)

    vdf2 = vaex.from_pandas(pdf1)
    output_path2 = str(tmpdir.join('sample_2.hdf5'))
    vdf2.export_hdf5(output_path2)

    shasum2 = hashlib.sha1()
    with open(output_path2, 'rb') as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            shasum2.update(data)

    assert shasum1.hexdigest() == shasum2.hexdigest()


def test_export_json(tmpdir, df_filtered):
    df = df_filtered
    path = tmpdir / 'test.json'
    df.export_json(path)
    df2 = vaex.from_json(path, orient='records')
    # for column in df.get_column_names():
    for column in ['x', 'name']:
        assert df[column].tolist() == df2[column].tolist()


def test_export_large_string(tmpdir):
    s = pa.array(["Hi", "there"], type=pa.large_string())
    df = vaex.from_arrays(s=s)
    df.export_arrow(tmpdir / "test.arrow")


def test_export_csv_badnames(tmpdir):
    data_path = os.path.join(os.path.dirname(__file__), "data/badname.csv")
    df = vaex.open(data_path)

    df = df.as_arrow()

    output_path = str(tmpdir.join('test.csv'))
    df.export_csv(output_path)

    df2 = vaex.open(output_path)
    assert df.get_column_names() == df2.get_column_names()
    assert df.shape == df2.shape


def test_export_hdf5_missing_values(tmpdir):
    df = vaex.from_dict({'x': pa.array([1, None, 5, None, 10]),
                          'y': pa.array([1.1, None, 5.5, None, 10.10]),
                          'z': ['Yes', None, 'No', None, 'Maybe']})
    export_path = str(tmpdir.join('tmp.hdf5'))
    df.export_hdf5(export_path)
    df2 = vaex.open(export_path)
    assert df2.shape == (5, 3)
    assert df2.x.tolist() == [1, None, 5, None, 10]
    assert df2.y.tolist() == [1.1, None, 5.5, None, 10.10]
    assert df2.z.tolist() == ['Yes', None, 'No', None, 'Maybe']
