from common import *
import os
import tempfile
import pandas as pd
import platform


@pytest.mark.skipif(platform.system().lower() == 'windows', reason="access violation?")
@pytest.mark.parametrize("filename", ["test.hdf5", "test.arrow", "test.parquet", "test.csv"])
def test_export_empty_string(tmpdir, filename):
    path = str(tmpdir.join(filename))
    s = np.array(["", ""])
    df = vaex.from_arrays(s=s)
    df.export(path)
    df = vaex.open(path)
    repr(df)

def test_export(ds_local, tmpdir):
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
        ds.export(path)
        df = vaex.open(path)

def test_export_open_hdf5(ds_local):
    ds = ds_local
    ds = ds.drop(ds.obj)
    filename = tempfile.mktemp(suffix='.hdf5')
    ds.export(filename)
    ds_opened = vaex.open(filename)
    assert list(ds) == list(ds_opened)

def test_export_open_csv(ds_local, tmpdir):
    df = ds_local
    path = str(tmpdir.join('test.csv'))
    df.export_csv(path, chunk_size=3, virtual=True)
    df_opened = vaex.from_csv(path)
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
    assert df['age'].fillnan(magic_value).tolist() == df_verify['age'].fillnan(magic_value).tolist()

def test_export_csv(df_local, tmpdir):
    df = df_local
    path = str(tmpdir.join('test.csv'))
    df.export_csv(path)

    assert '123456' in vaex.open(path)
