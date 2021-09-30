from concurrent.futures import ThreadPoolExecutor
import glob
import tempfile
import os
import pytest
import sys

import vaex

path = os.path.dirname(__file__)

if sys.platform.startswith("win"):
    pytest.skip("skipping windows, since it has issues re-opening files", allow_module_level=True)

csv2 = os.path.join(path, 'data', 'small2.csv')
csv3 = os.path.join(path, 'data', 'small2.csv')
h51 = os.path.join(path, 'data', 'small2.csv.hdf5')
h52 = os.path.join(path, 'data', 'small3.csv.hdf5')
target_path = os.path.join(path, 'data', 'output')


def test_open_convert_kwargs():
    csv2 = os.path.join(path, 'data', 'smæll2.csv')
    df = vaex.open(csv2, convert=True)
    assert df.x.dtype == 'integer'
    mtime = os.path.getmtime(csv2 + ".hdf5")
    # calling it a second time should reuse the hdf5 file
    df = vaex.open(csv2, convert=True)
    assert mtime == os.path.getmtime(csv2 + ".hdf5")

    # different argument should result in new conversion
    df = vaex.open(csv2, convert=True, dtype={'x': 'float64'})
    assert mtime != os.path.getmtime(csv2 + ".hdf5")
    assert df.x.dtype == 'float64'


def test_open_convert_non_hdf5():
    csv2 = os.path.join(path, 'data', 'smæll2.csv')
    df = vaex.open(csv2, convert='smæll2.parquet')


def test_open_convert_multithreaded():
    def do(i):
        fn = os.path.join(path, 'data', 'smæll2.csv')
        df = vaex.open(csv2, convert='smæll2.hdf5')
        df = vaex.open(csv2, convert='smæll2.parquet')
        df = vaex.open(csv2, convert='smæll2.arrow')
    for i in range(10):
        with ThreadPoolExecutor(4) as tpe:
            values = list(tpe.map(do, range(5)))
            assert values == [None] * 5
        for ext in ['hdf5', 'parquet', 'arrow']:
            os.remove(os.path.join(f'smæll2.{ext}'))


def test_open_convert_explicit_path():
    # explicit path, also a truthy
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    target = os.path.join(target_path, 'small2_via_chunks.hdf5')
    vaex.open(csv2, convert=target, chunk_size=1)
    assert os.path.exists(target)
    _cleanup_generated_files()


def test_open():
    with pytest.raises(IOError):
        vaex.open('doesnotexist')

    target = os.path.join(path, 'data', 'small2.csv.hdf5')
    vaex.open(csv2, convert=True)
    assert os.path.exists(target)
    _cleanup_generated_files()

    # convert and open several CSVs
    target = os.path.join(path, 'data', 'small2.csv_and_1_more.hdf5')
    vaex.open(os.path.join(path, 'data', 'small*.csv'), convert=True)
    assert os.path.exists(target)
    assert os.path.exists(h51)
    assert os.path.exists(h52)
    vaex.open(os.path.join(path, 'data', 'small?.csv.hdf5'))
    _cleanup_generated_files()

    # convert and open several "big" CSVs
    vaex.open(os.path.join(path, 'data', 'small*.csv'), convert=True)
    assert os.path.exists(target)
    _cleanup_generated_files()

    # convert can also be a path
    target = os.path.join(target_path, 'convert.hdf5')
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    vaex.open(os.path.join(path, 'data', 'small*.csv'), convert=target)
    assert os.path.exists(target)
    assert os.path.exists(h51)
    assert os.path.exists(h52)
    vaex.open(os.path.join(path, 'data', 'small?.csv.hdf5'))
    _cleanup_generated_files()

    target = os.path.join(target_path, 'custom.hdf5')
    vaex.open(os.path.join(path, 'data', 'small*.csv'), convert=target)
    assert os.path.exists(h51)
    assert os.path.exists(h52)
    assert os.path.exists(target)
    _cleanup_generated_files()

    # be nice to users when converting from unsupported format
    with tempfile.NamedTemporaryFile(suffix='.json') as in_f:
        in_f.write(b'[]')
        in_f.flush()
        with pytest.raises(IOError) as exc:
            vaex.open(in_f.name, convert=target)
        assert 'Cannot open' in str(exc.value)
    assert not os.path.exists(target)


def test_open_list():
    df2 = vaex.open(csv2)
    df3 = vaex.open(csv3)
    df = vaex.open([csv2, csv3])
    assert df.x.tolist() == (df2.x.tolist() + df3.x.tolist())


def test_open_nonstandard_extension(tmpdir):
    df = vaex.from_scalars(x=1, s='Hello')
    df.export_hdf5(tmpdir / 'this_is_hdf5.xyz')
    df = vaex.open(tmpdir / 'this_is_hdf5.xyz')
    assert df.x.tolist() == [1]
    assert df.s.tolist() == ['Hello']


def test_open_export_hdf5_groups(tmpdir):
    df1 = vaex.from_arrays(s=['Groningen', 'Ohrid'])
    df2 = vaex.from_arrays(z=[10, -10])
    df3 = vaex.from_arrays(first_name=['Reggie', 'Michael'], last_name=['Miller', 'Jordan'])

    df1.export_hdf5(tmpdir / 'hdf5_with_groups.hdf5', mode='a', group='my/table/cities')
    df2.export_hdf5(tmpdir / 'hdf5_with_groups.hdf5', mode='a', group='my/table/coords')
    df3.export_hdf5(tmpdir / 'hdf5_with_groups.hdf5', mode='a', group='my/table/players')

    df1_open = vaex.open(tmpdir / 'hdf5_with_groups.hdf5', group='my/table/cities')
    assert df1_open.s.tolist() == ['Groningen', 'Ohrid']
    df2_open = vaex.open(tmpdir / 'hdf5_with_groups.hdf5', group='my/table/coords')
    assert df2_open.z.tolist() == [10, -10]
    df3_open = vaex.open(tmpdir / 'hdf5_with_groups.hdf5', group='my/table/players')
    assert df3_open.first_name.tolist() == ['Reggie', 'Michael']
    assert df3_open.last_name.tolist() == ['Miller', 'Jordan']


def _cleanup_generated_files():
    for hdf5_file in glob.glob(os.path.join(path, 'data', '*.yaml')):
        os.remove(hdf5_file)
    for hdf5_file in glob.glob(os.path.join(path, 'data', 'output', '*.yaml')):
        os.remove(hdf5_file)
    for hdf5_file in glob.glob(os.path.join(path, 'data', 's*.hdf5')):
        os.remove(hdf5_file)
    for hdf5_file in glob.glob(os.path.join(path, 'data', 'output', '*.hdf5')):
        os.remove(hdf5_file)
    for hdf5_file in glob.glob(os.path.join(path, '..', 'smæll2*')):
        os.remove(hdf5_file)
