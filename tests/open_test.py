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


def _cleanup_generated_files():
    for hdf5_file in glob.glob(os.path.join(path, 'data', '*.yaml')):
        os.remove(hdf5_file)
    for hdf5_file in glob.glob(os.path.join(path, 'data', 'output', '*.yaml')):
        os.remove(hdf5_file)
    for hdf5_file in glob.glob(os.path.join(path, 'data', 'small*.hdf5')):
        os.remove(hdf5_file)
    for hdf5_file in glob.glob(os.path.join(path, 'data', 'output', '*.hdf5')):
        os.remove(hdf5_file)
