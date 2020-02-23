import glob
import tempfile
import os
import pytest
import sys

import vaex

path = os.path.dirname(__file__)

if sys.platform.startswith("win"):
    pytest.skip("skipping windows, since it has issues re-opening files", allow_module_level=True)


def test_open():
    with pytest.raises(IOError):
        vaex.open('doesnotexist')

    csv2 = os.path.join(path, 'data', 'small2.csv')
    h51 = os.path.join(path, 'data', 'small2.csv.hdf5')
    h52 = os.path.join(path, 'data', 'small3.csv.hdf5')
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
    vaex.open(os.path.join(path, 'data', 'small*.csv'), convert=True, chunk_size=2)
    assert os.path.exists(target)
    _cleanup_generated_files()

    # convert can also be a path
    target = os.path.join(path, 'data', 'convert.hdf5')
    vaex.open(os.path.join(path, 'data', 'small*.csv'), convert=target)
    assert os.path.exists(target)
    assert os.path.exists(h51)
    assert os.path.exists(h52)
    vaex.open(os.path.join(path, 'data', 'small?.csv.hdf5'))
    _cleanup_generated_files()

    target = os.path.join(path, 'data', 'custom.hdf5')
    vaex.open(os.path.join(path, 'data', 'small*.csv'), convert=target)
    assert os.path.exists(h51)
    assert os.path.exists(h52)
    assert os.path.exists(target)
    _cleanup_generated_files()

    # be nice to users when converting from unsupported format
    with tempfile.NamedTemporaryFile(suffix='json') as in_f:
        in_f.write(b'[]')
        in_f.flush()
        with pytest.raises(IOError) as exc:
            vaex.open(in_f.name, convert=target)
        assert 'format supported' in str(exc.value)
    assert not os.path.exists(target)


def _cleanup_generated_files():
    for hdf5_file in glob.glob(os.path.join(path, 'data', '*.hdf5')):
        os.remove(hdf5_file)
