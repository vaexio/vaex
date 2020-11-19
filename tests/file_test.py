from pathlib import Path
import pytest

import vaex.file
import pyarrow as pa


def test_parse():
    fs, path = vaex.file.parse('s3://vaex/testing/xys.hdf5?anonymous=true')
    assert fs is not None
    fs, path = vaex.file.parse('fsspec+s3://vaex/testing/xys.hdf5?anonymous=true')
    assert fs is not None
    fs, path = vaex.file.parse('/vaex/testing/xys.hdf5')
    assert fs is None

    s3list = ['s3://vaex/testing/xys.hdf5?anonymous=true']
    fs, path = vaex.file.parse(s3list)
    assert fs is not None
    assert path[0] in s3list[0]

    locallist = ['/vaex/testing/xys.hdf5?anonymous=true']
    fs, path = vaex.file.parse(locallist)
    assert fs is None
    assert path[0] in locallist[0]


def test_open_s3():
    with vaex.file.open('s3://vaex/testing/xys.hdf5?anonymous=true') as f:
        signature = f.read(4)
        assert signature == b"\x89\x48\x44\x46"


def test_stringify(tmpdir):
    assert vaex.file.stringyfy('bla') == 'bla'
    assert vaex.file.stringyfy(Path('bla')) == 'bla'
    path = (tmpdir / 'test.txt')
    with path.open('wb') as f:
        assert vaex.file.stringyfy(path) == str(path)
    with pa.OSFile(str(path), 'wb') as f:
        assert vaex.file.stringyfy(f) is None


def test_stringify(tmpdir):
    assert vaex.file.stringyfy('bla') == 'bla'
    assert vaex.file.stringyfy(Path('bla')) == 'bla'
    path = (tmpdir / 'test.txt')
    with path.open('wb') as f:
        assert vaex.file.stringyfy(path) == str(path)
    with pytest.raises(ValueError):
        with pa.OSFile(str(path), 'wb') as f:
            assert vaex.file.stringyfy(f) is None


def test_split_scheme(tmpdir):
    assert vaex.file.split_scheme('s3://vaex/testing/xys.hdf5') == ('s3', 'vaex/testing/xys.hdf5')
    assert vaex.file.split_scheme('/vaex/testing/xys.hdf5') == (None, '/vaex/testing/xys.hdf5')


def test_split_options(tmpdir):
    assert vaex.file.split_options('s3://vaex/testing/xys.hdf5?a=1&b=2') == ('s3://vaex/testing/xys.hdf5', {'a': '1', 'b': '2'})
    assert vaex.file.split_options('s3://vaex/testing/*.hdf5?a=1&b=2') == ('s3://vaex/testing/*.hdf5', {'a': '1', 'b': '2'})
    assert vaex.file.split_options('s3://vaex/testing/??.hdf5?a=1&b=2') == ('s3://vaex/testing/??.hdf5', {'a': '1', 'b': '2'})


def test_tokenize(tmpdir):
    assert vaex.file.tokenize(__file__) == vaex.file.tokenize(__file__)
    assert vaex.file.tokenize('s3://vaex/testing/xys.hdf5?anonymous=true') != vaex.file.tokenize('s3://vaex/testing/xys-masked.hdf5?anonymous=true')
    assert vaex.file.tokenize('s3://vaex/testing/xys.hdf5?anonymous=true') == vaex.file.tokenize('s3://vaex/testing/xys.hdf5?anonymous=true')
    assert vaex.file.tokenize('s3://vaex/testing/xys.hdf5', fs_options={'anonymous': True}) == vaex.file.tokenize('s3://vaex/testing/xys.hdf5?anonymous=true')


def test_memory_mappable():
    assert not vaex.file.memory_mappable('s3://vaex/testing/xys.hdf5')
    assert vaex.file.memory_mappable('/vaex/testing/xys.hdf5')


def test_is_file_object(tmpdir):
    path = tmpdir / 'test.dat'
    with open(path, 'wb') as f:
        path.write(b'vaex')
    assert vaex.file.is_file_object(vaex.file.open('s3://vaex/testing/xys.hdf5?anonymous=true'))
    assert vaex.file.is_file_object(vaex.file.open(path))


def test_open_local(tmpdir):
    path = tmpdir / 'test.dat'
    with open(path, 'wb') as f:
        path.write(b'vaex')
    with vaex.file.open(path) as f:
        assert f.read() == b'vaex'


def test_open_non_existing(tmpdir):
    path = tmpdir / 'test2.dat'
    with pytest.raises(IOError) as exc:
        f = open(path)
