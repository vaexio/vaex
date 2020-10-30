from pathlib import Path
import pytest

import vaex.file
import pyarrow as pa


def test_open_s3():
    with vaex.file.open('s3://vaex/testing/xys.hdf5') as f:
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


def test_memory_mappable():
    assert not vaex.file.memory_mappable('s3://vaex/testing/xys.hdf5')
    assert vaex.file.memory_mappable('/vaex/testing/xys.hdf5')


def test_is_file_object(tmpdir):
    path = tmpdir / 'test.dat'
    with open(path, 'wb') as f:
        path.write(b'vaex')
    assert vaex.file.is_file_object(vaex.file.open('s3://vaex/testing/xys.hdf5'))
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


# def test_open_with_file_handle(tmpdir):
#     path = tmpdir / 'test.dat'
#     with open(path, 'wb') as f:
#         path.write(b'vaex')
#     with vaex.file.open(path) as f:
#         with vaex.file.open(f) as f:
#             assert f.read() == b'vaex'
#     # with vaex.file.open('s3://vaex/testing/xys.hdf5') as fs3:
#     #     with vaex.file.open(fs3) as f:
#     #         signature = f.read(4)
#     #         assert signature == b"\x89\x48\x44\x46"
