from pathlib import Path
import io
import threading
import pytest
import fsspec.implementations.memory

import vaex.file
import vaex.file.asyncio
import pyarrow as pa
import time


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


def test_fingerprint(tmpdir):
    assert vaex.file.fingerprint(__file__) == vaex.file.fingerprint(__file__)
    assert vaex.file.fingerprint('s3://vaex/testing/xys.hdf5?anonymous=true') != vaex.file.fingerprint('s3://vaex/testing/xys-masked.hdf5?anonymous=true')
    assert vaex.file.fingerprint('s3://vaex/testing/xys.hdf5?anonymous=true') == vaex.file.fingerprint('s3://vaex/testing/xys.hdf5?anonymous=true')
    assert vaex.file.fingerprint('s3://vaex/testing/xys.hdf5', fs_options={'anonymous': True}) == vaex.file.fingerprint('s3://vaex/testing/xys.hdf5?anonymous=true')


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


def test_fsspec(df_trimmed):
    df_trimmed = df_trimmed.drop('obj')
    fs = fsspec.implementations.memory.MemoryFileSystem()
    df_trimmed.export('test.arrow', fs=fs)
    assert ('test.arrow' in fs.store) or ('/test.arrow' in fs.store)
    # with vaex.file.open('test.txt', fs=fs) as f:
    #     f.write('vaex')
    df_trimmed['test'] = df_trimmed.x + 10
    df_trimmed.state_write('test.json', fs=fs)
    df = vaex.open('test.arrow', fs=fs)
    df.state_load('test.json', fs=fs)
    assert df.test.tolist() == df_trimmed.test.tolist()

    import pyarrow

    fs = pyarrow.fs.FSSpecHandler(fs)
    fs = vaex.file.cache.FileSystemHandlerCached(fs, scheme="memory")
    df_trimmed.state_write("test.json", fs=fs)


def test_stream_sync():
    f = vaex.file.asyncio.WriteStream()
    def writer():
        with f:
            for i in range(10):
                f.write(chr(ord('a') + i).encode('ascii'))
    threading.Thread(target=writer, daemon=True).start()
    chunks = b''.join(list(f))
    assert chunks == b'abcdefghij'

    # with error
    f = vaex.file.asyncio.WriteStream()
    def writer():
        with f:
            for i in range(10):
                f.write(chr(ord('a') + i).encode('ascii'))
                if i == 4:
                    raise ValueError('test')
                # time.sleep()
    threading.Thread(target=writer, daemon=True).start()
    with pytest.raises(ValueError, match='test'):
        list(f)


@pytest.mark.asyncio
async def test_stream_async():
    f = vaex.file.asyncio.WriteStream()
    def writer():
        with f:
            for i in range(10):
                f.write(chr(ord('a') + i).encode('ascii'))
    threading.Thread(target=writer, daemon=True).start()
    chunks = b''.join(list([k async for k in f]))
    assert chunks == b'abcdefghij'

    # with error
    f = vaex.file.asyncio.WriteStream()
    def writer():
        with f:
            for i in range(10):
                f.write(chr(ord('a') + i).encode('ascii'))
                if i == 4:
                    raise ValueError('test')
    threading.Thread(target=writer, daemon=True).start()
    with pytest.raises(ValueError, match='test'):
        list(f)


def test_write_arrow_in_stream(tmpdir):
    df = vaex.from_arrays(x=[1,2])
    f = vaex.file.asyncio.WriteStream()
    def writer():
        with f:
            df.export_arrow(f)
    threading.Thread(target=writer, daemon=True).start()
    path = tmpdir / '1.arrow'
    with open(path, 'wb') as fdisk:
        for chunk in f:
            fdisk.write(chunk)
    df_disk = vaex.open(path)
    assert df_disk.x.tolist() == [1, 2]


def test_not_close_open_file():
    df = vaex.from_arrays(x=[1,2])
    with io.BytesIO() as f:
        df.export_arrow(f)
        assert not f.closed
    assert f.closed
