import vaex.file.cache
import os


def test_hdf5(tmpdir):
    path = str(tmpdir.join('test.hdf5'))
    df = vaex.from_arrays(x=[1,2], y=[3,4])
    df.export(path)
    fake_path = 's3://vaex/test.hdf5?profile_name=foo'
    length = os.stat(path).st_size
    with open(path, 'rb') as fp:
        fp.seek(0, 2)
        cache = vaex.file.cache.CachedFile(vaex.file.dup(fp), fake_path, str(tmpdir), block_size=2)
        df = vaex.hdf5.dataset.Hdf5MemoryMapped(cache)
        assert df.x.tolist() == [1, 2]
        assert df.y.tolist() == [3, 4]
        assert df.sum('x') == 3
        cache = vaex.file.dup(cache)
        cache.seek(0)
        data_cache = cache.read()
        # del df
    with open(path, 'rb') as f:
        length2 = f.seek(0, 2)
        length2 = f.tell()
        assert length == length2
        f.seek(0, 0)
        data = f.read()
        assert data_cache == data


def test_cache(tmpdir):
    path = str(tmpdir.join('test.txt'))
    data = b'1234567890'
    with open(path, 'wb') as f:
        f.write(data)
    fake_path = 's3://vaex/test.hdf5?profile_name=foo'
    with open(path, 'rb') as fp:
        cache = vaex.file.cache.CachedFile(fp, fake_path, str(tmpdir), block_size=2)
        assert cache.tell() == 0
        assert cache.read(1) == b'1'
        assert cache.block_reads == 1
        assert cache.reads == 1
        assert cache.data_file.data[0] == ord('1')
        assert cache.data_file.data[1] == ord('2')
        assert cache.data_file.data[2] == 0
        
        assert cache.mask_file.data[0] == 1
        assert cache.mask_file.data[1] == 0
        assert cache.mask_file.data[2] == 0

        assert cache.tell() == 1
        assert cache.read(1) == b'2'
        assert cache.block_reads == 1
        assert cache.reads == 1
        assert cache.data_file.data[0] == ord('1')
        assert cache.data_file.data[1] == ord('2')
        assert cache.data_file.data[2] == 0
        
        assert cache.mask_file.data[0] == 1
        assert cache.mask_file.data[1] == 0
        assert cache.mask_file.data[2] == 0


        assert cache.tell() == 2
        cache.seek(4)
        assert cache.read(1) == b'5'
        assert cache.block_reads == 2
        assert cache.reads == 2
        assert cache.data_file.data[0] == ord('1')
        assert cache.data_file.data[1] == ord('2')
        assert cache.data_file.data[2] == 0
        assert cache.data_file.data[3] == 0
        assert cache.data_file.data[4] == ord('5')
        assert cache.data_file.data[5] == ord('6')
        assert cache.data_file.data[6] == 0
        
        assert cache.mask_file.data[0] == 1
        assert cache.mask_file.data[1] == 0
        assert cache.mask_file.data[2] == 1
        assert cache.mask_file.data[3] == 0


        assert cache.tell() == 5
        cache.seek(8)
        assert cache.read(2) == b'90'
        assert cache.block_reads == 3
        assert cache.reads == 3
        assert cache.data_file.data[0] == ord('1')
        assert cache.data_file.data[1] == ord('2')
        assert cache.data_file.data[2] == 0
        assert cache.data_file.data[3] == 0
        assert cache.data_file.data[4] == ord('5')
        assert cache.data_file.data[5] == ord('6')
        assert cache.data_file.data[6] == 0
        assert cache.data_file.data[7] == 0
        assert cache.data_file.data[8] == ord('9')
        assert cache.data_file.data[9] == ord('0')
        
        assert cache.mask_file.data[0] == 1
        assert cache.mask_file.data[1] == 0
        assert cache.mask_file.data[2] == 1
        assert cache.mask_file.data[3] == 0
        assert cache.mask_file.data[4] == 1

        cache.seek(1)
        assert cache.read(8) == b'23456789'
        assert cache.block_reads == 5
        assert cache.reads == 5
        assert cache.data_file.data[0] == ord('1')
        assert cache.data_file.data[1] == ord('2')
        assert cache.data_file.data[2] == ord('3')
        assert cache.data_file.data[3] == ord('4')
        assert cache.data_file.data[4] == ord('5')
        assert cache.data_file.data[5] == ord('6')
        assert cache.data_file.data[6] == ord('7')
        assert cache.data_file.data[7] == ord('8')
        assert cache.data_file.data[8] == ord('9')
        assert cache.data_file.data[9] == ord('0')
        
        assert cache.mask_file.data[0] == 1
        assert cache.mask_file.data[1] == 1
        assert cache.mask_file.data[2] == 1
        assert cache.mask_file.data[3] == 1
        assert cache.mask_file.data[4] == 1
