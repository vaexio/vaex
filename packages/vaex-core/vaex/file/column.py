import ctypes
fcntl = None
try:
    import fcntl
except:
    pass
import threading
from ctypes import *
import os
import ast

import numpy as np
import vaex.column


#libc = cdll.LoadLibrary("libc.dylib")


def getsizeof(ar):
    return ar.nbytes


USE_CACHE = ast.literal_eval(os.environ.get('VAEX_USE_COLUMN_FILE_CACHE', 'False'))
GB = 1024**3
# in GB
CACHE_SIZE =  ast.literal_eval(os.environ.get('VAEX_COLUMN_FILE_CACHE_SIZE', '10'))
if USE_CACHE:
    from cachetools import LRUCache
    cache = LRUCache(maxsize=CACHE_SIZE*GB, getsizeof=getsizeof)
cache_lock = threading.RLock()
F_NOCACHE = 48


class ColumnFile(vaex.column.Column):
    def __init__(self, file, byte_offset, length, dtype, write=False, path=None):
        self.path = path or file.name
        self.file = file
        self.tls = threading.local()
        # keep a record of all duplicate file handles to we can close them
        self.file_handles = []
        self.tls.file = vaex.file.dup(file)
        self.file_handles.append(file)
        self.native = False
        # if hasattr(self.file, 'fileno') and osname
        #     fcntl.fcntl(self.file.fileno(), F_NOCACHE, 1)
            # self.native = True
        #libc.fcntl(self.file.fileno(), fcntl.F_NOCACHE, 1)
        #libc.fcntl(c_int(self.file.fileno()), c_int(fcntl.F_NOCACHE), c_int(1))
        self.byte_offset = byte_offset
        self.length = length
        self.dtype = np.dtype(dtype)
        self.shape = (length,)
        self.write = write

    def __del__(self):
        for f in self.file_handles:
            f.close()

    def __len__(self):
        return self.length

    def trim(self, i1, i2):
        itemsize = self.dtype.itemsize
        byte_offset = self.byte_offset + i1 * itemsize
        length = i2 - i1
        return ColumnFile(self.file, byte_offset, length, self.dtype, self.write, path=self.path)

    def __setitem__(self, slice, values):
        assert self.write, "trying to write to non-writable column"
        start, stop, step = slice.start, slice.stop, slice.step
        start = start or 0
        stop = stop or len(self)
        assert step in [None, 1]
        itemsize = self.dtype.itemsize
        N = stop - start

        page_size = 1024*4
        page_mask = page_size-1
        # TODO: check, it seems tobytes is slow
        ar_bytes = values.tobytes()
        assert len(ar_bytes) == N * itemsize
        # we want to read at a page boundary
        offset = self.byte_offset + start * itemsize

        # make sure we write at a multiple of the page size, if the content is smaller than
        if N * itemsize >= page_size:
            offset_optimal = math.ceil(offset/page_size) * page_size
            padding = offset_optimal - offset
            ar_ptr_pad = ctypes.c_char_p(ar_bytes[:padding])
            ar_ptr_opt = ctypes.c_char_p(ar_bytes[padding:])
            if offset != offset_optimal:
                bytes_write = libc.pwrite(ctypes.c_int32(self.file.fileno()), ar_ptr_pad, ctypes.c_uint64(padding), ctypes.c_uint64(offset))
                if (bytes_write) != padding:
                    raise IOError('write error: expected %d bytes, wrote %d, padding: %d' % (N * itemsize, bytes_write, padding))
            bytes_write = libc.pwrite(ctypes.c_int32(self.file.fileno()), ar_ptr_opt, ctypes.c_uint64(N * itemsize - padding), ctypes.c_uint64(offset_optimal))
            if (bytes_write) != N * itemsize - padding:
                raise IOError('write error: expected %d bytes, wrote %d, padding: %d' % (N * itemsize, bytes_write, padding))
        else:
            ar_ptr = ctypes.c_char_p(ar_bytes)
            bytes_write = libc.pwrite(ctypes.c_int32(self.file.fileno()), ar_ptr, ctypes.c_uint64(N * itemsize), ctypes.c_uint64(offset))
            if (bytes_write) != N * itemsize:
                raise IOError('write error: expected %d bytes, wrote %d, padding: %d' % (N * itemsize, bytes_write, padding))


    def __getitem__(self, slice):
        start, stop, step = slice.start, slice.stop, slice.step
        start = start or 0
        stop = stop or len(self)
        while start < 0:
            start += len(self)
        while stop < 0:
            stop += len(self)
        assert step in [None, 1]
        itemsize = self.dtype.itemsize
        items = stop - start
        key = (self.path, self.byte_offset, start, stop)
        if USE_CACHE:
            with cache_lock:
                ar = cache.get(key)
        else:
            ar = None
        if ar is None:
            N = items
            offset = self.byte_offset + start * itemsize
            if self.native:
                # fcntl.fcntl(self.file.fileno(), fcntl.F_NOCACHE, 1)
                page_size = 1024*4
                page_mask = page_size-1
                ar_bytes = np.zeros(N*itemsize + page_size, np.uint8)
                ar_ptr = ar_bytes.ctypes.data_as(ctypes.POINTER(ctypes.c_byte))
                # we want to read at a page boundary
                offset_optimal = offset & ~page_mask
                padding = offset - offset_optimal
                bytes_read = libc.pread(ctypes.c_int32(self.file.fileno()), ar_ptr, ctypes.c_uint64(N * itemsize + padding), ctypes.c_uint64(offset_optimal))
                if (bytes_read-padding) != N * itemsize:
                    raise IOError('read error: expected %d bytes, read %d, padding: %d' % (N * itemsize, bytes_read, padding))
                ar = np.frombuffer(ar_bytes, self.dtype, offset=padding, count=N)
            else:
                byte_length = items*itemsize
                offset = self.byte_offset + start * itemsize
                # Quick and safe way to get the thread local file handle:
                file = getattr(self.tls, 'file', None)
                if file is None:
                    with cache_lock:
                        file = getattr(self.tls, 'file', None)
                        if file is None:
                            file = self.tls.file = vaex.file.dup(self.file)
                            self.file_handles.append(file)
                # this is the fast path, that avoids a memory copy but gets a view on the underlying data
                # cache.py:CachedFile supports this
                if hasattr(file, '_as_numpy'):
                    ar = file._as_numpy(offset, byte_length, self.dtype)
                else:
                    # Traditinal file object go this slower route
                    # and they need per thread file object since the location (seek)
                    # is in the state of the file object

                    file.seek(offset)
                    data = file.read(byte_length)
                    ar = np.frombuffer(data, self.dtype, count=N)
            if USE_CACHE:
                with cache_lock:
                    cache[key] = ar
        return ar
