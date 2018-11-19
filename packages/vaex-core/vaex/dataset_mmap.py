__author__ = 'maartenbreddels'
import os
import mmap
import math
import itertools
import functools
import collections
import logging
import numpy as np
import numpy.ma
import vaex
import astropy.table
import astropy.units
from vaex.utils import ensure_string
import astropy.io.fits as fits
import re
import six
from vaex.dataset import DatasetLocal, DatasetArrays
import vaex.dataset
import vaex.file
from vaex.expression import Expression
import struct
import fcntl

logger = logging.getLogger("vaex.file")

dataset_type_map = {}

# h5py doesn't want to build at readthedocs
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
try:
    import h5py
except:
    if not on_rtd:
        raise

osname = vaex.utils.osname
no_mmap = os.environ.get('VAEX_NO_MMAP', False)

if no_mmap:
    from ctypes import *
    libc = cdll.LoadLibrary("libc.dylib")
    import ctypes
    import io

    from cachetools import LRUCache
    import threading
    GB = 1024**3
    def getsizeof(ar):
        return ar.nbytes
    cache = LRUCache(maxsize=10*GB, getsizeof=getsizeof)
    cache_lock = threading.RLock()
    F_NOCACHE = 48


    class ColumnReader(vaex.dataset.Column):
        def __init__(self, dataset, file, byte_offset, length, dtype):
            self.dataset = dataset
            self.file = file
            fcntl.fcntl(self.file.fileno(), F_NOCACHE, 1)
            #libc.fcntl(self.file.fileno(), fcntl.F_NOCACHE, 1)
            #libc.fcntl(c_int(self.file.fileno()), c_int(fcntl.F_NOCACHE), c_int(1))
            self.byte_offset = byte_offset
            self.length = length
            self.dtype = np.dtype(dtype)
            self.shape = (length,)

        def __len__(self):
            return self.length# // self.dtype.itemsize

        # def trim(self, i1, i2):
        #     return ColumnIndexed(self.dataset, self.indices[i1:i2], self.name)

        def trim(self, i1, i2):
            itemsize = self.dtype.itemsize
            byte_offset = self.byte_offset * i1 * itemsize
            length = i2 - i1
            return ColumnReader(self.dataset, io.FileIO(self.file.fileno(), 'rb'), byte_offset, length, self.dtype)

        def __setitem__(self, slice, values):
            start, stop, step = slice.start, slice.stop, slice.step
            start = start or 0
            stop = stop or len(self)
            assert step in [None, 1]
            itemsize = self.dtype.itemsize
            N = stop - start

            page_size = 1024*4
            page_mask = page_size-1
            ar_bytes = values.tobytes() #np.frombuffer(values.tobytes(), np.uint8)
            assert len(ar_bytes) == N * itemsize
            #print(ar_ptr, N, itemsize, start)
            # we want to read at a page boundary
            offset = self.byte_offset + start * itemsize
            

            #print(values[:2])
            # make sure we write at a multiple of the page size, if the content is smaller than
            if N * itemsize >= page_size:
                offset_optimal = math.ceil(offset/page_size) * page_size
                padding = offset_optimal - offset
                #ar_ptr_pad = ar_bytes[:padding].ctypes.data_as(ctypes.POINTER(ctypes.c_byte))
                #ar_ptr_opt = ar_bytes[padding:].ctypes.data_as(ctypes.POINTER(ctypes.c_byte))
                ar_ptr_pad = ctypes.c_char_p(ar_bytes[:padding])
                ar_ptr_opt = ctypes.c_char_p(ar_bytes[padding:])
                if offset != offset_optimal:
                    bytes_write = libc.pwrite(ctypes.c_int32(self.file.fileno()), ar_ptr_pad, ctypes.c_uint64(padding), ctypes.c_uint64(offset))
                    if (bytes_write) != padding:
                        raise IOError('write error: expected %d bytes, wrote %d, padding: %d' % (N * itemsize, bytes_write, padding))
                bytes_write = libc.pwrite(ctypes.c_int32(self.file.fileno()), ar_ptr_opt, ctypes.c_uint64(N * itemsize - padding), ctypes.c_uint64(offset_optimal))
                if (bytes_write) != N * itemsize - padding:
                    raise IOError('write error: expected %d bytes, wrote %d, padding: %d' % (N * itemsize, bytes_write, padding))
                #print(offset, offset_optimal, bytes_write, bytes_read+padding, padding, start, stop, N)
            else:
                #ar_ptr = ar_bytes.ctypes.data_as(ctypes.POINTER(ctypes.c_byte))
                ar_ptr = ctypes.c_char_p(ar_bytes)
                bytes_write = libc.pwrite(ctypes.c_int32(self.file.fileno()), ar_ptr, ctypes.c_uint64(N * itemsize), ctypes.c_uint64(offset))
                if (bytes_write) != N * itemsize:
                    raise IOError('write error: expected %d bytes, wrote %d, padding: %d' % (N * itemsize, bytes_write, padding))
            #ar = np.frombuffer(ar_bytes[padding:padding + N*itemsize], self.dtype)
            # ar = np.frombuffer(ar_bytes, self.dtype, offset=padding, count=N)


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
            #self.file.seek(self.byte_offset + start*itemsize)
            items = stop - start
            key = (self.dataset.path, self.byte_offset, start, stop)
            with cache_lock:
                ar = cache.get(key)
            if ar is None:
                fcntl.fcntl(self.file.fileno(), fcntl.F_NOCACHE, 1)
                #raw_data = self.file.read(items * itemsize)
                N = items
                page_size = 1024*4
                page_mask = page_size-1
                ar_bytes = np.zeros(N*itemsize + page_size, np.uint8)
                ar_ptr = ar_bytes.ctypes.data_as(ctypes.POINTER(ctypes.c_byte))
                #print(ar_ptr, N, itemsize, start)
                # we want to read at a page boundary
                offset = self.byte_offset + start * itemsize
                offset_optimal = offset & ~page_mask
                padding = offset - offset_optimal
                bytes_read = libc.pread(ctypes.c_int32(self.file.fileno()), ar_ptr, ctypes.c_uint64(N * itemsize + padding), ctypes.c_uint64(offset_optimal))
                #libc.lseek(ctypes.c_int32(self.file.fileno()), ctypes.c_uint64(offset)
                #bytes_read = libc.read(ctypes.c_int32(self.file.fileno()), ar_ptr, ctypes.c_uint64(N * itemsize))
                #print((bytes_read-padding), N * itemsize, N, itemsize, start, stop)
                if (bytes_read-padding) != N * itemsize:
                    raise IOError('read error: expected %d bytes, read %d, padding: %d' % (N * itemsize, bytes_read, padding))
                #ar = np.frombuffer(ar_bytes[padding:padding + N*itemsize], self.dtype)
                ar = np.frombuffer(ar_bytes, self.dtype, offset=padding, count=N)
                #print(bytes_read)
                #libc.pread(ctypes.c_int32(self.file.fileno()), ar_ptr,N * itemsize, self.byte_offset + start * itemsize)
                with cache_lock:
                    cache[key] = ar
            #else:
            #    print('.', end='')
            return ar
else:
    ColumnReader = False

class DatasetMemoryMapped(DatasetLocal):
    """Represents a dataset where the data is memory mapped for efficient reading"""

    # nommap is a hack to get in memory datasets working
    def __init__(self, filename, write=False, nommap=False, name=None):
        super(DatasetMemoryMapped, self).__init__(name=name or os.path.splitext(os.path.basename(filename))[0], path=os.path.abspath(filename) if filename is not None else None, column_names=[])
        self.filename = filename or "no file"
        self.write = write
        # self.name = name or os.path.splitext(os.path.basename(self.filename))[0]
        # self.path = os.path.abspath(filename) if filename is not None else None
        self.nommap = nommap
        if not nommap:
            self.file = open(self.filename, "rb+" if write else "rb")
            self.fileno = self.file.fileno()
            kwargs = {}
            if vaex.utils.osname == "windows":
                kwargs["access"] = mmap.ACCESS_READ | 0 if not write else mmap.ACCESS_WRITE
            else:
                kwargs["prot"] = mmap.PROT_READ | 0 if not write else mmap.PROT_WRITE
            self.mapping = mmap.mmap(self.fileno, 0, **kwargs)
            self.file_map = {filename: self.file}
            self.fileno_map = {filename: self.fileno}
            self.mapping_map = {filename: self.mapping}
        else:
            self.file_map = {}
            self.fileno_map = {}
            self.mapping_map = {}
        self._length_original = None
        # self._fraction_length = None
        self.nColumns = 0
        self.column_names = []
        self.rank1s = {}
        self.rank1names = []
        self.virtual_columns = collections.OrderedDict()

        self.axes = {}
        self.axis_names = []

        # these are replaced by variables
        # self.properties = {}
        # self.property_names = []

        self.current_slice = None
        self.fraction = 1.0

        self.selected_row_index = None
        self.selected_serie_index = 0
        self.row_selection_listeners = []
        self.serie_index_selection_listeners = []
        # self.mask_listeners = []

        self.all_columns = {}
        self.all_column_names = []
        self.global_links = {}

        self.offsets = {}
        self.strides = {}
        self.filenames = {}
        self.samp_id = None
        # self.variables = collections.OrderedDict()

    def close_files(self):
        for name, file in self.file_map.items():
            file.close()
        # on osx and linux this will give random bus errors (osx) or segfaults (linux)
        # on win32 however, we'll run out of file handles
        if vaex.utils.osname not in ["osx", "linux"]:
            for name, memmap in self.mapping_map.items():
                memmap.close()

    def has_snapshots(self):
        return len(self.rank1s) > 0

    def get_path(self):
        return self.path

    def addFile(self, filename, write=False):
        self.file_map[filename] = open(filename, "rb+" if write else "rb")
        self.fileno_map[filename] = self.file_map[filename].fileno()
        self.mapping_map[filename] = mmap.mmap(self.fileno_map[filename], 0, prot=mmap.PROT_READ | 0 if not write else mmap.PROT_WRITE)

    def selectSerieIndex(self, serie_index):
        self.selected_serie_index = serie_index
        for serie_index_selection_listener in self.serie_index_selection_listeners:
            serie_index_selection_listener(serie_index)
        self.signal_sequence_index_change.emit(self, serie_index)

    def matches_url(self, url):
        filename = url
        if filename.startswith("file:/"):
            filename = filename[5:]
        similar = os.path.splitext(os.path.abspath(self.filename))[0] == os.path.splitext(filename)[0]
        logger.info("matching urls: %r == %r == %r" % (os.path.splitext(self.filename)[0], os.path.splitext(filename)[0], similar))
        return similar

    def close(self):
        self.file.close()
        self.mapping.close()

    def addAxis(self, name, offset=None, length=None, dtype=np.float64, stride=1, filename=None):
        if filename is None:
            filename = self.filename
        mapping = self.mapping_map[filename]
        mmapped_array = np.frombuffer(mapping, dtype=dtype, count=length if stride is None else length * stride, offset=offset)
        if stride:
            mmapped_array = mmapped_array[::stride]
        self.axes[name] = mmapped_array
        self.axis_names.append(name)

    def _map_array(self, offset=None, length=None, dtype=np.float64, stride=1, filename=None, array=None, name='unknown'):
        if filename is None:
            filename = self.filename
        if not self.nommap:
            mapping = self.mapping_map[filename]

        if array is not None:
            length = len(array)

        # if self._length_original is not None and length != self._length_original:
        #     logger.error("inconsistent length", "length of column %s is %d, while %d was expected" % (name, length, self._length))
        # else:
            # self._length_unfiltered = length
            # self._length_original = length
            # if self.current_slice is None:
            #     self.current_slice = (0, length)
            #     self.fraction = 1.
            #     self._length = length
            #     self._index_end = self._length_unfiltered
            #     self._index_start = 0
            # print self.mapping, dtype, length if stride is None else length * stride, offset
        if 1:
            if array is not None:
                length = len(array)
                mmapped_array = array
                stride = None
                offset = None
                dtype = array.dtype
                column = array
            else:
                if offset is None:
                    print("offset is None")
                    sys.exit(0)

                #file = open(filename, 'rb') #self.file_map[filename]
                file = self.file_map[filename]
                if ColumnReader:
                    import io
                    column = ColumnReader(self, io.FileIO(file.fileno(), 'rb'), offset, length, dtype)
                    #column = ColumnReader(self, file, offset, length, dtype)
                else:
                    column = np.frombuffer(mapping, dtype=dtype, count=length if stride is None else length * stride, offset=offset)
                    if stride and stride != 1:
                        column = column[::stride]
            return column

    def addColumn(self, name, offset=None, length=None, dtype=np.float64, stride=1, filename=None, array=None):
        array = self._map_array(offset, length, dtype, stride, filename, array, name=name)
        if array is not None:
            if self._length_original is None:
                self._length_unfiltered = len(array)
                self._length_original = len(array)
                self._index_end = self._length_unfiltered
            self.columns[name] = array
            self.column_names.append(name)
            self._save_assign_expression(name, Expression(self, name))
            self.all_columns[name] = array
            self.all_column_names.append(name)
            # self.column_names.sort()
            self.nColumns += 1
            self.nRows = self._length_original
            self.offsets[name] = offset
            self.strides[name] = stride
            if filename is not None:
                self.filenames[name] = os.path.abspath(filename)
            self.dtypes[name] = dtype

    def addRank1(self, name, offset, length, length1, dtype=np.float64, stride=1, stride1=1, filename=None, transposed=False):
        if filename is None:
            filename = self.filename
        mapping = self.mapping_map[filename]
        if (not transposed and self._length is not None and length != self._length) or (transposed and self._length is not None and length1 != self._length):
            logger.error("inconsistent length", "length of column %s is %d, while %d was expected" % (name, length, self._length))
        else:
            if self.current_slice is None:
                self.current_slice = (0, length if not transposed else length1)
                self.fraction = 1.
                self._length_unfiltered = length if not transposed else length1
                self._length = self._length_unfiltered
                self._index_end = self._length_unfiltered
            self._length = length if not transposed else length1
            # print self.mapping, dtype, length if stride is None else length * stride, offset
            rawlength = length * length1
            rawlength *= stride
            rawlength *= stride1

            mmapped_array = np.frombuffer(mapping, dtype=dtype, count=rawlength, offset=offset)
            mmapped_array = mmapped_array.reshape((length1 * stride1, length * stride))
            mmapped_array = mmapped_array[::stride1, ::stride]

            self.rank1s[name] = mmapped_array
            self.rank1names.append(name)
            self.all_columns[name] = mmapped_array
            self.all_column_names.append(name)


class HansMemoryMapped(DatasetMemoryMapped):
    def __init__(self, filename, filename_extra=None):
        super(HansMemoryMapped, self).__init__(filename)
        self.pageSize, \
            self.formatSize, \
            self.numberParticles, \
            self.numberTimes, \
            self.numberParameters, \
            self.numberCompute, \
            self.dataOffset, \
            self.dataHeaderSize = struct.unpack("Q" * 8, self.mapping[:8 * 8])
        zerooffset = offset = self.dataOffset
        length = self.numberParticles + 1
        stride = self.formatSize // 8  # stride in units of the size of the element (float64)

        # TODO: ask Hans for the self.numberTimes-2
        lastoffset = offset + (self.numberParticles + 1) * (self.numberTimes - 2) * self.formatSize
        t_index = 3
        names = "x y z vx vy vz".split()
        midoffset = offset + (self.numberParticles + 1) * self.formatSize * t_index
        names = "x y z vx vy vz".split()

        for i, name in enumerate(names):
            self.addColumn(name + "_0", offset + 8 * i, length, dtype=np.float64, stride=stride)

        for i, name in enumerate(names):
            self.addColumn(name + "_last", lastoffset + 8 * i, length, dtype=np.float64, stride=stride)

        names = "x y z vx vy vz".split()

        if 1:
            stride = self.formatSize // 8
            # stride1 = self.numberTimes #*self.formatSize/8
            for i, name in enumerate(names):
                # TODO: ask Hans for the self.numberTimes-1
                self.addRank1(name, offset + 8 * i, length=self.numberParticles + 1, length1=self.numberTimes - 1, dtype=np.float64, stride=stride, stride1=1)

        if filename_extra is None:
            basename = os.path.basename(filename)
            if os.path.exists(basename + ".omega2"):
                filename_extra = basename + ".omega2"

        if filename_extra is not None:
            self.addFile(filename_extra)
            mapping = self.mapping_map[filename_extra]
            names = "J_r J_theta J_phi Theta_r Theta_theta Theta_phi Omega_r Omega_theta Omega_phi r_apo r_peri".split()
            offset = 0
            stride = 11
            # import pdb
            # pdb.set_trace()
            for i, name in enumerate(names):
                # TODO: ask Hans for the self.numberTimes-1
                self.addRank1(name, offset + 8 * i, length=self.numberParticles + 1, length1=self.numberTimes - 1, dtype=np.float64, stride=stride, stride1=1, filename=filename_extra)

                self.addColumn(name + "_0", offset + 8 * i, length, dtype=np.float64, stride=stride, filename=filename_extra)
                self.addColumn(name + "_last", offset + 8 * i + (self.numberParticles + 1) * (self.numberTimes - 2) * 11 * 8, length, dtype=np.float64, stride=stride, filename=filename_extra)

    @classmethod
    def can_open(cls, path, *args, **kwargs):
        return os.path.splitext(path)[-1] == ".bin"
        basename, ext = os.path.splitext(path)
        # if os.path.exists(basename + ".omega2"):
        # return True
        # return True

    @classmethod
    def get_options(cls, path):
        return []

    @classmethod
    def option_to_args(cls, option):
        return []


dataset_type_map["buist"] = HansMemoryMapped
