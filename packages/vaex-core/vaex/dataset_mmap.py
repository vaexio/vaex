__author__ = 'maartenbreddels'
import os
import mmap
import collections
import logging
import numpy as np
import numpy.ma
import vaex
from vaex.dataset import DatasetLocal
import vaex.dataset
import vaex.file
from vaex.expression import Expression
from vaex.file.column import ColumnFile
import struct

logger = logging.getLogger("vaex.file")

dataset_type_map = {}

osname = vaex.utils.osname
no_mmap = os.environ.get('VAEX_NO_MMAP', False)



class DatasetMemoryMapped(vaex.dataset.DatasetFile):
    """Represents a dataset where the data is memory mapped for efficient reading"""

    # nommap is a hack to get in memory datasets working
    def __init__(self, path, write=False, nommap=False):
        super().__init__(path=os.path.abspath(path), write=write)
        # self.name = name or os.path.splitext(os.path.basename(self.path))[0]
        # self.path = os.path.abspath(path) if path is not None else None
        self.nommap = nommap
        self.file_map = {}
        self.fileno_map = {}
        self.mapping_map = {}

        # self._mappings = []
        # self.samp_id = None

    def __getstate__(self):
        return {
            **super().__getstate__(),
            'nommap': self.nommap
        }

    def __setstate__(self, state):
        super().__setstate__(state)
        self.mapping_map = {}
        self.fileno_map = {}
        self.file_map = {}


    def _get_file(self, path):
        assert self.nommap
        if path not in self.file_map:
            file = open(path, "rb+" if self.write else "rb")
            self.file_map[path] = file
        return self.file_map[path]

    def _get_mapping(self, path):
        assert not self.nommap
        if path not in self.mapping_map:
            file = open(path, "rb+" if self.write else "rb")
            fileno = file.fileno()
            kwargs = {}
            if vaex.utils.osname == "windows":
                kwargs["access"] = mmap.ACCESS_READ | 0 if not self.write else mmap.ACCESS_WRITE
            else:
                kwargs["prot"] = mmap.PROT_READ | 0 if not self.write else mmap.PROT_WRITE
            mapping = mmap.mmap(fileno, 0, **kwargs)
            self.file_map[path] = file
            self.fileno_map[path] = fileno
            self.mapping_map[path] = mapping
        return self.mapping_map[path]

    def close_files(self):
        for name, file in self.file_map.items():
            file.close()
        # on osx and linux this will give random bus errors (osx) or segfaults (linux)
        # on win32 however, we'll run out of file handles
        if vaex.utils.osname not in ["osx", "linux"]:
            for name, memmap in self.mapping_map.items():
                memmap.close()

    # def addFile(self, path, write=False):
    #     self.file_map[path] = open(path, "rb+" if write else "rb")
    #     self.fileno_map[path] = self.file_map[path].fileno()
    #     self.mapping_map[path] = mmap.mmap(self.fileno_map[path], 0, prot=mmap.PROT_READ | 0 if not write else mmap.PROT_WRITE)

    # def matches_url(self, url):
    #     path = url
    #     if path.startswith("file:/"):
    #         path = path[5:]
    #     similar = os.path.splitext(os.path.abspath(self.path))[0] == os.path.splitext(path)[0]
    #     logger.info("matching urls: %r == %r == %r" % (os.path.splitext(self.path)[0], os.path.splitext(path)[0], similar))
    #     return similar

    def close(self):
        self.file.close()
        self.mapping.close()

    def _map_array(self, offset=None, length=None, dtype=np.float64, path=None):
        if path is None:
            path = self.path

        # self._mappings.append({
        #     'offset': offset,
        #     'length': length,
        #     'dtype': dtype,
        #     'path': path
        # })
        return self._do_map(path, offset, length, dtype)
    
    def _do_map(self, path, offset, length, dtype):        
        if self.nommap:
            # file = self.file_map[path]
            file = self._get_file(path)
            column = ColumnFile(file, offset, length, dtype, write=self.write, path=self.path)
        else:
            # mapping = self.mapping_map[path]
            mapping = self._get_mapping(path)
            column = np.frombuffer(mapping, dtype=dtype, count=length, offset=offset)
        return column

    # def _add_column(self, name, offset=None, length=None, dtype=np.float64, stride=1, path=None, array=None):
    #     array = self._map_array(offset, length, dtype, stride, path, array, name=name)
    #     if array is not None:
    #         if self._length_original is None:
    #             self._length_unfiltered = len(array)
    #             self._length_original = len(array)
    #             self._index_end = self._length_unfiltered
    #         self.columns[name] = array
    #         self.column_names.append(name)
    #         self._save_assign_expression(name, Expression(self, name))
    #         self.all_columns[name] = array
    #         self.all_column_names.append(name)
    #         # self.column_names.sort()
    #         self.nColumns += 1
    #         self.nRows = self._length_original
    #         self.offsets[name] = offset
    #         self.strides[name] = stride
    #         if path is not None:
    #             self.paths[name] = os.path.abspath(path)
    #         self.dtypes[name] = dtype

    # def _addRank1(self, name, offset, length, length1, dtype=np.float64, stride=1, stride1=1, path=None, transposed=False):
    #     if path is None:
    #         path = self.path
    #     mapping = self.mapping_map[path]
    #     if (not transposed and self._length is not None and length != self._length) or (transposed and self._length is not None and length1 != self._length):
    #         logger.error("inconsistent length", "length of column %s is %d, while %d was expected" % (name, length, self._length))
    #     else:
    #         if self.current_slice is None:
    #             self.current_slice = (0, length if not transposed else length1)
    #             self.fraction = 1.
    #             self._length_unfiltered = length if not transposed else length1
    #             self._length = self._length_unfiltered
    #             self._index_end = self._length_unfiltered
    #         self._length = length if not transposed else length1
    #         # print self.mapping, dtype, length if stride is None else length * stride, offset
    #         rawlength = length * length1
    #         rawlength *= stride
    #         rawlength *= stride1

    #         mmapped_array = np.frombuffer(mapping, dtype=dtype, count=rawlength, offset=offset)
    #         mmapped_array = mmapped_array.reshape((length1 * stride1, length * stride))
    #         mmapped_array = mmapped_array[::stride1, ::stride]

    #         self.rank1s[name] = mmapped_array
    #         self.rank1names.append(name)
    #         self.all_columns[name] = mmapped_array
    #         self.all_column_names.append(name)
