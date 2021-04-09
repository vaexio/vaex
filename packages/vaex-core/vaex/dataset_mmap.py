__author__ = 'maartenbreddels'
import os
import mmap
import collections
import logging
import numpy as np
import numpy.ma
import threading
import vaex
import vaex.dataset
import vaex.file
from vaex.expression import Expression
from vaex.file.column import ColumnFile
import struct

from vaex.multithreading import get_main_io_pool
from vaex.itertools import pmap, buffer, pwait

logger = logging.getLogger("vaex.file")

dataset_type_map = {}

osname = vaex.utils.osname
no_mmap = os.environ.get('VAEX_NO_MMAP', False)



class DatasetMemoryMapped(vaex.dataset.DatasetFile):
    """Represents a dataset where the data is memory mapped for efficient reading"""

    def __init__(self, path, write=False, nommap=False, fs_options={}, fs=None):
        super().__init__(path=path, write=write, fs_options=fs_options, fs=fs)
        self.nommap = nommap
        self.file_map = {}
        self.fileno_map = {}
        self.mapping_map = {}
        self.tls_map = collections.defaultdict(threading.local)

    def chunk_iterator(self, columns, chunk_size=None, reverse=False):
        if self.nommap:
            # we expect here a path like s3 fetching, which will benefit from multithreading
            pool = get_main_io_pool()
            def read(i1, i2, reader):
                return i1, i2, reader()
            chunks_generator = self._default_lazy_chunk_iterator(self._columns, columns, chunk_size)
            yield from pwait(buffer(pmap(read, chunks_generator, pool), pool._max_workers+3))
        else:
            yield from self._default_chunk_iterator(self._columns, columns, chunk_size, reverse=reverse)

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
            # TODO: we can think about adding this in py38
            # mapping.madvise(mmap.MADV_SEQUENTIAL)
            self.file_map[path] = file
            self.fileno_map[path] = fileno
            self.mapping_map[path] = mapping
        return self.mapping_map[path]

    def close(self):
        for name, memmap in self.mapping_map.items():
            memmap.close()
        for name, file in self.file_map.items():
            file.close()

    def _map_array(self, offset=None, length=None, dtype=np.float64, path=None):
        if path is None:
            path = self.path
        return self._do_map(path, offset, length, dtype)
    
    def _do_map(self, path, offset, length, dtype):        
        if self.nommap:
            file = self._get_file(path)
            column = ColumnFile(file, offset, length, dtype, write=self.write, path=self.path, tls=self.tls_map[path])
        else:
            mapping = self._get_mapping(path)
            column = np.frombuffer(mapping, dtype=dtype, count=length, offset=offset)
        return column
