__author__ = 'maartenbreddels'
import os
import mmap
import collections
import logging
import numpy as np
import numpy.ma
import vaex
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

    def __init__(self, path, write=False, nommap=False):
        super().__init__(path=path, write=write)
        self.nommap = nommap
        self.file_map = {}
        self.fileno_map = {}
        self.mapping_map = {}

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

    def close(self):
        for name, file in self.file_map.items():
            file.close()
        # on osx and linux this will give random bus errors (osx) or segfaults (linux)
        # on win32 however, we'll run out of file handles
        if vaex.utils.osname not in ["osx", "linux"]:
            for name, memmap in self.mapping_map.items():
                memmap.close()

    def _map_array(self, offset=None, length=None, dtype=np.float64, path=None):
        if path is None:
            path = self.path
        return self._do_map(path, offset, length, dtype)
    
    def _do_map(self, path, offset, length, dtype):        
        if self.nommap:
            file = self._get_file(path)
            column = ColumnFile(file, offset, length, dtype, write=self.write, path=self.path)
        else:
            mapping = self._get_mapping(path)
            column = np.frombuffer(mapping, dtype=dtype, count=length, offset=offset)
        return column
