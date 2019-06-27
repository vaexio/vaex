try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse
import logging
import os
import mmap

import numpy as np

import vaex.utils
import vaex.file


DEFAULT_BLOCK_SIZE = 1024*1024*1  # 1mb by default
logger = logging.getLogger("vaex.file")


class MMappedFile:
    """Small wrapper around a memory mapped file"""
    def __init__(self, path, length, dtype=np.uint8):
        self.path = path
        self.length = length
        if not os.path.exists(path):
            with open(self.path, 'wb') as fp:
                fp.seek(self.length-1)
                fp.write(b'\00')
                fp.flush()

        self.fp = open(self.path, 'rb+')
        kwargs = {}
        if vaex.utils.osname == "windows":
            kwargs["access"] = mmap.ACCESS_WRITE
        else:
            kwargs["prot"] = mmap.PROT_WRITE
        self.mmap = mmap.mmap(self.fp.fileno(), self.length)
        self.data = np.frombuffer(self.mmap, dtype=dtype, count=self.length)


def _to_block_ceil(index, block_size):
    return (index + block_size - 1) // block_size


def _to_block_floor(index, block_size):
    return index // block_size


def _to_index(block, block_size):
    return block * block_size


class CachedFile:
    def __init__(self, file, path=None, cache_dir=None, block_size=DEFAULT_BLOCK_SIZE, data_file=None, mask_file=None):
        """Decorator that wraps a file object (typically a s3) by caching the content locally on disk.

        The standard location for the cache is: ~/.vaex/file-cache/<protocol (e.g. s3)>/path/to/file.ext

        Arguments:
        :file file or callable: if callable, invoking it should give a file like object
        :path str: path of file, defaults of file.name
        :cache_dir str: path of cache dir, defaults to ~/.vaex/file-cache
        """
        self.name = path
        self.path = path
        self.file = file
        self.cache_dir = cache_dir
        self.block_size = block_size

        self.block_reads = 0
        self.reads = 0
        self.loc = 0

        if data_file is None or mask_file is None:
            o = urlparse(path)
            if cache_dir is None:
                self.cache_dir_path = vaex.utils.get_private_dir('file-cache', o.scheme, o.netloc, o.path[1:])
            else:
                # this path is used for testing
                self.cache_dir_path = os.path.join(cache_dir, 'file-cache', o.scheme, o.netloc, o.path[1:])
                if not os.path.exists(self.cache_dir_path):
                    os.makedirs(self.cache_dir_path)
            self.data_path = os.path.join(self.cache_dir_path, 'data')
            self.mask_path = os.path.join(self.cache_dir_path, 'mask')
            # if possible, we avoid using the file
            if os.path.exists(self.data_path):
                with open(self.data_path, 'rb') as f:
                    f.seek(0, 2)
                    self.length = f.tell()
            else:
                self._use_file()
                self.file.seek(0, 2)
                self.length = self.file.tell()
            self.mask_length = _to_block_ceil(self.length, self.block_size)

            logging.debug('cache path: %s', self.cache_dir_path)
            self.data_file = MMappedFile(self.data_path, self.length)
            self.mask_file = MMappedFile(self.mask_path, self.mask_length)
        else:
            self.data_file = data_file
            self.mask_file = mask_file
            self.length = self.data_file.length
            self.mask_length = self.mask_file.length

    def dup(self):
        if callable(self.file):
            file = self.file
        else:
            file = vaex.file.dup(self.file)
        return CachedFile(file, self.path, self.cache_dir, self.block_size, data_file=self.data_file, mask_file=self.mask_file)

    def tell(self):
        return self.loc

    def seek(self, loc, whence=0):
        if whence == 0:
            self.loc = loc
        elif whence == 1:
            self.loc = self.loc + loc
        elif whence == 2:
            self.loc = self.length + loc
        assert (self.loc >= 0) and (self.loc <= self.length)

    def _use_file(self):
        if callable(self.file):
            self.file = self.file()

    def read(self, length=-1):
        start = self.loc
        end = self.loc + length if length != -1 else self.length
        self._ensure_cached(start, end)
        self.loc = end
        # we have no other option than to return a copy of the data here
        return self.data_file.data[start:end].view('S1').tobytes()

    def __readinto(self, bytes):
        start = self.loc
        end = start + len(bytes)
        self._ensure_cached(start, end)
        bytes[:] = self.data_file.data[start:end]

    def _as_numpy(self, offset, byte_length, dtype):
        # quick route that avoids memory copies
        self._ensure_cached(offset, offset+byte_length)
        return self.data_file.data[offset:offset+byte_length].view(dtype)

    def _ensure_cached(self, start, end):
        block_start = _to_block_floor(start, self.block_size)
        block_end = _to_block_ceil(end, self.block_size)
        missing = self.mask_file.data[block_start:block_end] == 0
        # TODO: we could do the reading using multithreading or multiprocessing (processes)
        if np.all(missing):
            start_blocked = _to_index(block_start, self.block_size)
            end_blocked = _to_index(block_end, self.block_size)
            self._use_file()
            self.file.seek(start_blocked)
            data = self.file.read(end_blocked - start_blocked)
            self.data_file.data[start_blocked:end_blocked] = np.frombuffer(data, dtype=np.uint8)
            self.mask_file.data[block_start:block_end] = 1
            self.reads += 1
            self.block_reads += block_end - block_start
        elif np.any(missing):
            block_indices = np.arange(block_start, block_end, dtype=np.int64)
            missing_blocks = block_indices[missing]
            # TODO: we can group multiple blocks into 1 read
            for block_index in missing_blocks:
                start_blocked = _to_index(block_index, self.block_size)
                end_blocked = _to_index(block_index+1, self.block_size)
                self._use_file()
                self.file.seek(start_blocked)
                data = self.file.read(self.block_size)
                self.data_file.data[start_blocked:end_blocked] = np.frombuffer(data, dtype=np.uint8)
                self.mask_file.data[block_index] = 1
                self.reads += 1
                self.block_reads += 1

    def close(self):
        # if it is callable, the file is never opened
        if not callable(self.file):
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
