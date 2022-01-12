try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse
import logging
import os
import mmap


import numpy as np
from pyarrow.fs import FileSystemHandler


import vaex.utils
import vaex.file


DEFAULT_BLOCK_SIZE = 1024*1024*1  # 1mb by default
logger = logging.getLogger("vaex.file.cache")


class FileSystemHandlerCached(FileSystemHandler):
    """Proxies it to use the CachedFile
    """

    def __init__(self, fs, scheme, for_arrow=False):
        self.fs = fs
        self.scheme = scheme
        self.for_arrow = for_arrow
        self._file_cache = {}

    def __eq__(self, other):
        if isinstance(other, FileSystemHandlerCached):
            return self.fs == other.fs
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, FileSystemHandlerCached):
            return self.fs != other.fs
        return NotImplemented

    def __getattr__(self, name):
        return getattr(self.fs, name)

    def open_input_stream(self, path):
        from pyarrow import PythonFile

        def real_open():
            return self.fs.open_input_stream(path)
        full_path = f'{self.scheme}://{path}'
        # TODO: we may wait to cache the mmapped file
        if full_path not in self._file_cache:
            f = CachedFile(real_open, full_path, read_as_buffer=not self.for_arrow)
            self._file_cache[full_path] = f
        else:
            previous = self._file_cache[full_path]
            f = CachedFile(real_open, full_path, data_file=previous.data_file, mask_file=previous.mask_file)
        if not self.for_arrow:
            return f
        f = vaex.file.FileProxy(f, full_path, None)
        return PythonFile(f, mode="r")

    def open_input_file(self, path):
        from pyarrow import PythonFile

        def real_open():
            return self.fs.open_input_file(path)
        full_path = f'{self.scheme}://{path}'
        # TODO: we may wait to cache the mmapped file
        if full_path not in self._file_cache:
            f = CachedFile(real_open, full_path, read_as_buffer=not self.for_arrow)
            self._file_cache[full_path] = f
        else:
            previous = self._file_cache[full_path]
            f = CachedFile(real_open, full_path, data_file=previous.data_file, mask_file=previous.mask_file, read_as_buffer=not self.for_arrow)
        if not self.for_arrow:
            return f
        f = vaex.file.FileProxy(f, full_path, None)
        return PythonFile(f, mode="r")

    # these are forwarded
    def copy_file(self, *args, **kwargs):
        return self.fs.copy_file(*args, **kwargs)
    def create_dir(self, *args, **kwargs):
        return self.fs.create_dir(*args, **kwargs)
    def delete_dir(self, *args, **kwargs):
        return self.fs.delete_dir(*args, **kwargs)
    def delete_dir_contents(self, *args, **kwargs):
        return self.fs.delete_dir_contents(*args, **kwargs)
    def delete_file(self, *args, **kwargs):
        return self.fs.delete_file(*args, **kwargs)
    def delete_root_dir_contents(self, *args, **kwargs):
        return self.fs.delete_root_dir_contents(*args, **kwargs)
    def get_file_info(self, *args, **kwargs):
        return self.fs.get_file_info(*args, **kwargs)
    def get_file_info_selector(self, *args, **kwargs):
        return self.fs.get_file_info_selector(*args, **kwargs)
    def get_type_name(self, *args, **kwargs):
        return self.fs.get_type_name(*args, **kwargs)
    def move(self, *args, **kwargs):
        return self.fs.move(*args, **kwargs)
    def normalize_path(self, *args, **kwargs):
        return self.fs.normalize_path(*args, **kwargs)
    def open_append_stream(self, *args, **kwargs):
        return self.fs.open_append_stream(*args, **kwargs)
    def open_output_stream(self, *args, **kwargs):
        return self.fs.open_output_stream(*args, **kwargs)


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
        self.memoryview = memoryview(self.mmap)
        self.data = np.frombuffer(self.mmap, dtype=dtype, count=self.length)

    def __getitem__(self, item):
        return self.memoryview.__getitem__(item)


def _to_block_ceil(index, block_size):
    return (index + block_size - 1) // block_size


def _to_block_floor(index, block_size):
    return index // block_size


def _to_index(block, block_size):
    return block * block_size


class CachedFile:
    def __init__(self, file, path=None, cache_dir=None, block_size=DEFAULT_BLOCK_SIZE, data_file=None, mask_file=None, read_as_buffer=True):
        """Decorator that wraps a file object (typically a s3) by caching the content locally on disk.

        The standard location for the cache is: `${VAEX_FS_PATH}/<protocol (e.g. s3)>/path/to/file.ext`

        See `Configuration of paths<conf.html#cache-fs>`_ how to change this.

        Arguments:
        :file file or callable: if callable, invoking it should give a file like object
        :path str: path of file, defaults of file.name
        :cache_dir str: path of cache dir, defaults to `${VAEX_FS_PATH}`
        """
        self.name = path
        self.path = path
        self.file = file
        self.cache_dir = cache_dir
        self.block_size = block_size
        self.read_as_buffer = read_as_buffer

        self.block_reads = 0
        self.reads = 0
        self.loc = 0

        if data_file is None or mask_file is None:
            o = urlparse(path)
            if cache_dir is None:
                cache_dir = vaex.settings.fs.path
            self.cache_dir_path = os.path.join(cache_dir, o.scheme, o.netloc, o.path[1:])
            self.cache_dir_path = os.path.join(cache_dir, o.scheme, o.netloc, o.path[1:])
            lockname = os.path.join('file-cache', o.scheme, o.netloc, o.path[1:], 'create.lock')
            os.makedirs(self.cache_dir_path, exist_ok=True)
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
            with vaex.utils.file_lock(lockname):
                self.data_file = MMappedFile(self.data_path, self.length)
                self.mask_file = MMappedFile(self.mask_path, self.mask_length)
        else:
            self.data_file = data_file
            self.mask_file = mask_file
            self.length = self.data_file.length
            self.mask_length = self.mask_file.length

    def readable(self):
        return True

    def writable(self):
        return False

    def seekable(self):
        return True

    def closed(self):
        return self.file.closed()

    def flush(self):
        pass

    def dup(self):
        if callable(self.file):
            file = self.file
        else:
            file = lambda: vaex.file.dup(self.file)
        return CachedFile(file, self.path, self.cache_dir, self.block_size, data_file=self.data_file, mask_file=self.mask_file, read_as_buffer=self.read_as_buffer)

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
        buffer = self.data_file[start:end]
        # arrow 1 and 2 don't accept a non-bytes object via the PythonFile.read() path
        return buffer if self.read_as_buffer else buffer.tobytes()

    def readinto(self, buffer):
        start = self.loc
        end = start + len(buffer)
        self._ensure_cached(start, end)
        buffer[:] = self.data_file[start:end]
        self.loc = end
        return len(buffer)

    def read_buffer(self, byte_count):
        start = self.loc
        end = start + byte_count
        self._ensure_cached(start, end)
        self.loc = end
        return self.data_file[start:end]

    def _as_numpy(self, offset, byte_length, dtype):
        # quick route that avoids memory copies
        self._ensure_cached(offset, offset+byte_length)
        return np.frombuffer(self.data_file[offset:offset+byte_length], dtype)

    def _fetch_blocks(self, block_start, block_end):
        start_blocked = _to_index(block_start, self.block_size)
        end_blocked = min(self.length, _to_index(block_end, self.block_size))
        self._use_file()
        self.file.seek(start_blocked)
        bytes_read = self.file.readinto(self.data_file[start_blocked:end_blocked])
        expected = (end_blocked - start_blocked)
        assert bytes_read == expected, f'Read {bytes_read}, expected {expected} ({start_blocked}-{end_blocked} out of {self.length})'
        self.mask_file.data[block_start:block_end] = 1
        self.reads += 1
        self.block_reads += block_end - block_start

    def _ensure_cached(self, start, end):
        block_start = _to_block_floor(start, self.block_size)
        block_end = _to_block_ceil(end, self.block_size)
        missing = self.mask_file.data[block_start:block_end] == 0
        if np.all(missing):
            self._fetch_blocks(block_start, block_end)
        elif np.any(missing):
            i = block_start
            done = False
            while not done:
                # find first block that is not cached
                while i < block_end and self.mask_file.data[i] == 1:
                    i += 1
                if i == block_end:
                    break
                # find block that *is* cached
                j = i + 1
                while j < block_end and self.mask_file.data[j] == 0:
                    j += 1
                self._fetch_blocks(i, j)
                i = j

    def close(self):
        # if it is callable, the file is never opened
        if not callable(self.file):
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
