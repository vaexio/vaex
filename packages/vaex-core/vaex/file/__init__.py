__author__ = 'breddels'
import pathlib
import logging
from glob import glob as local_glob
import io
import os
import re
import sys
from urllib.parse import parse_qs
import pkg_resources

import vaex.file.cache


normal_open = open
logger = logging.getLogger("vaex.file")


class FileProxy:
    '''Wraps a file object, giving it a name a dup() method

    The dup is needed since a file is stateful, and needs to be duplicated in threads
    '''
    def __init__(self, file, name, dup):
        self.file = file
        self.name = name
        self.dup = dup
        self.closed = False

    def __iter__(self):
        raise NotImplementedError('This is just for looking like a file object to Pandas')

    def write(self, *args):
        return self.file.write(*args)

    def read(self, *args):
        return self.file.read(*args)

    def seek(self, *args):
        return self.file.seek(*args)

    def readinto(self, *args):
        return self.file.readinto(*args)

    def tell(self):
        return self.file.tell()

    def close(self):
        self.closed = True
        return self.file.close()

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.file.close()


def is_file_object(file):
    return hasattr(file, 'read') and hasattr(file, 'seek')


def file_and_path(file, mode='r', fs_options={}):
    if is_file_object(file):
        return file, stringyfy(file)
    else:
        file = open(file, mode=mode, fs_options=fs_options)
        return file, stringyfy(file)


def stringyfy(path):
    if hasattr(path, 'name'):  # passed in a file
        path = path.name
    try:
        # Pathlib support
        path = path.__fspath__()
    except AttributeError:
        pass
    if isinstance(path, str):
        return path


def split_scheme(path):
    path = stringyfy(path)
    if '://' in path:
        scheme, path = path[:path.index('://')], path[path.index('://')+3:]
    else:
        scheme = None
    return scheme, path


def memory_mappable(path):
    path = stringyfy(path)
    scheme, _ = split_scheme(path)
    return scheme is None


def split_options(path, fs_options={}):
    if isinstance(path, list):
        paths = []
        previous_options = None
        for path in path:
            path, options = split_options(path, fs_options)
            if previous_options is not None:
                if previous_options != options:
                    raise ValueError(f'Inconsistent set of fs_options given: {previous_options} {options}')
            else:
                previous_options = options
            paths.append(path)
        return paths, previous_options

    match = re.match(r'(.*?)\?((&?[^=&?]+=[^=&?]+)+)', path)
    if match:
        naked_path, query = match.groups()[:2]
    else:
        naked_path = path
        query = ''
    options = fs_options.copy()
    options.update({key: values[0] for key, values in parse_qs(query).items()})
    return naked_path, options


def _get_scheme_handler(path):
    scheme, _ = split_scheme(path)
    for entry in pkg_resources.iter_entry_points(group='vaex.file.scheme'):
        if entry.name == scheme:
            return entry.load()
    raise ValueError(f'Do not know how to open {path}, no handler for {scheme} is known')


def parse(path, fs_options={}):
    if isinstance(path, (list, tuple)):
        scheme, _ = split_scheme(path[0])
    else:
        scheme, _ = split_scheme(path)
    if not scheme:
        return None, path
    if isinstance(path, (list, tuple)):
        module = _get_scheme_handler(path[0])
        return module.parse(path[0], fs_options)[0], path
    else:
        module = _get_scheme_handler(path)
        return module.parse(path, fs_options)


def open(path, mode='rb', fs_options={}):
    fs, path = parse(path, fs_options=fs_options)
    if fs is None:
        return normal_open(path, mode)
    if mode == 'rb':
        def create():
            return fs.open_input_file(path)
    elif mode == "r":
        def create():
            return io.TextIOWrapper(fs.open_input_file(path))
    elif mode == 'wb':
        def create():
            return fs.open_output_stream(path)
    elif mode == "w":
        def create():
            return io.TextIOWrapper(fs.open_output_stream(path))
    else:
        raise ValueError(f'Only mode=rb/bw/r/w are supported, not {mode}')
    return FileProxy(create(), path, create)


def open_for_arrow(path, mode='rb', fs_options={}, mmap=False):
    '''When the file will be passed to arrow, we want file object arrow likes.

    This might avoid peformance issues with GIL, or call overhead.
    '''
    import pyarrow as pa
    if is_file_object(path):
        return path
    path = stringyfy(path)
    scheme, _ = split_scheme(path)
    if scheme is None:
        if fs_options:
            raise ValueError(f'fs_options not supported for local files. You passed: {repr(fs_options)}.')
        if mmap:
            return pa.memory_map(path, mode)
        else:
            return pa.OSFile(path, mode)
    else:
        return open(path, mode=mode, fs_options=fs_options).file


def dup(file):
    """Duplicate a file like object, s3 or cached file supported"""
    if isinstance(file, (vaex.file.cache.CachedFile, FileProxy)):
        return file.dup()
    else:
        return normal_open(file.name, file.mode)

def glob(path, fs_options={}):
    scheme, _ = split_scheme(path)
    if not scheme:
        return local_glob(path)
    module = _get_scheme_handler(path)
    return module.glob(path, fs_options)


def ext(path):
    path = stringyfy(path)
    path, options = split_options(path)
    return os.path.splitext(path)[1]
