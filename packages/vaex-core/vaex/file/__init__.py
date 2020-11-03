__author__ = 'breddels'
import pathlib
import logging
from glob import glob as local_glob
import os
import re
import sys
from urllib.parse import parse_qs

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
    if '://' in path:
        schema, path = path[:path.index('://')], path[path.index('://')+3:]
    else:
        schema = None
    return schema, path


def memory_mappable(path):
    path = stringyfy(path)
    scheme, _ = split_scheme(path)
    return scheme is None


def split_options(path, fs_options={}):
    match = re.match(r'(.*?)\?((&?[^=&?]+=[^=&?]+)+)', path)
    if match:
        naked_path, query = match.groups()[:2]
    else:
        naked_path = path
        query = ''
    options = fs_options.copy()
    options.update({key: values[0] for key, values in parse_qs(query).items()})
    return naked_path, options


def open_google_cloud(path, mode, fs_options={}):
    from .gcs import open
    return vaex.file.gcs.open(path, mode, fs_options=fs_options)


def open_s3_arrow(path, mode, fs_options={}):
    from .arrow import open_s3
    return open_s3(path, mode, fs_options=fs_options)


def open_s3fs(path, mode, fs_options={}):
    from .s3 import open
    return open(path, mode, fs_options=fs_options)


scheme_opener = {
    None: normal_open,
    's3': open_s3_arrow,
    's3fs': open_s3fs,
    'gs': open_google_cloud
}


def open(path, mode='rb', fs_options={}):
    if is_file_object(path):
        return path
    path = stringyfy(path)
    scheme, _ = split_scheme(path)
    opener = scheme_opener.get(scheme)
    if not opener:
        raise ValueError(f'Do not know how to open {path}')
    if scheme is None:
        if fs_options:
            raise ValueError(f'fs_options not supported for local files. You passed: {repr(fs_options)}.')
        return opener(path, mode)
    if scheme == 's3':
        # fallback to s3fs for windows
        import pyarrow as pa
        try:
            return opener(path, mode, fs_options=fs_options)
        except pa.lib.ArrowNotImplementedError:
            opener = scheme_opener['s3fs']
    return opener(path, mode, fs_options=fs_options)


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
        opener = scheme_opener.get(scheme)
        if not opener:
            raise ValueError(f'Do not know how to open {path}')
        return opener(path, mode, fs_options=fs_options).file


def dup(file):
    """Duplicate a file like object, s3 or cached file supported"""
    if isinstance(file, (vaex.file.cache.CachedFile, FileProxy)):
        return file.dup()
    else:
        return normal_open(file.name, file.mode)


def glob_s3(path, fs_options={}):
    from .s3 import glob
    return glob(path, fs_options=fs_options)


def glob_google_cloud(path, fs_options={}):
    from .gcs import glob
    return glob(path, fs_options=fs_options)


globber_map = {
    None: local_glob,
    's3': glob_s3,
    's3fs': glob_s3,
    'gs': glob_google_cloud
}


def glob(path, **kwargs):
    path = stringyfy(path)
    scheme, _ = split_scheme(path)
    globber = globber_map.get(scheme)
    if not globber:
        raise ValueError(f'Do not know how to glob {path}')
    return globber(path, **kwargs)


def ext(path):
    path = stringyfy(path)
    path, options = split_options(path)
    return os.path.splitext(path)[1]
