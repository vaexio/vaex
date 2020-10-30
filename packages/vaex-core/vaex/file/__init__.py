__author__ = 'breddels'
import pathlib
import logging
from glob import glob as local_glob
import os
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


def file_and_path(file, mode='r'):
    if is_file_object(file):
        return file, stringyfy(file)
    else:
        file = open(file, mode=mode)
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


def split_options(path, **kwargs):
    naked_path = path
    query = ''
    if '?' in naked_path:
        i = naked_path.index('?')
        naked_path, query = naked_path[:i], naked_path[i+1:]
    options = dict(kwargs)
    options.update({key: values[0] for key, values in parse_qs(query).items()})
    return naked_path, options


def open_google_cloud(path, mode, **kwargs):
    from .gcs import open
    return vaex.file.gcs.open(path, mode, **kwargs)


def open_s3_arrow(path, mode, **kwargs):
    from .arrow import open_s3
    return open_s3(path, mode, **kwargs)


def open_s3fs(path, mode, **kwargs):
    from .s3 import open
    return open(path, mode, **kwargs)


scheme_opener = {
    None: normal_open,
    's3': open_s3_arrow,
    's3fs': open_s3fs,
    'gs': open_google_cloud
}


def open(path, mode='rb', **kwargs):
    if is_file_object(path):
        return path
    path = stringyfy(path)
    scheme, _ = split_scheme(path)
    opener = scheme_opener.get(scheme)
    if not opener:
        raise ValueError(f'Do not know how to open {path}')
    return opener(path, mode, **kwargs)


def open_for_arrow(path, mode, **kwargs):
    '''When the file will be passed to arrow, we want file object arrow likes.

    This might avoid peformance issues with GIL, or call overhead.
    '''
    import pyarrow as pa
    if is_file_object(path):
        return path
    path = stringyfy(path)
    scheme, _ = split_scheme(path)
    if scheme is None:
        return pa.OSFile(path, mode)
    else:
        opener = scheme_opener.get(scheme)
        if not opener:
            raise ValueError(f'Do not know how to open {path}')
        return opener(path, mode, **kwargs).file


def dup(file):
    """Duplicate a file like object, s3 or cached file supported"""
    if isinstance(file, (vaex.file.cache.CachedFile, FileProxy)):
        return file.dup()
    else:
        return normal_open(file.name, file.mode)


def glob_s3(path, **kwargs):
    from .s3 import glob
    return glob(path, **kwargs)


def glob_google_cloud(path, **kwargs):
    from .gcs import glob
    return glob(path, **kwargs)


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
