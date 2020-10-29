__author__ = 'breddels'
import pathlib
import logging
import os
import sys
from urllib.parse import urlparse, parse_qs

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

    def read(self, *args):
        return self.file.read(*args)

    def seek(self, *args):
        return self.file.seek(*args)

    def tell(self):
        return self.file.tell()

    def close(self):
        return self.file.close()

    def __enter__(self, *args):
        pass

    def __exit__(self, *args):
        self.file.close()


def is_file_object(file):
    return hasattr(file, 'read')


def stringyify(path):
    if hasattr(path, 'name'):  # passed in a file 
        path = path.name
    try:
        # Pathlib support
        path = path.__fspath__()
    except AttributeError:
        pass
    return path


def memory_mappable(path):
    path = stringyify(path)
    o = urlparse(path)
    return o.scheme == ''


def split_options(path, **kwargs):
    o = urlparse(path)
    naked_path = path
    if '?' in naked_path:
        naked_path = naked_path[:naked_path.index('?')]
    options = dict(kwargs)
    options.update({key: values[0] for key, values in parse_qs(o.query).items()})
    return naked_path, options


def open_google_cloud(path, mode, **kwargs):
    from .gcs import open
    return vaex.file.gcs.open(path, mode, **kwargs)


def open_s3_arrow(path, mode, **kwargs):
    from .arrow import open_s3
    return open_s3(path, mode, **kwargs)


scheme_opener = {
    '': normal_open,
    's3': open_s3_arrow,
    'gs': open_google_cloud
}


def open(path, mode='rb', **kwargs):
    path = stringyify(path)
    o = urlparse(path)
    opener = scheme_opener.get(o.scheme)
    if not opener:
        raise ValueError(f'Do not know how to open {path}')
    return opener(path, mode, **kwargs)


def dup(file):
    """Duplicate a file like object, s3 or cached file supported"""
    if isinstance(file, (vaex.file.cache.CachedFile, FileProxy)):
        return file.dup()
    else:
        return normal_open(file.name, file.mode)
