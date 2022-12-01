__author__ = 'breddels'
import pathlib
import logging
from glob import glob as local_glob
import io
import os
import re
import sys
from urllib.parse import parse_qs
import warnings

import pyarrow as pa
import pyarrow.fs

import vaex.file.cache


try:
    from sys import version_info
    if version_info[:2] >= (3, 10):
        from importlib.metadata import entry_points
    else:
        from importlib_metadata import entry_points
except ImportError:
    import pkg_resources
    entry_points = pkg_resources.iter_entry_points

normal_open = open
logger = logging.getLogger("vaex.file")


class FileProxy:
    '''Wraps a file object, giving it a name a dup() method

    The dup is needed since a file is stateful, and needs to be duplicated in threads
    '''
    def __init__(self, file, name, dup, auto_close=True):
        self.file = file
        self.name = name
        self.dup = dup
        self.closed = False
        self.auto_close = auto_close

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
        if self.auto_close:
            self.file.close()

    def readable(self):
        return True

    def writable(self):
        return False

    def seekable(self):
        return True

    def closed(self):
        return False

    def flush(self):
        return self.file.flush()


def is_file_object(file):
    return hasattr(file, 'read') and hasattr(file, 'seek')


def file_and_path(file, mode='r', fs_options={}, fs=None):
    if is_file_object(file):
        return file, stringyfy(file)
    else:
        file = open(file, mode=mode, fs_options=fs_options, fs=fs)
        return file, stringyfy(file)


def is_path_like(path):
    try:
        stringyfy(path)
        return True
    except ValueError:
        return False


def stringyfy(path):
    """Get string from path like object of file like object

    >>> import sys, pytest
    >>> if sys.platform.startswith('win'):
    ...     pytest.skip('this doctest does not work on Windows')
    ...
    >>> stringyfy('/tmp/test')
    '/tmp/test'
    >>> from pathlib import Path
    >>> stringyfy(Path('/tmp/test'))
    '/tmp/test'
    """
    try:
        # Pathlib support
        path = path.__fspath__()
    except AttributeError:
        pass
    if hasattr(path, 'name'):  # passed in a file
        path = path.name
    if isinstance(path, str):
        return path
    raise ValueError(f'Cannot convert {path} to a path')


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
    path = stringyfy(path)
    match = re.match(r'(.*?)\?((&?[^=&?]+=[^=&?]+)+)', path)
    if match:
        naked_path, query = match.groups()[:2]
    else:
        naked_path = path
        query = ''
    options = fs_options.copy()
    options.update({key: values[0] for key, values in parse_qs(query).items()})
    return naked_path, options


def split_ext(path, fs_options={}):
    path, fs_options = split_options(path, fs_options=fs_options)
    base, ext = os.path.splitext(path)
    return base, ext, fs_options


def exists(path, fs_options={}, fs=None):
    """Checks if file exists.

    >>> vaex.file.exists('/you/do/not')
    False

    >>> vaex.file.exists('s3://vaex/taxi/nyc_taxi_2015_mini.parquet', fs_options={'anon': True})
    True
    """
    fs, path = parse(path, fs_options=fs_options, fs=fs)
    if fs is None:
        return os.path.exists(path)
    else:
        return fs.get_file_info([path])[0].type != pa.fs.FileType.NotFound


def _get_scheme_handler(path):
    scheme, _ = split_scheme(path)
    for entry in entry_points(group='vaex.file.scheme'):
        if entry.name == scheme:
            return entry.load()
    raise ValueError(f'Do not know how to open {path}, no handler for {scheme} is known')


def remove(path):
    scheme, path = split_scheme(path)
    if scheme:
        raise ValueError('Cannot delete non-local files yet')
    os.remove(path)


def parse(path, fs_options={}, fs=None, for_arrow=False):
    if fs is not None:
        if fs_options:
            warnings.warn(f'Passed fs_options while fs was specified, {fs_options} are ignored')
        if 'fsspec' in sys.modules:
            import fsspec
            if isinstance(fs, fsspec.AbstractFileSystem):
                fs = pa.fs.FSSpecHandler(fs)
            if for_arrow:
                fs = pyarrow.fs.PyFileSystem(fs)
        return fs, path
    if isinstance(path, (list, tuple)):
        scheme, _ = split_scheme(path[0])
    else:
        scheme, _ = split_scheme(path)
    if not scheme:
        return None, path
    if isinstance(path, (list, tuple)):
        module = _get_scheme_handler(path[0])
        return module.parse(path[0], fs_options, for_arrow=for_arrow)[0], path
    else:
        module = _get_scheme_handler(path)
        return module.parse(path, fs_options, for_arrow=for_arrow)


def create_dir(path, fs_options, fs=None):
    fs, path = parse(path, fs_options=fs_options, fs=fs)
    if fs is None:
        fs = pa.fs.LocalFileSystem()
    fs.create_dir(path, recursive=True)


def fingerprint(path, fs_options={}, fs=None):
    """Deterministic fingerprint for a file, useful in combination with dask or detecting file changes.

    Based on mtime (modification time), file size, and the path. May lead to
    false negative if the path changes, but not the content.

    >>> fingerprint('/data/taxi.parquet')  # doctest: +SKIP
    '0171ec50cb2cf71b8e4f813212063a19'

    >>> fingerprint('s3://vaex/taxi/nyc_taxi_2015_mini.parquet', fs_options={'anon': True})  # doctest: +SKIP
    '7c962e2d8c21b6a3681afb682d3bf91b'
    """
    fs, path = parse(path, fs_options, fs=fs)
    path = stringyfy(path)
    if fs is None:
        mtime = os.path.getmtime(path)
        size = os.path.getsize(path)
    else:
        info = fs.get_file_info([path])[0]
        mtime = info.mtime_ns
        size = info.size
    import vaex.cache
    return vaex.cache.fingerprint(('file', (path, mtime, size)))


def size(path, fs_options={}, fs=None):
    """Gives the file size in bytes

    >>> size(os.path.expanduser('~/.vaex/data/helmi-dezeeuw-2000-FeH-v2.hdf5'))  # doctest: +SKIP
    135323168

    >>> size('s3://vaex/taxi/nyc_taxi_2015_mini.parquet', fs_options={'anon': True})
    9820562
    """
    fs, path = parse(path, fs_options, fs=fs)
    path = stringyfy(path)
    if fs is None:
        return os.path.getsize(path)
    else:
        info = fs.get_file_info([path])[0]
        return info.size

def open(path, mode='rb', fs_options={}, fs=None, for_arrow=False, mmap=False, encoding="utf8"):
    '''Return a file like object, also accepts cloud based paths.

    If path is a file-like object already, it will be wrapped in a new file like object that will not
    close the original file.
    This makes is easier to write code like:

        def myfync(f_or_path):
            with vaex.file.open(f_or_path) as f:
                f.write(...)

    Without closing the file when called with an open file.
    '''
    if is_file_object(path):
        if hasattr(path, 'name'):
            name = path.name
        else:
            name = 'unkown name'
        def dup():
            raise RuntimeError('Cannot duplicate this file handle')
        return FileProxy(path, name=name, dup=dup, auto_close=False)
    fs, path = parse(path, fs_options=fs_options, fs=fs, for_arrow=for_arrow)
    if fs is None:
        path = stringyfy(path)
        if for_arrow:
            if fs_options:
                raise ValueError(f'fs_options not supported for local files. You passed: {repr(fs_options)}.')
            if mmap:
                return pa.memory_map(path, mode)
            else:
                return pa.OSFile(path, mode)
        else:
            if 'b' not in mode:
                return normal_open(path, mode, encoding=encoding)
            else:
                return normal_open(path, mode)
    if mode == 'rb':
        def create():
            return fs.open_input_file(path)
    elif mode == "r":
        def create():
            fa = fs.open_input_file(path)
            fp = FileProxy(fa, path, lambda: fs.open_input_file(path))
            return io.TextIOWrapper(fp, encoding=encoding)
    elif mode == 'wb':
        def create():
            return _make_argument_optional(fs.open_output_stream, metadata=None)(path)
    elif mode == "w":
        def create():
            fa =  _make_argument_optional(fs.open_output_stream, metadata=None)(path)
            fp = FileProxy(fa, path, lambda: fs.open_output_stream(path))
            return io.TextIOWrapper(fa, encoding=encoding)
    else:
        raise ValueError(f'Only mode=rb/bw/r/w are supported, not {mode}')
    return FileProxy(create(), path, create)


def _make_argument_optional(f, **defaults):
    # workaround for https://issues.apache.org/jira/browse/ARROW-13546
    # makes f act as if arguments have default values (or ignore when the argument does not exist)
    import inspect
    import functools
    sig = inspect.signature(f)
    params = sig.parameters
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        kwargs = kwargs.copy()
        for name, value in defaults.items():
            if name in params and name not in kwargs:
                kwargs[name] = value
        try:
            return f(*args, **kwargs)
        except TypeError:
            # fallback, just pass all args
            return f(*args, **kwargs, **defaults)

    return wrapper


def dup(file):
    """Duplicate a file like object, s3 or cached file supported"""
    if isinstance(file, (vaex.file.cache.CachedFile, FileProxy)):
        return file.dup()
    else:
        return normal_open(file.name, file.mode)

def glob(path, fs_options={}, fs=None):
    if fs:
        raise ValueError('globbing with custom fs not supported yet, please open an issue.')
    scheme, _ = split_scheme(path)
    if not scheme:
        return local_glob(path)
    module = _get_scheme_handler(path)
    return module.glob(path, fs_options)


def ext(path):
    path = stringyfy(path)
    path, options = split_options(path)
    return os.path.splitext(path)[1]
