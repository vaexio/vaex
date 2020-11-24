import pyarrow.fs


import vaex.utils
gcsfs = vaex.utils.optional_import('gcsfs', '>=0.6.2')
import vaex.file.cache
from . import split_options, split_scheme
from .cache import FileSystemHandlerCached

normal_open = open


def glob(path, fs_options={}):
    if '?' in path:
        __, query = path[:path.index('?')], path[path.index('?'):]
    else:
        query = ''
    path, options = split_options(path, fs_options)
    fs = gcsfs.GCSFileSystem(**options)
    return ['gs://' + k + query for k in fs.glob(path)]


def parse(path, fs_options={}, for_arrow=False):
    path = path.replace('fsspec+gs://', 'gs://')
    path, fs_options = split_options(path, fs_options)
    scheme, path = split_scheme(path)
    assert scheme == 'gs'
    use_cache = fs_options.pop('cache', 'true') in [True, 'true', 'True', '1']
    fs = gcsfs.GCSFileSystem(**fs_options)
    fs = pyarrow.fs.FSSpecHandler(fs)
    if use_cache:
        fs = FileSystemHandlerCached(fs, scheme='gs', for_arrow=for_arrow)
    if for_arrow:
        fs = pyarrow.fs.PyFileSystem(fs)
    return fs, path
