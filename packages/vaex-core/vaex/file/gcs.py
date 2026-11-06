import pyarrow.fs


import vaex.utils
gcsfs = vaex.utils.optional_import('gcsfs', '>=0.6.2')
import vaex.file.cache
from . import split_options, split_scheme
from .cache import FileSystemHandlerCached

normal_open = open


def _gcsfs_options(options):
    # accept the same anon/anonymous options as s3. gcsfs ignores them, and then
    # falls back to google default credentials, which probes the GCE metadata server
    # on every request and makes reads of a public bucket extremely slow
    options = options.copy()
    anon = (options.pop('anon', None) in [True, 'true', 'True', '1']) or (options.pop('anonymous', None) in [True, 'true', 'True', '1'])
    if anon:
        options['token'] = 'anon'
    return options


def glob(path, fs_options={}):
    if '?' in path:
        __, query = path[:path.index('?')], path[path.index('?'):]
    else:
        query = ''
    path, options = split_options(path, fs_options)
    fs = gcsfs.GCSFileSystem(**_gcsfs_options(options))
    return ['gs://' + k + query for k in fs.glob(path)]


def parse(path, fs_options={}, for_arrow=False):
    path = path.replace('fsspec+gs://', 'gs://')
    path, fs_options = split_options(path, fs_options)
    scheme, path = split_scheme(path)
    assert scheme == 'gs'
    use_cache = fs_options.pop('cache', 'true') in [True, 'true', 'True', '1']
    fs = gcsfs.GCSFileSystem(**_gcsfs_options(fs_options))
    fs = pyarrow.fs.FSSpecHandler(fs)
    if use_cache:
        fs = FileSystemHandlerCached(fs, scheme='gs', for_arrow=for_arrow)
    if for_arrow:
        fs = pyarrow.fs.PyFileSystem(fs)
    return fs, path
