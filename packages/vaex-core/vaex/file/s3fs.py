import vaex.utils
import pyarrow.fs
s3fs = vaex.utils.optional_import('s3fs')

import vaex.file.cache
from . import split_options, split_scheme
from .cache import FileSystemHandlerCached



def glob(path, fs_options={}):
    if '?' in path:
        __, query = path[:path.index('?')], path[path.index('?'):]
    else:
        query = ''
    scheme, _ = split_scheme(path)
    path = path.replace('s3fs://', 's3://')
    path, options = split_options(path, fs_options)
    # anon is for backwards compatibility
    if 'cache' in options:
        del options['cache']
    anon = (options.pop('anon', None) in [True, 'true', 'True', '1']) or (options.pop('anonymous', None) in [True, 'true', 'True', '1'])
    s3 = s3fs.S3FileSystem(anon=anon, **options)
    return [f'{scheme}://' + k + query for k in s3.glob(path)]


def parse(path, fs_options):
    path = path.replace('fsspec+s3://', 's3://')
    path, fs_options = split_options(path, fs_options)
    scheme, path = split_scheme(path)
    use_cache = fs_options.pop('cache', 'true') in ['true', 'True', '1']
    # anon is for backwards compatibility
    anon = (fs_options.pop('anon', None) in [True, 'true', 'True', '1']) or (fs_options.pop('anonymous', None) in [True, 'true', 'True', '1'])
    s3 = s3fs.S3FileSystem(anon=anon, default_fill_cache=False, **fs_options)
    fs = pyarrow.fs.FSSpecHandler(s3)
    if use_cache:
        fs = FileSystemHandlerCached(fs, scheme='s3')
    fs = pyarrow.fs.PyFileSystem(fs)
    return fs, path
