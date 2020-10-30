try:
    import s3fs
except Exception as e:
    import_exception = e
    s3fs = None

import vaex.file.cache
from . import split_options, FileProxy, split_scheme


normal_open = open


def is_s3_path(path):
    return path.startswith('s3fs://')

def dup(f):
    return f.s3.open(f.path, f.mode)


def glob(path, **kwargs):
    if '?' in path:
        __, query = path[:path.index('?')], path[path.index('?'):]
    else:
        query = ''
    scheme, _ = split_scheme(path)
    path = path.replace('s3fs://', 's3://')
    path, options = split_options(path, **kwargs)
    if s3fs is None:
        raise import_exception
    # anon is for backwards compatibility
    if 'cache' in options:
        del options['cache']
    anon = (options.pop('anon', None) in ['true', 'True', '1']) or (options.pop('anonymous', None) in ['true', 'True', '1'])
    s3 = s3fs.S3FileSystem(anon=anon, **options)
    return [f'{scheme}://' + k + query for k in s3.glob(path)]


def open(path, mode='rb', **kwargs):
    path = path.replace('s3fs://', 's3://')
    path, options = split_options(path, **kwargs)
    if s3fs is None:
        raise import_exception
    use_cache = options.pop('cache', 'true' if mode == 'rb' else 'false') in ['true', 'True', '1']
    # anon is for backwards compatibility
    anon = (options.pop('anon', None) in ['true', 'True', '1']) or (options.pop('anonymous', None) in ['true', 'True', '1'])
    s3 = s3fs.S3FileSystem(anon=anon, default_fill_cache=False, **options)
    def open():
        return s3.open(path, mode)
    if use_cache:
        fp = lambda: FileProxy(open(), path, open)
        fp = vaex.file.cache.CachedFile(fp, path)
    else:
        fp = FileProxy(open(), path, dup=open)
    return fp
