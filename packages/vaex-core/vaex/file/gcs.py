try:
    import gcsfs
except Exception as e:
    import_exception = e
    gcsfs = None

import vaex.file.cache
from . import split_options, FileProxy

normal_open = open


def is_gs_path(path):
    return path.startswith('gs://')


def dup(f):
    return f.gcsfs.open(f.path, f.mode)


def glob(path, **kwargs):
    if '?' in path:
        __, query = path[:path.index('?')], path[path.index('?'):]
    else:
        query = ''
    path, options = split_options(path, **kwargs)
    if gcsfs is None:
        raise import_exception
    # anon is for backwards compatibility
    anom = (options.pop('anon', None) in ['true', 'True', '1']) or (options.pop('anonymous', None) in ['true', 'True', '1'])
    fs = gcsfs.GCSFileSystem(**options)
    return ['gs://' + k + query for k in fs.glob(path)]


def open(path, mode='rb', **kwargs):
    if gcsfs is None:
        raise import_exception
    path, options = split_options(path, **kwargs)
    use_cache = options.pop('cache', 'true' if mode == 'rb' else 'false') in ['true', 'True', '1']
    # common iterface between s3 and gcs
    if (options.pop('anon', None) in ['true', 'True', '1']) or (options.pop('anonymous', None) in ['true', 'True', '1']):
        options['token'] = 'anon'
    fs = gcsfs.GCSFileSystem(**options)
    if use_cache:
        def gcs_open():
            return fs.open(path, mode)
        fp = lambda: FileProxy(gcs_open(), path, dup=gcs_open)
        fp = vaex.file.cache.CachedFile(fp, path)
    else:
        def gcs_open():
            return fs.open(path, mode)
        fp = FileProxy(gcs_open(), path, dup=gcs_open)
    return fp
