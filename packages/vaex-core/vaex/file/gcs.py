from urllib.parse import urlparse, parse_qs

try:
    import gcsfs
except Exception as e:
    import_exception = e
    gcsfs = None

import vaex.file.cache
from . import FileProxy

normal_open = open


def is_gs_path(path):
    return path.startswith('gs://')


def dup(f):
    return f.gcsfs.open(f.path, f.mode)


def open(path, mode='rb', **kwargs):
    if not is_gs_path(path):
        return normal_open(path, mode)
    if gcsfs is None:
        raise import_exception
    o = urlparse(path)
    assert o.scheme == 'gs'
    naked_path = path
    if '?' in naked_path:
        naked_path = naked_path[:naked_path.index('?')]
    # only use the first item
    options = {key: values[0] for key, values in parse_qs(o.query).items()}
    options.update(kwargs)
    use_cache = options.pop('cache', 'true' if mode == 'rb' else 'false') in ['true', 'True', '1']
    if 'cache' in options:
        del options['cache']
    fs = gcsfs.GCSFileSystem(**options)
    if use_cache:
        def gcs_open():
            return fs.open(naked_path, mode)
        fp = lambda: FileProxy(gcs_open(), naked_path, dup=gcs_open)
        fp = vaex.file.cache.CachedFile(fp, naked_path)
    else:
        def gcs_open():
            return fs.open(naked_path, mode)
        fp = FileProxy(gcs_open(), naked_path, dup=gcs_open)
    return fp
