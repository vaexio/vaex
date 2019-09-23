try:
    from urllib.parse import urlparse, parse_qs
except ImportError:
    from urlparse import urlparse, parse_qs

try:
    import s3fs
except Exception as e:
    import_exception = e
    s3fs = None

import vaex.file.cache


normal_open = open


def is_s3_path(path):
    return path.startswith('s3://')

def dup(f):
    return f.s3.open(f.path, f.mode)

def open(path, mode='rb', **kwargs):
    if not is_s3_path(path):
        return normal_open(path, mode)
    if s3fs is None:
        raise import_exception
    o = urlparse(path)
    assert o.scheme == 's3'
    naked_path = path
    if '?' in naked_path:
        naked_path = naked_path[:naked_path.index('?')]
    # only use the first item
    options = {key: values[0] for key, values in parse_qs(o.query).items()}
    options.update(kwargs)
    use_cache = options.get('cache', 'true') in ['true', 'True', '1']
    if 'cache' in options:
        del options['cache']
    anon = options.get('anon', 'false') in ['true', 'True', '1']
    if 'anon' in options:
        del options['anon']
    s3 = s3fs.S3FileSystem(anon=anon, default_block_size=1,
                           default_fill_cache=False, **options)
    if use_cache:
        fp = lambda: s3.open(naked_path, mode)
        fp = vaex.file.cache.CachedFile(fp, naked_path)
    else:
        fp = s3.open(naked_path, mode)
    return fp
