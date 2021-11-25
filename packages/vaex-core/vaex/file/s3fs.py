import vaex.utils
import pyarrow.fs
import warnings
s3fs = vaex.utils.optional_import('s3fs')

import vaex.file.cache
from . import split_options, split_scheme
from .cache import FileSystemHandlerCached
from .s3 import patch_profile
from ..cache import fingerprint


fs_cache = {}


def translate_options(fs_options):
    # translate options of arrow to s3fs
    fs_options = fs_options.copy()
    not_supported = {
        'role_arn', 'session_name', 'external_id', 'load_frequency', 'background_writes', 'profile', 'profile_name'
    }
    for key in not_supported:
        if key in fs_options:
             warnings.warn(f'The option {key} is not supported using s3fs instead of arrow, so it will be ignored')
             fs_options.pop(key)

    # If scheme key is present in the fs_options use that, else default it to https
    if 'endpoint_override' in fs_options.keys():
        if 'scheme' in fs_options.keys():
            fs_options['endpoint_override'] = fs_options.pop('scheme') + "://" + fs_options.pop('endpoint_override')
        else:
            fs_options['endpoint_override'] = "https://" + fs_options.pop('endpoint_override')

    # top level
    mapping = {
        'anonymous': 'anon',
    }
    for key in list(fs_options):
        if key in mapping:
            fs_options[mapping[key]] = fs_options.pop(key)

    # client kwargs
    mapping = {
        'access_key': 'aws_access_key_id',
        'secret_key': 'aws_secret_access_key',
        'session_token': 'aws_session_token',
        'region': 'region_name',
        'endpoint_override': 'endpoint_url',
    }
    fs_options['client_kwargs'] = fs_options.get('client_kwargs', {})
    for key in list(fs_options):
        if key in mapping:
            fs_options['client_kwargs'][mapping[key]] = fs_options.pop(key)
    return fs_options


def glob(path, fs_options={}):
    if '?' in path:
        __, query = path[:path.index('?')], path[path.index('?'):]
    else:
        query = ''
    scheme, _ = split_scheme(path)
    path = path.replace('s3fs://', 's3://')
    path, fs_options = split_options(path, fs_options)
    # anon is for backwards compatibility
    if 'cache' in fs_options:
        del fs_options['cache']
    # standardize value, and make bool
    anon = (fs_options.pop('anon', None) in [True, 'true', 'True', '1']) or (fs_options.pop('anonymous', None) in [True, 'true', 'True', '1'])
    fs_options = patch_profile(fs_options)
    fs_options = translate_options(fs_options)
    s3 = s3fs.S3FileSystem(anon=anon, **fs_options)
    return [f'{scheme}://' + k + query for k in s3.glob(path)]


def parse(path, fs_options, for_arrow=False):
    path = path.replace('fsspec+s3://', 's3://')
    path, fs_options = split_options(path, fs_options)
    scheme, path = split_scheme(path)
    use_cache = fs_options.pop('cache', 'true') in [True, 'true', 'True', '1']
    # standardize value, and make bool
    anon = (fs_options.pop('anon', None) in [True, 'true', 'True', '1']) or (fs_options.pop('anonymous', None) in [True, 'true', 'True', '1'])
    fs_options = patch_profile(fs_options)
    fs_options = translate_options(fs_options)

    bucket = path.split('/')[0]
    key = fingerprint(bucket, fs_options)
    if key not in fs_cache:
        s3 = s3fs.S3FileSystem(anon=anon, default_fill_cache=False, **fs_options)
        fs_cache[key] = s3
    else:
        s3 = fs_cache[key]
    fs = pyarrow.fs.FSSpecHandler(s3)
    if use_cache:
        fs = FileSystemHandlerCached(fs, scheme='s3', for_arrow=for_arrow)
    if for_arrow:
        fs = pyarrow.fs.PyFileSystem(fs)
    return fs, path
