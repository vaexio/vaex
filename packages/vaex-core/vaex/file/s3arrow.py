import os
import io
from .s3 import patch_profile

import pyarrow as pa
import pyarrow.fs

from . import split_options, FileProxy, split_scheme
from .cache import FileSystemHandlerCached


def glob(path, fs_options={}):
    from .s3 import glob
    return glob(path, fs_options)


def parse(path, fs_options, for_arrow=False):
    path, fs_options = split_options(path, fs_options)
    path = path.replace('arrow+s3://', 's3://')
    scheme, _ = split_scheme(path)
    assert scheme == 's3'
    # anon is for backwards compatibility
    fs_options['anonymous'] = (fs_options.pop('anon', None) in [True, 'true', 'True', '1']) or (fs_options.pop('anonymous', None) in [True, 'true', 'True', '1'])
    fs_options = patch_profile(fs_options)
    use_cache = fs_options.pop('cache', 'true') in [True, 'true', 'True', '1']
    if 'region' not in fs_options:
        # we use this to get the default region
        file_system, path = pa.fs.FileSystem.from_uri(path)
        # Remove this line for testing purposes to fake not having s3 support
        # raise pyarrow.lib.ArrowNotImplementedError('FOR TESTING')
        fs_options['region'] = file_system.region
    fs = pa.fs.S3FileSystem(**fs_options)
    if use_cache:
        fs = FileSystemHandlerCached(fs, scheme='s3', for_arrow=for_arrow)
        if for_arrow:
            fs = pyarrow.fs.PyFileSystem(fs)
    return fs, path
