import iniconfig
from urllib.parse import urlparse, parse_qs
import os

import pyarrow as pa
import pyarrow.fs

from . import split_options, FileProxy
from .cache import CachedFile


def open_s3(path, mode='rb', **kwargs):
    path, options = split_options(path, **kwargs)
    use_cache = options.pop('cache', 'true') in ['true', 'True', '1']
    # anon is for backwards compatibility
    options['anonymous'] = (options.pop('anon', None) in ['true', 'True', '1']) or (options.pop('anonymous', None) in ['true', 'True', '1'])
    if 'profile' in options:
        # TODO: ideally, Apache Arrow should take a profile argument
        profile = options.pop('profile')
        ic = iniconfig.IniConfig(os.path.expanduser('~/.aws/credentials'))
        options['access_key'] = ic[profile]['aws_access_key_id']
        options['secret_key'] = ic[profile]['aws_secret_access_key']
    if 'region' not in options:
        # we use this to get the default region
        file_system, path = pa.fs.FileSystem.from_uri(path)
        options['region'] = file_system.region
    file_system = pa.fs.S3FileSystem(**options)
    file = file_system.open_input_file(path)
    if use_cache:
        # we lazily open the file, if all is cached, we don't need to connect to s3
        if mode != "rb":
            raise ValueError(f'Cannot combine s3 with mode other than "rb", {mode} is not supported')
        def dup():
            return file_system.open_input_file(path)
        fp = lambda: FileProxy(file_system.open_input_file(path), path, dup)
        return CachedFile(fp, path)
    else:
        def arrow_open():
            if mode == "rb":
                return file_system.open_input_file(path)
            elif mode == "wb":
                # although a stream, will suffice in most cases
                return file_system.open_output_stream(path)
            else:
                raise ValueError(f'Only mode="rb" and mode="wb" supported, not {mode}')
        return FileProxy(arrow_open(), path, dup=arrow_open)
