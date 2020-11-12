import configparser
import os
import io

import pyarrow as pa
import pyarrow.fs

from . import split_options, FileProxy, split_scheme
from .cache import CachedFile


def open_s3(path, mode='rb', fs_options={}):
    path, options = split_options(path, fs_options)
    use_cache = options.pop('cache', 'true' if mode == 'rb' else 'false') in [True, 'true', 'True', '1']
    file_system, path = parse(path, options)
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
            elif mode == "r":
                return io.TextIOWrapper(file_system.open_input_file(path))
            elif mode == "wb":
                # although a stream, will suffice in most cases
                return file_system.open_output_stream(path)
            elif mode == "w":
                return io.TextIOWrapper(file_system.open_output_stream(path))
            else:
                raise ValueError(f'Only mode=rb/bw/r/w are supported, not {mode}')
        return FileProxy(arrow_open(), path, dup=arrow_open)


def parse(path, options):
    path, options = split_options(path, options)
    scheme, _ = split_scheme(path)
    if scheme == 's3':
        # anon is for backwards compatibility
        options['anonymous'] = (options.pop('anon', None) in [True, 'true', 'True', '1']) or (options.pop('anonymous', None) in [True, 'true', 'True', '1'])
        use_cache = options.pop('cache', 'false') in [True, 'true', 'True', '1']
        if use_cache:
            raise ValueError('cache=true not supported in combination with arrow')
        if 'profile' in options:
            # TODO: ideally, Apache Arrow should take a profile argument
            profile = options.pop('profile')
            config = configparser.ConfigParser()
            config.read('~/.aws/credentials')
            options['access_key'] = config[profile]['aws_access_key_id']
            options['secret_key'] = config[profile]['aws_secret_access_key']
        if 'region' not in options:
            # we use this to get the default region
            file_system, path = pa.fs.FileSystem.from_uri(path)
            options['region'] = file_system.region
        file_system = pa.fs.S3FileSystem(**options)
    elif scheme == 'gs':
        import gcsfs
        file_system = gcsfs.GCSFileSystem(**options)
    elif scheme:
        raise ValueError('scheme {scheme} in path {path} not supported yet, feel free to open an issue at https://github.com/vaexio/vaex/issues/')
    else:
        file_system = None
    return file_system, path
