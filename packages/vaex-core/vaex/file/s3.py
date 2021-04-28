import configparser
import warnings
import os

import pyarrow as pa

class CaseConfigParser(configparser.ConfigParser):
    def optionxform(self, optionstr):
        return optionstr


def patch_profile(fs_options):
    fs_options = fs_options.copy()
    if 'profile' in fs_options:
        profile = fs_options.pop('profile')
        config = CaseConfigParser()
        path = os.path.expanduser('~/.aws/credentials')
        config.read(path)
        warnings.warn(f'Reading key/secret from ~/.aws/credentials using profile: {path}')
        fs_options['access_key'] = config[profile]['aws_access_key_id']
        fs_options['secret_key'] = config[profile]['aws_secret_access_key']
    return fs_options


def parse(path, fs_options, for_arrow=False):
    from .s3arrow import parse
    fs_options = patch_profile(fs_options)
    try:
        # raise pa.lib.ArrowNotImplementedError('FOR TESTING')
        return parse(path, fs_options, for_arrow=for_arrow)
    except pa.lib.ArrowNotImplementedError:
        # fallback
        from .s3fs import parse
        return parse(path, fs_options, for_arrow=for_arrow)


def glob(path, fs_options={}):
    from .s3fs import glob
    fs_options = patch_profile(fs_options)
    return glob(path, fs_options)
