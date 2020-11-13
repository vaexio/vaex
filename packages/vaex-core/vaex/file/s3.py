import pyarrow as pa


def parse(path, fs_options):
    from .s3arrow import parse
    try:
        return parse(path, fs_options)
    except pa.lib.ArrowNotImplementedError:
        # fallback
        from .s3fs import parse
        return parse(path, fs_options)


def glob(path, fs_options={}):
    from .s3fs import glob
    return glob(path, fs_options)
