__author__ = 'breddels'
import logging
logger = logging.getLogger("vaex.file")

opener_classes = []
normal_open = open

def register(cls):
    opener_classes.append(cls)

import vaex.file.other
try:
    import vaex.hdf5 as hdf5
except ImportError:
    hdf5 = None
if hdf5:
    import vaex.hdf5.dataset

def can_open(path, *args, **kwargs):
    for name, class_ in list(vaex.file.other.dataset_type_map.items()):
        if class_.can_open(path, *args):
            return True


def open(path, *args, **kwargs):
    dataset_class = None
    openers = []
    for opener in opener_classes:
        if opener.can_open(path, *args, **kwargs):
            return opener.open(path, *args, **kwargs)
    if hdf5:
        openers.extend(hdf5.dataset.dataset_type_map.items())
    openers.extend(vaex.file.other.dataset_type_map.items())
    for name, class_ in list(openers):
        logger.debug("trying %r with class %r" % (path, class_))
        if class_.can_open(path, *args, **kwargs):
            logger.debug("can open!")
            dataset_class = class_
            break
    if dataset_class:
        dataset = dataset_class(path, *args, **kwargs)
        return dataset


def dup(file):
    """Duplicate a file like object, s3 or cached file supported"""
    if isinstance(file, vaex.file.cache.CachedFile):
        return file.dup()
    elif vaex.file.s3.s3fs is not None and isinstance(file, vaex.file.s3.s3fs.core.S3File):
        return vaex.file.s3.dup(file)
    else:
        return normal_open(file.name, file.mode)
