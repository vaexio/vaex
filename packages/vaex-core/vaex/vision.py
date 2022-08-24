__author__ = 'yonatanalexander'

import glob
import os
import pathlib
import functools
import numpy as np
import warnings
import io
import vaex
import vaex.utils

try:
    import PIL
    import base64
except:
    PIL = vaex.utils.optional_import("PIL.Image", modules="pillow")


def get_paths(path, suffix=None):
    if isinstance(path, list):
        return functools.reduce(lambda a, b: get_paths(a, suffix=suffix) + get_paths(b, suffix=suffix), path)
    if os.path.isfile(path):
        files = [path]
    elif os.path.isdir(path):
        files = []
        if suffix is not None:
            files = [str(path) for path in pathlib.Path(path).rglob(f"*{suffix}")]
        else:
            for suffix in ['jpg', 'png', 'jpeg', 'ppm', 'thumbnail']:
                files.extend([str(path) for path in pathlib.Path(path).rglob(f"*{suffix}")])
    elif isinstance(path, str) and len(glob.glob(path)) > 0:
        return glob.glob(path)
    else:
        raise ValueError(
            f"path: {path} do not point to an image, a directory of images, or a nested directory of images, or a glob path of files")
    # TODO validate the files without opening it
    return files


def _safe_apply(f, image_array):
    try:
        return f(image_array)
    except Exception as e:
        return None


def _infer(item):
    if hasattr(item, 'as_py'):
        item = item.as_py()
    if isinstance(item, np.ndarray):
        decode = numpy_2_pil
    elif isinstance(item, int):
        item = np.ndarray(item)
        decode = numpy_2_pil
    elif isinstance(item, bytes):
        decode = bytes_2_pil
    elif isinstance(item, str):
        if os.path.isfile(item):
            decode = PIL.Image.open
        else:
            decode = str_2_pil
    else:
        raise RuntimeError(f"Can't handle item {item}")
    return _safe_apply(decode, item)


@vaex.register_function(scope='vision')
def infer(images):
    images = [_infer(image) for image in images]
    return np.array(images, dtype="O")


@vaex.register_function(scope='vision')
def open(path, suffix=None):
    files = get_paths(path=path, suffix=suffix)
    df = vaex.from_arrays(path=files)
    df['image'] = df['path'].vision.infer()
    return df


@vaex.register_function(scope='vision')
def filename(images):
    images = [image.filename if hasattr(image, 'filename') else None for image in images]
    return np.array(images, dtype="O")


@vaex.register_function(scope='vision')
def resize(images, size, resample=3, **kwargs):
    images = [image.resize(size, resample=resample, **kwargs) for image in images]
    return np.array(images, dtype="O")


@vaex.register_function(scope='vision')
def to_numpy(images):
    images = [pil_2_numpy(image) for image in images]
    return np.array(images, dtype="O")


@vaex.register_function(scope='vision')
def to_bytes(arrays, format='png'):
    images = [pil_2_bytes(image_array, format=format) for image_array in arrays]
    return np.array(images, dtype="O")


@vaex.register_function(scope='vision')
def to_str(arrays, format='png', encoding=None):
    images = [pil_2_str(image_array, format=format, encoding=encoding) for image_array in arrays]
    return np.array(images, dtype="O")


@vaex.register_function(scope='vision')
def from_numpy(arrays):
    images = [_safe_apply(numpy_2_pil, image_array) for image_array in arrays]
    return np.array(images, dtype="O")


@vaex.register_function(scope='vision')
def from_bytes(arrays):
    images = [_safe_apply(bytes_2_pil, image_array) for image_array in arrays]
    return np.array(images, dtype="O")


@vaex.register_function(scope='vision')
def from_str(arrays):
    images = [_safe_apply(str_2_pil, image_array) for image_array in arrays]
    return np.array(images, dtype="O")


@vaex.register_function(scope='vision')
def from_path(arrays):
    images = [_safe_apply(PIL.Image.open, image_array) for image_array in vaex.array_types.tolist(arrays)]
    return np.array(images, dtype="O")


def rgba_2_pil(rgba):
    # TODO remove?
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        im = PIL.Image.fromarray(rgba[::-1], "RGBA")  # , "RGBA", 0, -1)
    return im


def numpy_2_pil(array):
    return PIL.Image.fromarray(np.uint8(array))


def pil_2_numpy(im):
    if im is not None:
        return np.array(im).astype(object)
    return None


def pil_2_bytes(im, format="png"):
    f = io.BytesIO()
    im.save(f, format)
    return base64.b64encode(f.getvalue())


def bytes_2_pil(b):
    return PIL.Image.open(io.BytesIO(base64.b64decode(b)))


def pil_2_str(im, format="png", encoding=None):
    args = [encoding] if encoding else []
    return pil_2_bytes(im, format=format).decode(*args)


def str_2_pil(im, encoding=None):
    args = [encoding] if encoding else []
    return bytes_2_pil(im.encode(*args))


def rgba_to_url(rgba):
    bit8 = rgba.dtype == np.uint8
    if not bit8:
        rgba = (rgba * 255.).astype(np.uint8)
    im = rgba_2_pil(rgba)
    data = pil_2_bytes(im)
    data = base64.b64encode(data)
    data = data.decode("ascii")
    imgurl = "data:image/png;base64," + data + ""
    return imgurl
