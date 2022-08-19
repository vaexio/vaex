__author__ = 'yonatanalexander'

import glob
import os
import pathlib
import functools
from io import StringIO, BytesIO
import collections
import numpy as np
import matplotlib.colors
import warnings
from base64 import b64encode
from PIL import Image
from io import BytesIO
import base64
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
    except:
        return None


def _infer(item):
    if isinstance(item, np.ndarray):
        decode = numpy_2_pil
    elif isinstance(item, bytes):
        decode = bytes_2_pil
    elif isinstance(item, str):
        if os.path.isfile(item):
            decode = PIL.Image.open()
        else:
            decode = base64_2_pil
    return _safe_apply(decode, item)


@vaex.register_function(scope='vision')
def infer(images):
    return np.array([_infer(image) for image in images], dtype="O")


@vaex.register_function(scope='vision')
def open(path, suffix=None):
    files = get_paths(path=path, suffix=suffix)
    df = vaex.from_arrays(path=files)
    df['image'] = df['path'].vision.from_path()
    return df


@vaex.register_function(scope='vision')
def resize(images, size, resample=3):
    images = [image.resize(size, resample=resample) for image in images]
    return np.array(images, dtype="O")


@vaex.register_function(scope='vision')
def to_numpy(images):
    images = [np.array(image.convert('RGB')).astype(object) for image in images]
    return np.array(images, dtype="O")


@vaex.register_function(scope='vision')
def to_bytes(arrays, format='png'):
    images = [pil_2_bytes(image_array, format=format) for image_array in arrays]
    return np.array(images, dtype="O")


@vaex.register_function(scope='vision')
def to_str(arrays, format='png'):
    images = [pil_2_base64(image_array, format) for image_array in arrays]
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
    images = [_safe_apply(base64_2_pil, image_array) for image_array in arrays]
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
    return Image.fromarray(array)


def pil_2_bytes(im, format="png"):
    f = BytesIO()
    im.save(f, format)
    return f.getvalue()


def bytes_2_pil(b):
    return PIL.Image.open(BytesIO(b))


def base64_2_pil(b):
    return PIL.Image.open(BytesIO(base64.b64decode(b)))


def pil_2_base64(im, format=format):
    return base64.b64encode(pil_2_bytes(im, format=format))


def rgba_to_url(rgba):
    bit8 = rgba.dtype == np.uint8
    if not bit8:
        rgba = (rgba * 255.).astype(np.uint8)
    im = rgba_2_pil(rgba)
    data = pil_2_bytes(im)
    data = b64encode(data)
    data = data.decode("ascii")
    imgurl = "data:image/png;base64," + data + ""
    return imgurl


pdf_modes = collections.OrderedDict()
pdf_modes["multiply"] = np.multiply
pdf_modes["screen"] = lambda a, b: a + b - a * b
pdf_modes["darken"] = np.minimum
pdf_modes["lighten"] = np.maximum

cairo_modes = collections.OrderedDict()
cairo_modes["saturate"] = (lambda aA, aB: np.minimum(1, aA + aB),
                           lambda aA, xA, aB, xB, aR: (np.minimum(aA, 1 - aB) * xA + aB * xB) / aR)
cairo_modes["add"] = (lambda aA, aB: np.minimum(1, aA + aB),
                      lambda aA, xA, aB, xB, aR: (aA * xA + aB * xB) / aR)

modes = list(pdf_modes.keys()) + list(cairo_modes.keys())


def background(shape, color="white", alpha=1, bit8=True):
    rgba = np.zeros(shape + (4,))
    rgba[:] = np.array(matplotlib.colors.colorConverter.to_rgba(color))
    rgba[..., 3] = alpha
    if bit8:
        return (rgba * 255).astype(np.uint8)
    else:
        return rgba


def fade(image_list, opacity=0.5, blend_mode="multiply"):
    result = image_list[0] * 1.
    for i in range(1, len(image_list)):
        result[result[..., 3] > 0, 3] = 1
        layer = image_list[i] * 1.0
        layer[layer[..., 3] > 0, 3] = opacity
        result = blend([result, layer], blend_mode=blend_mode)
    return result


def blend(image_list, blend_mode="multiply"):
    bit8 = image_list[0].dtype == np.uint8
    image_list = image_list[::-1]
    rgba_dest = image_list[0] * 1
    if bit8:
        rgba_dest = (rgba_dest / 255.).astype(np.float)
    for i in range(1, len(image_list)):
        rgba_source = image_list[i]
        if bit8:
            rgba_source = (rgba_source / 255.).astype(np.float)
        # assert rgba_source.dtype == image_list[0].dtype, "images have different types: first has %r, %d has %r" % (image_list[0].dtype, i, rgba_source.dtype)

        aA = rgba_source[:, :, 3]
        aB = rgba_dest[:, :, 3]

        if blend_mode in pdf_modes:
            aR = aA + aB * (1 - aA)
        else:
            aR = cairo_modes[blend_mode][0](aA, aB)
        mask = aR > 0
        for c in range(3):  # for r, g and b
            xA = rgba_source[..., c]
            xB = rgba_dest[..., c]
            if blend_mode in pdf_modes:
                f = pdf_modes[blend_mode](xB, xA)
                with np.errstate(divide='ignore', invalid='ignore'):  # these are fine, we are ok with nan's in vaex
                    result = ((1. - aB) * aA * xA + (1. - aA) * aB * xB + aA * aB * f) / aR
            else:
                result = cairo_modes[blend_mode][1](aA, xA, aB, xB, aR)
            with np.errstate(divide='ignore', invalid='ignore'):  # these are fine, we are ok with nan's in vaex
                result = (np.minimum(aA, 1 - aB) * xA + aB * xB) / aR
            # print(result)
            rgba_dest[:, :, c][(mask,)] = np.clip(result[(mask,)], 0, 1)
        rgba_dest[:, :, 3] = np.clip(aR, 0., 1)
    rgba = rgba_dest
    if bit8:
        rgba = (rgba * 255).astype(np.uint8)
    return rgba


def monochrome(I, color, vmin=None, vmax=None):
    """Turns a intensity array to a monochrome 'image' by replacing each intensity by a scaled 'color'

    Values in I between vmin  and vmax get scaled between 0 and 1, and values outside this range are clipped to this.

    Example
    >>> I = np.arange(16.).reshape(4,4)
    >>> color = (0, 0, 1) # red
    >>> rgb = vx.image.monochrome(I, color) # shape is (4,4,3)

    :param I: ndarray of any shape (2d for image)
    :param color: sequence of a (r, g and b) value
    :param vmin: normalization minimum for I, or np.nanmin(I) when None
    :param vmax: normalization maximum for I, or np.nanmax(I) when None
    :return:
    """
    if vmin is None:
        vmin = np.nanmin(I)
    if vmax is None:
        vmax = np.nanmax(I)
    normalized = (I - vmin) / (vmax - vmin)
    return np.clip(normalized[..., np.newaxis], 0, 1) * np.array(color)


def polychrome(I, colors, vmin=None, vmax=None, axis=-1):
    """Similar to monochrome, but now do it for multiple colors

    Example
    >>> I = np.arange(32.).reshape(4,4,2)
    >>> colors = [(0, 0, 1), (0, 1, 0)] # red and green
    >>> rgb = vx.image.polychrome(I, colors) # shape is (4,4,3)

    :param I: ndarray of any shape (3d will result in a 2d image)
    :param colors: sequence of [(r,g,b), ...] values
    :param vmin: normalization minimum for I, or np.nanmin(I) when None
    :param vmax: normalization maximum for I, or np.nanmax(I) when None
    :param axis: axis which to sum over, by default the last
    :return:
    """
    axes_length = len(I.shape)
    allaxes = list(range(axes_length))
    otheraxes = list(allaxes)
    otheraxes.remove((axis + axes_length) % axes_length)
    otheraxes = tuple(otheraxes)

    if vmin is None:
        vmin = np.nanmin(I, axis=otheraxes)
    if vmax is None:
        vmax = np.nanmax(I, axis=otheraxes)
    normalized = (I - vmin) / (vmax - vmin)
    return np.clip(normalized, 0, 1).dot(colors)


# c_r + c_g + c_b
def _repr_png_(self):
    from matplotlib import pylab
    fig, ax = pylab.subplots()
    self.plot(axes=ax, f=np.log1p)
    import vaex.utils
    if all([k is not None for k in [self.vx, self.vy, self.vcounts]]):
        N = self.vx.grid.shape[0]
        bounds = self.subspace_bounded.bounds
        print(bounds)
        positions = [vaex.utils.linspace_centers(bounds[i][0], bounds[i][1], N) for i in
                     range(self.subspace_bounded.subspace.dimension)]
        print(positions)
        mask = self.vcounts.grid > 0
        vx = np.zeros_like(self.vx.grid)
        vy = np.zeros_like(self.vy.grid)
        vx[mask] = self.vx.grid[mask] / self.vcounts.grid[mask]
        vy[mask] = self.vy.grid[mask] / self.vcounts.grid[mask]
        # vx = self.vx.grid / self.vcounts.grid
        # vy = self.vy.grid / self.vcounts.grid
        x2d, y2d = np.meshgrid(positions[0], positions[1])
        ax.quiver(x2d[mask], y2d[mask], vx[mask], vy[mask])
        # print x2d
        # print y2d
        # print vx
        # print vy
        # ax.quiver(x2d, y2d, vx, vy)
    ax.title.set_text(r"$\log(1+counts)$")
    ax.set_xlabel(self.subspace_bounded.subspace.expressions[0])
    ax.set_ylabel(self.subspace_bounded.subspace.expressions[1])
    # pylab.savefig
    # from .io import StringIO
    from six import StringIO
    file_object = StringIO()
    fig.canvas.print_png(file_object)
    pylab.close(fig)
    return file_object.getvalue()
