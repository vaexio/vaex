__author__ = 'maartenbreddels'
import PIL.Image
import PIL.ImageDraw
try:
    from StringIO import StringIO
    py3 = False
except ImportError:
    from io import StringIO, BytesIO
    py3 = True
import collections
import numpy as np
import matplotlib.colors
import warnings
from base64 import b64encode


def rgba_2_pil(rgba):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        im = PIL.Image.frombuffer("RGBA", rgba.shape[:2], rgba, 'raw')  # , "RGBA", 0, -1)
    return im


def pil_2_data(im, format="png"):
    if py3:  # py3 case
        f = BytesIO()
    else:
        f = StringIO()
    im.save(f, format)
    return f.getvalue()


def rgba_to_url(rgba):
    bit8 = rgba.dtype == np.uint8
    if not bit8:
        rgba = (rgba * 255.).astype(np.uint8)
    im = rgba_2_pil(rgba)
    data = pil_2_data(im)
    data = b64encode(data)
    if py3:
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
            rgba_dest[:, :, c][[mask]] = np.clip(result[[mask]], 0, 1)
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
