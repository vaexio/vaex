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
		im = PIL.Image.frombuffer("RGBA", rgba.shape[:2], rgba, 'raw') #, "RGBA", 0, -1)
	return im

def pil_2_data(im, format="png"):
	if py3: # py3 case
		f = BytesIO()
	else:
		f = StringIO()
	im.save(f, format)
	return f.getvalue()

def rgba_to_url(rgba):
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
						   lambda aA, xA, aB, xB, aR: (np.minimum(aA, 1-aB)*xA + aB*xB)/aR)
cairo_modes["add"] = (lambda aA, aB: np.minimum(1, aA + aB),
						   lambda aA, xA, aB, xB, aR: (aA*xA + aB*xB)/aR)

modes = list(pdf_modes.keys()) + list(cairo_modes.keys())
def background(shape, color="white", alpha=1, bit8=True):
	rgba = np.zeros(shape + (4,))
	rgba[:] = np.array(matplotlib.colors.colorConverter.to_rgba(color))
	rgba[...,3] = alpha
	if bit8:
		return (rgba*255).astype(np.uint8)
	else:
		return rgba


def fade(image_list, opacity=0.5, blend_mode="multiply"):
	result = image_list[0] * 1.
	for i in range(1, len(image_list)):
		result[result[...,3] > 0,3] = 1
		layer = image_list[i] * 1.0
		layer[layer[...,3] > 0,3] = opacity
		result = blend([result, layer], blend_mode=blend_mode)
	return result

def blend(image_list, blend_mode="multiply"):
	bit8 = image_list[0].dtype == np.uint8
	image_list = image_list[::-1]
	rgba_dest = image_list[0] * 1
	if bit8:
		rgba_dest = (rgba_dest/255.).astype(np.float)
	for i in range(1, len(image_list)):
		rgba_source  = image_list[i]
		if bit8:
			rgba_source = (rgba_source/255.).astype(np.float)
		#assert rgba_source.dtype == image_list[0].dtype, "images have different types: first has %r, %d has %r" % (image_list[0].dtype, i, rgba_source.dtype)

		aA = rgba_source[:,:,3]
		aB   = rgba_dest[:,:,3]

		if blend_mode in pdf_modes:
			aR = aA + aB * (1 - aA)
		else:
			aR = cairo_modes[blend_mode][0](aA, aB)
		mask = aR > 0
		for c in range(3): # for r, g and b
			xA = rgba_source[...,c]
			xB = rgba_dest[...,c]
			if blend_mode in pdf_modes:
				f = pdf_modes[blend_mode](xB, xA)
				result = ((1.-aB) * aA * xA  + (1.-aA) * aB * xB + aA * aB * f) / aR
			else:
				result = cairo_modes[blend_mode][1](aA, xA, aB, xB, aR)
			result = (np.minimum(aA, 1-aB)*xA + aB*xB)/aR
			#print(result)
			rgba_dest[:,:,c][[mask]] = np.clip(result[[mask]], 0, 1)
		rgba_dest[:,:,3] = np.clip(aR, 0., 1)
	rgba = rgba_dest
	if bit8:
		rgba = (rgba*255).astype(np.uint8)
	return rgba

