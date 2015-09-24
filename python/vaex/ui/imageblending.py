__author__ = 'breddels'
import collections
import numpy as np
modes = collections.OrderedDict()

modes["multiply"] = np.multiply
modes["screen"] = lambda a, b: a + b - a * b
modes["darken"] = np.minimum
modes["lighten"] = np.maximum




def blend(image_list, blend_mode="multiply"):
	rgba_dest = image_list[0] * 1.
	for i in range(1, len(image_list)):
		rgba_source  = image_list[i]
		alpha_source = rgba_source[:,:,3]
		alpha_dest   = rgba_dest[:,:,3]
		#print alpha_source.min(), alpha_source.max()
		alpha_result = alpha_source + alpha_dest * (1 - alpha_source)
		mask = alpha_result > 0
		for c in range(3): # for r, g and b
			f = modes[blend_mode](rgba_dest[:,:,c], rgba_source[:,:,c])
			result = ((1.-alpha_dest) * alpha_source * rgba_source[:,:,c]  + (1.-alpha_source) * alpha_dest * rgba_dest[:,:,c] + alpha_source * alpha_dest * f) / alpha_result
			rgba_dest[:,:,c][[mask]] = np.clip(result[[mask]], 0, 1)
		rgba_dest[:,:,3] = np.clip(alpha_result, 0., 1)
	#print rgba_dest[0]
	#for c in range(3):
	#	rgba_dest[:,:,c] = rgba_dest[:,:,c] * rgba_dest[:,:,3] + (1-rgba_dest[:,:,3])
	#print rgba_dest[0]
	#rgba_dest[:,:,3] = 1
	return rgba_dest

