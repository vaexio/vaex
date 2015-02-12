__author__ = 'breddels'
import collections
import numpy as np
modes = collections.OrderedDict()

modes["multiply"] = np.multiply
modes["screen"] = lambda a, b: a + b - a * b
modes["darken"] = np.minimum
modes["lighten"] = np.maximum




def blend(image_list):
	pass


