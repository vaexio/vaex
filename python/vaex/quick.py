__author__ = 'breddels'
import vaex as vx
import numpy as np
import scipy
from matplotlib import pylab

def example():
	import utils
	return vx.open(vx.utils.get_data_file("helmi-dezeeuw-2000-10p.hdf5"))
