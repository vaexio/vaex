# -*- coding: utf-8 -*-
import numpy
from numpy import *
import Image
im = Image.open('density.pgm')
#import pdb
#pdb.set_trace()
#im.show()
#dsa
data = array(im)
print data.shape

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


#plt.subplot(221)


def multisum(a, axes):
	correction = 0
	for axis in axes:
		a = numpy.sum(a, axis=axis-correction)
		correction += 1
	return a

dim = 2
quality = [[0.83, 0.91], [0.91, 0.41]]
for j in range(2):
	for i in range(j, 2):
		plt.subplot(2,2,1+i+2*j)
		if j == i:
			# marginalize
			axes = range(dim)
			axes.remove(i)
			data1d = multisum(data, axes)
			plt.plot(data1d)
		else:
			plt.imshow(data)
		plt.title("Quality: %.2f" % quality[i][j])
		
plt.show()
