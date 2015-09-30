# -*- coding: utf-8 -*-
# from http://www.ecse.rpi.edu/Homepages/wrf/Research/Short_Notes/pnpoly.html		

from numba import jit

#@jit('(f4[:], f4[:], f4[:], f4[:], u1[:], f8, f8, f8)', nopython=True)
#@jit('(f8[:], f8[:], f8[:], f8[:], u1[:], f8, f8, f8)', nopython=True)
@jit(nopython=True)
def pnpoly(vertx, verty, testx, testy, inside, meanx, meany, radius):
	nvert = len(vertx)
	ntest = len(testx)
	for k in range(ntest):
		distancesq = (testx[k] - meanx)**2 + (testy[k] - meany)**2
		inside[k] = 0
		if distancesq < radius**2: # quick check
			inside[k] = 0
			if 1:
				j = nvert-1
				for i in range(nvert):
					if (((verty[i]>testy[k]) != (verty[j]>testy[k])) and (testx[k] < (vertx[j]-vertx[i]) * (testy[k]-verty[i]) / (verty[j]-verty[i]) + vertx[i]) ):
						inside[k] = not inside[k]
					j = i
