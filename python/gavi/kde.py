import subspacefind
import numpy as np
import time

def kde1d(data, datamin=None, datamax=None, gridsize=128):
	if datamin is None:
		datamin = np.minimum(data)
	if datamax is None:
		datamax = np.maximum(data)
	N = len(data)
	prob = np.zeros(N, dtype=np.float64)
	density = np.zeros((gridsize,), dtype=np.float64)

	map = subspacefind.DensityMap1d(datamin, datamax, gridsize);
	sx1 = 2.0*(datamax-datamin)/N**(1./3);
	gmean = map.comp_data_probs(sx1, data, prob);
	map.adaptive_density(sx1, gmean, data, prob);
	map.fill(density)
	return density

def kde2d(x, y, xmin, xmax, ymin, ymax, gridshape=(128,128)):
	minima = [xmin, ymin]
	maxima = [xmax, ymax]
	columns = [x, y]
	print x
	print y
	N = len(x)
	import pdb
	#pdb.set_trace()
	assert len(y) == N
	prob = np.zeros(N, dtype=np.float64)
	density = np.zeros(gridshape, dtype=np.float64)

	k, l = 0, 1
	map = subspacefind.DensityMap2d(
			minima[k]-0.06125*(maxima[k]-minima[k]),
			maxima[k]+0.06125*(maxima[k]-minima[k]), gridshape[0],
			minima[l]-0.06125*(maxima[l]-minima[l]),
			maxima[l]+0.06125*(maxima[l]-minima[l]), gridshape[1]);
	hscale = 6.0/np.sqrt(N);
	#print columns[k].shape
	t0 = time.time()
	print "do gmean..."
	gmean = map.comp_data_probs_2d(hscale*(maxima[k]-minima[k]),
				hscale*(maxima[l]-minima[l]),
				columns[k].data, columns[l].data, prob);
	t1 = time.time()
	
	print "gmean", gmean
	print "step 1 completed", (t1-t0), "seconds"
	
	map.comp_density_2d (hscale*(maxima[k]-minima[k]), hscale*(maxima[l]-minima[l]), gmean,columns[k],columns[l],prob);
	#t2 = time.time()
	#print "step 2 completed", (t2-t1), "seconds"
	#print "step 1+2 completed", (t2-t0), "seconds"
	#map.pgm_write("density2d-py.pgm")
	#basename = "density2d-%s-%s" % tuple(column_names)
	#filename = basename + ".pgm"
	#print "writing out", filename
	#map.pgm_write(str(filename))
	map.fill(density)
	return density
