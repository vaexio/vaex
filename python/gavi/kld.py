# -*- coding: utf-8 -*-

import gavi.logging as logging
logger = logging.getLogger("gavi.kld")
import numpy as np
import gavi.histogram
import numba.dispatcher

class KlDivergenceShuffle(object):
	def __init__(self, dataset, pairs, gridsize=128):
		self.dataset = dataset
		self.pairs = pairs
		self.dimension = len(self.pairs[0])
		self.logger = logger.getLogger('kld')
		self.gridsize = gridsize
		logger.debug("dimension: %d, pairs: %s" % (self.dimension, self.pairs))
		
		
	def get_jobs(self):
		def job(pair):
			pass
		
		
def kld_shuffled(columns, Ngrid=128, datamins=None, datamaxes=None, offset=1):
	if datamins is None:
		datamins = np.array([np.nanmin(column) for column in columns])
	if datamaxes is None:
		datamaxes = np.array([np.nanmax(column) for column in columns])
	dim = len(columns)
	counts = np.zeros((Ngrid, ) * dim, dtype=np.float64)
	counts_shuffled = np.zeros((Ngrid, ) * dim, dtype=np.float64)
	D_kl = -1
	if len(columns) == 2:
		x, y = columns
		#print x
		#print y
		print x, y, counts, counts_shuffled, datamins[0], datamaxes[0], datamins[1], datamaxes[1], offset
		try:
			gavi.histogram.hist2d_and_shuffled(x, y, counts, counts_shuffled, datamins[0], datamaxes[0], datamins[1], datamaxes[1], offset)
		except:
			args = [x, y, counts, counts_shuffled, datamins[0], datamaxes[0], datamins[1], datamaxes[1], offset]
			sig = [numba.dispatcher.typeof_pyval(a) for a in args]
			print sig
			raise
			
		print "counts", sum(counts)
		deltax = [float(datamaxes[i] - datamins[i]) for i in range(dim)]
		dx = np.array([deltax[d]/counts.shape[d] for d in range(dim)])
		density = counts/np.sum(counts)# * np.sum(dx)
		density_shuffled = counts_shuffled / np.sum(counts_shuffled)# * np.sum(dx)
		mask = (density_shuffled > 0) & (density>0)
		#print density
		D_kl = np.sum(density[mask] * np.log(density[mask]/density_shuffled[mask]))# * np.sum(dx)
		#if D_kl < 0:
		#	import pdb
		#	pdb.set_trace()
	return D_kl
				
		
	
	
	
if __name__ == "__main__":
	import gavi.dataset
	import gavi.files
	from optparse import OptionParser
	parser = OptionParser() #usage="")

	#parser.add_option("-n", "--name",
	#                 help="dataset name [default=%default]", default="data", type=str)
	#parser.add_option("-o", "--output",
	#                 help="dataset output filename [by default the suffix of input filename will be replaced by hdf5]", default=None, type=str)
	(options, args) = parser.parse_args()
	if len(args) > 0:
		filename = args[0]
	else:
		filename = "gaussian4d-1e7.hdf5"
	
	path = gavi.files.get_datafile(filename)
	dataset = gavi.dataset.Hdf5MemoryMapped(path)
	#for column_name in dataset.column_names:
	#	print column_name
	
	subspace = dataset.subspace(1)[:3] * dataset.subspace(1)[:3]
	#subspace = Subspace(dataset, [("x",), ("y",), ("z",)]