
from numba import jit
import numba
#print numba.__version__
import math
#@jit('(f4[:],f4[:], i4[:,:], f4, f4, f4, f4)')
@jit
def hist1d(x, counts, dataminx, datamaxx):
	length = len(x)
	bincountx  = counts.shape
	for i in range(length):
		binNox = int(math.floor( ((x[i] - dataminx) / (float(datamaxx) - dataminx)) * float(bincountx)))
		if binNox >= 0 and binNox < bincountx:
			counts[binNox] += 1

@jit
def hist2d(x, y, counts, dataminx, datamaxx, dataminy, datamaxy):
	length = len(x)
	#counts = np.zeros((bincountx, bincounty), dtype=np.int32)
	bincountx, bincounty = counts.shape
	#print length
	#return bindata#
	for i in range(length):
		binNox = int(math.floor( ((x[i] - dataminx) / (float(datamaxx) - dataminx)) * float(bincountx)))
		binNoy = int(math.floor( ((y[i] - dataminy) / (float(datamaxy) - dataminy)) * float(bincounty)))
		if binNox >= 0 and binNox < bincountx and binNoy >= 0 and binNoy < bincounty:
			counts[binNox, binNoy] += 1
	#step = float(datamax-datamin)/bincount
	#return numpy.arange(datamin, datamax+step/2, step), binData
	return counts

@jit
def hist3d(x, y, z, counts, dataminx, datamaxx, dataminy, datamaxy, dataminz, datamaxz):
	length = len(x)
	#counts = np.zeros((bincountx, bincounty), dtype=np.int32)
	bincountx, bincounty, bincountz = counts.shape
	#print length
	#return bindata#
	for i in range(length):
		binNox = int(math.floor( ((x[i] - dataminx) / (float(datamaxx) - dataminx)) * float(bincountx)))
		binNoy = int(math.floor( ((y[i] - dataminy) / (float(datamaxy) - dataminy)) * float(bincounty)))
		binNoz = int(math.floor( ((z[i] - dataminz) / (float(datamaxz) - dataminz)) * float(bincountz)))
		if binNox >= 0 and binNox < bincountx and binNoy >= 0 and binNoy < bincounty and binNoz < bincountz:
			counts[binNox, binNoy, binNoz] += 1
	#step = float(datamax-datamin)/bincount
	#return numpy.arange(datamin, datamax+step/2, step), binData
	return counts
