# -*- coding: utf-8 -*-
import mmap
import os
import math
import time
import itertools
import functools
import collections
import sys
import platform
import vaex.export
import os
import re
from functools import reduce
import threading

import numpy as np
import numexpr as ne
import concurrent.futures
import astropy.table

from vaex.utils import Timer
import vaex.events
import vaex.ui.undo
import vaex.grids
import vaex.multithreading
import vaex.promise
import vaex.execution
import logging
import astropy.io.fits as fits


# h5py doesn't want to build at readthedocs
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
try:
	import h5py
except:
	if not on_rtd:
		raise

logger = logging.getLogger("vaex")
lock = threading.Lock()
dataset_type_map = {}


class Link(object):
	def __init__(self, dataset):
		self.dataset = dataset
		self.listeners = []
		
	def sendExpression(self, expression, receiver):
		for listener in self.listeners:
			if listener != receiver:
				listener.onChangeExpression(expression)
		if expression in self.dataset.global_links:
			# merge the listeners of this link to the other link
			logger.debug("merging with link %r" % expression)
			merging_link = self.dataset.global_links[expression]
			for receiver in merging_link.listeners:
				self.listeners.append(receiver)
		else:
			# add new mapping
			logger.debug("renamed link %r" % expression)
			self.dataset.global_links[expression] = self
		# remove old mapping
		for key, link in list(self.dataset.global_links.items()):
			logger.debug("link[%r] = %r" % (key, link))
			if (link == self) and key != expression: # remove dangling links
				logger.debug("removing link %r" % key)
				del self.dataset.global_links[key]
				
	def sendRanges(self, range_, receiver):
		for listener in self.listeners:
			if listener != receiver:
				listener.onChangeRange(range_)

	def sendRangesShow(self, range_, receiver):
		for listener in self.listeners:
			if listener != receiver:
				listener.onChangeRangeShow(range_)

	#
	@staticmethod
	def sendCompute(links, receivers):
		listener_set = set(list(itertools.chain.from_iterable([link.listeners for link in links])))

		for listener in listener_set:
			if listener not in receivers:
				listener.onCompute()

	def sendPlot(self, receiver):
		for listener in self.listeners:
			if listener != receiver:
				listener.onPlot()
				
				

executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
executor = vaex.execution.default_executor

class ColumnBase(object):
	def __init__(self, dataset, expression):
		self.dataset = dataset
		self.expression = expression

class Column(ColumnBase):
	def get(self, i1, i2):
		return self.dataset.columns[self.name][i1:i2]

class ColumnExpression(ColumnBase):
	def __init__(self, dataset, expression):
		super(ColumnExpression, self).__init__(dataset, expression)

#	def get(self, i1, i2):

class Task(vaex.promise.Promise):
	def __init__(self, dataset, expressions):
		vaex.promise.Promise.__init__(self)
		self.dataset = dataset
		self.expressions = expressions
		self.expressions_all = list(expressions)

	@property
	def dimension(self):
		return len(self.expressions)

class TaskMapReduce(Task):
	def __init__(self, dataset, expressions, map, reduce, converter=lambda x: x, info=False):
		Task.__init__(self, dataset, expressions)
		self._map = map
		self._reduce = reduce
		self.converter = converter
		self.info = info

	def map(self, thread_index, i1, i2, *blocks):
		if self.info:
			return self._map(thread_index, i1, i2, *blocks)
		else:
			return self._map(*blocks)#[self.map(block) for block in blocks]

	def reduce(self, results):
		return self.converter(reduce(self._reduce, results))



class TaskHistogram(Task):
	def __init__(self, dataset, subspace, expressions, size, limits, masked=False, weight=None):
		Task.__init__(self, dataset, expressions)
		self.subspace = subspace
		self.dtype = np.float64
		self.size = size
		self.limits = limits
		self.masked = masked
		self.weight = weight
		#self.grids = vaex.grids.Grids(self.dataset, self.dataset.executor.thread_pool, *expressions)
		#self.grids.ranges = limits
		#self.grids.grids["counts"] = vaex.grids.Grid(self.grids, size, self.dimension, None)
		shape = (self.dataset.executor.thread_pool.nthreads,) + ( self.size,) * self.dimension
		self.data = np.zeros(shape, dtype=self.dtype)
		self.ranges_flat = []
		self.minima = []
		self.maxima = []
		for limit in self.limits:
			self.ranges_flat.extend(limit)
			vmin, vmax = limit
			self.minima.append(vmin)
			self.maxima.append(vmax)
		if self.weight is not None:
			self.expressions_all.append(weight)
		#print self.ranges_flat

	def __repr__(self):
		name = self.__class__.__module__ + "." +self.__class__.__name__
		return "<%s(dataset=%r, expressions=%r, size=%r, limits=%r)> instance at 0x%x" % (name, self.dataset, self.expressions, self.size, self.limits, id(self))

	def map(self, thread_index, i1, i2, *blocks):
		class Info(object):
			pass
		info = Info()
		info.i1 = i1
		info.i2 = i2
		info.first = i1 == 0
		info.last = i2 == len(self.dataset)
		info.size = i2-i1
		#print "bin", i1, i2, info.last
		#self.grids["counts"].bin_block(info, *blocks)
		mask = self.dataset.mask
		data = self.data[thread_index]
		if self.masked:
			blocks = [block[mask[i1:i2]] for block in blocks]

		subblock_weight = None
		if len(blocks) == len(self.expressions) + 1:
			subblock_weight = blocks[-1]
			blocks = list(blocks[:-1])
		#print subblocks[0]
		#print subblocks[1]

		if self.dimension == 1:
			vaex.vaexfast.histogram1d(blocks[0], subblock_weight, data, *self.ranges_flat)
		elif self.dimension == 2:
			vaex.vaexfast.histogram2d(blocks[0], blocks[1], subblock_weight, data, *self.ranges_flat)
		elif self.dimension == 3:
			vaex.vaexfast.histogram3d(blocks[0], blocks[1], blocks[2], subblock_weight, data, *self.ranges_flat)
		else:
			blocks = list(blocks) # histogramNd wants blocks to be a list
			vaex.vaexfast.histogramNd(blocks, subblock_weight, data, self.minima, self.maxima)

		return i1
		#return map(self._map, blocks)#[self.map(block) for block in blocks]

	def reduce(self, results):
		for i in range(1, self.dataset.executor.thread_pool.nthreads):
			self.data[0] += self.data[i]
		return self.data[0]
		#return self.data

import scipy.ndimage.filters

class SubspaceGridded(object):
	def __init__(self, subspace_bounded, grid, vx=None, vy=None, vcounts=None):
		self.subspace_bounded = subspace_bounded
		self.grid = grid
		self.vx = vx
		self.vy = vy
		self.vcounts = vcounts

	def vector(self, weightx, weighty, size=32):
		counts = self.subspace_bounded.gridded_by_histogram(size=size)
		vx = self.subspace_bounded.gridded_by_histogram(size=size, weight=weightx)
		vy = self.subspace_bounded.gridded_by_histogram(size=size, weight=weighty)
		return SubspaceGridded(self.subspace_bounded, self.grid, vx=vx, vy=vy, vcounts=counts)

	def filter_gaussian(self, sigmas=1):
		return SubspaceGridded(self.subspace_bounded, scipy.ndimage.filters.gaussian_filter(self.grid, sigmas))

	def clip_relative(self, v1, v2):
		vmin = self.grid.min()
		vmax = self.grid.max()
		width = vmax - vmin
		return SubspaceGridded(self.subspace_bounded, np.clip(self.grid, vmin + v1 * width, vmin + v2 * width))

	def volr(self,  **kwargs):
		import vaex.notebook
		return vaex.notebook.volr(self, **kwargs)

	def plot(self, axes=None, **kwargs):
		self.subspace_bounded.subspace.plot(np.log1p(self.grid), self.subspace_bounded.bounds, axes=axes, **kwargs)

	def _repr_png_(self):
		from matplotlib import pylab
		fig, ax = pylab.subplots()
		self.plot(axes=ax, f=np.log1p)
		import vaex.utils
		if all([k is not None for k in [self.vx, self.vy, self.vcounts]]):
			N = self.vx.grid.shape[0]
			bounds = self.subspace_bounded.bounds
			print(bounds)
			positions = [vaex.utils.linspace_centers(bounds[i][0], bounds[i][1], N) for i in range(self.subspace_bounded.subspace.dimension)]
			print(positions)
			mask = self.vcounts.grid > 0
			vx = np.zeros_like(self.vx.grid)
			vy = np.zeros_like(self.vy.grid)
			vx[mask] = self.vx.grid[mask] / self.vcounts.grid[mask]
			vy[mask] = self.vy.grid[mask] / self.vcounts.grid[mask]
			#vx = self.vx.grid / self.vcounts.grid
			#vy = self.vy.grid / self.vcounts.grid
			x2d, y2d = np.meshgrid(positions[0], positions[1])
			ax.quiver(x2d[mask], y2d[mask], vx[mask], vy[mask])
			#print x2d
			#print y2d
			#print vx
			#print vy
			#ax.quiver(x2d, y2d, vx, vy)
		ax.title.set_text("$\log(1+counts)$")
		ax.set_xlabel(self.subspace_bounded.subspace.expressions[0])
		ax.set_ylabel(self.subspace_bounded.subspace.expressions[1])
		#pylab.savefig
		#from .io import StringIO
		from cStringIO import StringIO
		file_object = StringIO()
		fig.canvas.print_png(file_object)
		pylab.close(fig)
		return file_object.getvalue()

	def cube_png(self, f=np.log1p, colormap="afmhot", file="cube.png"):
		if self.grid.shape != ((128,) *3):
			logger.error("only 128**3 cubes are supported")
			return None
		colormap_name = "afmhot"
		import matplotlib.cm
		colormap = matplotlib.cm.get_cmap(colormap_name)
		mapping = matplotlib.cm.ScalarMappable(cmap=colormap)
		#pixmap = QtGui.QPixmap(32*2, 32)
		data = np.zeros((128*8, 128*16, 4), dtype=np.uint8)

		#mi, ma = 1*10**self.mod1, self.data3d.max()*10**self.mod2
		grid = f(self.grid)
		vmin, vmax = grid.min(), grid.max()
		grid_normalized = (grid-vmin) / (vmax-vmin)
		#intensity_normalized = (np.log(self.data3d + 1.) - np.log(mi)) / (np.log(ma) - np.log(mi));
		import PIL.Image
		for y2d in range(8):
			for x2d in range(16):
				zindex = x2d + y2d*16
				I = grid_normalized[zindex]
				rgba = mapping.to_rgba(I,bytes=True) #.reshape(Nx, 4)
				#print rgba.shape
				subdata = data[y2d*128:(y2d+1)*128, x2d*128:(x2d+1)*128]
				for i in range(3):
					subdata[:,:,i] = rgba[:,:,i]
				subdata[:,:,3] = (grid_normalized[zindex]*255).astype(np.uint8)# * 0 + 255
				if 0:
					filename = "cube%03d.png" % zindex
					img = PIL.Image.frombuffer("RGB", (128, 128), subdata[:,:,0:3] * 1)
					print(("saving to", filename))
					img.save(filename)
		img = PIL.Image.frombuffer("RGBA", (128*16, 128*8), data, 'raw') #, "RGBA", 0, -1)
		#filename = "cube.png"
		#print "saving to", file
		img.save(file, "png")

		if 0:
			filename = "colormap.png"
			print(("saving to", filename))
			height, width = self.colormap_data.shape[:2]
			img = PIL.Image.frombuffer("RGB", (width, height), self.colormap_data)
			img.save(filename)


class SubspaceBounded(object):
	def __init__(self, subspace, bounds):
		self.subspace = subspace
		self.bounds = bounds

	def histogram(self, size=256, weight=None):
		return self.subspace.histogram(limits=self.bounds, size=size, weight=weight)

	def gridded(self, size=256, weight=None):
		return self.gridded_by_histogram(size=size, weight=weight)

	def gridded_by_histogram(self, size=256, weight=None):
		grid = self.histogram(size=size, weight=weight)
		return SubspaceGridded(self, grid)




class Subspace(object):
	"""A Subspace represent a subset of columns or expressions from a dataset.

	subspace are not instantiated directly, but by 'calling' the dataset like this:

	>>> subspace_xy = some_dataset("x", "y")
	>>> subspace_r = some_dataset("sqrt(x**2+y**2)")

	See `vaex.dataset.Dataset` for more documentation.

	"""
	def __init__(self, dataset, expressions, executor, async, masked=False):
		"""

		:param Dataset dataset: the dataset the subspace refers to
		:param list[str] expressions: list of expressions that forms the subspace
		:param Executor executor: responsible for executing the tasks
		:param bool async: return answers directly, or as a promise
		:param bool masked: work on the selection or not
		:return:
		"""
		self.dataset = dataset
		self.expressions = expressions
		self.executor = executor
		self.async = async
		self.is_masked = masked

	def __repr__(self):
		name = self.__class__.__module__ + "." +self.__class__.__name__
		return "<%s(dataset=%r, expressions=%r, async=%r, is_masked=%r)> instance at 0x%x" % (name, self.dataset, self.expressions, self.async, self.is_masked, id(self))

	@property
	def dimension(self):
		return len(self.expressions)

	def get_selection(self):
		return self.dataset.get_selection("default") if self.is_masked else None

	def selected(self):
		return self.__class__(self.dataset, expressions=self.expressions, executor=self.executor, async=self.async, masked=True)

	def plot(self, grid=None, limits=None, center=None, weight=None, f=lambda x: x, axes=None, **kwargs):
		import pylab
		if limits is None:
			limits = self.limits_sigma()
		if center is not None:
			limits = np.array(limits) - np.array(center).reshape(2,1)
		if grid is None:
			grid = self.histogram(limits=limits, weight=weight)
		if axes is None:
			axes = pylab.gca()
		axes.imshow(f(grid), extent=np.array(limits).flatten(), origin="lower", **kwargs)

	def figlarge(self):
		import pylab
		pylab.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')

	#def bounded(self):
	#	return self.bounded_by_minmax()

	def bounded_by(self, limits):
		"""Returns a bounded subspace (SubspaceBounded) with limits as given by limits

		:param limits: sequence of [(min, max), ..., (min, max)] values
		:rtype: SubspaceBounded
		"""
		return SubspaceBounded(self, np.array(limits))

	def bounded_by_minmax(self):
		"""Returns a bounded subspace (SubspaceBounded) with limits given by Subspace.minmax()

		:rtype: SubspaceBounded
		"""
		bounds = self.minmax()
		return SubspaceBounded(self, bounds)

	bounded = bounded_by_minmax

	def bounded_by_sigmas(self, sigmas=3, square=False):
		"""Returns a bounded subspace (SubspaceBounded) with limits given by Subspace.limits_sigma()

		:rtype: SubspaceBounded
		"""
		bounds = self.limits_sigma(sigmas=sigmas, square=square)
		return SubspaceBounded(self, bounds)

	def minmax(self):
		"""Return a sequence of [(min, max), ..., (min, max)] corresponding to each expression in this subspace ignoring NaN.
		"""
		raise NotImplementedError
	def mean(self):
		"""Return a sequence of [mean, ... , mean] corresponding to the mean of each expression in this subspace ignoring NaN.
		"""
		raise NotImplementedError
	def var(self, means=None):
		"""Return a sequence of [var, ... , var] corresponding to the variance of each expression in this subspace ignoring NaN.
		"""
		raise NotImplementedError
	def sum(self):
		"""Return a sequence of [sum, ... , sum] corresponding to the sum of values of each expression in this subspace ignoring NaN."""
		raise NotImplementedError
	def histogram(self, limits, size=256, weight=None):
		"""Return a grid of shape (size, ..., size) corresponding to the dimensionality of this subspace containing the counts in each element

		The type of the grid of np.float64

		"""
		raise NotImplementedError
	def limits_sigma(self, sigmas=3, square=False):
		raise NotImplementedError

	def row(self, index):
		return np.array([self.dataset.evaluate(expression, i1=index, i2=index+1)[0] for expression in self.expressions])


class SubspaceLocal(Subspace):
	"""Subclass of subspace which implemented methods that can be run locally.
	"""

	def _toarray(self, list):
		return np.array(list)

	@property
	def pre(self):
		self.executor.pre

	@property
	def post(self):
		self.executor.post

	def _task(self, task):
		"""Helper function for returning tasks results, result when immediate is True, otherwise the task itself, which is a promise"""
		if self.async:
			# should return a task or a promise nesting it
			return self.executor.schedule(task)
		else:
			return self.executor.run(task)

	def minmax(self):
		def min_max_reduce(minmax1, minmax2):
			if minmax1 is None:
				return minmax2
			if minmax2 is None:
				return minmax1
			result = []
			for d in range(self.dimension):
				min1, max1 = minmax1[d]
				min2, max2 = minmax2[d]
				result.append((min(min1, min2), max(max1, max2)))
			return result
		def min_max_map(thread_index, i1, i2, *blocks):
			if self.is_masked:
				mask = self.dataset.mask
				blocks = [block[mask[i1:i2]] for block in blocks]
				is_empty = all(~mask[i1:i2])
				if is_empty:
					return None
			#with lock:
			#print blocks
			#with lock:
			#	print thread_index, i1, i2, blocks
			return [vaex.vaexfast.find_nan_min_max(block) for block in blocks]
		task = TaskMapReduce(self.dataset, self.expressions, min_max_map, min_max_reduce, self._toarray, info=True)
		return self._task(task)

	def mean(self):
		def mean_reduce(means1, means2):
			means = []
			for mean1, mean2 in zip(means1, means2):
				means.append(np.nanmean([mean1, mean2]))
			return means
		if self.is_masked:
			mask = self.dataset.mask
			task = TaskMapReduce(self.dataset, self.expressions, lambda thread_index, i1, i2, *blocks: [np.nanmean(block[mask[i1:i2]]) for block in blocks], mean_reduce, self._toarray, info=True)
		else:
			task = TaskMapReduce(self.dataset, self.expressions, lambda *blocks: [np.nanmean(block) for block in blocks], mean_reduce, self._toarray)
		return self._task(task)

	def var(self, means=None):
		# variances are linear, use the mean to reduce
		def vars_reduce(vars_and_counts1, vars_and_counts2):
			vars_and_counts = []
			for (var1, count1), (var2, count2) in zip(vars_and_counts1, vars_and_counts2):
				vars_and_counts.append( [np.nansum([var1*count1, var2*count2])/(count1+count2), count1+count2] )
			return vars_and_counts
		def remove_counts(vars_and_counts):
			return self._toarray(vars_and_counts)[:,0]
		if self.is_masked:
			mask = self.dataset.mask
			def var_map(thread_index, i1, i2, *blocks):
				if means is not None:
					return [(np.nanmean((block[mask[i1:i2]]-mean)**2), np.count_nonzero(~np.isnan(block[mask[i1:i2]]))) for block, mean in zip(blocks, means)]
				else:
					return [(np.nanmean(block[mask[i1:i2]]**2), np.count_nonzero(~np.isnan(block[mask[i1:i2]]))) for block in blocks]
			task = TaskMapReduce(self.dataset, self.expressions, var_map, vars_reduce, remove_counts, info=True)
		else:
			def var_map(*blocks):
				if means is not None:
					return [(np.nanmean((block-mean)**2), np.count_nonzero(~np.isnan(block))) for block, mean in zip(blocks, means)]
				else:
					return [(np.nanmean(block**2), np.count_nonzero(~np.isnan(block))) for block in blocks]
			task = TaskMapReduce(self.dataset, self.expressions, var_map, vars_reduce, remove_counts)
		return self._task(task)

	def correlation(self, means, vars):
		if self.dimension != 2:
			raise ValueError("correlation is only defined for 2d subspaces, not %dd" % self.dimension)

		meanx, meany = means
		sigmax, sigmay = vars[0]**0.5, vars[1]**0.5

		def remove_counts_and_normalize(covar_and_count):
			covar, counts = covar_and_count
			return covar/counts / (sigmax * sigmay)

		def covars_reduce(covar_and_count1, covar_and_count2):
			if covar_and_count1 is None:
				return covar_and_count2
			if covar_and_count2 is None:
				return covar_and_count1
			else:
				covar1, count1 = covar_and_count1
				covar2, count2 = covar_and_count2
				return [np.nansum([covar1, covar2]), count1+count2]

		mask = self.dataset.mask
		def covar_map(thread_index, i1, i2, *blocks):
			#return [(np.nanmean((block[mask[i1:i2]]-mean)**2), np.count_nonzero(~np.isnan(block[mask[i1:i2]]))) for block, mean in zip(blocks, means)]
			blockx, blocky = blocks
			if self.is_masked:
				blockx, blocky = blockx[mask[i1:i2]], blocky[mask[i1:i2]]
			counts = np.count_nonzero( ~(np.isnan(blockx) | np.isnan(blocky)) )
			if counts == 0:
				return None
			else:
				return np.nansum((blockx - meanx) * (blocky - meany)), counts

		task = TaskMapReduce(self.dataset, self.expressions, covar_map, covars_reduce, remove_counts_and_normalize, info=True)
		return self._task(task)

	def sum(self):
		if self.is_masked:
			mask = self.dataset.mask
			task = TaskMapReduce(self.dataset,\
								 self.expressions, lambda thread_index, i1, i2, *blocks: [np.nansum(block[mask[i1:i2]], dtype=np.float64) for block in blocks],\
								 lambda a, b: np.array(a) + np.array(b), self._toarray, info=True)
		else:
			task = TaskMapReduce(self.dataset, self.expressions, lambda *blocks: [np.nansum(block, dtype=np.float64) for block in blocks], lambda a, b: np.array(a) + np.array(b), self._toarray)
		return self._task(task)

	def histogram(self, limits, size=256, weight=None):
		task = TaskHistogram(self.dataset, self, self.expressions, size, limits, masked=self.is_masked, weight=weight)
		return self._task(task)

	def limits_sigma(self, sigmas=3, square=False):
		means = self.mean()
		stds = self.var(means=means)**0.5
		if square:
			stds = np.repeat(stds.mean(), len(stds))
		return np.array(list(zip(means-sigmas*stds, means+sigmas*stds)))

	def _not_needed_current(self):
		index = self.dataset.get_current_row()
		def find(thread_index, i1, i2, *blocks):
			if (index >= i1) and (index < i2):
				return [block[index-i1] for block in blocks]
			else:
				return None
		task = TaskMapReduce(self.dataset, self.expressions, find, lambda a, b: a if b is None else b, info=True)
		return self._task(task)

	def nearest(self, point, metric=None):
		metric = metric or [1.] * len(point)
		def nearest_in_block(thread_index, i1, i2, *blocks):
			if self.is_masked:
				mask = self.dataset.mask[i1:i2]
				if mask.sum() == 0:
					return None
				blocks = [block[mask] for block in blocks]
			distance_squared = np.sum( [(blocks[i]-point[i])**2.*metric[i] for i in range(self.dimension)], axis=0 )
			min_index = np.argmin(distance_squared)
			return min_index + i1, distance_squared[min_index]**0.5, [block[min_index] for block in blocks]
		def nearest_reduce(a, b):
			if a is None:
				return b
			if b is None:
				return a
			if a[1] < b[1]:
				return a
			else:
				return b
		if self.is_masked:
			pass
		mask = self.dataset.mask
		task = TaskMapReduce(self.dataset,\
							 self.expressions,
							 nearest_in_block,\
							 nearest_reduce, info=True)
		return self._task(task)







import vaex.events
import cgi

# mutex for numexpr (is not thread save)
ne_lock = threading.Lock()

class _BlockScope(object):
	def __init__(self, dataset, i1, i2, **variables):
		"""

		:param DatasetLocal dataset: the *local*  dataset
		:param i1: start index
		:param i2: end index
		:param values:
		:return:
		"""
		self.dataset = dataset
		self.i1 = i1
		self.i2 = i2
		self.variables = variables
		self.values = dict(self.variables)
		self.buffers = {}

	def move(self, i1, i2):
		length_new = i2 - i1
		length_old = self.i2 - self.i1
		if length_new > length_old: # old buffers are too small, discard them
			self.buffers = {}
		else:
			for name in list(self.buffers.keys()):
				self.buffers[name] = self.buffers[name][:length_new]
		self.i1 = i1
		self.i2 = i2
		self.values = dict(self.variables)

	def _ensure_buffer(self, column):
		if column not in self.buffers:
			self.buffers[column] = np.zeros(self.i2-self.i1)

	def evaluate(self, expression, out=None):
		result = ne.evaluate(expression, local_dict=self, out=out)
		return result

	def __getitem__(self, variable):
		#logger.debug("get " + variable)
		try:
			if variable in self.dataset.get_column_names():
				if self.dataset._needs_copy(variable):
					self._ensure_buffer(variable)
					self.values[variable] = self.buffers[variable] = self.dataset.columns[variable][self.i1:self.i2].astype(np.float64)
				else:
					self.values[variable] = self.dataset.columns[variable][self.i1:self.i2]
			elif variable in self.values:
				return self.values[variable]
			elif variable in list(self.dataset.virtual_columns.keys()):
				expression = self.dataset.virtual_columns[variable]
				self._ensure_buffer(variable)
				self.evaluate(expression, out=self.buffers[variable])
				self.values[variable] = self.buffers[variable]
			if variable not in self.values:
				raise KeyError("Unknown variables or column: %r" % (variable,))

			return self.values[variable]
		except:
			logger.exception("error in evaluating: %r" % variable)
			raise

main_executor = vaex.execution.Executor(vaex.multithreading.pool)
from vaex.execution import Executor


class Selection(object):
	def __init__(self, dataset, previous_selection, mode):
		self.dataset = dataset
		# we don't care about the previous selection if we simply replace the current selection
		self.previous_selection = previous_selection if mode != "replace" else None
		self.mode = mode

	def execute(self, executor, execute_fully=False):
		if execute_fully and self.previous_selection:
			self.previous_selection.execute(executor=executor, execute_fully=execute_fully)

class SelectionExpression(Selection):
	def __init__(self, dataset, boolean_expression, previous_selection, mode):
		super(SelectionExpression, self).__init__(dataset, previous_selection, mode)
		self.boolean_expression = boolean_expression

	def to_dict(self):
		previous = None
		if self.previous_selection:
			previous = self.previous_selection.to_dict()
		return dict(type="expression", boolean_expression=self.boolean_expression, mode=self.mode, previous_selection=previous)


	def execute(self, executor, execute_fully=False):
		super(SelectionExpression, self).execute(executor=executor, execute_fully=execute_fully)
		mode_function = _select_functions[self.mode]
		if self.boolean_expression is None:
			self.dataset._set_mask(None)
			promise = vaex.promise.Promise()
			promise.fulfill(None)
			return promise
		else:
			logger.debug("executing selection: %r, mode: %r", self.boolean_expression, self.mode)
			mask = np.zeros(len(self.dataset), dtype=np.bool)
			def map(thread_index, i1, i2, block):
				mask[i1:i2] = mode_function(None if self.dataset.mask is None else self.dataset.mask[i1:i2], block == 1)
				return 0
			def reduce(*args):
				None
			expr = self.dataset(self.boolean_expression, executor=executor)
			task = TaskMapReduce(self.dataset, [self.boolean_expression], lambda thread_index, i1, i2, *blocks: [map(thread_index, i1, i2, block) for block in blocks], reduce, info=True)
			def apply_mask(*args):
				#print "Setting mask"
				self.dataset._set_mask(mask)
			task.then(apply_mask)
			return expr._task(task)

class SelectionLasso(Selection):
	def __init__(self, dataset, boolean_expression_x, boolean_expression_y, xseq, yseq, previous_selection, mode):
		super(SelectionLasso, self).__init__(dataset, previous_selection, mode)
		self.boolean_expression_x = boolean_expression_x
		self.boolean_expression_y = boolean_expression_y
		self.xseq = xseq
		self.yseq = yseq

	def execute(self, executor, execute_fully=False):
		super(SelectionLasso, self).execute(executor=executor, execute_fully=execute_fully)
		mode_function = _select_functions[self.mode]
		x, y = np.array(self.xseq, dtype=np.float64), np.array(self.yseq, dtype=np.float64)
		meanx = x.mean()
		meany = y.mean()
		radius = np.sqrt((meanx-x)**2 + (meany-y)**2).max()

		mask = np.zeros(len(self.dataset), dtype=np.bool)
		def lasso(thread_index, i1, i2, blockx, blocky):
			vaex.vaexfast.pnpoly(x, y, blockx, blocky, mask[i1:i2], meanx, meany, radius)
			mask[i1:i2] = mode_function(None if self.dataset.mask is None else self.dataset.mask[i1:i2], mask[i1:i2])
			return 0
		def reduce(*args):
			None
		subspace = self.dataset(self.boolean_expression_x, self.boolean_expression_y, executor=executor)
		task = TaskMapReduce(self.dataset, subspace.expressions, lambda thread_index, i1, i2, blockx, blocky: lasso(thread_index, i1, i2, blockx, blocky), reduce, info=True)
		def apply_mask(*args):
			#print "Setting mask"
			self.dataset._set_mask(mask)
		task.then(apply_mask)
		return subspace._task(task)

	def to_dict(self):
		previous = None
		if self.previous_selection:
			previous = self.previous_selection.to_dict()
		return dict(type="lasso",
					boolean_expression_x=self.boolean_expression_x,
					boolean_expression_y=self.boolean_expression_y,
					xseq=vaex.utils.make_list(self.xseq),
					yseq=vaex.utils.make_list(self.yseq),
					mode=self.mode,
					previous_selection=previous)

def selection_from_dict(dataset, values):
	kwargs = dict(values)
	kwargs["dataset"] = dataset
	del kwargs["type"]
	if values["type"] == "lasso":
		kwargs["previous_selection"] = selection_from_dict(dataset, values["previous_selection"]) if values["previous_selection"] else None
		return SelectionLasso(**kwargs)
	elif values["type"] == "expression":
		kwargs["previous_selection"] = selection_from_dict(dataset, values["previous_selection"]) if values["previous_selection"] else None
		return SelectionExpression(**kwargs)
	else:
		raise ValueError, "unknown type: %r, in dict: %r" % (values["type"], values)


class Dataset(object):
	"""All datasets are encapsulated in this class, local or remote dataets

	Each dataset has a number of columns, and a number of rows, the length of the dataset.
	Most operations on the data are not done directly on the dataset, but on subspaces of it, using the
	Subspace class. Subspaces are created by 'calling' the dataset, like this:

	>> subspace_xy = some_dataset("x", "y")
	>> subspace_r = some_dataset("sqrt(x**2+y**2)")

	All Datasets have one 'selection', and all calculations by Subspace are done on the whole dataset (default)
	or for the selection. The following example shows how to use the selection.

	>>> some_dataset.select("x < 0")
	>>> subspace_xy = some_dataset("x", "y")
	>>> subspace_xy_selected = subspace_xy.selected()


	TODO: active fraction, length and shuffled



	:type signal_selection_changed: events.Signal
	:type executor: Executor
	"""
	def __init__(self, name, column_names, executor=None):
		self.name = name
		self.column_names = column_names
		self.executor = executor or main_executor
		self.signal_pick = vaex.events.Signal("pick")
		self.signal_sequence_index_change = vaex.events.Signal("sequence index change")
		self.signal_selection_changed = vaex.events.Signal("selection changed")
		self.signal_active_fraction_changed = vaex.events.Signal("active fraction changed")

		self.undo_manager = vaex.ui.undo.UndoManager()
		self.variables = collections.OrderedDict()
		self.variables["pi"] = np.pi
		self.variables["e"] = np.e
		self.virtual_columns = collections.OrderedDict()
		self._length = None
		self._full_length = None
		self._active_fraction = 1
		self._current_row = None

		self.mask = None # a bitmask for the selection does not work for server side

		# maps from name to list of Selection objets
		self.selection_histories = collections.defaultdict(list)
		# after an undo, the last one in the history list is not the active one, -1 means no selection
		self.selection_history_indices = collections.defaultdict(lambda: -1)
		self._auto_fraction= False

	def is_local(self): raise NotImplementedError

	def get_auto_fraction(self):
		return self._auto_fraction

	def set_auto_fraction(self, enabled):
		self._auto_fraction = enabled

	@classmethod
	def can_open(cls, path, *args, **kwargs):
		return False

	@classmethod
	def get_options(cls, path):
		return []

	@classmethod
	def option_to_args(cls, option):
		return []

	def __call__(self, *expressions, **kwargs):
		"""Return a Subspace for this dataset with the given expressions:

		Example:

		>>> subspace_xy = some_dataset("x", "y")

		:rtype: Subspace
		"""
		raise NotImplementedError

	def set_variable(self, name, value):
		self.variables[name] = value

	def get_variable(self, name):
		return self.variables[name]

	def evaluate(self, expression, i1=None, i2=None, out=None):
		raise NotImplementedError

	def _block_scope(self, i1, i2):
		return _BlockScope(self, i1, i2, **self.variables)

	def select(self, boolean_expression, mode="replace", selection_name="default"):
		"""Select rows based on the boolean_expression, if there was a previous selection, the mode is taken into account.

		if boolean_expression is None, remove the selection, has_selection() will returns false

		Note that per dataset, only one selection is possible.

		:param str boolean_expression: boolean expression, such as 'x < 0', '(x < 0) & (y > -10)' or None to remove the selection
		:param str mode: boolean operation to perform with the previous selection, "replace", "and", "or", "xor", "subtract"
		:return: None
		"""
		raise NotImplementedError

	def add_virtual_columns_matrix3d(self, x, y, z, xnew, ynew, znew, matrix, matrix_name):
		"""

		:param str x: name of x column
		:param str y:
		:param str z:
		:param str xnew: name of transformed x column
		:param str ynew:
		:param str znew:
		:param list[list] matrix: 2d array or list, with [row,column] order
		:param str matrix_name:
		:return:
		"""
		m = matrix_name
		for i in range(3):
			for j in range(3):
				self.set_variable(matrix_name +"_%d%d" % (i,j), matrix[i,j])
		self.virtual_columns[xnew] = "{m}_00 * {x} + {m}_01 * {y} + {m}_02 * {z}".format(**locals())
		self.virtual_columns[ynew] = "{m}_10 * {x} + {m}_11 * {y} + {m}_12 * {z}".format(**locals())
		self.virtual_columns[znew] = "{m}_20 * {x} + {m}_21 * {y} + {m}_22 * {z}".format(**locals())

	def add_virtual_columns_celestial(self, long_in, lat_in, long_out, lat_out, input=None, output=None, name_prefix="__celestial", radians=False):
		import kapteyn.celestial as c
		input = input or c.eq
		output = input or c.gal
		matrix = c.skymatrix((input,'j2000',c.fk5), output)[0]
		if not radians:
			long_in = "pi/180.*%s" % long_in
			lat_in = "pi/180.*%s" % lat_in
		x_in = name_prefix+"_in_x"
		y_in = name_prefix+"_in_y"
		z_in = name_prefix+"_in_z"
		x_out = name_prefix+"_out_x"
		y_out = name_prefix+"_out_y"
		z_out = name_prefix+"_out_z"
		self.add_virtual_column(x_in, "cos({long_in})*cos({lat_in})".format(**locals()))
		self.add_virtual_column(y_in, "sin({long_in})*cos({lat_in})".format(**locals()))
		self.add_virtual_column(z_in, "sin({lat_in})".format(**locals()))
		self.add_virtual_columns_matrix3d(x_in, y_in, z_in, x_out, y_out, z_out,\
										  matrix, name_prefix+"_matrix")
		long_out_expr = "arctan2({y_out},{x_out})".format(**locals())
		lat_out_expr = "arctan2({z_out},sqrt({x_out}**2+{y_out}**2))".format(**locals())
		if not radians:
			long_out_expr = "180./pi*%s" % long_out_expr
			lat_out_expr = "180./pi*%s" % lat_out_expr

		self.add_virtual_column(long_out, long_out_expr)
		self.add_virtual_column(lat_out, lat_out_expr)



	def add_virtual_columns_rotation(self, x, y, xnew, ynew, angle_degrees):
		"""

		:param str x: name of x column
		:param str y:
		:param str xnew: name of transformed x column
		:param str ynew:
		:param float angle_degrees: rotation in degrees, anti clockwise
		:return:
		"""
		theta = np.radians(angle_degrees)
		matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
		m = matrix_name = x +"_" +y + "_rot"
		for i in range(2):
			for j in range(2):
				self.set_variable(matrix_name +"_%d%d" % (i,j), matrix[i,j])
		self.virtual_columns[xnew] = "{m}_00 * {x} + {m}_01 * {y}".format(**locals())
		self.virtual_columns[ynew] = "{m}_10 * {x} + {m}_11 * {y}".format(**locals())


	def add_virtual_columns_spherical_to_cartesian(self, alpha, delta, distance, xname, yname, zname, radians=True):
		if not radians:
			alpha = "pi/180.*%s" % alpha
			delta = "pi/180.*%s" % delta
		self.virtual_columns[xname] = "-sin(%s) * cos(%s) * %s" % (alpha, delta, distance)
		self.virtual_columns[yname] = "cos(%s) * cos(%s) * %s" % (alpha, delta, distance)
		self.virtual_columns[zname] = "sin(%s) * %s" % (delta, distance)

	def add_virtual_columns_equatorial_to_galactic_cartesian(self, alpha, delta, distance, xname, yname, zname, radians=True, alpha_gp=np.radians(192.85948), delta_gp=np.radians(27.12825), l_omega=np.radians(32.93192)):
		"""From http://arxiv.org/pdf/1306.2945v2.pdf"""
		if not radians:
			alpha = "pi/180.*%s" % alpha
			delta = "pi/180.*%s" % delta
		# TODO: sort our x,y,z order and the l_omega
		self.virtual_columns[zname] = "{distance} * (cos({delta}) * cos({delta_gp}) * cos({alpha} - {alpha_gp}) + sin({delta}) * sin({delta_gp}))".format(**locals())
		self.virtual_columns[xname] = "{distance} * (cos({delta}) * sin({alpha} - {alpha_gp}))".format(**locals())
		self.virtual_columns[yname] = "{distance} * (sin({delta}) * cos({delta_gp}) - cos({delta}) * sin({delta_gp}) * cos({alpha} - {alpha_gp}))".format(**locals())

	def add_virtual_column(self, name, expression):
		"""Add a virtual column to the dataset

		Example:
		>>> dataset.add_virtual_column("r", "sqrt(x**2 + y**2 + z**2)")
		>>> dataset.select("r < 10")
		"""
		self.virtual_columns[name] = expression



	def __todo_repr_html_(self):
		html = """<div>%s - %s (length=%d)</div>""" % (cgi.escape(repr(self.__class__)), self.name, len(self))
		html += """<table>"""
		for column_name in self.get_column_names():
			html += "<tr><td>%s</td><td>type unknown</td></tr>" % (column_name)
		html += "</table>"
		return html


	def current_sequence_index(self):
		"""TODO"""
		return 0

	def has_current_row(self):
		return self._current_row is not None

	def get_current_row(self):
		"""Individual rows can be 'picked', this is the index (integer) of the current row, or None there is nothing picked"""
		return self._current_row

	def set_current_row(self, value):
		"""Set the current row, and emit the signal signal_pick"""
		if (value is not None) and ((value < 0) or (value >= len(self))):
			raise IndexError("index %d out of range [0,%d]" % (value, len(self)))
		self._current_row = value
		self.signal_pick.emit(self, value)

	def has_snapshots(self):
		return False

	def column_count(self):
		"""Returns the number of columns, not counting virtual ones"""
		return len(self.column_names)

	def get_column_names(self, virtual=False):
		"""Return a list of column names

		:rtype: list of str
 		"""
		return list(self.column_names) + ([key for key in self.virtual_columns.keys() if not key.startswith("__")] if virtual else [])

	def __len__(self):
		"""Returns the number of rows in the dataset, if active_fraction != 1, then floor(active_fraction*full_length) is returned"""
		return self._length

	def selected_length(self):
		"""Returns the number of rows that are selected"""
		raise NotImplementedError

	def full_length(self):
		"""the full length of the dataset, independant what active_fraction is"""
		return self._full_length

	def get_active_fraction(self):
		"""Value in the range (0, 1], to work only with a subset of rows
		"""
		return self._active_fraction

	def set_active_fraction(self, value):
		"""Sets the active_fraction, set picked row to None, and remove selection

		TODO: we may be able to keep the selection, if we keep the expression, and also the picked row
		"""
		if value != self._active_fraction:
			self._active_fraction = value
			#self._fraction_length = int(self._length * self._active_fraction)
			self.select(None)
			self.set_current_row(None)
			self._length = int(round(self.full_length() * self._active_fraction))
			self.signal_active_fraction_changed.emit(self, value)

	def get_selection(self, selection_name="default"):
		selection_history = self.selection_histories[selection_name]
		index = self.selection_history_indices[selection_name]
		if index == -1:
			return None
		else:
			return selection_history[index]

	def selection_undo(self, selection_name="default", executor=None):
		logger.debug("undo")
		executor = executor or self.executor
		assert self.selection_can_undo(selection_name=selection_name)
		selection_history = self.selection_histories[selection_name]
		index = self.selection_history_indices[selection_name]
		if index == 0:
			# special case, ugly solution to select nothing
			if self.is_local():
				result =  SelectionExpression(self, None, None, "replace").execute(executor=executor)
			else:
				# for remote we don't have to do anything, the index == -1 is enough
				# just emit the signal
				self.signal_selection_changed.emit(self)
				result = vaex.promise.Promise.fulfilled(None)
		else:
			previous = selection_history[index-1]
			if self.is_local():
				result = previous.execute(executor=executor, execute_fully=True) if previous else vaex.promise.Promise.fulfilled(None)
			else:
				self.signal_selection_changed.emit(self)
				result = vaex.promise.Promise.fulfilled(None)
		self.selection_history_indices[selection_name] -= 1
		logger.debug("undo: selection history is %r, index is %r", selection_history, self.selection_history_indices[selection_name])
		return result


	def selection_redo(self, selection_name="default", executor=None):
		logger.debug("redo")
		executor = executor or self.executor
		assert self.selection_can_redo(selection_name=selection_name)
		selection_history = self.selection_histories[selection_name]
		index = self.selection_history_indices[selection_name]
		next = selection_history[index+1]
		if self.is_local():
			result = next.execute(executor=executor)
		else:
			self.signal_selection_changed.emit(self)
			result = vaex.promise.Promise.fulfilled(None)
		self.selection_history_indices[selection_name] += 1
		logger.debug("redo: selection history is %r, index is %r", selection_history, index)
		return result

	def selection_can_undo(self, selection_name="default"):
		return self.selection_history_indices[selection_name] > -1

	def selection_can_redo(self, selection_name="default"):
		return (self.selection_history_indices[selection_name] + 1) < len(self.selection_histories[selection_name])

	def select(self, boolean_expression, mode="replace", selection_name="default", executor=None):
		if boolean_expression is None and not self.has_selection(selection_name=selection_name):
			pass # we don't want to pollute the history with many None selections
			self.signal_selection_changed.emit(self) # TODO: unittest want to know, does this make sense?
		else:
			def create(current):
				return SelectionExpression(self, boolean_expression, current, mode) if boolean_expression else None
			return self._selection(create, selection_name)

	def select_nothing(self, selection_name="default"):
		self.select(None, selection_name=selection_name)

	def select_lasso(self, expression_x, expression_y, xsequence, ysequence, mode="replace", selection_name="default", executor=None):
		def create(current):
			return SelectionLasso(self, expression_x, expression_y, xsequence, ysequence, current, mode)
		return self._selection(create, selection_name, executor=executor)

	def set_selection(self, selection, selection_name="default", executor=None):
		def create(current):
			return selection
		return self._selection(create, selection_name, executor=executor, execute_fully=True)


	def _selection(self, create_selection, selection_name, executor=None, execute_fully=False):
		"""select_lasso and select almost share the same code"""
		selection_history = self.selection_histories[selection_name]
		previous_index = self.selection_history_indices[selection_name]
		current = selection_history[previous_index] if selection_history else None
		selection = create_selection(current)
		executor = executor or self.executor
		if self.is_local():
			if selection:
				result = selection.execute(executor=executor, execute_fully=execute_fully)
			else:
				result = vaex.promise.Promise.fulfilled(None)
				self.signal_selection_changed.emit(self)
		else:
			self.signal_selection_changed.emit(self)
			result = vaex.promise.Promise.fulfilled(None)
		selection_history.append(selection)
		self.selection_history_indices[selection_name] += 1
		# clip any redo history
		del selection_history[self.selection_history_indices[selection_name]:-1]
		logger.debug("select selection history is %r, index is %r", selection_history, self.selection_history_indices[selection_name])
		return result

	def has_selection(self, selection_name="default"):
		return self.get_selection(selection_name) != None


def _select_replace(maskold, masknew):
	return masknew

def _select_and(maskold, masknew):
	return masknew if maskold is None else maskold & masknew

def _select_or(maskold, masknew):
	return masknew if maskold is None else maskold | masknew

def _select_xor(maskold, masknew):
	return masknew if maskold is None else maskold ^ masknew

def _select_subtract( maskold, masknew):
	return ~masknew if maskold is None else (maskold) & ~masknew

_select_functions = {"replace":_select_replace,\
					 "and":_select_and,
					 "or":_select_or,
					 "xor":_select_xor,
					 "subtract":_select_subtract
					 }
class DatasetLocal(Dataset):
	def __init__(self, name, path, column_names):
		super(DatasetLocal, self).__init__(name, column_names)
		self.path = path
		self.mask = None
		self.columns = collections.OrderedDict()

	def shallow_copy(self, virtual=True, variables=True):
		dataset = DatasetLocal(self.name, self.path, self.column_names)
		dataset.columns.update(self.columns)
		dataset._full_length = self._full_length
		dataset._length = self._length
		dataset._active_fraction = self._active_fraction
		if virtual:
			dataset.virtual_columns.update(self.virtual_columns)
		if variables:
			dataset.variables.update(self.variables)
		return dataset

	def is_local(self): return True

	def byte_size(self, selection=False):
		bytes_per_row = 0
		for column in list(self.columns.values()):
			dtype = column.dtype
			bytes_per_row += dtype.itemsize
		return bytes_per_row * self.length(selection=selection)


	def length(self, selection=False):
		if selection:
			return 0 if self.mask is None else np.sum(self.mask)
		else:
			return len(self)

	def __call__(self, *expressions, **kwargs):
		return SubspaceLocal(self, expressions, kwargs.get("executor") or self.executor, async=kwargs.get("async", False))

	def concat(self, other):
		datasets = []
		if isinstance(self, DatasetConcatenated):
			datasets.extend(self.datasets)
		else:
			datasets.extend([self])
		if isinstance(other, DatasetConcatenated):
			datasets.extend(other.datasets)
		else:
			datasets.extend([other])
		return DatasetConcatenated(datasets)

	def evaluate(self, expression, i1=None, i2=None, out=None):
		i1 = i1 or 0
		i2 = i2 or len(self)
		scope = _BlockScope(self, i1, i2, **self.variables)
		if out is not None:
			scope.buffers[expression] = out
		return scope.evaluate(expression)

	def export_hdf5(self, path, column_names=None, byteorder="=", shuffle=False, selection=False, progress=None):
		"""
		:param DatasetLocal dataset: dataset to export
		:param str path: path for file
		:param lis[str] column_names: list of column names to export or None for all columns
		:param str byteorder: = for native, < for little endian and > for big endian
		:param bool shuffle: export rows in random order
		:param bool selection: export selection or not
		:param progress: progress callback that gets a progress fraction as argument and should return True to continue
		:return:
		"""
		vaex.export.export_hdf5(self, path, column_names, byteorder, shuffle, selection, progress=progress)

	def export_fits(self, path, column_names=None, shuffle=False, selection=False, progress=None):
		"""
		:param DatasetLocal dataset: dataset to export
		:param str path: path for file
		:param lis[str] column_names: list of column names to export or None for all columns
		:param bool shuffle: export rows in random order
		:param bool selection: export selection or not
		:param progress: progress callback that gets a progress fraction as argument and should return True to continue
		:return:
		"""
		vaex.export.export_fits(self, path, column_names, shuffle, selection, progress=progress)

	def _needs_copy(self, column_name):
		return not \
			(column_name in self.column_names  \
			and not isinstance(self.columns[column_name], vaex.dataset._ColumnConcatenatedLazy)\
			and self.columns[column_name].dtype.type==np.float64 \
			and self.columns[column_name].strides[0] == 8 \
			and column_name not in self.virtual_columns)
				#and False:

	def selected_length(self):
		return np.sum(self.mask) if self.has_selection() else None



	def _set_mask(self, mask):
		self.mask = mask
		self._has_selection = mask is not None
		self.signal_selection_changed.emit(self)


class _ColumnConcatenatedLazy(object):
	def __init__(self, datasets, column_name):
		self.datasets = datasets
		self.column_name = column_name
		dtypes = [dataset.columns[self.column_name].dtype for dataset in datasets]
		self.dtype = np.find_common_type(dtypes, [])
		self.shape = (len(self), ) + self.datasets[0].columns[self.column_name].shape[1:]
		for i in range(1, len(datasets)):
			c0 = self.datasets[0].columns[self.column_name]
			ci = self.datasets[i].columns[self.column_name]
			if c0.shape[1:] != ci.shape[1:]:
				raise ValueError("shape of of column %s, array index 0, is %r and is incompatible with the shape of the same column of array index %d, %r" % (self.column_name, c0.shape, i, ci.shape))

	def __len__(self):
		return sum(len(ds) for ds in self.datasets)

	def __getitem__(self, slice):
		start, stop, step = slice.start, slice.stop, slice.step
		start = start or 0
		stop = stop or len(self)
		assert step in [None, 1]
		datasets = iter(self.datasets)
		current_dataset = next(datasets)
		offset = 0
		#print "#@!", start, stop, [len(dataset) for dataset in self.datasets]
		while start >= offset + len(current_dataset):
			#print offset
			offset += len(current_dataset)
			#try:
			current_dataset = next(datasets)
			#except StopIteration:
				#logger.exception("requested start:stop %d:%d when max was %d, offset=%d" % (start, stop, offset+len(current_dataset), offset))
				#raise
			#	break
		# this is the fast path, no copy needed
		if stop <= offset + len(current_dataset):
			return current_dataset.columns[self.column_name][start-offset:stop-offset].astype(self.dtype)
		else:
			copy = np.zeros(stop-start, dtype=self.dtype)
			copy_offset = 0
			#print "!!>", start, stop, offset, len(current_dataset), current_dataset.columns[self.column_name]
			while offset < stop: #> offset + len(current_dataset):
				part = current_dataset.columns[self.column_name][start-offset:min(len(current_dataset), stop-offset)]
				#print "part", part, copy_offset,copy_offset+len(part)
				copy[copy_offset:copy_offset+len(part)] = part
				#print copy[copy_offset:copy_offset+len(part)]
				offset += len(current_dataset)
				copy_offset += len(part)
				start = offset
				if offset < stop:
					current_dataset = next(datasets)
			return copy


class DatasetConcatenated(DatasetLocal):
	def __init__(self, datasets, name=None):
		super(DatasetConcatenated, self).__init__(None, None, [])
		self.datasets = datasets
		self.name = name or "-".join(ds.name for ds in self.datasets)
		first, tail = datasets[0], datasets[1:]
		for column_name in first.get_column_names():
			if all([column_name in dataset.get_column_names() for dataset in tail]):
				self.column_names.append(column_name)
		self.columns = {}
		for column_name in self.get_column_names():
			self.columns[column_name] = _ColumnConcatenatedLazy(datasets, column_name)

		for name in list(first.virtual_columns.keys()):
			if all([first.virtual_columns[name] == dataset.virtual_columns.get(name, None) for dataset in tail]):
				self.virtual_columns[name] = first.virtual_columns[name]
		for dataset in datasets:
			for name, value in list(dataset.variables.items()):
				self.set_variable(name, value)


		self._full_length = sum(ds.full_length() for ds in self.datasets)
		self._length = self.full_length()

class DatasetArrays(DatasetLocal):
	def __init__(self, name="arrays"):
		super(DatasetArrays, self).__init__(None, None, [])
		self.name = name

	#def __len__(self):
	#	return len(self.columns.values()[0])

	def add_column(self, name, data):
		self.column_names.append(name)
		self.columns[name] = data
		#self._length = len(data)
		if self._full_length is None:
			self._full_length = len(data)
		else:
			assert self.full_length() == len(data), "columns should be of equal length"
		self._length = int(round(self.full_length() * self._active_fraction))
		#self.set_active_fraction(self._active_fraction)


class DatasetMemoryMapped(DatasetLocal):

		
	# nommap is a hack to get in memory datasets working
	def __init__(self, filename, write=False, nommap=False, name=None):
		super(DatasetMemoryMapped, self).__init__(name=name or os.path.splitext(os.path.basename(filename))[0], path=os.path.abspath(filename) if filename is not None else None, column_names=[])
		self.filename = filename or "no file"
		self.write = write
		#self.name = name or os.path.splitext(os.path.basename(self.filename))[0]
		#self.path = os.path.abspath(filename) if filename is not None else None
		self.nommap = nommap
		if not nommap:
			self.file = open(self.filename, "r+" if write else "r")
			self.fileno = self.file.fileno()
			self.mapping = mmap.mmap(self.fileno, 0, prot=mmap.PROT_READ | 0 if not write else mmap.PROT_WRITE )
			self.file_map = {filename: self.file}
			self.fileno_map = {filename: self.fileno}
			self.mapping_map = {filename: self.mapping}
		else:
			self.file_map = {}
			self.fileno_map = {}
			self.mapping_map = {}
		self._length = None
		#self._fraction_length = None
		self.nColumns = 0
		self.column_names = []
		self.rank1s = {}
		self.rank1names = []
		self.virtual_columns = collections.OrderedDict()
		
		self.axes = {}
		self.axis_names = []
		
		# these are replaced by variables
		#self.properties = {}
		#self.property_names = []

		self.current_slice = None
		self.fraction = 1.0
		
		
		self.selected_row_index = None
		self.selected_serie_index = 0
		self.row_selection_listeners = []
		self.serie_index_selection_listeners = []
		#self.mask_listeners = []

		self.all_columns = {}
		self.all_column_names = []
		self.global_links = {}
		
		self.offsets = {}
		self.strides = {}
		self.filenames = {}
		self.dtypes = {}
		self.samp_id = None
		#self.variables = collections.OrderedDict()
		
		self.undo_manager = vaex.ui.undo.UndoManager()

	def has_snapshots(self):
		return len(self.rank1s) > 0

	def get_path(self):
		return self.path
		
	def addFile(self, filename, write=False):
		self.file_map[filename] = open(filename, "r+" if write else "r")
		self.fileno_map[filename] = self.file_map[filename].fileno()
		self.mapping_map[filename] = mmap.mmap(self.fileno_map[filename], 0, prot=mmap.PROT_READ | 0 if not write else mmap.PROT_WRITE )


	def selectSerieIndex(self, serie_index):
		self.selected_serie_index = serie_index
		for serie_index_selection_listener in self.serie_index_selection_listeners:
			serie_index_selection_listener(serie_index)
		self.signal_sequence_index_change.emit(self, serie_index)
			
	def matches_url(self, url):
		filename = url
		if filename.startswith("file:/"):
			filename = filename[5:]
		similar = os.path.splitext(os.path.abspath(self.filename))[0] == os.path.splitext(filename)[0]
		logger.info("matching urls: %r == %r == %r" % (os.path.splitext(self.filename)[0], os.path.splitext(filename)[0], similar) )
		return similar

	def close(self):
		self.file.close()
		self.mapping.close()
		
	def addAxis(self, name, offset=None, length=None, dtype=np.float64, stride=1, filename=None):
		if filename is None:
			filename = self.filename
		mapping = self.mapping_map[filename]
		mmapped_array = np.frombuffer(mapping, dtype=dtype, count=length if stride is None else length * stride, offset=offset)
		if stride:
			mmapped_array = mmapped_array[::stride]
		self.axes[name] = mmapped_array
		self.axis_names.append(name)
	
		
	def addColumn(self, name, offset=None, length=None, dtype=np.float64, stride=1, filename=None, array=None):
		if filename is None:
			filename = self.filename
		if not self.nommap:
			mapping = self.mapping_map[filename]
			
		if array is not None:
			length = len(array)
			
		if self._length is not None and length != self._length:
			logger.error("inconsistent length", "length of column %s is %d, while %d was expected" % (name, length, self._length))
		else:
			if self.current_slice is None:
				self.current_slice = (0, length)
				self.fraction = 1.
				self._full_length = length
				self._length = length
			self._length = length
			#print self.mapping, dtype, length if stride is None else length * stride, offset
			if array is not None:
				length = len(array)
				mmapped_array = array
				stride = None
				offset = None
				dtype = array.dtype
			else:
				if offset is None:
					print("offset is None")
					sys.exit(0)
				mmapped_array = np.frombuffer(mapping, dtype=dtype, count=length if stride is None else length * stride, offset=offset)
				if stride:
					#import pdb
					#pdb.set_trace()
					mmapped_array = mmapped_array[::stride]
			self.columns[name] = mmapped_array
			self.column_names.append(name)
			self.all_columns[name] = mmapped_array
			self.all_column_names.append(name)
			#self.column_names.sort()
			self.nColumns += 1
			self.nRows = self._length
			self.offsets[name] = offset
			self.strides[name] = stride
			if filename is not None:
				self.filenames[name] = os.path.abspath(filename)
			self.dtypes[name] = dtype
			
	def addRank1(self, name, offset, length, length1, dtype=np.float64, stride=1, stride1=1, filename=None, transposed=False):
		if filename is None:
			filename = self.filename
		mapping = self.mapping_map[filename]
		if (not transposed and self._length is not None and length != self._length) or (transposed and self._length is not None and length1 != self._length):
			logger.error("inconsistent length", "length of column %s is %d, while %d was expected" % (name, length, self._length))
		else:
			if self.current_slice is None:
				self.current_slice = (0, length if not transposed else length1)
				self.fraction = 1.
				self._full_length = length if not transposed else length1
				self._length = self._full_length
			self._length = length if not transposed else length1
			#print self.mapping, dtype, length if stride is None else length * stride, offset
			rawlength = length * length1
			rawlength *= stride
			rawlength *= stride1

			mmapped_array = np.frombuffer(mapping, dtype=dtype, count=rawlength, offset=offset)
			mmapped_array = mmapped_array.reshape((length1*stride1, length*stride))
			mmapped_array = mmapped_array[::stride1,::stride]

			self.rank1s[name] = mmapped_array
			self.rank1names.append(name)
			self.all_columns[name] = mmapped_array
			self.all_column_names.append(name)
			


import struct
class HansMemoryMapped(DatasetMemoryMapped):
	def __init__(self, filename, filename_extra=None):
		super(HansMemoryMapped, self).__init__(filename)
		self.pageSize, \
		self.formatSize, \
		self.numberParticles, \
		self.numberTimes, \
		self.numberParameters, \
		self.numberCompute, \
		self.dataOffset, \
		self.dataHeaderSize = struct.unpack("Q"*8, self.mapping[:8*8])
		zerooffset = offset = self.dataOffset
		length = self.numberParticles+1
		stride = self.formatSize/8 # stride in units of the size of the element (float64)
		
		# TODO: ask Hans for the self.numberTimes-2
		lastoffset = offset + (self.numberParticles+1)*(self.numberTimes-2)*self.formatSize
		t_index = 3
		names = "x y z vx vy vz".split()
		midoffset = offset + (self.numberParticles+1)*self.formatSize*t_index
		names = "x y z vx vy vz".split()

		for i, name in enumerate(names):
			self.addColumn(name+"_0", offset+8*i, length, dtype=np.float64, stride=stride)
			
		for i, name in enumerate(names):
			self.addColumn(name+"_last", lastoffset+8*i, length, dtype=np.float64, stride=stride)
		

		names = "x y z vx vy vz".split()

		if 1:
			stride = self.formatSize/8 
			#stride1 = self.numberTimes #*self.formatSize/8 
			for i, name in enumerate(names):
				# TODO: ask Hans for the self.numberTimes-1
				self.addRank1(name, offset+8*i, length=self.numberParticles+1, length1=self.numberTimes-1, dtype=np.float64, stride=stride, stride1=1)

		if filename_extra is None:
			basename = os.path.basename(filename)
			if os.path.exists(basename + ".omega2"):
				filename_extra = basename + ".omega2"

		if filename_extra is not None:
			self.addFile(filename_extra)
			mapping = self.mapping_map[filename_extra]
			names = "J_r J_theta J_phi Theta_r Theta_theta Theta_phi Omega_r Omega_theta Omega_phi r_apo r_peri".split()
			offset = 0
			stride = 11
			#import pdb
			#pdb.set_trace()
			for i, name in enumerate(names):
				# TODO: ask Hans for the self.numberTimes-1
				self.addRank1(name, offset+8*i, length=self.numberParticles+1, length1=self.numberTimes-1, dtype=np.float64, stride=stride, stride1=1, filename=filename_extra)

				self.addColumn(name+"_0", offset+8*i, length, dtype=np.float64, stride=stride, filename=filename_extra)
				self.addColumn(name+"_last", offset+8*i + (self.numberParticles+1)*(self.numberTimes-2)*11*8, length, dtype=np.float64, stride=stride, filename=filename_extra)
			
			
	@classmethod
	def can_open(cls, path, *args, **kwargs):
		basename, ext = os.path.splitext(path)
		if os.path.exists(basename + ".omega2"):
			return True
		return False

	@classmethod
	def get_options(cls, path):
		return []

	@classmethod
	def option_to_args(cls, option):
		return []
dataset_type_map["buist"] = HansMemoryMapped


class FitsBinTable(DatasetMemoryMapped):
	def __init__(self, filename, write=False):
		super(FitsBinTable, self).__init__(filename, write=write)
		fitsfile = fits.open(filename)
		for table in fitsfile:
			if isinstance(table, fits.BinTableHDU):
				table_offset = table._data_offset
				#import pdb
				#pdb.set_trace()
				if table.columns[0].dim is not None: # for sure not a colfits
					dim = eval(table.columns[0].dim) # TODO: can we not do an eval here? not so safe
					if dim[0] == 1 and len(dim) == 2: # we have colfits format
						logger.debug("colfits file!")
						offset = table_offset
						for i in range(len(table.columns)):
							column = table.columns[i]
							cannot_handle = False

							# flatlength == length * arraylength
							flatlength, fitstype = int(column.format[:-1]),column.format[-1]
							arraylength, length = arrayshape = eval(column.dim)

							# numpy dtype code, like f8, i4
							dtypecode = astropy.io.fits.column.FITS2NUMPY[fitstype]


							dtype = np.dtype((">" +dtypecode, arraylength))
							if 0:
								if arraylength > 1:
									dtype = np.dtype((">" +dtypecode, arraylength))
								else:
									if dtypecode == "a": # I think numpy needs by default a length 1
										dtype = np.dtype(dtypecode + "1")
									else:
										dtype = np.dtype(">" +dtypecode)
								#	bytessize = 8

							bytessize = dtype.itemsize
							logger.debug("%r", (column.name, dtype, column.format, column.dim, length, bytessize, arraylength))
							if (flatlength > 0) and dtypecode != "a": # TODO: support strings
								logger.debug("%r", (column.name, offset, dtype, length))
								if arraylength == 1:
									self.addColumn(column.name, offset=offset, dtype=dtype, length=length)
								else:
									for i in range(arraylength):
										name = column.name+"_" +str(i)
										self.addColumn(name, offset=offset+bytessize*i/arraylength, dtype=">" +dtypecode, length=length, stride=arraylength)
							if flatlength > 0: # flatlength can be
								offset += bytessize * length

				else:
					logger.debug("adding table: %r" % table)
					for column in table.columns:
						array = column.array[:]
						array = column.array[:] # 2nd time it will be a real np array
						#import pdb
						#pdb.set_trace()
						if array.dtype.kind in "fi":
							self.addColumn(column.name, array=array)

	@classmethod
	def can_open(cls, path, *args, **kwargs):
		return os.path.splitext(path)[1] == ".fits"
	
	@classmethod
	def get_options(cls, path):
		return [] # future: support multiple tables?
	
	@classmethod
	def option_to_args(cls, option):
		return []

dataset_type_map["fits"] = FitsBinTable

class Hdf5MemoryMapped(DatasetMemoryMapped):
	def __init__(self, filename, write=False):
		super(Hdf5MemoryMapped, self).__init__(filename, write=write)
		self.h5file = h5py.File(self.filename, "r+" if write else "r")
		try:
			self.load()
		finally:
			self.h5file.close()
		
	@classmethod
	def can_open(cls, path, *args, **kwargs):
		h5file = None
		try:
			with open(path, "rb") as f:
				signature = open(path, "rb").read(4)
				hdf5file = signature == "\x89\x48\x44\x46"
		except:
			logger.error("could not read 4 bytes from %r", path)
			return
		if hdf5file:
			try:
				h5file = h5py.File(path, "r")
			except:
				logger.exception("could not open file as hdf5")
				return False
			if h5file is not None:
				with h5file:
					return ("data" in h5file) or ("columns" in h5file)
			else:
				logger.debug("file %s has no data or columns group" % path)
		return False
			
	
	@classmethod
	def get_options(cls, path):
		return []
	
	@classmethod
	def option_to_args(cls, option):
		return []

	def load(self):
		if "data" in self.h5file:
			self.load_columns(self.h5file["/data"])
		if "columns" in self.h5file:
			self.load_columns(self.h5file["/columns"])
		if "properties" in self.h5file:
			self.load_variables(self.h5file["/properties"]) # old name, kept for portability
		if "variables" in self.h5file:
			self.load_variables(self.h5file["/variables"])
		if "axes" in self.h5file:
			self.load_axes(self.h5file["/axes"])
			
	#def 
	def load_axes(self, axes_data):
		for name in axes_data:
			axis = axes_data[name]
			logger.debug("loading axis %r" % name)
			offset = axis.id.get_offset() 
			shape = axis.shape
			assert len(shape) == 1 # ony 1d axes
			#print name, offset, len(axis), axis.dtype
			self.addAxis(name, offset=offset, length=len(axis), dtype=axis.dtype)
			#self.axis_names.append(axes_data)
			#self.axes[name] = np.array(axes_data[name])
			
	def load_variables(self, h5variables):
		for key, value in list(h5variables.attrs.items()):
			self.variables[key] = value
			
			
	def load_columns(self, h5data):
		#print h5data
		# make sure x y x etc are first
		first = "x y z vx vy vz".split()
		finished = set()
		for column_name in first + list(h5data):
			if column_name in h5data and column_name not in finished:
				#print type(column_name)
				column = h5data[column_name]
				if hasattr(column, "dtype"):
					#print column, column.shape
					offset = column.id.get_offset() 
					if offset is None:
						raise Exception("columns doesn't really exist in hdf5 file")
					shape = column.shape
					if True: #len(shape) == 1:
						self.addColumn(column_name, offset, len(column), dtype=column.dtype)
					else:

						#transposed = self._length is None or shape[0] == self._length
						transposed = shape[1] < shape[0]
						self.addRank1(column_name, offset, shape[1], length1=shape[0], dtype=column.dtype, stride=1, stride1=1, transposed=transposed)
						#if len(shape[0]) == self._length:
						#	self.addRank1(column_name, offset, shape[1], length1=shape[0], dtype=column.dtype, stride=1, stride1=1)
						#self.addColumn(column_name+"_0", offset, shape[1], dtype=column.dtype)
						#self.addColumn(column_name+"_last", offset+(shape[0]-1)*shape[1]*column.dtype.itemsize, shape[1], dtype=column.dtype)
						#self.addRank1(name, offset+8*i, length=self.numberParticles+1, length1=self.numberTimes-1, dtype=np.float64, stride=stride, stride1=1, filename=filename_extra)
			finished.add(column_name)
			
	def close(self):
		super(Hdf5MemoryMapped, self).close()
		self.h5file.close()
		
	def __expose_array(self, hdf5path, column_name):
		array = self.h5file[hdf5path]
		array[0] = array[0] # without this, get_offset returns None, probably the array isn't really created
		offset = array.id.get_offset() 
		self.remap()
		self.addColumn(column_name, offset, len(array), dtype=array.dtype)
		
	def __add_column(self, column_name, dtype=np.float64, length=None):
		array = self.h5data.create_dataset(column_name, shape=(self._length if length is None else length,), dtype=dtype)
		array[0] = array[0] # see above
		offset = array.id.get_offset() 
		self.h5file.flush()
		self.remap()
		self.addColumn(column_name, offset, len(array), dtype=array.dtype)

dataset_type_map["h5vaex"] = Hdf5MemoryMapped

class AmuseHdf5MemoryMapped(Hdf5MemoryMapped):
	def __init__(self, filename, write=False):
		super(AmuseHdf5MemoryMapped, self).__init__(filename, write=write)
		
	@classmethod
	def can_open(cls, path, *args, **kwargs):
		h5file = None
		try:
			h5file = h5py.File(path, "r")
		except:
			return False
		if h5file is not None:
			with h5file:
				return ("particles" in h5file)# or ("columns" in h5file)
		return False

	def load(self):
		particles = self.h5file["/particles"]
		for group_name in particles:
			#import pdb
			#pdb.set_trace()
			group = particles[group_name]
			self.load_columns(group["attributes"])
			
			column_name = "keys"
			column = group[column_name]
			offset = column.id.get_offset() 
			self.addColumn(column_name, offset, len(column), dtype=column.dtype)

dataset_type_map["amuse"] = AmuseHdf5MemoryMapped


gadget_particle_names = "gas halo disk bulge stars dm".split()

class Hdf5MemoryMappedGadget(DatasetMemoryMapped):
	def __init__(self, filename, particle_name=None, particle_type=None):
		if "#" in filename:
			filename, index = filename.split("#")
			index = int(index)
			particle_type = index
			particle_name = gadget_particle_names[particle_type]
		elif particle_type is not None:
			self.particle_name = gadget_particle_names[self.particle_type]
			self.particle_type = particle_type
		elif particle_name is not None:
			if particle_name.lower() in gadget_particle_names:
				self.particle_type = gadget_particle_names.index(particle_name.lower())
				self.particle_name = particle_name.lower()
			else:
				raise ValueError("particle name not supported: %r, expected one of %r" % (particle_name, " ".join(gadget_particle_names)))
		else:
			raise Exception("expected particle type or name as argument, or #<nr> behind filename")
		super(Hdf5MemoryMappedGadget, self).__init__(filename)
		self.particle_type = particle_type
		self.particle_name = particle_name
		self.name = self.name + "-" + self.particle_name
		h5file = h5py.File(self.filename, 'r')
		#for i in range(1,4):
		key = "/PartType%d" % self.particle_type
		if key not in h5file:
			raise KeyError("%s does not exist" % key)
		particles = h5file[key]
		for name in list(particles.keys()):
			#name = "/PartType%d/Coordinates" % i
			data = particles[name]
			if isinstance(data, h5py.highlevel.Dataset): #array.shape
				array = data
				shape = array.shape
				if len(shape) == 1:
					offset = array.id.get_offset()
					if offset is not None:
						self.addColumn(name, offset, data.shape[0], dtype=data.dtype)
				else:
					if name == "Coordinates":
						offset = data.id.get_offset() 
						if offset is None:
							print((name, "is not of continuous layout?"))
							sys.exit(0)
						bytesize = data.dtype.itemsize
						self.addColumn("x", offset, data.shape[0], dtype=data.dtype, stride=3)
						self.addColumn("y", offset+bytesize, data.shape[0], dtype=data.dtype, stride=3)
						self.addColumn("z", offset+bytesize*2, data.shape[0], dtype=data.dtype, stride=3)
					elif name == "Velocity":
						offset = data.id.get_offset() 
						self.addColumn("vx", offset, data.shape[0], dtype=data.dtype, stride=3)
						self.addColumn("vy", offset+bytesize, data.shape[0], dtype=data.dtype, stride=3)
						self.addColumn("vz", offset+bytesize*2, data.shape[0], dtype=data.dtype, stride=3)
					elif name == "Velocities":
						offset = data.id.get_offset() 
						self.addColumn("vx", offset, data.shape[0], dtype=data.dtype, stride=3)
						self.addColumn("vy", offset+bytesize, data.shape[0], dtype=data.dtype, stride=3)
						self.addColumn("vz", offset+bytesize*2, data.shape[0], dtype=data.dtype, stride=3)
					else:
						logger.error("unsupported column: %r of shape %r" % (name, array.shape))
		if "Header" in h5file:
			for name in "Redshift Time_GYR".split():
				if name in h5file["Header"].attrs:
					value = h5file["Header"].attrs[name]
					logger.debug("property[{name!r}] = {value}".format(**locals()))
					self.variables[name] = value
					#self.property_names.append(name)
		
		name = "particle_type"
		value = particle_type
		logger.debug("property[{name}] = {value}".format(**locals()))
		self.variables[name] = value
		#self.property_names.append(name)

	@classmethod
	def can_open(cls, path, *args, **kwargs):
		if len(args) == 2:
			particleName = args[0]
			particleType = args[1]
		elif "particle_name" in kwargs:
			particle_type = gadget_particle_names.index(kwargs["particle_name"].lower())
		elif "particle_type" in kwargs:
			particle_type = kwargs["particle_type"]
		elif "#" in path:
			filename, index = path.split("#")
			particle_type = gadget_particle_names[index]
		else:
			return False
		h5file = None
		try:
			h5file = h5py.File(path, "r")
		except:
			return False
		has_particles = False
		#for i in range(1,6):
		key = "/PartType%d" % particle_type
		exists = key in h5file
		h5file.close()
		return exists

		#has_particles = has_particles or (key in h5file)
		#return has_particles
			
	
	@classmethod
	def get_options(cls, path):
		return []
	
	@classmethod
	def option_to_args(cls, option):
		return []


dataset_type_map["gadget-hdf5"] = Hdf5MemoryMappedGadget

class InMemory(DatasetMemoryMapped):
	def __init__(self, name):
		super(InMemory, self).__init__(filename=None, nommap=True, name=name)


class SoneiraPeebles(InMemory):
	def __init__(self, dimension, eta, max_level, L):
		super(SoneiraPeebles, self).__init__(name="soneira-peebles")
		#InMemory.__init__(self)
		def todim(value):
			if isinstance(value, (tuple, list)):
				assert len(value) == dimension, "either a scalar or sequence of length equal to the dimension"
				return value
			else:
				return [value] * dimension

		eta = eta
		max_level = max_level
		N = eta**(max_level)
		# array[-1] is used as a temp storage
		array = np.zeros((dimension+1, N), dtype=np.float64)
		L = todim(L)

		for d in range(dimension):
			vaex.vaexfast.soneira_peebles(array[d], 0, 1, L[d], eta, max_level)
		order = np.zeros(N, dtype=np.int64)
		vaex.vaexfast.shuffled_sequence(order);
		for i, name in zip(list(range(dimension)), "x y z w v u".split()):
			#np.take(array[i], order, out=array[i])
			reorder(array[i], array[-1], order)
			self.addColumn(name, array=array[i])

dataset_type_map["soneira-peebles"] = Hdf5MemoryMappedGadget


class Zeldovich(InMemory):
	def __init__(self, dim=2, N=256, n=-2.5, t=None, seed=None, scale=1, name="zeldovich approximation"):
		super(Zeldovich, self).__init__(name=name)
		
		if seed is not None:
			np.random.seed(seed)
		#sys.exit(0)
		shape = (N,) * dim
		A = np.random.normal(0.0, 1.0, shape)
		F = np.fft.fftn(A) 
		K = np.fft.fftfreq(N, 1./(2*np.pi))[np.indices(shape)]
		k = (K**2).sum(axis=0)
		k_max = np.pi
		F *= np.where(np.sqrt(k) > k_max, 0, np.sqrt(k**n) * np.exp(-k*4.0))
		F.flat[0] = 0
		#pylab.imshow(np.where(sqrt(k) > k_max, 0, np.sqrt(k**-2)), interpolation='nearest')
		grf = np.fft.ifftn(F).real
		Q = np.indices(shape) / float(N-1) - 0.5
		s = np.array(np.gradient(grf)) / float(N)
		#pylab.imshow(s[1], interpolation='nearest')
		#pylab.show()
		s /= s.max() * 100.
		#X = np.zeros((4, 3, N, N, N))
		#for i in range(4):
		#if t is None:
		#	s = s/s.max()
		t = t or 1.
		X = Q + s * t

		for d, name in zip(list(range(dim)), "xyzw"):
			self.addColumn(name, array=X[d].reshape(-1) * scale)
		for d, name in zip(list(range(dim)), "xyzw"):
			self.addColumn("v"+name, array=s[d].reshape(-1) * scale)
		for d, name in zip(list(range(dim)), "xyzw"):
			self.addColumn(name+"0", array=Q[d].reshape(-1) * scale)
		return
		
dataset_type_map["zeldovich"] = Zeldovich
		
		
class AsciiTable(DatasetMemoryMapped):
	def __init__(self, filename):
		super(AsciiTable, self).__init__(filename, nommap=True)
		import asciitable
		table = asciitable.read(filename)
		logger.debug("done parsing ascii table")
		#import pdb
		#pdb.set_trace()
		#names = table.array.dtype.names
		names = table.dtype.names

		#data = table.array.data
		for i in range(len(table.dtype)):
			name = table.dtype.names[i]
			type = table.dtype[i]
			if type.kind in ["f", "i"]: # only store float and int
				#datagroup.create_dataset(name, data=table.array[name].astype(np.float64))
				#dataset.addMemoryColumn(name, table.array[name].astype(np.float64))
				self.addColumn(name, array=table[name])
		#dataset.samp_id = table_id
		#self.list.addDataset(dataset)
		#return dataset

	@classmethod
	def can_open(cls, path, *args, **kwargs):
		can_open = path.endswith(".asc")
		logger.debug("%r can open: %r"  %(cls.__name__, can_open))
		return can_open
dataset_type_map["ascii"] = AsciiTable

class MemoryMappedGadget(DatasetMemoryMapped):
	def __init__(self, filename):
		super(MemoryMappedGadget, self).__init__(filename)
		#h5file = h5py.File(self.filename)
		import vaex.file.gadget
		length, posoffset, veloffset, header = vaex.file.gadget.getinfo(filename)
		self.addColumn("x", posoffset, length, dtype=np.float32, stride=3)
		self.addColumn("y", posoffset+4, length, dtype=np.float32, stride=3)
		self.addColumn("z", posoffset+8, length, dtype=np.float32, stride=3)
		
		self.addColumn("vx", veloffset, length, dtype=np.float32, stride=3)
		self.addColumn("vy", veloffset+4, length, dtype=np.float32, stride=3)
		self.addColumn("vz", veloffset+8, length, dtype=np.float32, stride=3)
dataset_type_map["gadget-plain"] = MemoryMappedGadget

class DatasetAstropyTable(DatasetArrays):
	def __init__(self, filename, format, **kwargs):
		DatasetArrays.__init__(self, filename)
		self.filename = filename
		self.table = astropy.table.Table.read(filename, format=format, **kwargs)

		#data = table.array.data
		for i in range(len(self.table.dtype)):
			name = self.table.dtype.names[i]
			type = self.table.dtype[i]
			clean_name = re.sub("[^a-zA-Z_]", "_", name)
			if type.kind in ["f", "i"]: # only store float and int
				#datagroup.create_dataset(name, data=table.array[name].astype(np.float64))
				#dataset.addMemoryColumn(name, table.array[name].astype(np.float64))
				masked_array = self.table[name].data
				if type.kind in ["f"]:
					masked_array.data[masked_array.mask] = np.nan
				if type.kind in ["i"]:
					masked_array.data[masked_array.mask] = 0
				self.add_column(clean_name, self.table[name].data)
			if type.kind in ["S"]:
				self.add_column(clean_name, self.table[name].data)

		#dataset.samp_id = table_id
		#self.list.addDataset(dataset)
		#return dataset


import astropy.io.votable
class VOTable(DatasetAstropyTable):
	def __init__(self, filename):
		super(VOTable, self).__init__(filename, format="votable")

	@classmethod
	def can_open(cls, path, *args, **kwargs):
		can_open = path.endswith(".vot")
		logger.debug("%r can open: %r"  %(cls.__name__, can_open))
		return can_open

dataset_type_map["votable"] = VOTable



class DatasetNed(DatasetAstropyTable):
	def __init__(self, code="2012AJ....144....4M"):
		url = "http://ned.ipac.caltech.edu/cgi-bin/objsearch?refcode={code}&hconst=73&omegam=0.27&omegav=0.73&corr_z=1&out_csys=Equatorial&out_equinox=J2000.0&obj_sort=RA+or+Longitude&of=xml_main&zv_breaker=30000.0&list_limit=5&img_stamp=YES&search_type=Search"\
			.format(code=code)
		super(DatasetNed, self).__init__(url, format="votable", use_names_over_ids=True)
		self.name = "ned:" + code

dataset_type_map["ned"] = DatasetNed

def can_open(path, *args, **kwargs):
	for name, class_ in list(dataset_type_map.items()):
		if class_.can_open(path, *args):
			return True
		
def load_file(path, *args, **kwargs):
	dataset_class = None
	for name, class_ in list(vaex.dataset.dataset_type_map.items()):
		logger.debug("trying %r with class %r" % (path, class_))
		if class_.can_open(path, *args, **kwargs):
			logger.debug("can open!")
			dataset_class = class_
			break
	if dataset_class:
		dataset = dataset_class(path, *args)
		return dataset

from .remote import ServerRest, SubspaceRemote, DatasetRemote