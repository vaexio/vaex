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
import six
from vaex.utils import ensure_string
import vaex.utils

import numpy as np
import numexpr as ne
import concurrent.futures
import astropy.table
import astropy.units

from vaex.utils import Timer
import vaex.events
import vaex.ui.undo
import vaex.grids
import vaex.multithreading
import vaex.promise
import vaex.execution
import vaex.expresso
import logging
import astropy.io.fits as fits
import vaex.kld

# py2/p3 compatibility
try:
	from urllib.parse import urlparse
except ImportError:
	from urlparse import urlparse


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

#executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
#executor = vaex.execution.default_executor


class Task(vaex.promise.Promise):
	"""
	:type: signal_progress: Signal
	"""
	def __init__(self, dataset=None, expressions=[], name="task"):
		vaex.promise.Promise.__init__(self)
		self.dataset = dataset
		self.expressions = expressions
		self.expressions_all = list(expressions)
		self.signal_progress = vaex.events.Signal("progress (float)")
		self.progress_fraction = 0
		self.signal_progress.connect(self._set_progress)
		self.cancelled = False
		self.name = name

	def _set_progress(self, fraction):
		self.progress_fraction = fraction
		return not self.cancelled # don't cancel

	def cancel(self):
		self.cancelled = True

	@property
	def dimension(self):
		return len(self.expressions)

	@classmethod
	def create(cls):
		ret = Task()
		return ret

	def create_next(self):
		ret = Task(self.dataset, [])
		self.signal_progress.connect(ret.signal_progress.emit)
		return ret

class TaskMapReduce(Task):
	def __init__(self, dataset, expressions, map, reduce, converter=lambda x: x, info=False, name="task"):
		Task.__init__(self, dataset, expressions, name=name)
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


#import numba
#@numba.jit(nopython=True, nogil=True)
#def histogram_numba(x, y, weight, grid, xmin, xmax, ymin, ymax):
#    scale_x = 1./ (xmax-xmin);
#    scale_y = 1./ (ymax-ymin);
#    counts_length_y, counts_length_x = grid.shape
#    for i in range(len(x)):
#        value_x = x[i];
#        value_y = y[i];
#        scaled_x = (value_x - xmin) * scale_x;
#        scaled_y = (value_y - ymin) * scale_y;
#
#        if ( (scaled_x >= 0) & (scaled_x < 1) &  (scaled_y >= 0) & (scaled_y < 1) ) :
#            index_x = (int)(scaled_x * counts_length_x);
#            index_y = (int)(scaled_y * counts_length_y);
#            grid[index_y, index_x] += 1;

class TaskHistogram(Task):
	def __init__(self, dataset, subspace, expressions, size, limits, masked=False, weight=None):
		self.size = size
		self.limits = limits
		Task.__init__(self, dataset, expressions, name="histogram")
		self.subspace = subspace
		self.dtype = np.float64
		self.masked = masked
		self.weight = weight
		#self.grids = vaex.grids.Grids(self.dataset, self.dataset.executor.thread_pool, *expressions)
		#self.grids.ranges = limits
		#self.grids.grids["counts"] = vaex.grids.Grid(self.grids, size, self.dimension, None)
		shape = (self.subspace.executor.thread_pool.nthreads,) + ( self.size,) * self.dimension
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
			#if subblock_weight is None:
			#	#print "speedup?"
			#	histogram_numba(blocks[0], blocks[1], subblock_weight, data, *self.ranges_flat)
			#else:
				vaex.vaexfast.histogram2d(blocks[0], blocks[1], subblock_weight, data, *self.ranges_flat)
		elif self.dimension == 3:
			vaex.vaexfast.histogram3d(blocks[0], blocks[1], blocks[2], subblock_weight, data, *self.ranges_flat)
		else:
			blocks = list(blocks) # histogramNd wants blocks to be a list
			vaex.vaexfast.histogramNd(blocks, subblock_weight, data, self.minima, self.maxima)

		return i1
		#return map(self._map, blocks)#[self.map(block) for block in blocks]

	def reduce(self, results):
		for i in range(1, self.subspace.executor.thread_pool.nthreads):
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
		self.subspace_bounded.subspace.plot(np.log1p(self.grid), limits=self.subspace_bounded.bounds, axes=axes, **kwargs)

	def mean_line(self, axis=0, **kwargs):
		from matplotlib import pylab
		assert axis in [0,1]
		other_axis = 0 if axis == 1 else 1
		xmin, xmax = self.subspace_bounded.bounds[axis]
		ymin, ymax = self.subspace_bounded.bounds[other_axis]
		x = vaex.utils.linspace_centers(xmin, xmax, self.grid.shape[axis])
		y = vaex.utils.linspace_centers(ymin, ymax, self.grid.shape[other_axis])
		print(y)
		if axis == 0:
			counts = np.sum(self.grid, axis=axis)
			means = np.sum(self.grid * y[np.newaxis,:].T, axis=axis)/counts
		else:
			counts = np.sum(self.grid, axis=axis)
			means = np.sum(self.grid * y[:,np.newaxis].T, axis=axis)/counts
		if axis == 0:
			result = pylab.plot(x, means, **kwargs)
		else:
			result = pylab.plot(means, x, **kwargs)

		self.subspace_bounded.lim()
		return result, x, means



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

	def lim(self):
		from matplotlib import pylab
		xmin, xmax = self.bounds[0]
		ymin, ymax = self.bounds[1]
		pylab.xlim(xmin, xmax)
		pylab.ylim(ymin, ymax)


class Subspaces(object):
	"""
	:type: subspaces: list[Subspace]

	"""
	def __init__(self, subspaces):
		self.subspaces = subspaces
		self.expressions = set()
		first_subspace = self.subspaces[0]
		self.async = first_subspace.async
		self.dimension = first_subspace.dimension
		self.dataset = self.subspaces[0].dataset
		for subspace in self.subspaces:
			assert subspace.dataset == self.subspaces[0].dataset
			assert subspace.async == self.subspaces[0].async
			assert subspace.dimension == self.subspaces[0].dimension, "subspace is of dimension %s, while first subspace if of dimension %s" % (subspace.dimension, self.subspaces[0].dimension)
			#assert subspace.sele== self.subspaces[0].async
			self.expressions.update(subspace.expressions)
		self.expressions = list(self.expressions)
		self.subspace = self.dataset(*list(self.expressions), async=self.async, executor=first_subspace.executor)

	#def _repr_html_(self):



	def __len__(self):
		return len(self.subspaces)

	def names(self, seperator=" "):
		return [seperator.join(subspace.expressions) for subspace in self.subspaces]

	def expressions_list(self):
		return [subspace.expressions for subspace in self.subspaces]

	def selected(self):
		return Subspaces([subspace.selected() for subspace in self.subspaces])

	def _unpack(self, values):
		value_map = dict(zip(self.expressions, values))
		return [[value_map[ex] for ex in subspace.expressions] for subspace in self.subspaces]

	def _pack(self, values):
		value_map = {}
		for subspace_values, subspace in zip(values, self.subspaces):
			for value, expression in zip(subspace_values, subspace.expressions):
				if expression in value_map:
					if isinstance(value, np.ndarray):
						assert np.all(value_map[expression] == value), "inconsistency in subspaces, value for expression %r is %r in one case, and %r in the other" % (expression, value, value_map[expression])
					else:
						assert value_map[expression] == value, "inconsistency in subspaces, value for expression %r is %r in one case, and %r in the other" % (expression, value, value_map[expression])
				else:
					value_map[expression] = value
		return [value_map[expression] for expression in self.expressions]


	def minmax(self):
		if self.async:
			return self.subspace.minmax().then(self._unpack)
		else:
			return self._unpack(self.subspace.minmax())

	def limits_sigma(self, sigmas=3, square=False):
		if self.async:
			return self.subspace.limits_sigma(sigmas=sigmas, square=square).then(self._unpack)
		else:
			return self._unpack(self.subspace.limits_sigma(sigmas=sigmas, square=square))

	def mutual_information(self, limits=None, size=256):
		if limits is not None:
			limits = self._pack(limits)
		def mutual_information(limits):
			return vaex.promise.listPromise([vaex.promise.Promise.fulfilled(subspace.mutual_information(subspace_limits, size=size)) for subspace_limits, subspace in zip(limits, self.subspaces)])
			#return histograms
		if limits is None:
			limits_promise = vaex.promise.Promise.fulfilled(self.subspace.minmax())
		else:
			limits_promise = vaex.promise.Promise.fulfilled(limits)
		limits_promise = limits_promise.then(self._unpack)
		promise = limits_promise.then(mutual_information)
		return promise if self.async else promise.get()



	def mean(self):
		if self.async:
			return self.subspace.mean().then(self._unpack)
		else:
			means = self.subspace.mean()
			return self._unpack(means)

	def var(self, means=None):
		# 'pack' means, and check if it makes sence
		if means is not None:
			means = self._pack(means)
		def var(means):
			return self.subspace.var(means=means)
		if self.async:
			#if means is None:
			#	return self.subspace.mean().then(var).then(self._unpack)
			#else:
			return var(means).then(self._unpack)
		else:
			#if means is None:
			#	means = self.subspace.mean()
			#logger.debug("means: %r", means)
			return self._unpack(var(means=means))

	def correlation(self, means=None, vars=None):
		def var(means):
			return self.subspace.var(means=means)
		def correlation(means_and_vars):
			means, vars = means_and_vars
			means, vars = self._unpack(means), self._unpack(vars)
			#return self.subspace.correlation(means=means, vars=vars)
			return vaex.promise.listPromise([subspace.correlation(means=subspace_mean, vars=subspace_var) for subspace_mean, subspace_var, subspace in zip(means, vars, self.subspaces)])
		if means is not None:
			means = self._pack(means)
		if vars is not None:
			vars = self._pack(vars)
		if self.async:
			if means is None:
				mean_promise = self.subspace.mean()
			else:
				mean_promise = vaex.promise.Promise.fulfilled(means)
			if vars is None:
				var_promise = mean_promise.then(var)
			else:
				var_promise = vaex.promise.Promise.fulfilled(vars)
			mean_and_var_calculated = vaex.promise.listPromise(mean_promise, var_promise)
			return mean_and_var_calculated.then(correlation)
		else:
			if means is None:
				means = self.subspace.mean()
			if vars is None:
				vars = self.subspace.var(means=means)
			means = self._unpack(means)
			vars = self._unpack(vars)
			return [subspace.correlation(means=subspace_mean, vars=subspace_var) for subspace_mean, subspace_var, subspace in zip(means, vars, self.subspaces)]
			#return correlation((means, vars))





	#def bounded_by(self, limits_list):
	#	return SubspacesBounded(SubspaceBounded(subspace, limits) for subspace, limit in zip(self.subspaces, limits_list))

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

	def plot(self, grid=None, size=256, limits=None, square=False, center=None, weight=None, figsize=None, aspect="auto", f=lambda x: x, axes=None, xlabel=None, ylabel=None, **kwargs):
		"""Plot the subspace using sane defaults to get a quick look at the data.

		:param grid: A 2d numpy array with the counts, if None it will be calculated using limits provided and Subspace.histogram
		:param size: Passed to Subspace.histogram
		:param limits: Limits for the subspace in the form [[xmin, xmax], [ymin, ymax]], if None it will be calculated using Subspace.limits_sigma
		:param square: argument passed to Subspace.limits_sigma
		:param Executor executor: responsible for executing the tasks
		:param figsize: (x, y) tuple passed to pylab.figure for setting the figure size
		:param aspect: Passed to matplotlib's axes.set_aspect
		:param xlabel: String for label on x axis (may contain latex)
		:param ylabel: Same for y axis
		:param kwargs: extra argument passed to axes.imshow, useful for setting the colormap for instance, e.g. cmap='afmhot'
		:return: matplotlib.image.AxesImage

		 """
		import pylab
		if limits is None:
			limits = self.limits_sigma()
		if grid is None:
			grid = self.histogram(limits=limits, size=size, weight=weight)
		if figsize is not None:
			pylab.figure(num=None, figsize=figsize, dpi=80, facecolor='w', edgecolor='k')
		if axes is None:
			axes = pylab.gca()
		#if xlabel:
		pylab.xlabel(xlabel or self.expressions[0])
		#if ylabel:
		pylab.ylabel(ylabel or self.expressions[1])
		#axes.set_aspect(aspect)
		return axes.imshow(f(grid), extent=np.array(limits).flatten(), origin="lower", aspect=aspect, **kwargs)

	def figlarge(self, size=(10,10)):
		import pylab
		pylab.figure(num=None, figsize=size, dpi=80, facecolor='w', edgecolor='k')

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

	def _task(self, task, progressbar=False):
		"""Helper function for returning tasks results, result when immediate is True, otherwise the task itself, which is a promise"""
		if self.async:
			# should return a task or a promise nesting it
			return self.executor.schedule(task)
		else:
			import vaex.utils
			callback = None
			try:
				if progressbar == True:
					def update(fraction):
						bar.update(fraction)
						return True
					bar = vaex.utils.progressbar(task.name)
					callback = self.executor.signal_progress.connect(update)
				result = self.executor.run(task)
				if progressbar == True:
					bar.finish()
					sys.stdout.write('\n')
				return result
			finally:
				if callback:
					self.executor.signal_progress.disconnect(callback)

	def minmax(self, progressbar=False):
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
		task = TaskMapReduce(self.dataset, self.expressions, min_max_map, min_max_reduce, self._toarray, info=True, name="minmax")
		return self._task(task, progressbar=progressbar)

	def mean(self):
		return self._moment(1)

	def _moment(self, moment=1):
		def mean_reduce(means_and_counts1, means_and_counts2):
			means_and_counts = []
			for (mean1, count1), (mean2, count2) in zip(means_and_counts1, means_and_counts2):
				means_and_counts.append( [np.nansum([mean1*count1, mean2*count2])/(count1+count2), count1+count2] )
			return means_and_counts
		def remove_counts(means_and_counts):
			return self._toarray(means_and_counts)[:,0]
		def mean_map(thread_index, i1, i2, *blocks):
			if self.is_masked:
				mask = self.dataset.mask
				return [(np.nanmean(block[mask[i1:i2]]**moment), np.count_nonzero(~np.isnan(block[mask[i1:i2]]))) for block in blocks]
			else:
				return [(np.nanmean(block**moment), np.count_nonzero(~np.isnan(block))) for block in blocks]
		task = TaskMapReduce(self.dataset, self.expressions, mean_map, mean_reduce, remove_counts, info=True)
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

	def correlation(self, means=None, vars=None):
		if self.dimension != 2:
			raise ValueError("correlation is only defined for 2d subspaces, not %dd" % self.dimension)

		def do_correlation(means, vars):
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
		if means is None:
			if self.async:
				means_wrapper = [None]
				def do_vars(means):
					means_wrapper[0] = means
					return self.var(means)
				def do_correlation_wrapper(vars):
					return do_correlation(means_wrapper[0], vars)
				return self.mean().then(do_vars).then(do_correlation_wrapper)
			else:
				means = self.mean()
				vars = self.var(means=means)
				return do_correlation(means, vars)
		else:
			if vars is None:
				if self.async:
					def do_correlation_wrapper(vars):
						return do_correlation(means, vars)
					return self.vars(means=means).then(do_correlation_wrapper)
				else:
					vars = self.var(means)
					return do_correlation(means, vars)
			else:
				if means is None:
					means = self.mean()
				if vars is None:
					vars = self.var(means=means)
				return do_correlation(means, vars)

	def sum(self):
		nansum = lambda x: np.nansum(x, dtype=np.float64)
		# TODO: we can speed up significantly using our own nansum, probably the same for var and mean
		# nansum = vaex.vaexfast.nansum
		if self.is_masked:
			mask = self.dataset.mask
			task = TaskMapReduce(self.dataset,\
								 self.expressions, lambda thread_index, i1, i2, *blocks: [nansum(block[mask[i1:i2]]) for block in blocks],\
								 lambda a, b: np.array(a) + np.array(b), self._toarray, info=True)
		else:
			task = TaskMapReduce(self.dataset, self.expressions, lambda *blocks: [nansum(block) for block in blocks], lambda a, b: np.array(a) + np.array(b), self._toarray)
		return self._task(task)

	def histogram(self, limits, size=256, weight=None, progressbar=False, ):
		task = TaskHistogram(self.dataset, self, self.expressions, size, limits, masked=self.is_masked, weight=weight)
		return self._task(task, progressbar=progressbar)

	def mutual_information(self, limits=None, grid=None, size=256):
		if limits is None:
			limits_done = Task.fulfilled(self.minmax())
		else:
			limits_done = Task.fulfilled(limits)
		if grid is None:
			if limits is None:
				histogram_done = limits_done.then(lambda limits: self.histogram(limits, size=size))
			else:
				histogram_done = Task.fulfilled(self.histogram(limits, size=size))
		else:
			histogram_done = Task.fulfilled(grid)
		mutual_information_promise = histogram_done.then(vaex.kld.mutual_information)
		return mutual_information_promise if self.async else mutual_information_promise.get()


	def limits_sigma(self, sigmas=3, square=False):
		if self.async:
			means_wrapper = [None]
			def do_vars(means):
				means_wrapper[0] = means
				return self.var(means)
			def do_limits(vars):
				stds = vars**0.5
				means = means_wrapper[0]
				if square:
					stds = np.repeat(stds.mean(), len(stds))
				return np.array(list(zip(means-sigmas*stds, means+sigmas*stds)))
			return self.mean().then(do_vars).then(do_limits)
		else:
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
			min_index_global = min_index = np.argmin(distance_squared)
			if self.is_masked: # we skipped some indices, so correct for that
				min_index_global = np.argmin((np.cumsum(mask) - 1 - min_index)**2)
			#with lock:
			#	print i1, i2, min_index, distance_squared, [block[min_index] for block in blocks]
			return min_index_global.item() + i1, distance_squared[min_index].item()**0.5, [block[min_index].item() for block in blocks]
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

globals_for_eval = {}
globals_for_eval.update(np.__dict__)

class UnitScope(object):
	def __init__(self, dataset, value=None):
		self.dataset = dataset
		self.value = value

	def __getitem__(self, variable):
		if variable in self.dataset.units:
			unit = self.dataset.units[variable]
			return (self.value * unit) if self.value is not None else unit
		elif variable in self.dataset.virtual_columns:
			return eval(self.dataset.virtual_columns[variable], globals_for_eval, self)
		elif variable in self.dataset.variables:
			return astropy.units.dimensionless_unscaled # TODO units for variables?
		else:
			raise KeyError("unkown variable %s" % variable)

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
			logger.debug("creating column for: %s", column)
			self.buffers[column] = np.zeros(self.i2-self.i1)

	def evaluate(self, expression, out=None):
		try:
			#logger.debug("try avoid evaluating: %s", expression)
			result = self[expression]
		except:
			#logger.debug("no luck, eval: %s", expression)
			#result = ne.evaluate(expression, local_dict=self, out=out)
			#logger.debug("in eval")
			#eval("def f(")
			result = eval(expression, {}, self)
			self.values[expression] = result
			#if out is not None:
			#	out[:] = result
			#	result = out
			#logger.debug("out eval")
		#logger.debug("done with eval of %s", expression)
		return result

	def __getitem__(self, variable):
		#logger.debug("get " + variable)
		#return self.dataset.columns[variable][self.i1:self.i2]
		if variable in np.__dict__:
			return np.__dict__[variable]
		try:
			if variable in self.dataset.get_column_names():
				if self.dataset._needs_copy(variable):
					#self._ensure_buffer(variable)
					#self.values[variable] = self.buffers[variable] = self.dataset.columns[variable][self.i1:self.i2].astype(np.float64)
					self.values[variable] = self.dataset.columns[variable][self.i1:self.i2].astype(np.float64)
				else:
					self.values[variable] = self.dataset.columns[variable][self.i1:self.i2]
			elif variable in self.values:
				return self.values[variable]
			elif variable in list(self.dataset.virtual_columns.keys()):
				expression = self.dataset.virtual_columns[variable]
				#self._ensure_buffer(variable)
				self.values[variable] = self.evaluate(expression)#, out=self.buffers[variable])
				#self.values[variable] = self.buffers[variable]
			if variable not in self.values:
				raise KeyError("Unknown variables or column: %r" % (variable,))

			return self.values[variable]
		except:
			#logger.exception("error in evaluating: %r" % variable)
			raise

main_executor = None# vaex.execution.Executor(vaex.multithreading.pool)
from vaex.execution import Executor
def get_main_executor():
	global main_executor
	if main_executor is None:
		main_executor = vaex.execution.Executor(vaex.multithreading.get_main_pool())
	return main_executor


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

class SelectionInvert(Selection):
	def __init__(self, dataset, previous_selection):
		super(SelectionInvert, self).__init__(dataset, previous_selection, "")

	def to_dict(self):
		previous = None
		if self.previous_selection:
			previous = self.previous_selection.to_dict()
		return dict(type="invert", previous_selection=previous)


	def execute(self, executor, execute_fully=False):
		super(SelectionInvert, self).execute(executor=executor, execute_fully=execute_fully)
		self.dataset.mask = ~self.dataset.mask
		self.dataset._set_mask(self.dataset.mask)
		return
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
	elif values["type"] == "invert":
		kwargs["previous_selection"] = selection_from_dict(dataset, values["previous_selection"]) if values["previous_selection"] else None
		return SelectionInvert(**kwargs)
	else:
		raise ValueError("unknown type: %r, in dict: %r" % (values["type"], values))

# name maps to numpy function
# <vaex name>:<numpy name>
function_mapping = [name.strip().split(":") if ":" in name else (name, name) for name in """
sinc
sin
cos
tan
arcsin
arccos
arctan2
sinh
cosh
tanh
arcsinh
arccosh
arctanh
log
log10
log1p
exp
expm1
sqrt
abs
""".strip().split()]
expression_namespace = {}
for name, numpy_name in function_mapping:
	if not hasattr(np, numpy_name):
		raise SystemError("numpy does not have: %s" % numpy_name)
	else:
		expression_namespace[name] = getattr(np, numpy_name)


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
		self.executor = executor or get_main_executor()
		self.signal_pick = vaex.events.Signal("pick")
		self.signal_sequence_index_change = vaex.events.Signal("sequence index change")
		self.signal_selection_changed = vaex.events.Signal("selection changed")
		self.signal_active_fraction_changed = vaex.events.Signal("active fraction changed")
		self.signal_column_changed = vaex.events.Signal("a column changed") # (dataset, column_name, change_type=["add", "remove", "change"])
		self.signal_variable_changed = vaex.events.Signal("a variable changed")

		self.undo_manager = vaex.ui.undo.UndoManager()
		self.variables = collections.OrderedDict()
		self.variables["pi"] = np.pi
		self.variables["e"] = np.e
		self.variables["km_in_au"] = 149597870700/1000.
		self.variables["seconds_per_year"] = 31557600
		# leads to k = 4.74047 to go from au/year to km/s
		self.virtual_columns = collections.OrderedDict()
		self._length = None
		self._full_length = None
		self._active_fraction = 1
		self._current_row = None

		self.description = None
		self.ucds = {}
		self.units = {}
		self.descriptions = {}

		self.favorite_selections = collections.OrderedDict()

		self.mask = None # a bitmask for the selection does not work for server side

		# maps from name to list of Selection objets
		self.selection_histories = collections.defaultdict(list)
		# after an undo, the last one in the history list is not the active one, -1 means no selection
		self.selection_history_indices = collections.defaultdict(lambda: -1)
		self._auto_fraction= False

	def close_files(self):
		"""Close any possible open file handles, the dataset not not be usable afterwards"""
		pass

	def byte_size(self, selection=False):
		bytes_per_row = 0
		for column in list(self.get_column_names()):
			dtype = self.dtype(column)
			bytes_per_row += dtype.itemsize
		return bytes_per_row * self.length(selection=selection)


	def dtype(self, expression):
		if expression in self.get_column_names():
			return self.columns[expression].dtype
		else:
			return np.zeros(1, dtype=np.float64).dtype

	def unit(self, expression, default=None):
		"""Returns the unit (an astropy.unit.Units object) for the expression

		:Example:
		>>> import vaex as vx
		>>> ds = vx.example()
		>>> ds.unit("x")
		Unit("kpc")
		>>> ds.unit("x*L")
		Unit("km kpc2 / s")


		:param expression: Expression, which can be a column name
		:param default: if no unit is known, it will return this
		:return: The resulting unit of the expression
		:rtype: astropy.units.Unit
		"""
		try:
			# if an expression like pi * <some_expr> it will evaluate to a quantity instead of a unit
			unit_or_quantity = eval(expression, globals_for_eval, UnitScope(self))
			return unit_or_quantity.unit if hasattr(unit_or_quantity, "unit") else unit_or_quantity
		except:
			#logger.exception("error evaluating unit expression: %s", expression)
			# astropy doesn't add units, so we try with a quatiti
			try:
				return eval(expression, globals_for_eval, UnitScope(self, 1.)).unit
			except:
				#logger.exception("error evaluating unit expression: %s", expression)
				return default

	def ucd_find(self, *ucds):
		"""Find a set of columns (names) which have the ucd, or part of the ucd

		Prefixed with a ^, it will only match the first part of the ucd

		:Example:
		>>> dataset.ucd_find('pos.eq.ra', 'pos.eq.dec')
		['RA', 'DEC']
		>>> dataset.ucd_find('pos.eq.ra', 'doesnotexist')
		>>> dataset.ucds[dataset.ucd_find('pos.eq.ra')]
		'pos.eq.ra;meta.main'
		>>> dataset.ucd_find('meta.main')]
		'dec'
		>>> dataset.ucd_find('^meta.main')]
		>>>
		"""
		if len(ucds) == 1:
			ucd = ucds[0]
			if ucd[0] == "^": # we want it to start with
				ucd = ucd[1:]
				columns = [name for name in self.get_column_names(virtual=True) if self.ucds.get(name, "").startswith(ucd)]
			else:
				columns = [name for name in self.get_column_names(virtual=True) if ucd in self.ucds.get(name, "")]
			return None if len(columns) == 0 else columns[0]
		else:
			columns = [self.ucd_find(ucd) for ucd in ucds]
			return None if None in columns else columns

	def add_favorite_selection(self, name, selection_name="default"):
		selection = self.get_selection(selection_name=selection_name)
		if selection:
			self.favorite_selections[name] = selection
			self.store_favorite_selections()
		else:
			raise ValueError("no selection exists")

	def remove_favorite_selection(self, name):
		del self.favorite_selections[name]
		self.store_favorite_selections()

	def apply_favorite_selection(self, name, selection_name="default", executor=None):
		self.set_selection(self.favorite_selections[name], selection_name=selection_name, executor=executor)

	def store_favorite_selections(self):
		path = os.path.join(self.get_private_dir(create=True), "favorite_selection.yaml")
		selections = collections.OrderedDict([(key,value.to_dict()) for key,value in self.favorite_selections.items()])
		vaex.utils.write_json_or_yaml(path, selections)

	def load_favorite_selections(self):
		try:
			path = os.path.join(self.get_private_dir(create=True), "favorite_selection.yaml")
			if os.path.exists(path):
				selections_dict = vaex.utils.read_json_or_yaml(path)
				for key, value in selections_dict.items():
					self.favorite_selections[key] = selection_from_dict(self, value)
		except:
			logger.exception("non fatal error")


	def get_private_dir(self, create=False):
		"""Each datasets has a directory where files are stored for metadata etc

		:Example:
		>>> import vaex as vx
		>>> ds = vx.example()
		>>> ds.get_private_dir()
		'/Users/users/breddels/.vaex/datasets/_Users_users_breddels_vaex-testing_data_helmi-dezeeuw-2000-10p.hdf5'

		:param bool create: is True, it will create the directory if it does not exist

		"""
		if self.is_local():
			name = os.path.abspath(self.path).replace("/", "_")
		else:
			server = self.server
			name = "%s_%s_%s_%s" % (server.hostname, server.port, server.base_path.replace("/", "_"), self.name)
		dir = os.path.join(vaex.utils.get_private_dir(), "datasets", name)
		if create and not os.path.exists(dir):
			os.makedirs(dir)
		return dir


	def remove_virtual_meta(self):
		"""Removes the file with the virtual column etc, it does not change the current virtual columns etc"""
		dir = self.get_private_dir(create=True)
		path = os.path.join(dir, "virtual_meta.yaml")
		try:
			if os.path.exists(path):
				os.remove(path)
			if not os.listdir(dir):
				os.rmdir(dir)
		except:
			logger.exception("error while trying to remove %s or %s", path, dir)
	#def remove_meta(self):
	#	path = os.path.join(self.get_private_dir(create=True), "meta.yaml")
	#	os.remove(path)

	def write_virtual_meta(self):
		"""Writes virtual columns, variables and their ucd,description and units

		The default implementation is to write this to a file called virtual_meta.yaml in the directory defined by
		:func:`Dataset.get_private_dir`. Other implementation may store this in the dataset file itself.

		This method is called after virtual columns or variables are added. Upon opening a file, :func:`Dataset.update_virtual_meta`
		is called, so that the information is not lost between sessions.

		Note: opening a dataset twice may result in corruption of this file.

		"""
		path = os.path.join(self.get_private_dir(create=True), "virtual_meta.yaml")
		virtual_names = list(self.virtual_columns.keys())  + list(self.variables.keys())
		units = {key:str(value) for key, value in self.units.items() if key in virtual_names}
		ucds = {key:value for key, value in self.ucds.items() if key in virtual_names}
		descriptions = {key:value for key, value in self.descriptions.items() if key in virtual_names}
		meta_info = dict(virtual_columns=self.virtual_columns,
						 variables=self.variables,
						 ucds=ucds, units=units, descriptions=descriptions)
		vaex.utils.write_json_or_yaml(path, meta_info)

	def update_virtual_meta(self):
		"""Will read back the virtual column etc, written by :func:`Dataset.write_virtual_meta`. This will be done when opening a dataset."""
		try:
			path = os.path.join(self.get_private_dir(create=False), "virtual_meta.yaml")
			if os.path.exists(path):
					meta_info = vaex.utils.read_json_or_yaml(path)
					self.virtual_columns.update(meta_info["virtual_columns"])
					self.variables.update(meta_info["variables"])
					self.ucds.update(meta_info["ucds"])
					self.descriptions.update(meta_info["descriptions"])
					units = {key:astropy.units.Unit(value) for key, value in meta_info["units"].items()}
					self.units.update(units)
		except:
			logger.exception("non fatal error")

	def write_meta(self):
		"""Writes all meta data, ucd,description and units

		The default implementation is to write this to a file called meta.yaml in the directory defined by
		:func:`Dataset.get_private_dir`. Other implementation may store this in the dataset file itself.
		(For instance the vaex hdf5 implementation does this)

		This method is called after virtual columns or variables are added. Upon opening a file, :func:`Dataset.update_meta`
		is called, so that the information is not lost between sessions.

		Note: opening a dataset twice may result in corruption of this file.

		"""
		#raise NotImplementedError
		path = os.path.join(self.get_private_dir(create=True), "meta.yaml")
		units = {key:str(value) for key, value in self.units.items()}
		meta_info = dict(description=self.description,
						 ucds=self.ucds, units=units, descriptions=self.descriptions,
)
		vaex.utils.write_json_or_yaml(path, meta_info)

	def update_meta(self):
		"""Will read back the ucd, descriptions, units etc, written by :func:`Dataset.write_meta`. This will be done when opening a dataset."""
		try:
			path = os.path.join(self.get_private_dir(create=False), "meta.yaml")
			if os.path.exists(path):
				meta_info = vaex.utils.read_json_or_yaml(path)
				self.description = meta_info["description"]
				self.ucds.update(meta_info["ucds"])
				self.descriptions.update(meta_info["descriptions"])
				#self.virtual_columns.update(meta_info["virtual_columns"])
				#self.variables.update(meta_info["variables"])
				units = {key:astropy.units.Unit(value) for key, value in meta_info["units"].items()}
				self.units.update(units)
		except:
			logger.exception("non fatal error, but could read/understand %s", path)



	def is_local(self):
		"""Returns True if the dataset is a local dataset, False when a remote dataset"""
		raise NotImplementedError

	def get_auto_fraction(self):
		return self._auto_fraction

	def set_auto_fraction(self, enabled):
		self._auto_fraction = enabled

	@classmethod
	def can_open(cls, path, *args, **kwargs):
		"""Tests if this class can open the file given by path"""
		return False

	@classmethod
	def get_options(cls, path):
		return []

	@classmethod
	def option_to_args(cls, option):
		return []

	def subspace(self, *expressions, **kwargs):
		"""Return a :class:`Subspace` for this dataset with the given expressions:

		Example:

		>>> subspace_xy = some_dataset("x", "y")

		:rtype: Subspace
		:param list[str] expressions: list of expressions
		:param kwargs:
		:return:
		"""
		return self(*expressions, **kwargs)

	def subspaces(self, expressions_list=None, dimensions=None, exclude=None, **kwargs):
		"""Generate a Subspaces object, based on a custom list of expressions or all possible combinations based on
		dimension

		:param expressions_list: list of list of expressions, where the inner list defines the subspace
		:param dimensions: if given, generates a subspace with all possible combinations for that dimension
		:param exclude: list of
		"""
		if dimensions is not None:
			expressions_list = list(itertools.combinations(self.get_column_names(), dimensions))
			if exclude is not None:
				import six
				def excluded(expressions):
					if callable(exclude):
						return exclude(expressions)
					elif isinstance(exclude, six.string_types):
						return exclude in expressions
					elif isinstance(exclude, (list, tuple)):
						#$#expressions = set(expressions)
						for e in exclude:
							if isinstance(e, six.string_types):
								if e in expressions:
									return True
							elif isinstance(e, (list, tuple)):
								if set(e).issubset(expressions):
									return True
							else:
								raise ValueError("elements of exclude should contain a string or a sequence of strings")
					else:
						raise ValueError("exclude should contain a string, a sequence of strings, or should be a callable")
					return False
				# test if any of the elements of exclude are a subset of the expression
				expressions_list = [expr for expr in expressions_list if not excluded(expr)]
			logger.debug("expression list generated: %r", expressions_list)
		return Subspaces([self(*expressions, **kwargs) for expressions in expressions_list])


	def __call__(self, *expressions, **kwargs):
		"""Alias/shortcut for :func:`Dataset.subspace`"""
		raise NotImplementedError

	def set_variable(self, name, expression_or_value):
		"""Set the variable to an expression or value defined by expression_or_value

		:Example:
		>>> ds.set_variable("a", 2.)
		>>> ds.set_variable("b", "a**2")
		>>> ds.get_variable("b")
		'a**2'
		>>> ds.evaluate_variable("b")
		4.0

		:param name: Name of the variable
		:param expression: value or expression
		"""
		self.variables[name] = expression_or_value
		self.write_virtual_meta()

	def get_variable(self, name):
		"""Returns the variable given by name, it will not evaluate it.

		For evaluation, see :func:`Dataset.evaluate_variable`, see also :func:`Dataset.set_variable`

		"""
		return self.variables[name]

	def evaluate_variable(self, name):
		"""Evaluates the variable given by name"""
		if isinstance(self.variables[name], six.string_types):
			# TODO: this does not allow more than one level deep variable, like a depends on b, b on c, c is a const
			value = eval(self.variables[name], expression_namespace, self.variables)
			return value
		else:
			return self.variables[name]

	def evaluate(self, expression, i1=None, i2=None, out=None):
		"""Evaluate an expression, and return a numpy array with the results for the full column or a part of it.

		Note that this is not how vaex should be used, since it means a copy of the data needs to fit in memory.

		To get partial results, use i1 and i2/

		:param str expression: Name/expression to evaluate
		:param int i1: Start row index, default is the start (0)
		:param int i2: End row index, default is the length of the dataset
		:param ndarray out: Output array, to which the result may be written (may be used to reuse an array, or write to
		a memory mapped array)
		:return:
		"""
		raise NotImplementedError

	def validate_expression(self, expression):
		"""Validate an expression (may throw Exceptions)"""
		#return self.evaluate(expression, 0, 2)
		vars = set(self.get_column_names(True, True)) | set(self.variables.keys())
		funcs = set(expression_namespace.keys())
		return vaex.expresso.validate_expression(expression, vars, funcs)

	def _block_scope(self, i1, i2):
		variables = {key:self.evaluate_variable(key) for key in self.variables.keys()}
		return _BlockScope(self, i1, i2, **variables)

	def select(self, boolean_expression, mode="replace", selection_name="default"):
		"""Select rows based on the boolean_expression, if there was a previous selection, the mode is taken into account.

		if boolean_expression is None, remove the selection, has_selection() will returns false

		Note that per dataset, only one selection is possible.

		:param str boolean_expression: boolean expression, such as 'x < 0', '(x < 0) || (y > -10)' or None to remove the selection
		:param str mode: boolean operation to perform with the previous selection, "replace", "and", "or", "xor", "subtract"
		:return: None
		"""
		raise NotImplementedError

	def add_virtual_columns_matrix3d(self, x, y, z, xnew, ynew, znew, matrix, matrix_name, matrix_is_expression=False):
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
		matrix_list = [[None for i in range(3)] for j in range(3)]
		for i in range(3):
			for j in range(3):
				if matrix_is_expression:
					self.add_virtual_column(matrix_name +"_%d%d" % (i,j), matrix[i][j])
				else:
					#self.set_variable(matrix_name +"_%d%d" % (i,j), matrix[i,j])
					matrix_list[i][j] = matrix[i,j].item()
		if not matrix_is_expression:
			self.add_variable(matrix_name, matrix_list)

		if matrix_is_expression:
			self.virtual_columns[xnew] = "{m}_00 * {x} + {m}_01 * {y} + {m}_02 * {z}".format(**locals())
			self.virtual_columns[ynew] = "{m}_10 * {x} + {m}_11 * {y} + {m}_12 * {z}".format(**locals())
			self.virtual_columns[znew] = "{m}_20 * {x} + {m}_21 * {y} + {m}_22 * {z}".format(**locals())
		else:
			self.virtual_columns[xnew] = "{m}[0][0] * {x} + {m}[0][1] * {y} + {m}[0][2] * {z}".format(**locals())
			self.virtual_columns[ynew] = "{m}[1][0] * {x} + {m}[1][1] * {y} + {m}[1][2] * {z}".format(**locals())
			self.virtual_columns[znew] = "{m}[2][0] * {x} + {m}[2][1] * {y} + {m}[2][2] * {z}".format(**locals())
		self.write_virtual_meta()

	def add_virtual_columns_eq2ecl(self, long_in, lat_in, long_out="lambda_", lat_out="beta", input=None, output=None, name_prefix="__celestial_eq2ecl", radians=False):
		"""Add ecliptic coordates (long_out, lat_out) from equatorial coordinates.

		:param long_in: Name/expression for right ascension
		:param lat_in: Name/expression for declination
		:param long_out:  Output name for lambda coordinate
		:param lat_out: Output name for beta coordinate
		:param input:
		:param output:
		:param name_prefix:
		:param radians: input and output in radians (True), or degrees (False)
		:return:
		"""
		import kapteyn.celestial as c
		self.add_virtual_columns_celestial(long_in, lat_in, long_out, lat_out, input=input or c.equatorial, output=output or c.ecliptic, name_prefix=name_prefix, radians=radians)

	def add_virtual_columns_eq2gal(self, long_in, lat_in, long_out="l", lat_out="b", input=None, output=None, name_prefix="__celestial_eq2gal", radians=False):
		"""Add galactic coordates (long_out, lat_out) from equatorial coordinates.

		:param long_in: Name/expression for right ascension
		:param lat_in: Name/expression for declination
		:param long_out:  Output name for galactic longitude
		:param lat_out: Output name for galactic latitude
		:param input:
		:param output:
		:param name_prefix:
		:param radians: input and output in radians (True), or degrees (False)
		:return:
		"""
		import kapteyn.celestial as c
		self.add_virtual_columns_celestial(long_in, lat_in, long_out, lat_out, input=input or c.equatorial, output=output or c.galactic, name_prefix=name_prefix, radians=radians)

	def add_virtual_columns_proper_motion_eq2gal(self, long_in, lat_in, pm_long, pm_lat, pm_long_out, pm_lat_out, name_prefix="__proper_motion_eq2gal", radians=False):
		"""Transform/rotate proper motions from equatorial to galactic coordinates

		Taken from http://arxiv.org/abs/1306.2945

		:param long_in: Name/expression for right ascension
		:param lat_in: Name/expression for declination
		:param pm_long: Proper motion for ra
		:param pm_lat: Proper motion for dec
		:param pm_long_out:  Output name for output proper motion on l direction
		:param pm_lat_out: Output name for output proper motion on b direction
		:param name_prefix:
		:param radians: input and output in radians (True), or degrees (False)
		:return:
		"""
		import kapteyn.celestial as c
		"""mu_gb =  mu_dec*(cdec*sdp-sdec*cdp*COS(ras))/cgb $
		  - mu_ra*cdp*SIN(ras)/cgb"""
		if not radians:
			long_in = "pi/180.*%s" % long_in
			lat_in = "pi/180.*%s" % lat_in
		c1 = name_prefix + "_C1"
		c2 = name_prefix + "_C2"
		self.add_variable("right_ascension_galactic_pole", np.radians(192.85).item())
		self.add_variable("declination_galactic_pole", np.radians(27.12).item())
		self.add_virtual_column(c1, "sin(declination_galactic_pole) * cos({lat_in}) - cos(declination_galactic_pole)*sin({lat_in})*cos({long_in}-right_ascension_galactic_pole)".format(**locals()))
		self.add_virtual_column(c2, "cos(declination_galactic_pole) * sin({long_in}-right_ascension_galactic_pole)".format(**locals()))
		self.add_virtual_column(pm_long_out, "({c1} * {pm_long} + {c2} * {pm_lat})/sqrt({c1}**2+{c2}**2)".format(**locals()))
		self.add_virtual_column(pm_lat_out, "(-{c2} * {pm_long} + {c1} * {pm_lat})/sqrt({c1}**2+{c2}**2)".format(**locals()))

		#mu

		#self.add_virtual_columns_celestial(long_in, lat_in, long_out, lat_out, input=input or c.equatorial, output=output or c.galactic, name_prefix=name_prefix, radians=radians)

	def add_virtual_columns_lbrvr_proper_motion2vcartesian(self, long_in, lat_in, distance, pm_long, pm_lat, vr, vx, vy, vz, name_prefix="__lbvr_proper_motion2vcartesian", center_v=(0,0,0), center_v_name="solar_motion", radians=False):
		"""Convert radial velocity and galactic proper motions (and positions) to cartesian velocities wrt the center_v
		Based on http://adsabs.harvard.edu/abs/1987AJ.....93..864J

		:param long_in: Name/expression for galactic longitude
		:param lat_in: Name/expression for galactic latitude
		:param distance: Name/expression for heliocentric distance
		:param pm_long: Name/expression for the galactic proper motion in latitude direction (pm_l*, so cosine(b) term should be included)
		:param pm_lat: Name/expression for the galactic proper motion in longitude direction
		:param vr: Name/expression for the radial velocity
		:param vx: Output name for the cartesian velocity x-component
		:param vy: Output name for the cartesian velocity y-component
		:param vz: Output name for the cartesian velocity z-component
		:param name_prefix:
		:param center_v: Extra motion that should be added, for instance lsr + motion of the sun wrt the galactic restframe
		:param center_v_name:
		:param radians: input and output in radians (True), or degrees (False)
		:return:
		"""
		k = 4.74057
		if 0:
			v_sun_lsr = array([10.0, 5.2, 7.2])
			v_lsr_gsr = array([0., 220, 0])
			theta0 = 122.932
			#d_NGP = 27.128336111111111
			#al_NGP = 192.10950833333334
			al_NGP = 192.85948
			d_NGP = 27.12825
			c = numpy.matrix([
					[cosd(al_NGP),  sind(al_NGP), 0],
					[sind(al_NGP), -cosd(al_NGP), 0],
					[0, 0, 1]])
			b = numpy.matrix([
					[-sind(d_NGP), 0, cosd(d_NGP)],
					[0, -1, 0],
					[cosd(d_NGP), 0, sind(d_NGP)]])
			a = numpy.matrix([
					[cosd(theta0),  sind(theta0), 0],
					[sind(theta0), -cosd(theta0), 0],
					[0, 0, 1]])
			T = a*b*c
		self.add_variable("k", k)
		A = [["cos({a})*cos({d})",  "-sin({a})", "-cos({a})*sin({d})"],
			 ["sin({a})*cos({d})", "cos({a})", "-sin({a})*sin({d})"],
			 ["sin({d})", "0", "cos({d})"]]
		a = long_in
		d = lat_in
		if not radians:
			a = "pi/180.*%s" % a
			d = "pi/180.*%s" % d
		for i in range(3):
			for j in range(3):
				A[i][j] = A[i][j].format(**locals())
		self.add_virtual_columns_matrix3d(vr, "k*{pm_long}*{distance}".format(**locals()), "k*{pm_lat}*{distance}".format(**locals()), name_prefix +vx, name_prefix +vy, name_prefix +vz,\
										  A, name_prefix+"_matrix", matrix_is_expression=True)
		self.add_variable(center_v_name, center_v)
		self.add_virtual_column(vx, "%s + %s[0]" % (name_prefix +vx, center_v_name))
		self.add_virtual_column(vy, "%s + %s[1]" % (name_prefix +vy, center_v_name))
		self.add_virtual_column(vz, "%s + %s[2]" % (name_prefix +vz, center_v_name))


	def add_virtual_columns_celestial(self, long_in, lat_in, long_out, lat_out, input=None, output=None, name_prefix="__celestial", radians=False):
		import kapteyn.celestial as c
		input = input or c.equatorial
		output = output or c.galactic
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
		#long_out_expr = "arctan2({y_out},{x_out})".format(**locals())
		#lat_out_expr = "arctan2({z_out},sqrt({x_out}**2+{y_out}**2))".format(**locals())
		#if not radians:
		#	long_out_expr = "180./pi*%s" % long_out_expr
		#	lat_out_expr = "180./pi*%s" % lat_out_expr
		transform = "" if radians else "*180./pi"
		x = x_out
		y = y_out
		z = z_out
		#self.add_virtual_column(long_out, "((arctan2({y}, {x})+2*pi) % (2*pi)){transform}".format(**locals()))
		self.add_virtual_column(long_out, "arctan2({y}, {x}){transform}".format(**locals()))
		self.add_virtual_column(lat_out, "(-arccos({z}/sqrt({x}**2+{y}**2+{z}**2))+pi/2){transform}".format(**locals()))

		#self.add_virtual_column(long_out, long_out_expr)
		#self.add_virtual_column(lat_out, lat_out_expr)



	def add_virtual_columns_rotation(self, x, y, xnew, ynew, angle_degrees):
		"""Rotation in 2d

		:param str x: Name/expression of x column
		:param str y: idem for y
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


	def add_virtual_columns_spherical_to_cartesian(self, alpha, delta, distance, xname, yname, zname, center=None, center_name="solar_position", radians=True):
		if not radians:
			alpha = "pi/180.*%s" % alpha
			delta = "pi/180.*%s" % delta
		if center is not None:
			self.add_variable(center_name, center)
		if center:
			solar_mod = " + " +center_name+"[0]"
		else:
			solar_mod = ""
		self.add_virtual_column(xname, "cos(%s) * cos(%s) * %s%s" % (alpha, delta, distance, solar_mod))
		if center:
			solar_mod = " + " +center_name+"[1]"
		else:
			solar_mod = ""
		self.add_virtual_column(yname, "sin(%s) * cos(%s) * %s%s" % (alpha, delta, distance, solar_mod))
		if center:
			solar_mod = " + " +center_name+"[2]"
		else:
			solar_mod = ""
		self.add_virtual_column(zname, "sin(%s) * %s%s" % (delta, distance, solar_mod))

	def add_virtual_columns_cartesian_to_spherical(self, x, y, z, alpha, delta, distance, radians=True, center=None, center_name="solar_position"):
		transform = "" if radians else "*180./pi"

		if center is not None:
			self.add_variable(center_name, center)
		if center is not None and center[0] != 0:
			x = "({x} - {center_name}[0])".format(**locals())
		if center is not None and center[1] != 0:
			y = "({y} - {center_name}[1])".format(**locals())
		if center is not None and center[2] != 0:
			z = "({z} - {center_name}[2])".format(**locals())
		self.add_virtual_column(distance, "sqrt({x}**2 + {y}**2 + {z}**2)".format(**locals()))
		#self.add_virtual_column(alpha, "((arctan2({y}, {x}) + 2*pi) % (2*pi)){transform}".format(**locals()))
		self.add_virtual_column(alpha, "arctan2({y}, {x}){transform}".format(**locals()))
		self.add_virtual_column(delta, "(-arccos({z}/{distance})+pi/2){transform}".format(**locals()))
		#self.add_virtual_column(long_out, "((arctan2({y}, {x})+2*pi) % (2*pi)){transform}".format(**locals()))
		#self.add_virtual_column(lat_out, "(-arccos({z}/sqrt({x}**2+{y}**2+{z}**2))+pi/2){transform}".format(**locals()))

	def add_virtual_columns_aitoff(self, alpha, delta, x, y, radians=True):
		"""Add aitoff (https://en.wikipedia.org/wiki/Aitoff_projection) projection

		:param alpha: azimuth angle
		:param delta: polar angle
		:param x: output name for x coordinate
		:param y: output name for y coordinate
		:param radians: input and output in radians (True), or degrees (False)
		:return:
		"""
		transform = "" if radians else "*pi/180."
		aitoff_alpha = "__aitoff_alpha_%s_%s" % (alpha, delta)
		# sanatize
		aitoff_alpha = re.sub("[^a-zA-Z_]", "_", aitoff_alpha)

		self.add_virtual_column(aitoff_alpha, "arccos(cos({delta}{transform})*cos({alpha}{transform}/2))".format(**locals()))
		self.add_virtual_column(x, "2*cos({delta}{transform})*sin({alpha}{transform}/2)/sinc({aitoff_alpha}/pi)/pi".format(**locals()))
		self.add_virtual_column(y, "sin({delta}{transform})/sinc({aitoff_alpha}/pi)/pi".format(**locals()))

	def add_virtual_columns_equatorial_to_galactic_cartesian(self, alpha, delta, distance, xname, yname, zname, radians=True, alpha_gp=np.radians(192.85948), delta_gp=np.radians(27.12825), l_omega=np.radians(32.93192)):
		"""From http://arxiv.org/pdf/1306.2945v2.pdf"""
		if not radians:
			alpha = "pi/180.*%s" % alpha
			delta = "pi/180.*%s" % delta
		self.virtual_columns[zname] = "{distance} * (cos({delta}) * cos({delta_gp}) * cos({alpha} - {alpha_gp}) + sin({delta}) * sin({delta_gp}))".format(**locals())
		self.virtual_columns[xname] = "{distance} * (cos({delta}) * sin({alpha} - {alpha_gp}))".format(**locals())
		self.virtual_columns[yname] = "{distance} * (sin({delta}) * cos({delta_gp}) - cos({delta}) * sin({delta_gp}) * cos({alpha} - {alpha_gp}))".format(**locals())
		self.write_virtual_meta()

	def add_virtual_column(self, name, expression):
		"""Add a virtual column to the dataset

		:param: str name: name of virtual column
		:param: expression: expression for the column

		:Example:
		>>> dataset.add_virtual_column("r", "sqrt(x**2 + y**2 + z**2)")
		>>> dataset.select("r < 10")
		"""
		type = "change" if name in self.virtual_columns else "add"
		self.virtual_columns[name] = expression
		self.signal_column_changed.emit(self, name, "add")
		self.write_virtual_meta()

	def delete_virtual_column(self, name):
		"""Deletes a virtual column from a dataset"""
		del self.virtual_columns[name]
		self.signal_column_changed.emit(self, name, "delete")
		self.write_virtual_meta()

	def add_variable(self, name, expression):
		"""Add a variable column to the dataset

		:param: str name: name of virtual varible
		:param: expression: expression for the variable

		Variable may refer to other variables, and virtual columns and expression may refer to variables

		:Example:
		>>> dataset.add_variable("center")
		>>> dataset.add_virtual_column("x_prime", "x-center")
		>>> dataset.select("x_prime < 0")
		"""
		self.variables[name] = expression
		self.signal_variable_changed.emit(self, name, "add")
		self.write_virtual_meta()

	def delete_variable(self, name):
		"""Deletes a variable from a dataset"""
		del self.variables[name]
		self.signal_variable_changed.emit(self, name, "delete")
		self.write_virtual_meta()



	def _repr_html_(self):
		"""Representation for Jupyter"""
		html = """<div>%s - %s (length=%d)</div>""" % (cgi.escape(repr(self.__class__)), self.name, len(self))
		html += """<table>"""
		for column_name in self.get_column_names():
			html += "<tr><td>%s</td><td>%s</td></tr>" % (column_name, self.dtype(column_name).name)
		html += "</table>"
		return html


	def __current_sequence_index(self):
		"""TODO"""
		return 0

	def has_current_row(self):
		"""Returns True/False is there currently is a picked row"""
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

	def __has_snapshots(self):
		# currenly disabled
		return False

	def column_count(self):
		"""Returns the number of columns, not counting virtual ones"""
		return len(self.column_names)

	def get_column_names(self, virtual=False, hidden=False):
		"""Return a list of column names


		:param virtual: If True, also return virtual columns
		:param hidden: If True, also return hidden columns
		:rtype: list of str
 		"""
		return list(self.column_names) + ([key for key in self.virtual_columns.keys() if (hidden or (not key.startswith("__")))] if virtual else [])

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
		"""Get the current selection object (mostly for internal use atm)"""
		selection_history = self.selection_histories[selection_name]
		index = self.selection_history_indices[selection_name]
		if index == -1:
			return None
		else:
			return selection_history[index]

	def selection_undo(self, selection_name="default", executor=None):
		"""Undo selection, for the selection_name"""
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
		"""Redo selection, for the selection_name"""
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
		"""Can selection selection_name be undone?"""
		return self.selection_history_indices[selection_name] > -1

	def selection_can_redo(self, selection_name="default"):
		"""Can selection selection_name be redone?"""
		return (self.selection_history_indices[selection_name] + 1) < len(self.selection_histories[selection_name])

	def select(self, boolean_expression, mode="replace", selection_name="default", executor=None):
		"""Perform a selection, defined by the boolean expression, and combined with the previous selection using the given mode

		Selections are recorded in a history tree, per selection_name, undo/redo can be done for them seperately

		:param str boolean_expression: Any valid column expression, with comparison operators
		:param str mode: Possible boolean operator: replace/and/or/xor/subtract
		:param str selection_name: history tree or selection 'slot' to use
		:param executor:
		:return:
		"""
		if boolean_expression is None and not self.has_selection(selection_name=selection_name):
			pass # we don't want to pollute the history with many None selections
			self.signal_selection_changed.emit(self) # TODO: unittest want to know, does this make sense?
		else:
			def create(current):
				return SelectionExpression(self, boolean_expression, current, mode) if boolean_expression else None
			return self._selection(create, selection_name)

	def select_nothing(self, selection_name="default"):
		"""Select nothing"""
		self.select(None, selection_name=selection_name)

	def select_lasso(self, expression_x, expression_y, xsequence, ysequence, mode="replace", selection_name="default", executor=None):
		"""For performance reasons, a lasso selection is handled differently.

		:param str expression_x: Name/expression for the x coordinate
		:param str expression_y: Name/expression for the y coordinate
		:param xsequence: list of x numbers defining the lasso, together with y
		:param ysequence:
		:param str mode: Possible boolean operator: replace/and/or/xor/subtract
		:param str selection_name:
		:param executor:
		:return:
		"""


		def create(current):
			return SelectionLasso(self, expression_x, expression_y, xsequence, ysequence, current, mode)
		return self._selection(create, selection_name, executor=executor)

	def select_inverse(self, selection_name="default", executor=None):
		"""Invert the selection, i.e. what is selected will not be, and vice versa

		:param str selection_name:
		:param executor:
		:return:
		"""


		def create(current):
			return SelectionInvert(self, current)
		return self._selection(create, selection_name, executor=executor)

	def set_selection(self, selection, selection_name="default", executor=None):
		"""Sets the selection object

		:param selection: Selection object
		:param selection_name: selection 'slot'
		:param executor:
		:return:
		"""
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
		"""Returns True of there is a selection"""
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
	"""Base class for datasets that work with local file/data"""
	def __init__(self, name, path, column_names):
		super(DatasetLocal, self).__init__(name, column_names)
		self.path = path
		self.mask = None
		self.columns = collections.OrderedDict()

	@property
	def data(self):
		"""Gives direct access to the data as numpy-like arrays.

		Convenient when working with ipython in combination with small datasets, since this gives tab-completion

		Columns can be accesed by there names, which are attributes. The attribues are subclasses of numpy.ndarray
		and have the following extra properties:

		* ucd - The ucd for the column
		* description - Text description for column
		* unit - astropy unit object (astropy.units.Unit)

		:Example:
		>>> ds = vx.example()
		>>> r = np.sqrt(ds.data.x**2 + ds.data.y**2)

		"""
		class Data(object):
			pass
		class VaexColumn(np.ndarray):
			pass

		data = Data()
		for name, array in self.columns.items():
			#c = VaexColumn()
			column = array.view(VaexColumn)
			column.unit = self.unit(name)
			column.ucd = self.ucds.get(name)
			column.description = self.descriptions.get(name)
			setattr(data, name, column)
		return data

	def shallow_copy(self, virtual=True, variables=True):
		"""Creates a (shallow) copy of the dataset

		It will link to the same data, but will have its own state, e.g. virtual columns, variables, selection etc

		"""
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

	def is_local(self):
		"""The local implementation of :func:`Dataset.evaluate`, always returns True"""
		return True


	def length(self, selection=False):
		"""Get the length of the datasets, for the selection of the whole dataset.

		If selection is False, it returns len(dataset)

		TODO: Implement this in DatasetRemote, and move the method up in :func:`Dataset.length`

		:param selection: When True, will return the number of selected rows
		:return:
		"""
		if selection:
			return 0 if self.mask is None else np.sum(self.mask)
		else:
			return len(self)

	def __call__(self, *expressions, **kwargs):
		"""The local implementation of :func:`Dataset.__call__`"""
		return SubspaceLocal(self, expressions, kwargs.get("executor") or self.executor, async=kwargs.get("async", False))

	def concat(self, other):
		"""Concatenates two datasets, adding the rows of one the other dataset to the current, returned in a new dataset.

		No copy of the data is made.

		:param other: The other dataset that is concatenated with this dataset
		:return: New dataset with the rows concatenated
		:rtype: DatasetConcatenated
		"""
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
		"""The local implementation of :func:`Dataset.evaluate`"""
		i1 = i1 or 0
		i2 = i2 or len(self)
		scope = _BlockScope(self, i1, i2, **self.variables)
		if out is not None:
			scope.buffers[expression] = out
		return scope.evaluate(expression)

	def export_hdf5(self, path, column_names=None, byteorder="=", shuffle=False, selection=False, progress=None, virtual=True):
		"""Exports the dataset to a vaex hdf5 file

		:param DatasetLocal dataset: dataset to export
		:param str path: path for file
		:param lis[str] column_names: list of column names to export or None for all columns
		:param str byteorder: = for native, < for little endian and > for big endian
		:param bool shuffle: export rows in random order
		:param bool selection: export selection or not
		:param progress: progress callback that gets a progress fraction as argument and should return True to continue,
			or a default progress bar when progress=True
		:param: bool virtual: When True, export virtual columns
		:return:
		"""
		vaex.export.export_hdf5(self, path, column_names, byteorder, shuffle, selection, progress=progress)

	def export_fits(self, path, column_names=None, shuffle=False, selection=False, progress=None):
		"""Exports the dataset to a fits file that is compatible with TOPCAT colfits format

		:param DatasetLocal dataset: dataset to export
		:param str path: path for file
		:param lis[str] column_names: list of column names to export or None for all columns
		:param bool shuffle: export rows in random order
		:param bool selection: export selection or not
		:param progress: progress callback that gets a progress fraction as argument and should return True to continue,
			or a default progress bar when progress=True
		:param: bool virtual: When True, export virtual columns
		:return:
		"""
		vaex.export.export_fits(self, path, column_names, shuffle, selection, progress=progress)

	def _needs_copy(self, column_name):
		return not \
			(column_name in self.column_names  \
			and not isinstance(self.columns[column_name], vaex.dataset._ColumnConcatenatedLazy)\
			and not isinstance(self.columns[column_name], vaex.dataset.DatasetTap.TapColumn)\
			and self.columns[column_name].dtype.type==np.float64 \
			and self.columns[column_name].strides[0] == 8 \
			and column_name not in self.virtual_columns)
				#and False:

	def selected_length(self):
		"""The local implementation of :func:`Dataset.selected_length`"""
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
	"""Represents a set of datasets all concatenated. See :func:`DatasetLocal.concat` for usage.
	"""
	def __init__(self, datasets, name=None):
		super(DatasetConcatenated, self).__init__(None, None, [])
		self.datasets = datasets
		self.name = name or "-".join(ds.name for ds in self.datasets)
		self.path =  "-".join(ds.path for ds in self.datasets)
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
	"""Represent an in-memory dataset of numpy arrays, see :func:`from_arrays` for usage."""
	def __init__(self, name="arrays"):
		super(DatasetArrays, self).__init__(None, None, [])
		self.name = name
		self.path = "/has/no/path/"+name

	#def __len__(self):
	#	return len(self.columns.values()[0])

	def add_column(self, name, data):
		"""Add a column to the dataset

		:param str name: name of column
		:param data: numpy array with the data
		"""
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
	"""Represents a dataset where the data is memory mapped for efficient reading"""

		
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

	def close_files(self):
		for name, file in self.file_map.items():
			file.close()

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
		return os.path.splitext(path)[-1] == ".bin"
		basename, ext = os.path.splitext(path)
		#if os.path.exists(basename + ".omega2"):
		#	return True
		#return True

	@classmethod
	def get_options(cls, path):
		return []

	@classmethod
	def option_to_args(cls, option):
		return []
dataset_type_map["buist"] = HansMemoryMapped

def _python_save_name(name, used=[]):
	first, rest = name[0], name[1:]
	name = re.sub("[^a-zA-Z_]", "_", first) +  re.sub("[^a-zA-Z_0-9]", "_", rest)
	if name in used:
		nr = 1
		while name + ("_%d" % nr) in used:
			nr += 1
		name = name + ("_%d" % nr)
	return name

class FitsBinTable(DatasetMemoryMapped):
	def __init__(self, filename, write=False):
		super(FitsBinTable, self).__init__(filename, write=write)
		with fits.open(filename) as fitsfile:
			for table in fitsfile:
				if isinstance(table, fits.BinTableHDU):
					table_offset = table._data_offset
					#import pdb
					#pdb.set_trace()
					if table.columns[0].dim is not None: # for sure not a colfits
						dim = eval(table.columns[0].dim) # TODO: can we not do an eval here? not so safe
						if len(dim) == 2 and dim[0] <= dim[1]: # we have colfits format
							logger.debug("colfits file!")
							offset = table_offset
							for i in range(len(table.columns)):
								column = table.columns[i]
								cannot_handle = False
								column_name = _python_save_name(column.name, used=self.columns.keys())

								ucd_header_name = "TUCD%d" % (i+1)
								if ucd_header_name in table.header:
									self.ucds[column_name] = table.header[ucd_header_name]
								#unit_header_name = "TUCD%d" % (i+1)
								#if ucd_header_name in table.header:
								if column.unit:
									try:
										self.units[column_name] = astropy.units.Unit(column.unit)
									except:
										logger.debug("could not understand unit: %s" % column.unit)

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
										self.addColumn(column_name, offset=offset, dtype=dtype, length=length)
									else:
										for i in range(arraylength):
											name = column_name+"_" +str(i)
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
		self.update_meta()
		self.update_virtual_meta()
		self.load_favorite_selections()

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
	"""Implements the vaex hdf5 file format"""
	def __init__(self, filename, write=False):
		super(Hdf5MemoryMapped, self).__init__(filename, write=write)
		self.h5file = h5py.File(self.filename, "r+" if write else "r")
		self.h5table_root_name = None
		try:
			self._load()
		finally:
			self.h5file.close()

	def write_meta(self):
		"""ucds, descriptions and units are written as attributes in the hdf5 file, instead of a seperate file as
		 the default :func:`Dataset.write_meta`.
		 """
		with h5py.File(self.filename, "r+") as h5file_output:
			h5table_root = h5file_output[self.h5table_root_name]
			if self.description is not None:
				h5table_root.attrs["description"] = self.description
			for column_name in self.columns.keys():
				h5dataset = h5table_root[column_name]
				for name, values in [("ucd", self.ucds), ("unit", self.units), ("description", self.descriptions)]:
					if column_name in values:
						value = str(values[column_name])
						h5dataset.attrs[name] = value
	@classmethod
	def can_open(cls, path, *args, **kwargs):
		h5file = None
		try:
			with open(path, "rb") as f:
				signature = f.read(4)
				hdf5file = signature == b"\x89\x48\x44\x46"
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

	def _load(self):
		if "data" in self.h5file:
			self._load_columns(self.h5file["/data"])
			self.h5table_root_name = "/data"
		# TODO: shall we rename it vaex... ?
		# if "vaex" in self.h5file:
		#	self.load_columns(self.h5file["/vaex"])
		#	h5table_root = "/vaex"
		if "columns" in self.h5file:
			self._load_columns(self.h5file["/columns"])
			self.h5table_root_name = "/columns"
		if "properties" in self.h5file:
			self._load_variables(self.h5file["/properties"]) # old name, kept for portability
		if "variables" in self.h5file:
			self._load_variables(self.h5file["/variables"])
		if "axes" in self.h5file:
			self._load_axes(self.h5file["/axes"])
		self.update_meta()
		self.update_virtual_meta()
		self.load_favorite_selections()

	#def 
	def _load_axes(self, axes_data):
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
			
	def _load_variables(self, h5variables):
		for key, value in list(h5variables.attrs.items()):
			self.variables[key] = value
			
			
	def _load_columns(self, h5data):
		#print h5data
		# make sure x y x etc are first
		first = "x y z vx vy vz".split()
		finished = set()
		if "description" in h5data.attrs:
			self.description = ensure_string(h5data.attrs["description"])
		for column_name in first + list(h5data):
			if column_name in h5data and column_name not in finished:
				#print type(column_name)
				column = h5data[column_name]
				if "ucd" in column.attrs:
					self.ucds[column_name] = ensure_string(column.attrs["ucd"])
				if "description" in column.attrs:
					self.descriptions[column_name] = ensure_string(column.attrs["description"])
				if "unit" in column.attrs:
					try:
						unitname = ensure_string(column.attrs["unit"])
						if unitname and unitname != "None":
							self.units[column_name] = astropy.units.Unit(unitname)
					except:
						logger.exception("error parsing unit: %s", column.attrs["unit"])
				if "units" in column.attrs: # Amuse case
					unitname = ensure_string(column.attrs["units"])
					logger.debug("amuse unit: %s", unitname)
					if unitname == "(0.01 * system.get('S.I.').base('length'))":
						self.units[column_name] = astropy.units.Unit("cm")
					if unitname == "((0.01 * system.get('S.I.').base('length')) * (system.get('S.I.').base('time')**-1))":
						self.units[column_name] = astropy.units.Unit("cm/s")
					if unitname == "(0.001 * system.get('S.I.').base('mass'))":
						self.units[column_name] = astropy.units.Unit("gram")


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
	"""Implements reading Amuse hdf5 files `amusecode.org <http://amusecode.org/>`_"""
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

	def _load(self):
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
		self.update_meta()
		self.update_virtual_meta()
		self.load_favorite_selections()

dataset_type_map["amuse"] = AmuseHdf5MemoryMapped


gadget_particle_names = "gas halo disk bulge stars dm".split()

class Hdf5MemoryMappedGadget(DatasetMemoryMapped):
	"""Implements reading `Gadget2 <http://wwwmpa.mpa-garching.mpg.de/gadget/>`_ hdf5 files """
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
					value = h5file["Header"].attrs[name].decode("utf-8")
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


class SoneiraPeebles(DatasetArrays):
	def __init__(self, dimension, eta, max_level, L):
		super(SoneiraPeebles, self).__init__(name="soneira-peebles")
		#InMemory.__init__(self)
		def todim(value):
			if isinstance(value, (tuple, list)):
				assert len(value) >= dimension, "either a scalar or sequence of length equal to or larger than the dimension"
				return value[:dimension]
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
		for d, name in zip(list(range(dimension)), "x y z w v u".split()):
			self.add_column(name, array[d])
		if 0:
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
	def __init__(self, filename=None, format=None, table=None, **kwargs):
		if table is None:
			self.filename = filename
			self.format = format
			DatasetArrays.__init__(self, filename)
			self.read_table()
		else:
			#print vars(table)
			#print dir(table)
			DatasetArrays.__init__(self, table.meta["name"])
			self.table = table
			#self.name

		#data = table.array.data
		for i in range(len(self.table.dtype)):
			name = self.table.dtype.names[i]
			column = self.table[name]
			type = self.table.dtype[i]
			#clean_name = re.sub("[^a-zA-Z_]", "_", name)
			clean_name = _python_save_name(name, self.columns.keys())
			if type.kind in ["f", "i"]: # only store float and int
				#datagroup.create_dataset(name, data=table.array[name].astype(np.float64))
				#dataset.addMemoryColumn(name, table.array[name].astype(np.float64))
				masked_array = self.table[name].data
				if "ucd" in column._meta:
					self.ucds[clean_name] = column._meta["ucd"]
				if column.description:
					self.descriptions[clean_name] = column.description
				if hasattr(masked_array, "mask"):
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

	def read_table(self):
		self.table = astropy.table.Table.read(self.filename, format=self.format, **kwargs)

import astropy.io.votable
import string
class VOTable(DatasetArrays):
	def __init__(self, filename):
		DatasetArrays.__init__(self, filename)
		self.filename = filename
		self.path = filename
		votable = astropy.io.votable.parse(self.filename)

		self.first_table = votable.get_first_table()
		self.description = self.first_table.description

		for field in self.first_table.fields:
			name = field.name
			data = self.first_table.array[name].data
			type = self.first_table.array[name].dtype
			clean_name = re.sub("[^a-zA-Z_0-9]", "_", name)
			if clean_name in string.digits:
				clean_name = "_" + clean_name
			self.ucds[clean_name] = field.ucd
			self.units[clean_name] = field.unit
			self.descriptions[clean_name] = field.description
			if type.kind in ["f", "i"]: # only store float and int
				masked_array = self.first_table.array[name]
				if type.kind in ["f"]:
					masked_array.data[masked_array.mask] = np.nan
				if type.kind in ["i"]:
					masked_array.data[masked_array.mask] = 0
				self.add_column(clean_name, self.first_table.array[name].data)
			#if type.kind in ["S"]:
			#	self.add_column(clean_name, self.first_table.array[name].data)

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

class DatasetTap(DatasetArrays):
	class TapColumn(object):
		def __init__(self, tap_dataset, column_name, column_type, ucd):
			self.tap_dataset = tap_dataset
			self.column_name = column_name
			self.column_type = column_type
			self.ucd = ucd
			self.alpha_min = 0
			length = len(tap_dataset)
			steps = length/1e6 # try to do it in chunks
			self.alpha_step = 360/steps
			self.alpha_max = self.alpha_min + self.alpha_step
			logger.debug("stepping in alpha %f" % self.alpha_step)
			self.data = []
			self.offset = 0
			self.shape = (length,)
			self.dtype = DatasetTap.type_map[self.column_type]().dtype
			self.left_over_chunk = None
			self.rows_left = length
			import tempfile
			self.download_file = tempfile.mktemp(".vot")

		def __getitem__(self, slice):
			start, stop, step = slice.start, slice.stop, slice.step
			required_length = stop - start
			assert start >= self.offset
			chunk_data = self.left_over_chunk
			enough = False if chunk_data is None else len(chunk_data) >= required_length
			if chunk_data is not None:
				logger.debug("start %s offset %s chunk length %s", start, self.offset, len(chunk_data))
				#assert len(chunk_data) == start - self.offset
			if enough:
				logger.debug("we can skip the query, already have results from previous query")
			while not enough:
				adql_query = "SELECT {column_name} FROM {table_name} WHERE alpha >= {alpha_min} AND alpha < {alpha_max} ORDER BY alpha ASC"\
					.format(column_name=self.column_name, table_name=self.tap_dataset.table_name, alpha_min=self.alpha_min, alpha_max=self.alpha_max)
				logger.debug("executing: %s" % adql_query)
				logger.debug("executing: %s" % adql_query.replace(" ", "+"))


				url = self.tap_dataset.tap_url + "/sync?REQUEST=doQuery&LANG=ADQL&MAXREC=10000000&FORMAT=votable&QUERY=" +adql_query.replace(" ", "+")
				import urllib2
				response = urllib2.urlopen(url)
				with open(self.download_file, "w") as f:
					f.write(response.read())
				votable = astropy.io.votable.parse(self.download_file)
				data = votable.get_first_table().array[self.column_name].data
				# TODO: respect masked array
				#table = astropy.table.Table.read(url, format="votable") #, show_progress=False)
				#data = table[self.column_name].data.data.data
				logger.debug("new chunk is of lenght %d", len(data))
				self.rows_left -= len(data)
				logger.debug("rows left %d", self.rows_left)
				if chunk_data is None:
					chunk_data = data
				else:
					chunk_data = np.concatenate([chunk_data, data])
				if len(chunk_data) >= required_length:
					enough = True
				logger.debug("total chunk is of lenght %d, enough: %s", len(chunk_data), enough)
				self.alpha_min += self.alpha_step
				self.alpha_max += self.alpha_step


			result, self.left_over_chunk = chunk_data[:required_length], chunk_data[required_length:]
			#print(result)
			logger.debug("left over is of length %d", len(self.left_over_chunk))
			return result #np.zeros(N, dtype=self.dtype)



	type_map = {
		'REAL':np.float32,
	    'SMALLINT':np.int32,
		'DOUBLE':np.float64,
		'BIGINT':np.int64,
		'INTEGER':np.int32,
		'BOOLEAN':np.bool8
	}
	#not supported types yet 'VARCHAR',', u'BOOLEAN', u'INTEGER', u'CHAR
	def __init__(self, tap_url="http://gaia.esac.esa.int/tap-server/tap/g10_smc", table_name=None):
		logger.debug("tap url: %r", tap_url)
		self.tap_url = tap_url
		self.table_name = table_name
		if table_name is None: # let us try to infer the table name
			if tap_url.endswith("tap") or tap_url.endswith("tap/"):
				pass # this mean we really didn't provide one
			else:
				index = tap_url.rfind("tap/")
				if index != -1:
					self.tap_url, self.table_name = tap_url[:index+4], self.tap_url[index+4:]
					logger.debug("inferred url is %s, and table name is %s", self.tap_url, self.table_name)

		if self.tap_url.startswith("tap+"): # remove tap+ part from tap+http(s), only keep http(s) part
			self.tap_url = self.tap_url[len("tap+"):]
		import requests
		super(DatasetTap, self).__init__(self.table_name)
		self.req = requests.request("get", self.tap_url+"/tables/")
		self.path = "tap+" +self.tap_url + "/" + table_name

		#print dir(self.req)
		from bs4 import BeautifulSoup
			#self.soup = BeautifulSoup(req.response)
		tables = BeautifulSoup(self.req.content, 'xml')
		self.tap_tables = collections.OrderedDict()
		for table in tables.find_all("table"):
			#print table.find("name").string, table.description.string, table["gaiatap:size"]
			table_name = unicode(table.find("name").string)
			table_size = int(table["esatapplus:size"])
			#print table_name, table_size
			logger.debug("tap table %r ", table_name)
			columns = []
			for column in table.find_all("column"):
				column_name = unicode(column.find("name").string)
				column_type = unicode(column.dataType.string)
				ucd = column.ucd.string if column.ucd else None
				unit = column.unit.string if column.unit else None
				description = column.description.string if column.description else None
				#print "\t", column_name, column_type, ucd
				#types.add()
				columns.append((column_name, column_type, ucd, unit, description))
			self.tap_tables[table_name] = (table_size, columns)
		if not self.tap_tables:
			raise ValueError("no tables or wrong url")
		for name, (table_size, columns) in self.tap_tables.items():
			logger.debug("table %s has length %d", name, table_size)
		self._full_length, self._tap_columns = self.tap_tables[self.table_name]
		self._length = self._full_length
		logger.debug("selected table table %s has length %d", self.table_name, self._full_length)
		#self.column_names = []
		#self.columns = collections.OrderedDict()
		for column_name, column_type, ucd, unit, description in self._tap_columns:
			logger.debug("  column %s has type %s and ucd %s, unit %s and description %s", column_name, column_type, ucd, unit, description)
			if column_type in self.type_map.keys():
				self.column_names.append(column_name)
				if ucd:
					self.ucds[column_name] = ucd
				if unit:
					self.units[column_name] = unit
				if description:
					self.descriptions[column_name] = description
				self.columns[column_name] = self.TapColumn(self, column_name, column_type, ucd)
			else:
				logger.warning("  type of column %s is not supported, it will be skipped", column_name)


	@classmethod
	def can_open(cls, path, *args, **kwargs):
		can_open = False
		url = None
		try:
			url = urlparse(path)
		except:
			return False
		if url.scheme:
			if url.scheme.startswith("tap+http"): # will also catch https
				can_open = True
		logger.debug("%r can open: %r"  %(cls.__name__, can_open))
		return can_open


dataset_type_map["tap"] = DatasetTap

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

from vaex.remote import ServerRest, SubspaceRemote, DatasetRemote
from vaex.events import Signal

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(prog='python -m vaex.dataset')
	subparsers = parser.add_subparsers(help='sub-command help', dest="task")
	parser_soneira = subparsers.add_parser('soneira', help='create soneira peebles dataset')
	parser_soneira.add_argument('output', help='output file')
	parser.add_argument("columns", help="list of columns to export", nargs="*")
	parser_soneira.add_argument('--dimension','-d', type=int, help='dimensions', default=4)
	#parser_soneira.add_argument('--eta','-e', type=int, help='dimensions', default=3)
	parser_soneira.add_argument('--max-level','-m', type=int, help='dimensions', default=28)
	parser_soneira.add_argument('--lambdas','-l', type=int, help='lambda values for fractal', default=[1.1, 1.3, 1.6, 2.])
	args = parser.parse_args()
	print(args.task)
	if args.task == "soneira":
		if vaex.utils.check_memory_usage(4*8*2**args.max_level, vaex.utils.confirm_on_console):
			dataset = SoneiraPeebles(args.dimension, 2, args.max_level, args.lambdas)

	if args.columns:
		columns = args.columns
	else:
		columns = None
	if columns is None:
		columns = dataset.get_column_names()
	for column in columns:
		if column not in dataset.get_column_names():
			print("column %r does not exist, run with --list or -l to list all columns")
			sys.exit(1)

	base, output_ext = os.path.splitext(args.output)
	with vaex.utils.progressbar("exporting") as progressbar:
		def update(p):
			progressbar.update(p)
			return True
		if output_ext == ".hdf5":
			dataset.export_hdf5(args.output, column_names=columns, progress=update)
		else:
			print("extension %s not supported, only .fits and .hdf5 are" % output_ext)

	#vaex.set_log_level_debug()
	#ds = DatasetTap()
	#ds.columns["alpha"][0:100]


def alias_main(argv):
	import argparse
	parser = argparse.ArgumentParser(argv[0])
	#parser.add_argument('--verbose', '-v', action='count', default=0)
	#parser.add_argument('--quiet', '-q', default=False, action='store_true', help="do not output anything")
	#parser.add_argument('--list', '-l', default=False, action='store_true', help="list columns of input")
	#parser.add_argument('--progress', help="show progress (default: %(default)s)", default=True, action='store_true')
	#parser.add_argument('--no-progress', dest="progress", action='store_false')
	#parser.add_argument('--shuffle', "-s", dest="shuffle", action='store_true', default=False)

	subparsers = parser.add_subparsers(help='type of task', dest="task")

	parser_list = subparsers.add_parser('list', help='list aliases')

	parser_add = subparsers.add_parser('add', help='add alias')
	parser_add.add_argument('name', help='name of alias')
	parser_add.add_argument('path', help='path/filename for alias')
	parser.add_argument('-f', '--force', help="force/overwrite existing alias", default=False, action='store_true')

	parser_remove = subparsers.add_parser('remove', help='remove alias')
	parser_remove.add_argument('name', help='name of alias')

	args = parser.parse_args(argv[1:])
	import vaex
	if args.task == "add":
		vaex.aliases[args.name] = args.path
	if args.task == "remove":
		del vaex.aliases[args.name]
	if args.task == "list":
		for name in sorted(vaex.aliases.keys()):
			print("%s: %s" % (name, vaex.aliases[name]))


def make_stat_parser(name):
	import argparse
	parser = argparse.ArgumentParser(name)
	#parser.add_argument('--verbose', '-v', action='count', default=0)
	#parser.add_argument('--quiet', '-q', default=False, action='store_true', help="do not output anything")
	#parser.add_argument('--list', '-l', default=False, action='store_true', help="list columns of input")
	#parser.add_argument('--progress', help="show progress (default: %(default)s)", default=True, action='store_true')
	#parser.add_argument('--no-progress', dest="progress", action='store_false')
	#parser.add_argument('--shuffle', "-s", dest="shuffle", action='store_true', default=False)

	#subparsers = parser.add_subparsers(help='type of task', dest="task")

	#parser_list = subparsers.add_parser('list', help='list aliases')

	#parser = subparsers.add_parser('add', help='add alias')
	parser.add_argument('dataset', help='path or name of dataset')
	parser.add_argument('--fraction', "-f", dest="fraction", type=float, default=1.0, help="fraction of input dataset to export")
	return parser

def stat_main(argv):
	parser = make_stat_parser(argv[0])
	args = parser.parse_args(argv[1:])
	import vaex
	dataset = vaex.open(args.dataset)
	if dataset is None:
		print("Cannot open input: %s" % args.dataset)
		sys.exit(1)
	print("dataset:")
	print("  length: %s" % len(dataset))
	print("  full_length: %s" % dataset.full_length())
	print("  name: %s" % dataset.name)
	print("  path: %s" % dataset.path)
	print("  columns: ")
	desc = dataset.description
	if desc:
		print("    description: %s" % desc)
	for name in dataset.get_column_names():
		print("   - %s: " % name)
		desc = dataset.descriptions.get(name)
		if desc:
			print("  \tdescription: %s" % desc)
		unit = dataset.unit(name)
		if unit:
			print("   \tunit: %s" % unit)
		dtype = dataset.dtype(name)
		print("   \ttype: %s" % dtype.name)

