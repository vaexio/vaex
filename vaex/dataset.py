# -*- coding: utf-8 -*-
import os
import math
import time
import itertools
import functools
import collections
import sys
import platform
import warnings
import os
import re
from functools import reduce
import threading
import six
import vaex.utils
import vaex.image
import numpy as np
import concurrent.futures
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
import vaex.kld
from .delayed import delayed, delayed_args, delayed_list

# py2/p3 compatibility
try:
	from urllib.parse import urlparse
except ImportError:
	from urlparse import urlparse

sys_is_le = sys.byteorder == 'little'

logger = logging.getLogger("vaex")
lock = threading.Lock()
default_shape = 128
#executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
#executor = vaex.execution.default_executor

def _parse_f(f):
	if f is None:
		return lambda x: x
	elif isinstance(f, six.string_types):
		if f == "identity":
			return lambda x: x
		else:
			if hasattr(np, f):
				return getattr(np, f)
			else:
				raise ValueError("do not understand f = %s, should be a function, string 'identity' or a function from numpy such as 'log', 'log1p'" % f)
	else:
		return f

def _normalize(a, axis=None):
	a = np.copy(a) # we're gonna modify inplace, better copy iy
	mask = np.isfinite(a)
	a[~mask] = np.nan # put inf to nan
	allaxis = list(range(len(a.shape)))
	if axis is not None:
		if type(axis) == int:
			axis = [axis]
		for ax in axis:
			allaxis.remove(ax)
		axis=tuple(allaxis)
	vmin = np.nanmin(a)
	vmax = np.nanmax(a)
	a = a - np.nanmin(a, axis=axis, keepdims=True)
	a /= np.nanmax(a, axis=axis, keepdims=True)
	return a, vmin, vmax

def _parse_n(n):
	if isinstance(n, six.string_types):
		if n == "normalize":
			return _normalize
			#return lambda x: x
		else:
			raise ValueError("do not understand n = %s, should be a function, or string 'normalize'" % n)
	else:
		return n


def _parse_reduction(name, colormap, colors):
	if name.startswith("stack.fade"):
		def _reduce_stack_fade(grid):
			return grid[...,-1] # return last..
		return _reduce_stack_fade
	elif name.startswith("colormap"):
		import matplotlib
		cmap = matplotlib.cm.get_cmap(colormap)
		def f(grid):
			return cmap(grid)
		return f
	elif name.startswith("stack.color"):
		def f(grid, colors=colors, colormap=colormap):
			import matplotlib
			colormap = matplotlib.cm.get_cmap(colormap)
			if isinstance(colors, six.string_types):
				colors = matplotlib.cm.get_cmap(colors)
			if isinstance(colors, matplotlib.colors.Colormap):
				group_count = grid.shape[-1]
				colors = [colors(k/float(group_count-1.)) for k in range(group_count) ]
			else:
				colors = [matplotlib.colors.colorConverter.to_rgba(k) for k in colors]
			#print grid.shape
			total = np.nansum(grid, axis=0)/grid.shape[0]
			#grid /= total
			#mask = total > 0
			#alpha = total - total[mask].min()
			#alpha[~mask] = 0
			#alpha = total / alpha.max()
			#print np.nanmax(total), np.nanmax(grid)
			return colormap(total)
			rgba = grid.dot(colors)
			#def _norm(data):
			#	mask = np.isfinite(data)
			#	data = data - data[mask].min()
			#	data /= data[mask].max()
			#	return data
			#rgba[...,3] = (f(alpha))
			#rgba[...,3] = 1
			rgba[total == 0,3] = 0.
			#mask = alpha > 0
			#if 1:
			#	for i in range(3):
			#		rgba[...,i] /= total
			#		#rgba[...,i] /= rgba[...,0:3].max()
			#		rgba[~mask,i] = background_color[i]
			#rgba = (np.swapaxes(rgba, 0, 1))
			return rgba
		return f

	else:
		raise ValueError("do not understand reduction = %s, should be a ..." % name)

import numbers
def _is_string(x):
	return isinstance(x, six.string_types)
def _issequence(x):
	return isinstance(x, (tuple, list, np.ndarray))
def _isnumber(x):
	return isinstance(x, numbers.Number)
def _is_limit(x):
	return isinstance(x, (tuple, list, np.ndarray)) and all([_isnumber(k) for k in x])
def _ensure_list(x):
	return [x] if not _issequence(x) else x

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
		shape1 = ( self.size,) * self.dimension
		try:
			self.size[0]
			shape1 = tuple(self.size)
		except:
			pass
		shape = (self.subspace.executor.thread_pool.nthreads,) + shape1
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
		#mask = self.dataset.mask
		data = self.data[thread_index]
		if self.masked:
			mask = self.dataset.evaluate_selection_mask("default", i1=i1, i2=i2)
			blocks = [block[mask] for block in blocks]

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
			#vaex.vaexfast.statisticNd([blocks[0], blocks[1]], subblock_weight, data, self.minima, self.maxima, 0)
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

class StatOp(object):
	def __init__(self, code, fields, reduce_function=np.nansum):
		self.code = code
		self.fields = fields
		self.reduce_function = reduce_function

	def init(self, grid):
		pass

	def reduce(self, grid, axis=0):
		return self.reduce_function(grid, axis=axis)

class StatOpMinMax(StatOp):
	def __init__(self, code, fields):
		super(StatOpMinMax, self).__init__(code, fields)

	def init(self, grid):
		grid[...,0] = np.inf
		grid[...,1] = -np.inf

	def reduce(self, grid, axis=0):
		out = np.zeros(grid.shape[1:], dtype=grid.dtype)
		out[...,0] = np.nanmin(grid[...,0], axis=axis)
		out[...,1] = np.nanmax(grid[...,1], axis=axis)
		return out

OP_ADD1 = StatOp(0, 1)
OP_COUNT = StatOp(1, 1)
OP_MIN_MAX = StatOpMinMax(2, 2)
OP_ADD_WEIGHT_MOMENTS_01 = StatOp(3, 2, np.nansum)
OP_ADD_WEIGHT_MOMENTS_012 = StatOp(4, 3, np.nansum)

def _expand(x, dimension, type=tuple):
	if _issequence(x):
		assert len(x) == dimension, "wants to expand %r to dimension %d" % (x, dimension)
		return type(x)
	else:
		return type((x,) * dimension)

def _expand_shape(shape, dimension):
	if isinstance(shape, (tuple, list)):
		assert len(shape) == dimension, "wants to expand shape %r to dimension %d" % (shape, dimension)
		return tuple(shape)
	else:
		return (shape,) * dimension

def _expand_limits(limits, dimension):
	if isinstance(limits, (tuple, list, np.ndarray)) and \
			(isinstance(limits[0], (tuple, list, np.ndarray)) or isinstance(limits[0], six.string_types)):
		assert len(limits) == dimension, "wants to expand shape %r to dimension %d" % (limits, dimension)
		return tuple(limits)
	else:
		return [limits,] * dimension

class TaskStatistic(Task):
	def __init__(self, dataset, expressions, shape, limits, masked=False, weight=None, op=OP_ADD1, selection=None):
		if not isinstance(expressions, (tuple, list)):
			expressions = [expressions]
		self.shape = _expand_shape(shape, len(expressions))
		self.limits = limits
		self.weight = weight
		self.selection_waslist, [self.selections,] = vaex.utils.listify(selection)
		Task.__init__(self, dataset, expressions, name="statisticNd")
		self.dtype = np.float64
		self.masked = masked
		self.op = op

		self.shape_total = (self.dataset.executor.thread_pool.nthreads,) + (len(self.selections), ) + self.shape + (op.fields,)
		self.grid = np.zeros(self.shape_total, dtype=self.dtype)
		self.op.init(self.grid)
		self.minima = []
		self.maxima = []
		limits = np.array(self.limits)
		if len(limits) != 0:
			logger.debug("limits = %r", limits)
			assert limits.shape[-1] == 2, "expected last dimension of limits to have a length of 2 (not %d, total shape: %s), of the form [[xmin, xmin], ... [zmin, zmax]], not %s" % (limits.shape[-1], limits.shape, limits)
			if len(limits.shape) == 1: # short notation: [xmin, max], instead of [[xmin, xmax]]
				limits = [limits]
			logger.debug("limits = %r", limits)
			for limit in limits:
				vmin, vmax = limit
				self.minima.append(float(vmin))
				self.maxima.append(float(vmax))
		if self.weight is not None:
			self.expressions_all.append(weight)


	def __repr__(self):
		name = self.__class__.__module__ + "." +self.__class__.__name__
		return "<%s(dataset=%r, expressions=%r, shape=%r, limits=%r, weight=%r, selections=%r)> instance at 0x%x" % (name, self.dataset, self.expressions, self.shape, self.limits, self.weight, self.selections, id(self))

	def map(self, thread_index, i1, i2, *blocks):
		class Info(object):
			pass
		info = Info()
		info.i1 = i1
		info.i2 = i2
		info.first = i1 == 0
		info.last = i2 == len(self.dataset)
		info.size = i2-i1

		this_thread_grid = self.grid[thread_index]
		for i, selection in enumerate(self.selections):
			if selection:
				mask = self.dataset.evaluate_selection_mask(selection, i1=i1, i2=i2)
				if mask is None:
					raise ValueError("performing operation on selection while no selection present")
				selection_blocks = [block[mask] for block in blocks]
			else:
				selection_blocks = [block for block in blocks]
			little_endians = len([k for k in selection_blocks if k.dtype.byteorder in ["<", "="]])
			if not ((len(selection_blocks) == little_endians) or little_endians == 0):
				def _to_native(ar):
					if ar.dtype.byteorder not in ["<", "="]:
						dtype = ar.dtype.newbyteorder()
						return ar.astype(dtype)
					else:
						return ar

				selection_blocks = [_to_native(k) for k in selection_blocks]
			subblock_weight = None
			if len(selection_blocks) == len(self.expressions) + 1:
				subblock_weight = selection_blocks[-1]
				selection_blocks = list(selection_blocks[:-1])
			if len(selection_blocks) == 0 and subblock_weight is None:
				if self.op == OP_ADD1: # special case for counting '*' (i.e. the number of rows)
					if selection:
						this_thread_grid[i][0] += np.sum(mask)
					else:
						this_thread_grid[i][0] += i2-i1
				else:
					raise ValueError("Nothing to compute for OP %s" % self.op.code)

			blocks = list(blocks) # histogramNd wants blocks to be a list
			vaex.vaexfast.statisticNd(selection_blocks, subblock_weight, this_thread_grid[i], self.minima, self.maxima, self.op.code)
		return i2-i1
		#return map(self._map, blocks)#[self.map(block) for block in blocks]

	def reduce(self, results):
		#for i in range(1, self.subspace.executor.thread_pool.nthreads):
		#	self.data[0] += self.data[i]
		#return self.data[0]
		#return self.data
		grid = self.op.reduce(self.grid)
		# If selection was a string, we just return the single selection
		return grid if self.selection_waslist else grid[0]



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

	def is_selected(self):
		return self.is_masked

	def selected(self):
		return self.__class__(self.dataset, expressions=self.expressions, executor=self.executor, async=self.async, masked=True)

	def asynchronous(self):
		return self.__class__(self.dataset, expressions=self.expressions, executor=self.executor, async=True, masked=self.is_masked)

	def image_rgba_save(self, filename, data=None, rgba8=None, **kwargs):
		if rgba8 is not None:
			data = self.image_rgba_data(rgba8=rgba8, **kwargs)
		if data is None:
			data = self.image_rgba_data(**kwargs)
		with open(filename, "wb") as f:
			f.write(data)

	def image_rgba_notebook(self, data=None, rgba8=None, **kwargs):
		if rgba8 is not None:
			data = self.image_rgba_data(rgba8=rgba8, **kwargs)
		if data is None:
			data = self.image_rgba_data(**kwargs)
		from IPython.display import display, Image
		return Image(data=data)
		
	def image_rgba_data(self, rgba8=None, format="png", pil_draw=False, **kwargs):
		import PIL.Image
		import PIL.ImageDraw
		import StringIO
		if rgba8 is None:
			rgba8 = self.image_rgba(**kwargs)
		img = PIL.Image.frombuffer("RGBA", rgba8.shape[:2], rgba8, 'raw') #, "RGBA", 0, -1)
		if pil_draw:
			draw = PIL.ImageDraw.Draw(img)
			pil_draw(draw)

		f = StringIO.StringIO()
		img.save(f, format)
		return f.getvalue()

	def image_rgba_url(self, rgba8=None, **kwargs):
		if rgba8 is None:
			rgba8 = self.image_rgba(**kwargs)
		import PIL.Image
		img = PIL.Image.frombuffer("RGBA", rgba8.shape[:2], rgba8, 'raw') #, "RGBA", 0, -1)
		import StringIO
		f = StringIO.StringIO()
		img.save(f, "png")
		from base64 import b64encode
		imgurl = "data:image/png;base64," + b64encode(f.getvalue()) + ""
		return imgurl

	def normalize_grid(self, grid):
		grid = grid * 1 # copy
		mask = (grid > 0) & np.isfinite(grid)
		if grid.sum():
			grid -= grid[mask].min()
			grid /= grid[mask].max()
		else:
			grid[:] = 0
		return grid

	def limits(self, value, square=False):
		"""TODO: doc + server side implementation"""
		if isinstance(value, six.string_types):
			import re
			match = re.match("(\d*)(\D*)", value)
			if match is None:
				raise ValueError("do not understand limit specifier %r, examples are 90%, 3sigma")
			else:
				value, type = match.groups()
				import ast
				value = ast.literal_eval(value)
				type = type.strip()
				if type in ["s", "sigma"]:
					return self.limits_sigma(value)
				elif type in ["ss", "sigmasquare"]:
					return self.limits_sigma(value, square=True)
				elif type in ["%", "percent"]:
					return self.limits_percentage(value)
				elif type in ["%s", "%square", "percentsquare"]:
					return self.limits_percentage(value, square=True)
		if value is None:
			return self.limits_percentage(square=square)
		else:
			return value

	def image_rgba(self, grid=None, size=256, limits=None, square=False, center=None, weight=None, weight_stat="mean", figsize=None,
			 aspect="auto", f=lambda x: x, axes=None, xlabel=None, ylabel=None,
			 group_by=None, group_limits=None, group_colors='jet', group_labels=None, group_count=10, cmap="afmhot",
		     vmin=None, vmax=None,
			 pre_blend=False, background_color="white", background_alpha=1., normalize=True, color=None):
		f = _parse_f(f)
		if grid is None:
			limits = self.limits(limits)
			if limits is None:
				limits = self.limits_sigma()
			if group_limits is None and group_by:
				group_limits = tuple(self.dataset(group_by).minmax()[0]) + (group_count,)
			if weight_stat == "mean" and weight is not None:
				grid = self.bin_mean(weight, limits=limits, size=size, group_limits=group_limits, group_by=group_by)
			else:
				grid = self.histogram(limits=limits, size=size, weight=weight, group_limits=group_limits, group_by=group_by)
			if grid is None: # cancel occured
				return
		import matplotlib.cm
		background_color = np.array(matplotlib.colors.colorConverter.to_rgb(background_color))
		if group_by:
			gmin, gmax, group_count = group_limits
			if isinstance(group_colors, six.string_types):
				group_colors = matplotlib.cm.get_cmap(group_colors)
			if isinstance(group_colors, matplotlib.colors.Colormap):
				group_count = group_limits[2]
				colors = [group_colors(k/float(group_count-1.)) for k in range(group_count) ]
			else:
				colors = [matplotlib.colors.colorConverter.to_rgba(k) for k in group_colors]
			total = np.sum(grid, axis=0).T
			#grid /= total
			mask = total > 0
			alpha = total - total[mask].min()
			alpha[~mask] = 0
			alpha = total / alpha.max()
			rgba = grid.T.dot(colors)
			def _norm(data):
				mask = np.isfinite(data)
				data = data - data[mask].min()
				data /= data[mask].max()
				return data
			rgba[...,3] = (f(alpha))
			#rgba[...,3] = 1
			rgba[total == 0,3] = 0.
			mask = alpha > 0
			if 1:
				for i in range(3):
					rgba[...,i] /= total
					#rgba[...,i] /= rgba[...,0:3].max()
					rgba[~mask,i] = background_color[i]
			rgba = (np.swapaxes(rgba, 0, 1))
		else:
			if color:
				color = np.array(matplotlib.colors.colorConverter.to_rgba(color))
				rgba = np.zeros(grid.shape + (4,))
				rgba[...,0:4] = color
				data = f(grid)
				mask = (grid > 0) & np.isfinite(data)
				if vmin is None:
					vmin = data[mask].min()
				if vmax is None:
					vmax = data[mask].max()
				if mask.sum():
					data -= vmin
					data /= vmax
					data[~mask] = 0
				else:
					data[:] = 0
				rgba[...,3] = data
			else:
				cmap = matplotlib.cm.get_cmap(cmap)
				data = f(grid)
				if normalize:
					mask = (data > 0) & np.isfinite(data)
					if vmin is None:
						vmin = data[mask].min()
					if vmax is None:
						vmax = data[mask].max()
					if mask.sum():
						data -= vmin
						data /= vmax
					else:
						data[:] = 0
					data[~mask] = 0
				data = np.clip(data, 0, 1)
				rgba = cmap(data)
				if normalize:
					rgba[~mask,3] = 0
				rgba[...,3] = 1#data
			#rgba8 = np.swapaxes(rgba8, 0, 1)
		#white = np.ones_like(rgba[...,0:3])
		if pre_blend:
			#rgba[...,3] = background_alpha
			rgb = rgba[...,:3].T
			alpha = rgba[...,3].T
			rgb[:] = rgb * alpha + background_color[:3].reshape(3,1,1) * (1-alpha)
			alpha[:] = alpha + background_alpha * (1-alpha)
		rgba= np.clip(rgba, 0, 1)
		rgba8 = (rgba*255).astype(np.uint8)
		return rgba8

	def plot_vectors(self, expression_x, expression_y, limits, wx=None, wy=None, counts=None, size=32, axes=None, **kwargs):
		import pylab
		# refactor: should go to bin_means_xy
		if counts is None:
			counts = self.histogram(size=size, limits=limits)
		if wx is None:
			wx = self.histogram(size=size, weight=expression_x, limits=limits)
		if wy is None:
			wy = self.histogram(size=size, weight=expression_y, limits=limits)
		N = size
		positions = [vaex.utils.linspace_centers(limits[i][0], limits[i][1], N) for i in range(self.dimension)]
		#print(positions)
		mask = counts > 0
		vx = wx/counts
		vy = wy/counts
		vx[counts==0] = 0
		vy[counts==0] = 0
		#vx = self.vx.grid / self.vcounts.grid
		#vy = self.vy.grid / self.vcounts.grid
		x2d, y2d = np.meshgrid(positions[0], positions[1])
		if axes is None:
			axes = pylab.gca()
		axes.quiver(x2d[mask], y2d[mask], vx[mask], vy[mask], **kwargs)

	def plot(self, grid=None, size=256, limits=None, square=False, center=None, weight=None, weight_stat="mean", figsize=None,
			 aspect="auto", f="identity", axes=None, xlabel=None, ylabel=None,
			 group_by=None, group_limits=None, group_colors='jet', group_labels=None, group_count=None,
			 vmin=None, vmax=None,
			 cmap="afmhot",
			 **kwargs):
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
		f = _parse_f(f)
		limits = self.limits(limits)
		if limits is None:
			limits = self.limits_sigma()
		#if grid is None:
		if group_limits is None and group_by:
			group_limits = tuple(self.dataset(group_by).minmax()[0]) + (group_count,)
		#	grid = self.histogram(limits=limits, size=size, weight=weight, group_limits=group_limits, group_by=group_by)
		if figsize is not None:
			pylab.figure(num=None, figsize=figsize, dpi=80, facecolor='w', edgecolor='k')
		if axes is None:
			axes = pylab.gca()
		fig = pylab.gcf()
		#if xlabel:
		pylab.xlabel(xlabel or self.expressions[0])
		#if ylabel:
		pylab.ylabel(ylabel or self.expressions[1])
		#axes.set_aspect(aspect)
		rgba8 = self.image_rgba(grid=grid, size=size, limits=limits, square=square, center=center, weight=weight, weight_stat=weight_stat,
			 f=f, axes=axes,
			 group_by=group_by, group_limits=group_limits, group_colors=group_colors, group_count=group_count,
			vmin=vmin, vmax=vmax,
			 cmap=cmap)
		import matplotlib
		if group_by:
			if isinstance(group_colors, six.string_types):
				group_colors = matplotlib.cm.get_cmap(group_colors)
			if isinstance(group_colors, matplotlib.colors.Colormap):
				group_count = group_limits[2]
				colors = [group_colors(k/float(group_count-1.)) for k in range(group_count) ]
			else:
				colors = [matplotlib.colors.colorConverter.to_rgba(k) for k in group_colors]
			colormap = matplotlib.colors.ListedColormap(colors)
			gmin, gmax, group_count = group_limits#[:2]
			delta = (gmax - gmin) / (group_count-1.)
			norm = matplotlib.colors.Normalize(gmin-delta/2, gmax+delta/2)
			sm = matplotlib.cm.ScalarMappable(norm, colormap)
			sm.set_array(1) # make matplotlib happy (strange behavious)
			colorbar = fig.colorbar(sm)
			if group_labels:
				colorbar.set_ticks(np.arange(gmin, gmax+delta/2, delta))
				colorbar.set_ticklabels(group_labels)
			else:
				colorbar.set_ticks(np.arange(gmin, gmax+delta/2, delta))
				colorbar.set_ticklabels(map(lambda x: "%f" % x, np.arange(gmin, gmax+delta/2, delta)))
			colorbar.ax.set_ylabel(group_by)
			#matplotlib.colorbar.ColorbarBase(axes, norm=norm, cmap=colormap)
			im = axes.imshow(rgba8, extent=np.array(limits).flatten(), origin="lower", aspect=aspect, **kwargs)
		else:
			norm = matplotlib.colors.Normalize(0, 23)
			sm = matplotlib.cm.ScalarMappable(norm, cmap)
			sm.set_array(1) # make matplotlib happy (strange behavious)
			colorbar = fig.colorbar(sm)
			im = axes.imshow(rgba8, extent=np.array(limits).flatten(), origin="lower", aspect=aspect, **kwargs)
			colorbar = None
		return im, colorbar

	def plot1d(self, grid=None, size=64, limits=None, weight=None, figsize=None, f="identity", axes=None, xlabel=None, ylabel=None, **kwargs):
		"""Plot the subspace using sane defaults to get a quick look at the data.

		:param grid: A 2d numpy array with the counts, if None it will be calculated using limits provided and Subspace.histogram
		:param size: Passed to Subspace.histogram
		:param limits: Limits for the subspace in the form [[xmin, xmax], [ymin, ymax]], if None it will be calculated using Subspace.limits_sigma
		:param figsize: (x, y) tuple passed to pylab.figure for setting the figure size
		:param xlabel: String for label on x axis (may contain latex)
		:param ylabel: Same for y axis
		:param kwargs: extra argument passed to ...,

		 """
		import pylab
		f = _parse_f(f)
		limits = self.limits(limits)
		assert self.dimension == 1, "can only plot 1d, not %s" % self.dimension
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
		#pylab.ylabel(ylabel or self.expressions[1])
		pylab.ylabel("counts" or ylabel)
		#axes.set_aspect(aspect)
		N = len(grid)
		xmin, xmax = limits[0]
		return pylab.plot(np.arange(N) / (N-1.0) * (xmax-xmin) + xmin, f(grid,), drawstyle="steps", **kwargs)
		#pylab.ylim(-1, 6)


	def plot_histogram_bq(self, f="identity", size=64, limits=None, color="red", bq_cleanup=True):
		import vaex.ext.bqplot
		limits = self.limits(limits)
		plot = vaex.ext.bqplot.BqplotHistogram(self, color, size, limits)
		if not hasattr(self, "_bqplot"):
			self._bqplot = {}
			self._bqplot["cleanups"] = []
		else:
			if bq_cleanup:
				for cleanup in self._bqplot["cleanups"]:
					cleanup()
			self._bqplot["cleanups"] = []

		def cleanup(callback=plot.callback):
			self.dataset.signal_selection_changed.disconnect(callback=callback)
		self._bqplot["cleanups"].append(cleanup)

		return plot


	def plot_bq(self, grid=None, size=256, limits=None, square=False, center=None, weight=None, figsize=None,
			 aspect="auto", f="identity", fig=None, axes=None, xlabel=None, ylabel=None, title=None,
			 group_by=None, group_limits=None, group_colors='jet', group_labels=None, group_count=None,
			 cmap="afmhot", scales=None, tool_select=False, bq_cleanup=True,
			 **kwargs):
		import vaex.ext.bqplot
		import bqplot.interacts
		import bqplot.pyplot as p
		import ipywidgets as widgets
		import bqplot as bq
		f = _parse_f(f)
		limits = self.limits(limits)
		import vaex.ext.bqplot
		vaex.ext.bqplot.patch()
		if not hasattr(self, "_bqplot"):
			self._bqplot = {}
			self._bqplot["cleanups"] = []
		else:
			if bq_cleanup:
				for cleanup in self._bqplot["cleanups"]:
					cleanup()
			self._bqplot["cleanups"] = []
		if limits is None:
			limits = self.limits_sigma()
		#if fig is None:
		if scales is None:
			x_scale = bq.LinearScale(min=limits[0][0], max=limits[0][1])
			y_scale = bq.LinearScale(min=limits[1][0], max=limits[1][1])
			scales = {'x': x_scale, 'y': y_scale}
		else:
			x_scale = scales["x"]
			y_scale = scales["y"]
		if 1:
			fig = p.figure() # actually, bqplot doesn't return it
			fig = p.current_figure()
			fig.fig_color = "black" # TODO, take the color from the colormap
			fig.padding_y = 0
			# if we don't do this, bqplot may flip some axes... report this bug
			x = np.arange(10)
			y = x**2
			p.plot(x, y, scales=scales)
			#p.xlim(*limits[0])
			#p.ylim(*limits[1])
			#if grid is None:
		if group_limits is None and group_by:
			group_limits = tuple(self.dataset(group_by).minmax()[0]) + (group_count,)
		#fig = p.
		#if xlabel:
		fig.axes[0].label = xlabel or self.expressions[0]
		#if ylabel:
		fig.axes[1].label = ylabel or self.expressions[1]
		if title:
			fig.title = title
		#axes.set_aspect(aspect)
		rgba8 = self.image_rgba(grid=grid, size=size, limits=limits, square=square, center=center, weight=weight,
			 f=f, axes=axes,
			 group_by=group_by, group_limits=group_limits, group_colors=group_colors, group_count=group_count,
			 cmap=cmap)
		#x_scale = p._context["scales"]["x"]
		#y_scale = p._context["scales"]["y"]
		src="http://localhost:8888/kernelspecs/python2/logo-64x64.png"
		import bqplot.marks
		im = vaex.ext.bqplot.Image(src=src, scales=scales, x=0, y=0, width=1, height=1)
		if 0:
			size = 20
			x_data = np.arange(size)
			line = bq.Lines(x=x_data, y=np.random.randn(size), scales={'x': x_scale, 'y': y_scale},
							stroke_width=3, colors=['red'])


			ax_x = bq.Axis(scale=x_scale, tick_format='0.2f', grid_lines='solid')
			ax_y = bq.Axis(scale=y_scale, orientation='vertical', tick_format='0.2f', grid_lines='solid')
			panzoom = bq.PanZoom(scales={'x': [x_scale], 'y': [y_scale]})
			lasso = bqplot.interacts.LassoSelector()
			brush = bqplot.interacts.BrushSelector(x_scale=x_scale, y_scale=y_scale, color="green")
			fig = bq.Figure(marks=[line,im], axes=[ax_x, ax_y], min_width=100, min_height=100, interaction=panzoom)
		else:
			fig.marks = list(fig.marks) + [im]
		def make_image(executor, limits):
			#print "make image" * 100
			self.executor = executor
			if self.dataset.has_selection():
				sub = self.selected()
			else:
				sub = self
			return sub.image_rgba(limits=limits, size=size, f=f)
		progress = widgets.FloatProgress(value=0.0, min=0.0, max=1.0, step=0.01)
		updater = vaex.ext.bqplot.DebouncedThreadedUpdater(self, size, im, make_image, progress_widget=progress)
		def update_image():
			limits = [x_scale.min, x_scale.max], [y_scale.min, y_scale.max]
			#\print limits
			#print "update...", limits
			#vxbq.debounced_threaded_update(self.dataset, im, make_image2, limits=limits)
			updater.update(limits)
		def update(*args):
			update_image()
		y_scale.observe(update, "min")
		y_scale.observe(update, "max")
		x_scale.observe(update, "min")
		x_scale.observe(update, "max")
		update_image()
		#fig = kwargs.pop('figure', p.current_figure())
		tools = []
		tool_actions = []
		panzoom = bq.PanZoom(scales={'x': [x_scale], 'y': [y_scale]})
		tool_actions_map = {u"m":panzoom}
		tool_actions.append(u"m")

		fig.interaction = panzoom
		if tool_select:
			brush = bqplot.interacts.BrushSelector(x_scale=x_scale, y_scale=y_scale, color="green")
			tool_actions_map["b"] = brush
			tool_actions.append("b")
			def update_selection(*args):
				def f():
					if brush.selected:
						(x1, y1), (x2, y2) = brush.selected
						ex1, ex2 = self.expressions
						mode = modes_names[modes_labels.index(button_selection_mode.value)]
						self.dataset.select_rectangle(ex1, ex2, limits=[[x1, x2], [y1, y2]], mode=mode)
					else:
						self.dataset.select_nothing()
				updater.update_select(f)
			brush.observe(update_selection, "selected")
			#fig.interaction = brush
			#callback = self.dataset.signal_selection_changed.connect(lambda dataset: update_image())
			callback = self.dataset.signal_selection_changed.connect(lambda dataset: updater.update_direct_safe())
			def cleanup(callback=callback):
				self.dataset.signal_selection_changed.disconnect(callback=callback)
			self._bqplot["cleanups"].append(cleanup)

			button_select_nothing = widgets.Button(icon="fa-trash-o")
			def select_nothing(button):
				self.dataset.select_nothing()
			button_select_nothing.on_click(select_nothing)
			tools.append(button_select_nothing)
			modes_names = "replace and or xor subtract".split()
			modes_labels = "= & | ^ -".split()
			button_selection_mode = widgets.ToggleButtons(description='',options=modes_labels)
			tools.append(button_selection_mode)
		def change_interact(*args):
			#print "change", args
			fig.interaction = tool_actions_map[button_action.value]
		#tool_actions = ["m", "b"]
		#tool_actions = [("m", "m"), ("b", "b")]
		button_action = widgets.ToggleButtons(description='',options=tool_actions, icons=["fa-arrows", "fa-pencil-square-o"])
		button_action.observe(change_interact, "value")
		tools.insert(0,button_action)
		button_action.value = "m" #tool_actions[-1]
		if len(tools) == 1:
			tools = []
		tools = widgets.HBox(tools)

		box_layout = widgets.Layout(display='flex',
						flex_flow='column',
						#border='solid',
						width='100%', height="100%")
		fig.fig_margin = {'bottom': 40, 'left': 60, 'right': 10, 'top': 40}
		#fig.min_height = 700
		#fig.min_width = 400
		fig.layout = box_layout
		return widgets.VBox([fig, progress, tools])

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
				elif progressbar:
					callback = self.executor.signal_progress.connect(progressbar)
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
				mask = self.dataset.evaluate_selection_mask("default", i1=i1, i2=i2)
				blocks = [block[mask] for block in blocks]
				is_empty = all(~mask)
				if is_empty:
					return None
			#with lock:
			#print blocks
			#with lock:
			#	print thread_index, i1, i2, blocks
			return [vaex.vaexfast.find_nan_min_max(block) for block in blocks]
			if 0: # TODO: implement using statisticNd and benchmark
				minmaxes = np.zeros((len(blocks), 2), dtype=float)
				minmaxes[:,0] = np.inf
				minmaxes[:,1] = -np.inf
				for i, block in enumerate(blocks):
					vaex.vaexfast.statisticNd([], block, minmaxes[i,:], [], [], 2)
				#minmaxes[~np.isfinite(minmaxes)] = np.nan
				return minmaxes
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
				mask = self.dataset.evaluate_selection_mask("default", i1=i1, i2=i2)
				return [(np.nanmean(block[mask]**moment), np.count_nonzero(~np.isnan(block[mask]))) for block in blocks]
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
			def var_map(thread_index, i1, i2, *blocks):
				mask = self.dataset.evaluate_selection_mask("default", i1=i1, i2=i2)
				if means is not None:
					return [(np.nanmean((block[mask]-mean)**2), np.count_nonzero(~np.isnan(block[mask]))) for block, mean in zip(blocks, means)]
				else:
					return [(np.nanmean(block[mask]**2), np.count_nonzero(~np.isnan(block[mask]))) for block in blocks]
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
		nansum = vaex.vaexfast.nansum
		if self.is_masked:
			task = TaskMapReduce(self.dataset,\
								 self.expressions, lambda thread_index, i1, i2, *blocks: [nansum(block[self.dataset.evaluate_selection_mask("default", i1=i1, i2=i2)])
																						  for block in blocks],\
								 lambda a, b: np.array(a) + np.array(b), self._toarray, info=True)
		else:
			task = TaskMapReduce(self.dataset, self.expressions, lambda *blocks: [nansum(block) for block in blocks], lambda a, b: np.array(a) + np.array(b), self._toarray)
		return self._task(task)

	def histogram(self, limits, size=256, weight=None, progressbar=False, group_by=None, group_limits=None):
		expressions = self.expressions
		if group_by:
			expressions = list(expressions) + [group_by]
			limits = list(limits) + [group_limits[:2]] #[[group_limits[0] - 0,5, group_limits[1]+0.5]]
			#assert group_limits[2] == 1
			size = (group_limits[2],) + (size,) * (len(expressions) -1)
		task = TaskHistogram(self.dataset, self, expressions, size, limits, masked=self.is_masked, weight=weight)
		return self._task(task, progressbar=progressbar)

	def bin_mean(self, expression, limits, size=256, progressbar=False, group_by=None, group_limits=None):
		# todo, fix progressbar into two...
		counts = self.histogram(limits=limits, size=size, progressbar=progressbar, group_by=group_by, group_limits=group_limits)
		weighted  =self.histogram(limits=limits, size=size, progressbar=progressbar, group_by=group_by, group_limits=group_limits,
								weight=expression)
		mean = weighted/counts
		mean[counts==0] = np.nan
		return mean

	def bin_mean_cyclic(self, expression, max_value, limits, size=256, progressbar=False, group_by=None, group_limits=None):
		# todo, fix progressbar into two...
		meanx = self.bin_mean(limits=limits, size=size, progressbar=progressbar, group_by=group_by, group_limits=group_limits,
								expression="cos((%s)/%r*2*pi)" % (expression, max_value))
		meany = self.bin_mean(limits=limits, size=size, progressbar=progressbar, group_by=group_by, group_limits=group_limits,
								expression="sin((%s)/%r*2*pi)" % (expression, max_value))
		angles = np.arctan2(meany, meanx)
		values =  ((angles+2*np.pi) % (2*np.pi)) / (2*np.pi) * max_value
		length =  np.sqrt(meanx**2+meany**2)
		length[~np.isfinite(meanx)] = np.nan
		return values, length

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


	def limits_percentage(self, percentage=99.73, square=False):
		limits = []
		for expr in self.expressions:
			subspace = self.dataset(expr)
			if self.is_selected():
				subspace = subspace.selected()
			limits_minmax = subspace.minmax()
			vmin, vmax = limits_minmax[0]
			size = 1024*16
			counts = subspace.histogram(size=size, limits=limits_minmax)
			cumcounts = np.concatenate([[0], np.cumsum(counts)])
			cumcounts /= cumcounts.max()
			# TODO: this is crude.. see the details!
			f = (1-percentage/100.)/2
			x = np.linspace(vmin, vmax, size+1)
			l = scipy.interp([f,1-f], cumcounts, x)
			limits.append(l)
		return limits

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
				mask = self.dataset.evaluate_selection_mask("default", i1=i1, i2=i2)
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
		task = TaskMapReduce(self.dataset,\
							 self.expressions,
							 nearest_in_block,\
							 nearest_reduce, info=True)
		return self._task(task)







import vaex.events
import cgi

# mutex for numexpr (is not thread save)
ne_lock = threading.Lock()

class UnitScope(object):
	def __init__(self, dataset, value=None):
		self.dataset = dataset
		self.value = value

	def __getitem__(self, variable):
		if variable in self.dataset.units:
			unit = self.dataset.units[variable]
			return (self.value * unit) if self.value is not None else unit
		elif variable in self.dataset.virtual_columns:
			return eval(self.dataset.virtual_columns[variable], expression_namespace, self)
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
		self.i1 = int(i1)
		self.i2 = int(i2)
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
		self.i1 = int(i1)
		self.i2 = int(i2)
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
			result = eval(expression, expression_namespace, self)
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
		if variable in expression_namespace:
			return expression_namespace[variable]
		try:
			if variable in self.dataset.get_column_names(strings=True):
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

class _BlockScopeSelection(object):
	def __init__(self, dataset, i1, i2, selection=None):
		self.dataset = dataset
		self.i1 = i1
		self.i2 = i2
		self.selection = selection

	def evaluate(self, expression):
		if expression is True:
			expression = "default"
		try:
			return eval(expression, expression_namespace, self)
		except:
			import traceback as tb
			tb.print_stack()
			raise
		#if selection is None:
		selection = self.get_selection(name)
		#if selection is None:
		#	return None
		cache = self._selection_mask_caches.get(expression)
		if cache:
			key = (i1, i2)
			value = cache.get(key)
			logger.debug("cache for %r is %r", name, value)
			if value is None or value[0] != selection:
				logger.debug("creating new mask")
				mask = selection.evaluate(name, i1, i2)
				cache[key] = selection, mask
			else:
				selection_in_cache, mask = value
		return mask
		result = eval(expression, expression_namespace, self)
		return result

	def __getitem__(self, variable):
		logger.debug("getitem for selection: %s", variable)
		try:
			selection = self.selection or self.dataset.get_selection(variable)
			logger.debug("selection: %s %r", selection, self.dataset.selection_histories)
			key = (self.i1, self.i2)
			if selection:
				cache = self.dataset._selection_mask_caches.get(variable)
				if cache:
					selection_in_cache, mask = cache.get(key, (None, None))
					logger.debug("mask for %r is %r", variable, mask)
					if selection_in_cache == selection:
						return mask
				logger.debug("was not cached")
				mask = selection.evaluate(variable, self.i1, self.i2)
				if cache:
					cache[key] = selection, mask
				return mask
			else:
					if variable in expression_namespace:
						return expression_namespace[variable]
					elif variable in self.dataset.get_column_names(strings=True):
						return self.dataset.columns[variable][self.i1:self.i2]
					elif variable in list(self.dataset.virtual_columns.keys()):
						expression = self.dataset.virtual_columns[variable]
						#self._ensure_buffer(variable)
						return self.evaluate(expression)#, out=self.buffers[variable])
						#self.values[variable] = self.buffers[variable]
					raise KeyError("Unknown variables or column: %r" % (variable,))
		except:
			import traceback as tb
			tb.print_exc()
			logger.exception("error in evaluating: %r" % variable)
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

	def evaluate(self, name, i1, i2):
		if self.previous_selection:
			previous_mask = self.dataset.evaluate_selection_mask(name, i1, i2, selection=self.previous_selection)
		else:
			previous_mask = None
		current_mask = self.dataset.evaluate(self.boolean_expression, i1, i2).astype(np.bool)
		if previous_mask is None:
			logger.debug("setting mask")
			mask = current_mask
		else:
			logger.debug("combining previous mask with current mask using op %r", self.mode)
			mode_function = _select_functions[self.mode]
			mask = mode_function(previous_mask, current_mask)
		return mask


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

	def evaluate(self, name, i1, i2):
		previous_mask = self.dataset.evaluate_selection_mask(name, i1, i2, selection=self.previous_selection)
		return ~previous_mask


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

	def evaluate(self, name, i1, i2):
		if self.previous_selection:
			previous_mask = self.dataset.evaluate_selection_mask(name, i1, i2, selection=self.previous_selection)
		else:
			previous_mask = None
		current_mask = np.zeros(i2-i1, dtype=np.bool)
		x, y = np.array(self.xseq, dtype=np.float64), np.array(self.yseq, dtype=np.float64)
		meanx = x.mean()
		meany = y.mean()
		radius = np.sqrt((meanx-x)**2 + (meany-y)**2).max()
		blockx = self.dataset.evaluate(self.boolean_expression_x, i1=i1, i2=i2)
		blocky = self.dataset.evaluate(self.boolean_expression_y, i1=i1, i2=i2)
		vaex.vaexfast.pnpoly(x, y, blockx, blocky, current_mask, meanx, meany, radius)
		if previous_mask is None:
			logger.debug("setting mask")
			mask = current_mask
		else:
			logger.debug("combining previous mask with current mask using op %r", self.mode)
			mode_function = _select_functions[self.mode]
			mask = mode_function(previous_mask, current_mask)
		return mask

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
arctan
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
where
""".strip().split()]
expression_namespace = {}
for name, numpy_name in function_mapping:
	if not hasattr(np, numpy_name):
		raise SystemError("numpy does not have: %s" % numpy_name)
	else:
		expression_namespace[name] = getattr(np, numpy_name)
import pandas as pd
def dayofweek(x):
    x = x.astype("<M8[ns]")
    return pd.Series(x).dt.dayofweek.values.astype(np.float64)
expression_namespace["dayofweek"] = dayofweek
def hourofday(x):
    x = x.astype("<M8[ns]")
    return pd.Series(x).dt.hour.values.astype(np.float64)
expression_namespace["hourofday"] = hourofday

_doc_snippets = {}
_doc_snippets["expression"] = "expression or list of expressions, e.g. 'x', or ['x, 'y']"
_doc_snippets["expression_single"] = "if previous argument is not a list, this argument should be given"
_doc_snippets["binby"] = "List of expressions for constructing a binned grid"
_doc_snippets["limits"] = """description for the min and max values for the expressions, e.g. 'minmax', '99.7%', [0, 10], or a list of, e.g. [[0, 10], [0, 20], 'minmax']"""
_doc_snippets["shape"] = """shape for the array where the statistic is calculated on, if only an integer is given, it is used for all dimensions, e.g. shape=128, shape=[128, 256]"""
_doc_snippets["percentile_limits"] = """description for the min and max values to use for the cumulative histogram, should currently only be 'minmax'"""
_doc_snippets["percentile_shape"] = """shape for the array where the cumulative histogram is calculated on, integer type"""
_doc_snippets["selection"] = """Name of selection to use (or True for the 'default'), or all the data (when selection is None or False)"""
_doc_snippets["async"] = """Do not return the result, but a proxy for asynchronous calculations (currently only for internal use)"""
_doc_snippets["expression_limits"] = _doc_snippets["expression"]

_doc_snippets["return_stat_scalar"] = """Numpy array with the given shape, or a scalar when no binby argument is given, with the statistic"""
_doc_snippets["return_limits"] = """List in the form [[xmin, xmax], [ymin, ymax], .... ,[zmin, zmax]] or [xmin, xmax] when expression is not a list"""

def docsubst(f):
	f.__doc__ = f.__doc__.format(**_doc_snippets)
	return f

_functions_statistics_1d = []

def stat_1d(f):
	_functions_statistics_1d.append(f)
	return f
class Dataset(object):
	"""All datasets are encapsulated in this class, local or remote dataets

	Each dataset has a number of columns, and a number of rows, the length of the dataset.

	The most common operations are:
	Dataset.plot
	>>>
	>>>


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
		self._index_start = 0
		self._index_end = None

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

	def map_reduce(self, map, reduce, arguments, async=False):
		#def map_wrapper(*blocks):
		task = TaskMapReduce(self, arguments, map, reduce, info=False)
		self.executor.schedule(task)
		return self._async(async, task)


	@docsubst
	def mutual_information(self, x, y=None, mi_limits=None, mi_shape=256, binby=[], limits=None, shape=default_shape, sort=False, selection=False, async=False):
		"""Estimate the mutual information between and x and y on a grid with shape mi_shape and mi_limits, possible on a grid defined by binby

		If sort is True, the mutual information is returned in sorted (descending) order and the list of expressions is returned in the same order

		Examples:

		>>> ds.mutual_information("x", "y")
		array(0.1511814526380327)
		>>> ds.mutual_information([["x", "y"], ["x", "z"], ["E", "Lz"]])
		array([ 0.15118145,  0.18439181,  1.07067379])
		>>> ds.mutual_information([["x", "y"], ["x", "z"], ["E", "Lz"]], sort=True)
		(array([ 1.07067379,  0.18439181,  0.15118145]),
		[['E', 'Lz'], ['x', 'z'], ['x', 'y']])


		:param x: {expression}
		:param y: {expression}
		:param limits: {limits}
		:param shape: {shape}
		:param binby: {binby}
		:param limits: {limits}
		:param shape: {shape}
		:param sort: return mutual information in sorted (descending) order, and also return the correspond list of expressions when sorted is True
		:param selection: {selection}
		:param async: {async}
		:return: {return_stat_scalar},
		"""
		if y is None:
			waslist, [x,] = vaex.utils.listify(x)
		else:
			waslist, [x,y] = vaex.utils.listify(x, y)
			x = list(zip(x, y))
			if mi_limits:
				mi_limits = [mi_limits]
		#print("x, mi_limits", x, mi_limits)
		limits = self.limits(binby, limits, async=True)
		#print("$"*80)
		mi_limits = self.limits(x, mi_limits, async=True)
		#print("@"*80)

		@delayed
		def calculate(counts):
			# TODO: mutual information doesn't take axis arguments, so ugly solution for now
			fullshape = _expand_shape(shape, len(binby))
			out = np.zeros((fullshape), dtype=float)
			if len(fullshape) == 0:
				out = vaex.kld.mutual_information(counts)
				#print("count> ", np.sum(counts))
			elif len(fullshape) == 1:
				for i in range(fullshape[0]):
					out[i] = vaex.kld.mutual_information(counts[...,i])
					#print("counti> ", np.sum(counts[...,i]))
				#print("countt> ", np.sum(counts))
			elif len(fullshape) == 2:
				for i in range(fullshape[0]):
					for j in range(fullshape[1]):
						out[i,j] = vaex.kld.mutual_information(counts[...,i,j])
			elif len(fullshape) == 3:
				for i in range(fullshape[0]):
					for j in range(fullshape[1]):
						for k in range(fullshape[2]):
							out[i,j,k] = vaex.kld.mutual_information(counts[...,i,j,k])
			else:
				raise ValueError("binby with dim > 3 is not yet supported")
			return out
		@delayed
		def has_limits(limits, mi_limits):
			if not _issequence(binby):
				limits = [list(limits)]
			values = []
			for expressions, expression_limits in zip(x, mi_limits):
				#print("mi for", expressions, expression_limits)
				#total_shape =  _expand_shape(mi_shape, len(expressions)) + _expand_shape(shape, len(binby))
				total_shape =  _expand_shape(mi_shape, len(expressions)) + _expand_shape(shape, len(binby))
				#print("expressions", expressions)
				#print("total_shape", total_shape)
				#print("limits", limits,expression_limits)
				#print("limits>", list(limits) + list(expression_limits))
				counts = self.count(binby=list(expressions) + list(binby), limits=list(expression_limits)+list(limits),
						   shape=total_shape, async=True, selection=selection)
				values.append(calculate(counts))
			return values

		@delayed
		def finish(mi_list):
			if sort:
				mi_list = np.array(mi_list)
				indices = np.argsort(mi_list)[::-1]
				sorted_x = list([x[k] for k in indices])
				return mi_list[indices], sorted_x
			else:
				return np.array(vaex.utils.unlistify(waslist, mi_list))
		values = finish(delayed_list(has_limits(limits, mi_limits)))
		return self._async(async, values)

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

	@docsubst
	def count(self, expression=None, binby=[], limits=None, shape=default_shape, selection=False, async=False, progress=None):
		"""Count the number of non-NaN values (or all, if expression is None or "*")

		Examples:


		>>> ds.count("*")
		330000.0
		>>> ds.count("*", binby=["x"], shape=4)
		array([  10925.,  155427.,  152007.,   10748.])

		:param expression: {expression}
		:param binby: {binby}
		:param limits: {limits}
		:param shape: {shape}
		:param selection: {selection}
		:param async: {async}
		:return: {return_stat_scalar}
		"""
		logger.debug("count(%r, binby=%r, limits=%r)", expression, binby, limits)
		@delayed
		def calculate(expression, limits):
			if expression in ["*", None]:
				#if not binby: # if we have nothing to iterate over, the statisticNd code won't do anything
				#\3	return np.array([self.length(selection=selection)], dtype=float)
				#else:
				task = TaskStatistic(self, binby, shape, limits, op=OP_ADD1, selection=selection)
			else:
				task = TaskStatistic(self, binby, shape, limits, weight=expression, op=OP_COUNT, selection=selection)
			self.executor.schedule(task)
			i = expressions.index(expression)
			task.signal_progress.connect(progressbars[i])
			task.signal_progress.connect(progressbar)
			return task
		@delayed
		def finish(*stats_args):
			stats = np.array(stats_args)
			counts = stats[...,0]
			return vaex.utils.unlistify(waslist, counts)
		waslist, [expressions,] = vaex.utils.listify(expression)
		progressbar, progressbars = vaex.utils.progressbars(progress, len(expressions))
		limits = self.limits(binby, limits, async=True)
		stats = [calculate(expression, limits) for expression in expressions]
		var = finish(*stats)
		return self._async(async, var)

	@docsubst
	@stat_1d
	def mean(self, expression, binby=[], limits=None, shape=default_shape, selection=False, async=False):
		"""Calculate the mean for expression, possible on a grid defined by binby.

		Examples:

		>>> ds.mean("x")
		-0.067131491264005971
		>>> ds.mean("(x**2+y**2)**0.5", binby="E", shape=4)
		array([  2.43483742,   4.41840721,   8.26742458,  15.53846476])

		:param expression: {expression}
		:param binby: {binby}
		:param limits: {limits}
		:param shape: {shape}
		:param selection: {selection}
		:param async: {async}
		:return: {return_stat_scalar}
		"""
		logger.debug("mean of %r, with binby=%r, limits=%r, shape=%r, selection=%r, async=%r", expression, binby, limits, shape, selection, async)
		@delayed
		def calculate(expression, limits):
			task = TaskStatistic(self, binby, shape, limits, weight=expression, op=OP_ADD_WEIGHT_MOMENTS_01, selection=selection)
			self.executor.schedule(task)
			return task
		@delayed
		def finish(*stats_args):
			stats = np.array(stats_args)
			counts = stats[...,0]
			mean = stats[...,1] / counts
			return vaex.utils.unlistify(waslist, mean)
		waslist, [expressions,] = vaex.utils.listify(expression)
		limits = self.limits(binby, limits, async=True)
		stats = [calculate(expression, limits) for expression in expressions]
		var = finish(*stats)
		return self._async(async, var)

	@docsubst
	@stat_1d
	def sum(self, expression, binby=[], limits=None, shape=default_shape, selection=False, async=False):
		"""Calculate the sum for the given expression, possible on a grid defined by binby

		Examples:

		>>> ds.sum("L")
		304054882.49378014
		>>> ds.sum("L", binby="E", shape=4)
		array([  8.83517994e+06,   5.92217598e+07,   9.55218726e+07,
				 1.40008776e+08])

		:param expression: {expression}
		:param binby: {binby}
		:param limits: {limits}
		:param shape: {shape}
		:param selection: {selection}
		:param async: {async}
		:return: {return_stat_scalar}
		"""
		@delayed
		def calculate(expression, limits):
			task = TaskStatistic(self, binby, shape, limits, weight=expression, op=OP_ADD_WEIGHT_MOMENTS_01, selection=selection)
			self.executor.schedule(task)
			return task
		@delayed
		def finish(*stats_args):
			stats = np.array(stats_args)
			sum = stats[...,1]
			return vaex.utils.unlistify(waslist, sum)
		waslist, [expressions,] = vaex.utils.listify(expression)
		limits = self.limits(binby, limits, async=True)
		stats = [calculate(expression, limits) for expression in expressions]
		s = finish(*stats)
		return self._async(async, s)

	@docsubst
	@stat_1d
	def std(self, expression, binby=[], limits=None, shape=default_shape, selection=False, async=False):
		"""Calculate the standard deviation for the given expression, possible on a grid defined by binby


		>>> ds.std("vz")
		110.31773397535071
		>>> ds.std("vz", binby=["(x**2+y**2)**0.5"], shape=4)
		array([ 123.57954851,   85.35190177,   61.14345748,   38.0740619 ])

		:param expression: {expression}
		:param binby: {binby}
		:param limits: {limits}
		:param shape: {shape}
		:param selection: {selection}
		:param async: {async}
		:return: {return_stat_scalar}
		"""
		@delayed
		def finish(var):
			return var**0.5
		return self._async(async, finish(self.var(expression, binby=binby, limits=limits, shape=shape, selection=selection, async=True)))

	@docsubst
	@stat_1d
	def var(self, expression, binby=[], limits=None, shape=default_shape, selection=False, async=False):
		"""Calculate the sample variance for the given expression, possible on a grid defined by binby

		Examples:

		>>> ds.var("vz")
		12170.002429456246
		>>> ds.var("vz", binby=["(x**2+y**2)**0.5"], shape=4)
		array([ 15271.90481083,   7284.94713504,   3738.52239232,   1449.63418988])
		>>> ds.var("vz", binby=["(x**2+y**2)**0.5"], shape=4)**0.5
		array([ 123.57954851,   85.35190177,   61.14345748,   38.0740619 ])
		>>> ds.std("vz", binby=["(x**2+y**2)**0.5"], shape=4)
		array([ 123.57954851,   85.35190177,   61.14345748,   38.0740619 ])

		:param expression: {expression}
		:param binby: {binby}
		:param limits: {limits}
		:param shape: {shape}
		:param selection: {selection}
		:param async: {async}
		:return: {return_stat_scalar}
		"""
		@delayed
		def calculate(expression, limits):
			task = TaskStatistic(self, binby, shape, limits, weight=expression, op=OP_ADD_WEIGHT_MOMENTS_012, selection=selection)
			self.executor.schedule(task)
			return task
		@delayed
		def finish(*stats_args):
			stats = np.array(stats_args)
			counts = stats[...,0]
			mean = stats[...,1] / counts
			raw_moments2 = stats[...,2] / counts
			variance = (raw_moments2-mean**2)
			return vaex.utils.unlistify(waslist, variance)
		waslist, [expressions,] = vaex.utils.listify(expression)
		limits = self.limits(binby, limits, async=True)
		stats = [calculate(expression, limits) for expression in expressions]
		var = finish(*stats)
		return self._async(async, var)


	@docsubst
	def covar(self, x, y, binby=[], limits=None, shape=default_shape, selection=False, async=False):
		"""Calculate the covariance cov[x,y] between and x and y, possible on a grid defined by binby

		Examples:

		>>> ds.covar("x**2+y**2+z**2", "-log(-E+1)")
		array(52.69461456005138)
		>>> ds.covar("x**2+y**2+z**2", "-log(-E+1)")/(ds.std("x**2+y**2+z**2") * ds.std("-log(-E+1)"))
		0.63666373822156686
		>>> ds.covar("x**2+y**2+z**2", "-log(-E+1)", binby="Lz", shape=4)
		array([ 10.17387143,  51.94954078,  51.24902796,  20.2163929 ])



		:param x: {expression}
		:param y: {expression}
		:param binby: {binby}
		:param limits: {limits}
		:param shape: {shape}
		:param selection: {selection}
		:param async: {async}
		:return: {return_stat_scalar}
		"""
		@delayed
		def cov(mean_x, mean_y, mean_xy):
			return mean_xy - mean_x * mean_y

		waslist, [xlist,ylist] = vaex.utils.listify(x, y)
		#print("limits", limits)
		limits = self.limits(binby, limits, selection=selection, async=True)
		#print("limits", limits)

		@delayed
		def calculate(limits):
			covars = [cov(
						self.mean(x, binby=binby, limits=limits, shape=shape, selection=selection, async=True),
						self.mean(y, binby=binby, limits=limits, shape=shape, selection=selection, async=True),
						self.mean("(%s)*(%s)" % (x, y), binby=binby, limits=limits, shape=shape, selection=selection, async=True),
					)
					  for x, y in zip(xlist, ylist)]
			return covars

		covars = calculate(limits)
		@delayed
		def finish(covars):
			value = np.array(vaex.utils.unlistify(waslist, covars))
			return value
		return self._async(async, finish(delayed_list(covars)))

	@docsubst
	def correlation(self, x, y=None, binby=[], limits=None, shape=default_shape, sort=False, sort_key=np.abs, selection=False, async=False):
		"""Calculate the correlation coefficient cov[x,y]/(std[x]*std[y]) between and x and y, possible on a grid defined by binby

		Examples:


		>>> ds.correlation("x**2+y**2+z**2", "-log(-E+1)")
		array(0.6366637382215669)
		>>> ds.correlation("x**2+y**2+z**2", "-log(-E+1)", binby="Lz", shape=4)
		array([ 0.40594394,  0.69868851,  0.61394099,  0.65266318])

		:param x: {expression}
		:param y: {expression}
		:param binby: {binby}
		:param limits: {limits}
		:param shape: {shape}
		:param selection: {selection}
		:param async: {async}
		:return: {return_stat_scalar}
		"""
		@delayed
		def corr(cov):
			return cov[...,0,1] / (cov[...,0,0] * cov[...,1,1])**0.5

		if y is None:
			if not isinstance(x, (tuple, list)):
				raise ValueError("if y not given, x is expected to be a list or tuple, not %r" % x)
			if _issequence(x) and not _issequence(x[0]) and len(x) == 2:
				x = [x]
			if not(_issequence(x) and all([_issequence(k) and len(k) == 2 for k in x])):
				raise ValueError("if y not given, x is expected to be a list of lists with length 2, not %r" % x)
			#waslist, [xlist,ylist] = vaex.utils.listify(*x)
			waslist = True
			xlist, ylist = zip(*x)
			#print xlist, ylist
		else:
			waslist, [xlist,ylist] = vaex.utils.listify(x, y)
		limits = self.limits(binby, limits, selection=selection, async=True)

		@delayed
		def echo(limits):
			logger.debug(">>>>>>>>: %r %r", limits, np.array(limits).shape)
		echo(limits)

		@delayed
		def calculate(limits):
			correlation = [corr(
						self.cov(x, y, binby=binby, limits=limits, shape=shape, selection=selection, async=True),
					)
					  for x, y in zip(xlist, ylist)]
			return correlation

		correlations = calculate(limits)
		@delayed
		def finish(correlations):
			if sort:
				correlations = np.array(correlations)
				indices = np.argsort(sort_key(correlations) if sort_key else correlations)[::-1]
				sorted_x = list([x[k] for k in indices])
				return correlations[indices], sorted_x
			value = np.array(vaex.utils.unlistify(waslist, correlations))
			return value
		return self._async(async, finish(delayed_list(correlations)))

	@docsubst
	def cov(self, x, y=None, binby=[], limits=None, shape=default_shape, selection=False, async=False):
		"""Calculate the covariance matrix for x and y or more expressions, possible on a grid defined by binby

		Either x and y are expressions, e.g:

		>>> ds.cov("x", "y")

		Or only the x argument is given with a list of expressions, e,g.:

		>> ds.cov(["x, "y, "z"])

		Examples:

		>>> ds.cov("x", "y")
		array([[ 53.54521742,  -3.8123135 ],
       [ -3.8123135 ,  60.62257881]])
       >>> ds.cov(["x", "y", "z"])
       array([[ 53.54521742,  -3.8123135 ,  -0.98260511],
       [ -3.8123135 ,  60.62257881,   1.21381057],
       [ -0.98260511,   1.21381057,  25.55517638]])

		>>> ds.cov("x", "y", binby="E", shape=2)
		array([[[  9.74852878e+00,  -3.02004780e-02],
        [ -3.02004780e-02,   9.99288215e+00]],

       [[  8.43996546e+01,  -6.51984181e+00],
        [ -6.51984181e+00,   9.68938284e+01]]])


		:param x: {expression}
		:param y: {expression_single}
		:param binby: {binby}
		:param limits: {limits}
		:param shape: {shape}
		:param selection: {selection}
		:param async: {async}
		:return: {return_stat_scalar}, the last dimensions are of shape (2,2)
		"""
		@delayed
		def cov_matrix(mean_x, mean_y, var_x, var_y, mean_xy):
			cov = mean_xy - mean_x * mean_y
			return np.array([[var_x, cov], [cov, var_y]]).T

		if y is None:
			if not _issequence(x):
				raise ValueError("if y argument is not given, x is expected to be sequence, not %r", x)
			expressions = x
		else:
			expressions = [x, y]
		N = len(expressions)
		binby = _ensure_list(binby)
		shape = _expand_shape(shape, len(binby))
		limits = self.limits(binby, limits, selection=selection, async=True)

		@delayed
		def calculate_matrix(means, vars, raw_mixed):
			#print(">>> %r" % means)
			raw_mixed = list(raw_mixed) # lists can pop
			cov_matrix = np.zeros(shape + (N,N), dtype=float)
			for i in range(N):
				for j in range(i+1):
					if i != j:
						cov_matrix[...,i,j] = raw_mixed.pop(0) - means[i] * means[j]
						cov_matrix[...,j,i] = cov_matrix[...,i,j]
					else:
						cov_matrix[...,i,j] = vars[i]
			return cov_matrix


		@delayed
		def calculate(limits):
			# calculate the right upper triangle
			means = [self.mean(expression, binby=binby, limits=limits, shape=shape, selection=selection, async=True) for expression in expressions]
			vars  = [self.var (expression, binby=binby, limits=limits, shape=shape, selection=selection, async=True) for expression in expressions]
			raw_mixed = []
			for i in range(N):
				for j in range(i+1):
					if i != j:
						raw_mixed.append(self.mean("(%s)*(%s)" % (expressions[i], expressions[j]), binby=binby, limits=limits, shape=shape, selection=selection, async=True))
			return calculate_matrix(delayed_list(means), delayed_list(vars), delayed_list(raw_mixed))

		covars = calculate(limits)
		return self._async(async, covars)

	@docsubst
	@stat_1d
	def minmax(self, expression, binby=[], limits=None, shape=default_shape, selection=False, async=False):
		"""Calculate the minimum and maximum for expressions, possible on a grid defined by binby


		Example:

		>>> ds.minmax("x")
		array([-128.293991,  271.365997])
		>>> ds.minmax(["x", "y"])
		array([[-128.293991 ,  271.365997 ],
			   [ -71.5523682,  146.465836 ]])
		>>> ds.minmax("x", binby="x", shape=5, limits=[-10, 10])
		array([[-9.99919128, -6.00010443],
			   [-5.99972439, -2.00002384],
			   [-1.99991322,  1.99998057],
			   [ 2.0000093 ,  5.99983597],
			   [ 6.0004878 ,  9.99984646]])

		:param expression: {expression}
		:param binby: {binby}
		:param limits: {limits}
		:param shape: {shape}
		:param selection: {selection}
		:param async: {async}
		:return: {return_stat_scalar}, the last dimension is of shape (2)
		"""
		@delayed
		def calculate(expression, limits):
			task =  TaskStatistic(self, binby, shape, limits, weight=expression, op=OP_MIN_MAX, selection=selection)
			self.executor.schedule(task)
			return task
		@delayed
		def finish(*minmax_list):
			value = vaex.utils.unlistify(waslist, np.array(minmax_list))
			return value
		waslist, [expressions,] = vaex.utils.listify(expression)
		limits = self.limits(binby, limits, selection=selection, async=True)
		tasks = [calculate(expression, limits) for expression in expressions]
		result = finish(*tasks)
		return self._async(async, result)

	@docsubst
	@stat_1d
	def min(self, expression, binby=[], limits=None, shape=default_shape, selection=False, async=False):
		"""Calculate the minimum for given expressions, possible on a grid defined by binby


		Example:

		>>> ds.min("x")
		array(-128.293991)
		>>> ds.min(["x", "y"])
		array([-128.293991 ,  -71.5523682])
		>>> ds.min("x", binby="x", shape=5, limits=[-10, 10])
		array([-9.99919128, -5.99972439, -1.99991322,  2.0000093 ,  6.0004878 ])

		:param expression: {expression}
		:param binby: {binby}
		:param limits: {limits}
		:param shape: {shape}
		:param selection: {selection}
		:param async: {async}
		:return: {return_stat_scalar}, the last dimension is of shape (2)
		"""
		@delayed
		def finish(result):
			return result[...,0]
		return self._async(async, finish(self.minmax(expression, binby=binby, limits=limits, shape=shape, selection=selection, async=async)))

	@docsubst
	@stat_1d
	def max(self, expression, binby=[], limits=None, shape=default_shape, selection=False, async=False):
		"""Calculate the maximum for given expressions, possible on a grid defined by binby


		Example:

		>>> ds.max("x")
		array(271.365997)
		>>> ds.max(["x", "y"])
		array([ 271.365997,  146.465836])
		>>> ds.max("x", binby="x", shape=5, limits=[-10, 10])
		array([-6.00010443, -2.00002384,  1.99998057,  5.99983597,  9.99984646])

		:param expression: {expression}
		:param binby: {binby}
		:param limits: {limits}
		:param shape: {shape}
		:param selection: {selection}
		:param async: {async}
		:return: {return_stat_scalar}, the last dimension is of shape (2)
		"""
		@delayed
		def finish(result):
			return result[...,1]
		return self._async(async, finish(self.minmax(expression, binby=binby, limits=limits, shape=shape, selection=selection, async=async)))

	@docsubst
	@stat_1d
	def median(self, expression, percentage=50., binby=[], limits=None, shape=default_shape, percentile_shape=1024*16, percentile_limits="minmax", selection=False, async=False):
		"""Calculate the median , possible on a grid defined by binby

		NOTE: this value is approximated by calculating the cumulative distribution on a grid defined by
		percentile_shape and percentile_limits


		:param expression: {expression}
		:param binby: {binby}
		:param limits: {limits}
		:param shape: {shape}
		:param percentile_limits: {percentile_limits}
		:param percentile_shape: {percentile_shape}
		:param selection: {selection}
		:param async: {async}
		:return: {return_stat_scalar}
		"""
		return self.percentile(expression, 50, binby=binby, limits=limits, shape=shape, percentile_shape=percentile_shape, percentile_limits=percentile_limits, selection=selection, async=async)

	@docsubst
	def percentile(self, expression, percentage=50., binby=[], limits=None, shape=default_shape, percentile_shape=1024*16, percentile_limits="minmax", selection=False, async=False):
		"""Calculate the percentile given by percentage, possible on a grid defined by binby

		NOTE: this value is approximated by calculating the cumulative distribution on a grid defined by
		percentile_shape and percentile_limits


		>>> ds.percentile("x", 10), ds.percentile("x", 90)
		(array([-8.3220355]), array([ 7.92080358]))
		>>> ds.percentile("x", 50, binby="x", shape=5, limits=[-10, 10])
		array([[-7.56462982],
			   [-3.61036641],
			   [-0.01296306],
			   [ 3.56697863],
			   [ 7.45838367]])


		:param expression: {expression}
		:param binby: {binby}
		:param limits: {limits}
		:param shape: {shape}
		:param percentile_limits: {percentile_limits}
		:param percentile_shape: {percentile_shape}
		:param selection: {selection}
		:param async: {async}
		:return: {return_stat_scalar}
		"""
		if not isinstance(binby, (tuple, list)):
			binby = [binby]
		else:
			binby = binby
		@delayed
		def calculate(expression, shape, limits):
			#print(binby + [expression], shape, limits)
			task =  TaskStatistic(self, [expression] + binby, shape, limits, op=OP_ADD1, selection=selection)
			self.executor.schedule(task)
			return task
		@delayed
		def finish(percentile_limits, *counts_list):
			medians = []
			for i, counts in enumerate(counts_list):
				counts = counts[0]
				#print("percentile_limits", percentile_limits)
				#print("counts=", counts)
				#print("counts shape=", counts.shape)
				# F is the 'cumulative distribution'
				F = np.cumsum(counts, axis=0)
				# we'll fill empty values with nan later on..
				ok = F[-1,...] > 0
				F /= np.max(F, axis=(0))
				#print(F[-1])
				# find indices around 0.5 for each bin
				i2 = np.apply_along_axis(lambda x: x.searchsorted(percentage/100., side='left'), axis = 0, arr = F)
				i1 = i2 - 1
				i1 = np.clip(i1, 0, percentile_shapes[i]-1)
				i2 = np.clip(i2, 0, percentile_shapes[i]-1)

				# interpolate between i1 and i2
				#print("cum", F)
				#print("i1", i1)
				#print("i2", i2)
				pmin, pmax = percentile_limits[i]

				# np.choose seems buggy, use the equivalent code instead
				#a = i1
				#c = F
				F1 = np.array([F[i1[I]][I] for I in np.ndindex(i1.shape)])
				F1 = F1.reshape(F.shape[1:])

				#a = i2
				F2 = np.array([F[i2[I]][I] for I in np.ndindex(i2.shape)])
				F2 = F2.reshape(F.shape[1:])

				#print("F1,2", F1, F2)

				offset = (percentage/100.-F1)/(F2-F1)
				median = pmin + (i1+offset) / float(percentile_shapes[i]-1.) * (pmax-pmin)
				#print("offset", offset)
				#print(pmin + (i1+offset) / float(percentile_shapes[i]-1.) * (pmax-pmin))
				#print(pmin + (i1) / float(percentile_shapes[i]-1.) * (pmax-pmin))
				#print(median)

				# empty values should be set to nan
				median[~ok] = np.nan
				medians.append(median)
			value = np.array(vaex.utils.unlistify(waslist, medians))
			return value
		waslist, [expressions, ] = vaex.utils.listify(expression)
		shape = _expand_shape(shape, len(binby))
		percentile_shapes = _expand_shape(percentile_shape, len(expressions))
		if percentile_limits:
			percentile_limits = _expand_limits(percentile_limits, len(expressions))
		limits = self.limits(binby, limits, selection=selection, async=True)
		percentile_limits = self.limits(expressions, percentile_limits, selection=selection, async=True)
		@delayed
		def calculation(limits, percentile_limits):
			tasks = [calculate(expression, (percentile_shape, ) + tuple(shape), list(percentile_limits) + list(limits))
					 for    percentile_shape,  percentile_limit, expression
					 in zip(percentile_shapes, percentile_limits, expressions)]
			return finish(percentile_limits, delayed_args(*tasks))
			#return tasks
		result = calculation(limits, percentile_limits)
		return self._async(async, result)

	def _async(self, async, task, progressbar=False):
		if async:
			return task
		else:
			self.executor.execute()
			return task.get()

	@docsubst
	def limits_percentage(self, expression, percentage=99.73, square=False, async=False):
		"""Calculate the [min, max] range for expression, containing approximately a percentage of the data as defined
		by percentage.

		The range is symmetric around the median, i.e., for a percentage of 90, this gives the same results as:


		>>> ds.limits_percentage("x", 90)
		array([-12.35081376,  12.14858052]
		>>> ds.percentile("x", 5), ds.percentile("x", 95)
		(array([-12.36813152]), array([ 12.13275818]))

		NOTE: this value is approximated by calculating the cumulative distribution on a grid.
		NOTE 2: The values above are not exactly the same, since percentile and limits_percentage do not share the same code

		:param expression: {expression_limits}
		:param float percentage: Value between 0 and 100
		:param async: {async}
		:return: {return_limits}
		"""
		#percentiles = self.percentile(expression, [100-percentage/2, 100-(100-percentage/2.)], async=True)
		#return self._async(async, percentiles)
		#print(percentage)
		logger.info("limits_percentage for %r, with percentage=%r", expression, percentage)
		waslist, [expressions,] = vaex.utils.listify(expression)
		limits = []
		for expr in expressions:
			subspace = self(expr)
			limits_minmax = subspace.minmax()
			vmin, vmax = limits_minmax[0]
			size = 1024*16
			counts = subspace.histogram(size=size, limits=limits_minmax)
			cumcounts = np.concatenate([[0], np.cumsum(counts)])
			cumcounts /= cumcounts.max()
			# TODO: this is crude.. see the details!
			f = (1-percentage/100.)/2
			x = np.linspace(vmin, vmax, size+1)
			l = scipy.interp([f,1-f], cumcounts, x)
			limits.append(l)
		#return limits
		return vaex.utils.unlistify(waslist, limits)


	def __percentile_old(self, expression, percentage=99.73, selection=False):
		limits = []
		waslist, percentages = vaex.utils.listify(percentage)
		values = []
		for percentage in percentages:
			subspace = self(expression)
			if selection:
				subspace = subspace.selected()
			limits_minmax = subspace.minmax()
			vmin, vmax = limits_minmax[0]
			size = 1024*16
			counts = subspace.histogram(size=size, limits=limits_minmax)
			cumcounts = np.concatenate([[0], np.cumsum(counts)])
			cumcounts /= cumcounts.max()
			# TODO: this is crude.. see the details!
			f = percentage/100.
			x = np.linspace(vmin, vmax, size+1)
			l = scipy.interp([f], cumcounts, x)
			values.append(l[0])
		return vaex.utils.unlistify(waslist, values)

	@docsubst
	def limits(self, expression, value=None, square=False, selection=None, async=False):
		"""Calculate the [min, max] range for expression, as described by value, which is '99.7%' by default.

		If value is a list of the form [minvalue, maxvalue], it is simply returned, this is for convenience when using mixed
		forms.

		Example:

		>>> ds.limits("x")
		array([-28.86381927,  28.9261226 ])
		>>> ds.limits(["x", "y"])
		(array([-28.86381927,  28.9261226 ]), array([-28.60476934,  28.96535249]))
		>>> ds.limits(["x", "y"], "minmax")
		(array([-128.293991,  271.365997]), array([ -71.5523682,  146.465836 ]))
		>>> ds.limits(["x", "y"], ["minmax", "90%"])
		(array([-128.293991,  271.365997]), array([-13.37438402,  13.4224423 ]))
		>>> ds.limits(["x", "y"], ["minmax", [0, 10]])
		(array([-128.293991,  271.365997]), [0, 10])

		:param expression: {expression_limits}
		:param value: {limits}
		:param selection: {selection}
		:param async: {async}
		:return: {return_limits}
		"""
		if expression == []:
			return []
		waslist, [expressions, ] = vaex.utils.listify(expression)
		#values =
		#values = _expand_limits(value, len(expressions))
		#logger.debug("limits %r", list(zip(expressions, values)))
		if value is None:
			value = "99.73%"
		#print("value is seq/limit?", _issequence(value), _is_limit(value), value)
		if _is_limit(value) or not _issequence(value):
			values = (value,) * len(expressions)
		else:
			values = value

		#print("expressions 1)", expressions)
		#print("values      1)", values)

		initial_expressions, initial_values = expressions, values
		expression_values = dict()
		for expression, value in zip(expressions, values):
			#print(">>>", expression, value)
			if _issequence(expression):
				expressions = expression
			else:
				expressions = [expression]
			if _is_limit(value) or not _issequence(value):
				values = (value,) * len(expressions)
			else:
				values = value
			#print("expressions 2)", expressions)
			#print("values      2)", values)
			for expression, value in zip(expressions, values):
				if not _is_limit(value): # if a
					#value = tuple(value) # list is not hashable
					expression_values[(expression, value)] = None

		#print("##### 1)", expression_values.keys())

		limits_list = []
		#for expression, value in zip(expressions, values):
		for expression, value in expression_values.keys():
			if isinstance(value, six.string_types):
				if value == "minmax":
					limits = self.minmax(expression, selection=selection, async=True)
				else:
					import re
					match = re.match("([\d.]*)(\D*)", value)
					if match is None:
						raise ValueError("do not understand limit specifier %r, examples are 90%, 3sigma")
					else:
						number, type = match.groups()
						import ast
						number = ast.literal_eval(number)
						type = type.strip()
						if type in ["s", "sigma"]:
							limits =  self.limits_sigma(number)
						elif type in ["ss", "sigmasquare"]:
							limits =  self.limits_sigma(number, square=True)
						elif type in ["%", "percent"]:
							limits =  self.limits_percentage(expression, number, async=True)
						elif type in ["%s", "%square", "percentsquare"]:
							limits =  self.limits_percentage(expression, number, square=True, async=async)
			elif value is None:
				limits = self.limits_percentage(expression, square=square, async=True)
			else:
				limits =  value
			limits_list.append(limits)
			if limits is None:
				raise ValueError("limit %r not understood" % value)
			expression_values[(expression, value)] = limits
			logger.debug("!!!!!!!!!! limits: %r %r", limits, np.array(limits).shape)
			@delayed
			def echo(limits):
				logger.debug(">>>>>>>> limits: %r %r", limits, np.array(limits).shape)
			echo(limits)

		limits_list = delayed_args(*limits_list)
		@delayed
		def finish(limits_list):
			#print("##### 2)", expression_values.keys())
			limits_outer = []
			for expression, value in zip(initial_expressions, initial_values):
				#print(">>>3", expression, value)
				if _issequence(expression):
					expressions = expression
					waslist2 = True
				else:
					expressions = [expression]
					waslist2 = False
				if _is_limit(value) or not _issequence(value):
					values = (value,) * len(expressions)
				else:
					values = value
				#print("expressions 3)", expressions)
				#print("values      3)", values)
				limits = []
				for expression, value in zip(expressions, values):
					#print("get", (expression, value))
					if not _is_limit(value):
						value = expression_values[(expression, value)]
						if not _is_limit(value):
							#print(">>> value", value)
							value = value.get()
					limits.append(value)
					#if not _is_limit(value): # if a
					#	#value = tuple(value) # list is not hashable
					#	expression_values[(expression, value)] = expression_values[(expression, value)].get()
					#else:
					#	#value = tuple(value) # list is not hashable
					#	expression_values[(expression, value)] = ()
				if waslist2:
					limits_outer.append(limits)
				else:
					limits_outer.append(limits[0])
			#logger.debug(">>>>>>>> complete list of limits: %r %r", limits_list, np.array(limits_list).shape)

			#print("limits", limits_outer)
			return vaex.utils.unlistify(waslist, limits_outer)
		return self._async(async, finish(limits_list))

	def __minmax__old(self, expression, progressbar=False, selection=None):
		waslist, expressions = vaex.utils.listify(expression)
		subspace = self(*expressions)
		if selection:
			subspace = subspace.selected()
		return vaex.utils.unlistify(waslist, subspace.minmax())

	def histogram(self, expressions, limits, shape=256, weight=None, progressbar=False, selection=None):
		subspace = self(*expressions)
		if selection:
			subspace = subspace.selected()
		return subspace.histogram(limits=limits, size=shape, weight=weight)#, progressbar=progressbar)

	def __mean_old(self, expression, binby=[], limits=None, shape=256, progressbar=False, selection=None):
		if len(binby) == 0:
			subspace = self(expression)
			if selection:
				subspace = subspace.selected()
			return subspace.mean()[0]
		else:
			# todo, fix progressbar into two...
			limits = self.limits(binby, limits)
			subspace = self(*binby)
			if selection:
				subspace = subspace.selected()
			counts = self.count(expression, binby=binby, limits=limits, shape=shape, progressbar=progressbar, selection=selection)
			summed = self.sum  (expression, binby=binby, limits=limits, shape=shape, progressbar=progressbar, selection=selection)
			mean = summed/counts
			mean[counts==0] = np.nan
			return mean

	def mode(self, expression, binby=[], limits=None, shape=256, mode_shape=64, mode_limits=None, progressbar=False, selection=None):
		if len(binby) == 0:
			raise ValueError("only supported with binby argument given")
		else:
			# todo, fix progressbar into two...
			try:
				len(shape)
				shape = tuple(shape)
			except:
				shape = len(binby) * (shape,)
			shape = (mode_shape,) + shape
			subspace = self(*(list(binby) + [expression]))
			if selection:
				subspace = subspace.selected()

			limits = self.limits(list(binby), limits)
			mode_limits = self.limits([expression], mode_limits)
			limits = list(limits) + list(mode_limits)
			counts = subspace.histogram(limits=limits, size=shape, progressbar=progressbar)

			indices = np.argmax(counts, axis=0)
			pmin, pmax = limits[-1]
			centers = np.linspace(pmin, pmax, mode_shape+1)[:-1]# ignore last bin
			centers += (centers[1] - centers[0])/2 # and move half a bin to the right

			modes = centers[indices]
			ok = counts.sum(axis=0) > 0
			modes[~ok] = np.nan
			return modes

	def __sum_old(self, expression, binby=[], limits=None, shape=256, progressbar=False, selection=None):
		if len(binby) == 0:
			subspace = self(expression)
			if selection:
				subspace = subspace.selected()
			return subspace.sum()[0]
		else:
			# todo, fix progressbar into two...
			subspace = self(*binby)
			if selection:
				subspace = subspace.selected()
			summed = subspace.histogram(limits=limits, size=shape, progressbar=progressbar,
									weight=expression)
			return summed

	def __count_old(self, expression=None, binby=[], limits=None, shape=256, progressbar=False, selection=None):
		if len(binby) == 0:
			subspace = self("(%s)*0+1" % expression)
			if selection:
				subspace = subspace.selected()
			return subspace.sum()[0]
		else:
			limits = self.limits(list(binby), limits)
			subspace = self(*binby)
			if selection:
				subspace = subspace.selected()
			summed = subspace.histogram(limits=limits, size=shape, progressbar=progressbar,
									weight=("(%s)*0+1" % expression) if expression is not None else None)
			return summed

	def plot_bq(self, x, y, grid=None, size=256, limits=None, square=False, center=None, weight=None, figsize=None,
			 aspect="auto", f="identity", fig=None, axes=None, xlabel=None, ylabel=None, title=None,
			 group_by=None, group_limits=None, group_colors='jet', group_labels=None, group_count=None,
			 cmap="afmhot", scales=None, tool_select=False, bq_cleanup=True,
			 **kwargs):
		"""Use bqplot to create an interactive plot, this method is subject to change, it is currently a tech demo"""
		subspace = self(x, y)
		return subspace.plot_bq(grid, size, limits, square, center, weight, figsize, aspect, f, fig, axes, xlabel, ylabel, title,
								group_by, group_limits, group_colors, group_labels, group_count, cmap, scales, tool_select, bq_cleanup, **kwargs)


	def healpix_count(self, expression=None, healpix_expression="source_id/34359738368", healpix_max_level=12, healpix_level=8, binby=None, limits=None, shape=default_shape, **kwargs):
		#if binby is None:
		import healpy as hp
		reduce_level = healpix_max_level - healpix_level
		NSIDE = 2**healpix_level
		nmax = hp.nside2npix(NSIDE)
		scaling = 4**reduce_level
		expr = "%s/%s" % (healpix_expression, scaling)
		binby = [expr] + ([] if binby is None else _ensure_list(binby))
		shape = (nmax,) + _expand_shape(shape, len(binby)-1)
		limits = [[0, nmax]] + ([] if limits is None else limits)
		return self.count(expression, binby=binby, limits=limits, shape=shape, **kwargs)

	def healpix_plot(self, healpix_expression="source_id/34359738368", healpix_max_level=12, healpix_level=8, what="count(*)", selection=None,
					 grid=None,
					 healpix_input="equatorial", healpix_output="galactic", f=None,
					 colormap="afmhot", grid_limits=None, image_size =800, nest=True,
					 figsize=None, interactive=False,title="", smooth=None, show=False,
					 rotation=(0,0,0)):
		#plot_level = healpix_level #healpix_max_level-reduce_level
		import healpy as hp
		import pylab as plt
		if grid is None:
			reduce_level = healpix_max_level - healpix_level
			NSIDE=2**healpix_level
			nmax = hp.nside2npix(NSIDE)
			#print nmax, np.sqrt(nmax)
			scaling = 4**reduce_level
			#print nmax
			grid = self._stat(what=what, binby="%s/%s" % (healpix_expression, scaling), limits=[0., nmax], shape=nmax, selection=selection)
		if grid_limits:
			grid_min, grid_max = grid_limits
		else:
			grid_min = grid_max = None
		f_org = f
		f = _parse_f(f)
		if smooth:
			if nest:
				grid = hp.reorder(grid, inp="NEST", out="RING")
				nest = False
			#grid[np.isnan(grid)] = np.nanmean(grid)
			grid = hp.smoothing(grid, sigma=np.radians(smooth))
		fgrid = f(grid)
		coord_map = dict(equatorial='C', galactic='G', ecliptic="E")
		fig = plt.gcf()
		if figsize is not None:
			fig.set_size_inches(*figsize)
		what_label = what
		if f_org:
			what_label = f_org + " " + what_label
		f = hp.mollzoom if interactive else hp.mollview
		f(fgrid, unit=what_label, rot=rotation, nest=nest ,title=title, coord=[coord_map[healpix_input], coord_map[healpix_output]], cmap=colormap, hold=True, xsize=image_size,
					min=grid_min, max=grid_max)#, min=6-1, max=8.7-1)
		if show:
			plt.show()

	@docsubst
	@stat_1d
	def _stat(self, what="count(*)", what_kwargs={}, binby=[], limits=None, shape=default_shape, selection=False, async=False):
		"""Calculate the sum for the given expression, possible on a grid defined by binby

		Examples:

		>>> ds.sum("L")
		304054882.49378014
		>>> ds.sum("L", binby="E", shape=4)
		array([  8.83517994e+06,   5.92217598e+07,   9.55218726e+07,
				 1.40008776e+08])

		:param expression: {expression}
		:param binby: {binby}
		:param limits: {limits}
		:param shape: {shape}
		:param selection: {selection}
		:param async: {async}
		:return: {return_stat_scalar}
		"""
		waslist_what, [whats,] = vaex.utils.listify(what)
		limits = self.limits(binby, limits, async=True)
		waslist_selection, [selections] = vaex.utils.listify(selection)
		binby = _ensure_list(binby)

		what_labels = []
		shape = _expand_shape(shape, len(binby))
		total_grid = np.zeros( (len(whats), len(selections)) + shape, dtype=float)
		@delayed
		def copy_grids(grids):
			total_grid[index] = grid
		@delayed
		def get_whats(limits):
			grids = []
			for j, what in enumerate(whats):
				what = what.strip()
				index = what.index("(")
				import re
				groups = re.match("(.*)\((.*)\)", what).groups()
				if groups and len(groups) == 2:
					function = groups[0]
					arguments = groups[1].strip()
					if "," in arguments:
						arguments = arguments.split(",")
					functions = ["mean", "sum", "std", "var", "correlation", "covar", "min", "max"]
					unit_expression = None
					if function in ["mean", "sum", "std", "min", "max"]:
						unit_expression = arguments
					if function in ["var"]:
						unit_expression = "(%s) * (%s)" % (arguments, arguments)
					if function in ["covar"]:
						unit_expression = "(%s) * (%s)" % arguments
					if unit_expression:
						unit = self.unit(unit_expression)
						if unit:
							what_units = unit.to_string('latex_inline')
					if function in functions:
						grid = getattr(self, function)(arguments, binby=binby, limits=limits, shape=shape, selection=selections)
					elif function == "count":
						grid = self.count(arguments, binby, shape=shape, limits=limits, selection=selections)
					else:
						raise ValueError("Could not understand method: %s, expected one of %r'" % (function, functions))
					#what_labels.append(what_label)
					grids.append(grid)

			#else:
			#	raise ValueError("Could not understand 'what' argument %r, expected something in form: 'count(*)', 'mean(x)'" % what)
			return grids

		grids = get_whats(limits)
		#print grids
		#grids = delayed_args(*grids)
		@delayed
		def finish(grids):
			for i, grid in enumerate(grids):
				total_grid[i] = grid
			return total_grid[slice(None, None, None) if waslist_what else 0, slice(None, None, None) if waslist_selection else 0]
		s = finish(grids)
		return self._async(async, s)

	#def plot(self, x=None, y=None, z=None, axes=[], row=None, agg=None, extra=["selection:none,default"], reduce=["colormap", "stack.fade"], f="log", n="normalize", naxis=None,
	def plot(self, x=None, y=None, z=None, what="count(*)", vwhat=None, reduce=["colormap"], f=None,
			 normalize="normalize", normalize_axis="what",
			 vmin=None, vmax=None,
			 shape=256, vshape=32, limits=None, grid=None, colormap="afmhot", # colors=["red", "green", "blue"],
			figsize=None, xlabel=None, ylabel=None, aspect="auto", tight_layout=True, interpolation="nearest", show=False,
			colorbar=True,
			selection=None, selection_labels=None, title=None,
		 	background_color="white", pre_blend=False, background_alpha=1.,
			visual=dict(x="x", y="y", layer="z", fade="selection", row="subspace", column="what"),
			smooth_pre=None, smooth_post=None,
			wrap=True, wrap_columns=4,
			return_extra=False, hardcopy=None):
		"""Declarative plotting of statistical plots using matplotlib, supports subplots, selections, layers

		Instead of passing x and y, pass a list as x argument for multiple panels. Give what a list of options to have multiple
		panels. When both are present then will be origanized in a column/row order.

		This methods creates a 6 dimensional 'grid', where each dimension can map the a visual dimension.
		The grid dimensions are:

		 * x: shape determined by shape, content by x argument or the first dimension of each space
		 * y:   ,,
		 * z:  related to the z argument
		 * selection: shape equals length of selection argument
		 * what: shape equals length of what argument
		 * space: shape equals length of x argument if multiple values are given

		 By default, this its shape is (1, 1, 1, 1, shape, shape) (where x is the last dimension)

		The visual dimensions are

		 * x: x coordinate on a plot / image (default maps to grid's x)
		 * y: y   ,,                         (default maps to grid's y)
		 * layer: each image in this dimension is blended togeher to one image (default maps to z)
		 * fade: each image is shown faded after the next image (default mapt to selection)
		 * row: rows of subplots (default maps to space)
		 * columns: columns of subplot (default maps to what)

		All these mappings can be changes by the visual argument, some examples:

		>>> ds.plot('x', 'y', what=['mean(x)', 'correlation(vx, vy)'])

		Will plot each 'what' as a column

		>>> ds.plot('x', 'y', selection=['FeH < -3', '(FeH >= -3) & (FeH < -2)'], visual=dict(column='selection'))

		Will plot each selection as a column, instead of a faded on top of each other.





		:param x: Expression to bin in the x direction (by default maps to x), or list of pairs, like [['x', 'y'], ['x', 'z']], if multiple pairs are given, this dimension maps to rows by default
		:param y:                          y           (by default maps to y)
		:param z: Expression to bin in the z direction, followed by a :start,end,shape  signature, like 'FeH:-3,1:5' will produce 5 layers between -10 and 10 (by default maps to layer)
		:param what: What to plot, count(*) will show a N-d histogram, mean('x'), the mean of the x column, sum('x') the sum, std('x') the standard deviation, correlation('vx', 'vy') the correlation coefficient. Can also be a list of values, like ['count(x)', std('vx')], (by default maps to column)
		:param reduce:
		:param f: transform values by: 'identity' does nothing 'log' or 'log10' will show the log of the value
		:param normalize: normalization function, currently only 'normalize' is supported
		:param normalize_axis: which axes to normalize on, None means normalize by the global maximum.
		:param vmin: instead of automatic normalization, (using normalize and normalization_axis) scale the data between vmin and vmax to [0, 1]
		:param vmax: see vmin
		:param shape: shape/size of the n-D histogram grid
		:param limits: list of [[xmin, xmax], [ymin, ymax]], or a description such as 'minmax', '99%'
		:param grid: if the binning is done before by yourself, you can pass it
		:param colormap: matplotlib colormap to use
		:param figsize: (x, y) tuple passed to pylab.figure for setting the figure size
		:param xlabel:
		:param ylabel:
		:param aspect:
		:param tight_layout: call pylab.tight_layout or not
		:param colorbar: plot a colorbar or not
		:param interpolation: interpolation for imshow, possible options are: 'nearest', 'bilinear', 'bicubic', see matplotlib for more
		:param return_extra:
		:return:
		"""
		import pylab
		import matplotlib
		n = _parse_n(normalize)
		if type(shape) == int:
			shape = (shape,) * 2
		binby = []
		for expression in [y,x]:
			if expression is not None:
				binby = [expression] + binby
		fig = pylab.gcf()
		if figsize is not None:
			fig.set_size_inches(*figsize)
		import re

		what_units = None
		whats = _ensure_list(what)
		selections = _ensure_list(selection)

		if y is None:
			waslist, [x,] = vaex.utils.listify(x)
		else:
			waslist, [x,y] = vaex.utils.listify(x, y)
			x = list(zip(x, y))
			limits = [limits]

		# every plot has its own vwhat for now
		vwhats = _expand_limits(vwhat, len(x)) # TODO: we're abusing this function..
		logger.debug("x: %s", x)
		limits = self.limits(x, limits)
		logger.debug("limits: %r", limits)

		labels = {}
		shape = _expand_shape(shape, 2)
		vshape = _expand_shape(shape, 2)
		if z is not None:
			match = re.match("(.*):(.*),(.*),(.*)", z)
			if match:
				groups = match.groups()
				import ast
				z_expression = groups[0]
				logger.debug("found groups: %r", list(groups))
				z_limits = [ast.literal_eval(groups[1]), ast.literal_eval(groups[2])]
				z_shape = ast.literal_eval(groups[3])
				#for pair in x:
				x = [[z_expression] + list(k) for k in x]
				limits = np.array([[z_limits]  + list(k) for k in limits])
				shape =  (z_shape,)+ shape
				vshape =  (z_shape,)+ vshape
				logger.debug("x = %r", x)
				values = np.linspace(z_limits[0], z_limits[1], num=z_shape+1)
				labels["z"] = list(["%s <= %s < %s" % (v1, z_expression, v2) for v1, v2 in zip(values[:-1], values[1:])])
			else:
				raise ValueError("Could not understand 'z' argument %r, expected something in form: 'column:-1,10:5'" % facet)
		else:
			z_shape = 1


		# z == 1
		if z is None:
			total_grid = np.zeros( (len(x), len(whats), len(selections), 1) + shape, dtype=float)
			total_vgrid = np.zeros( (len(x), len(whats), len(selections), 1) + vshape, dtype=float)
		else:
			total_grid = np.zeros( (len(x), len(whats), len(selections)) + shape, dtype=float)
			total_vgrid = np.zeros( (len(x), len(whats), len(selections)) + vshape, dtype=float)
		logger.debug("shape of total grid: %r", total_grid.shape)
		axis = dict(plot=0, what=1, selection=2)
		xlimits = limits
		if xlabel is None:
			xlabels = []
			ylabels = []
			for i, (binby, limits) in enumerate(zip(x, xlimits)):
				xlabels.append(self.label(binby[0]))
				ylabels.append(self.label(binby[1]))
		else:
			xlabels = _expand(xlabel, len(x))
			ylabels = _expand(ylabel, len(x))
		labels["subspace"] = (xlabels, ylabels)

		grid_axes = dict(x=-1, y=-2, z=-3, selection=-4, what=-5, subspace=-6)
		visual_axes = dict(x=-1, y=-2, layer=-3, fade=-4, column=-5, row=-6)
		#visual_default=dict(x="x", y="y", z="layer", selection="fade", subspace="row", what="column")
		visual_default=dict(x="x", y="y", layer="z", fade="selection", row="subspace", column="what")
		invert = lambda x: dict((v, k) for k, v in x.iteritems())
		#visual_default_reverse = invert(visual_default)
		#visual_ = visual_default
		#visual = dict(visual) # copy for modification
		# add entries to avoid mapping multiple times to the same axis
		free_visual_axes = visual_default.keys()
		#visual_reverse = invert(visual)
		logger.debug("1: %r %r", visual, free_visual_axes)
		for visual_name, grid_name in visual.items():
			if visual_name in free_visual_axes:
				free_visual_axes.remove(visual_name)
			else:
				raise ValueError("visual axes %s used multiple times" % visual_name)
		logger.debug("2: %r %r", visual, free_visual_axes)
		for visual_name, grid_name in visual_default.items():
			if visual_name in free_visual_axes and grid_name not in visual.values():
				free_visual_axes.remove(visual_name)
				visual[visual_name] = grid_name
		logger.debug("3: %r %r", visual, free_visual_axes)
		for visual_name, grid_name in visual_default.items():
			if visual_name not in free_visual_axes and grid_name not in visual.values():
				visual[free_visual_axes.pop(0)] = grid_name

		logger.debug("4: %r %r", visual, free_visual_axes)


		visual_reverse = invert(visual)
		# TODO: the meaning of visual and visual_reverse is changed below this line, super confusing
		visual, visual_reverse = visual_reverse, visual
		move = {}
		for grid_name, visual_name in visual.items():
			if visual_axes[visual_name] in visual.values():
				index = visual.values().find(visual_name)
				key = visual.keys()[index]
				raise ValueError("trying to map %s to %s while, it is already mapped by %s" % (grid_name, visual_name, key))
			move[grid_axes[grid_name]] = visual_axes[visual_name]
		logger.debug("grid shape: %r", total_grid.shape)
		logger.debug("visual: %r", visual.items())
		#normalize_axis = _ensure_list(normalize_axis)

		fs = _expand(f, total_grid.shape[grid_axes[normalize_axis]])
		#assert len(vwhat)
		#labels["y"] = ylabels
		what_labels = []
		if grid is None:
			for i, (binby, limits) in enumerate(zip(x, xlimits)):
				for j, what in enumerate(whats):
					if what:
						what = what.strip()
						index = what.index("(")
						import re
						groups = re.match("(.*)\((.*)\)", what).groups()
						if groups and len(groups) == 2:
							function = groups[0]
							arguments = groups[1].strip()
							if "," in arguments:
								arguments = arguments.split(",")
							functions = ["mean", "sum", "std", "var", "correlation", "covar", "min", "max", "median"]
							unit_expression = None
							if function in ["mean", "sum", "std", "min", "max", "median"]:
								unit_expression = arguments
							if function in ["var"]:
								unit_expression = "(%s) * (%s)" % (arguments, arguments)
							if function in ["covar"]:
								unit_expression = "(%s) * (%s)" % arguments
							if unit_expression:
								unit = self.unit(unit_expression)
								if unit:
									what_units = unit.to_string('latex_inline')
							if function in functions:
								grid = getattr(self, function)(arguments, binby=binby, limits=limits, shape=shape, selection=selections)
							elif function == "count":
								grid = self.count(arguments, binby, shape=shape, limits=limits, selection=selections)
							else:
								raise ValueError("Could not understand method: %s, expected one of %r'" % (function, functions))
							if i == 0:# and j == 0:
								what_label = whats[j]
								if what_units:
									what_label += " (%s)" % what_units
								if fs[j]:
									what_label = fs[j] + " " + what_label
								what_labels.append(what_label)
						else:
							raise ValueError("Could not understand 'what' argument %r, expected something in form: 'count(*)', 'mean(x)'" % what)
					else:
						grid = self.histogram(binby, size=shape, limits=limits, selection=selection)
					total_grid[i,j,:,:] = grid[:,None,...]
			labels["what"] = what_labels
		else:
			total_grid = np.broadcast_to(grid, (1,) * 4 + grid.shape)

		#			visual=dict(x="x", y="y", selection="fade", subspace="facet1", what="facet2",)
		def _selection_name(name):
			if name in [None, False]:
				return "selection: all"
			elif name in ["default", True]:
				return "selection: default"
			else:
				return "selection: %s" % name
		if selection_labels is None:
			labels["selection"] = list([_selection_name(k) for k in selections])
		else:
			labels["selection"] = selection_labels

		visual_grid = np.moveaxis(total_grid, move.keys(), move.values())
		logger.debug("visual grid shape: %r", visual_grid.shape)
		#grid = total_grid
		#print(grid.shape)
		#grid = self.reduce(grid, )
		axes = []
		#cax = pylab.subplot(1,1,1)

		background_color = np.array(matplotlib.colors.colorConverter.to_rgb(background_color))


		#if grid.shape[axis["selection"]] > 1:#  and not facet:
		#	rgrid = vaex.image.fade(rgrid)
		#	finite_mask = np.any(finite_mask, axis=0) # do we really need this
		#	print(rgrid.shape)
		#facet_row_axis = axis["what"]
		import math
		facet_columns = None
		facets = visual_grid.shape[visual_axes["row"]] * visual_grid.shape[visual_axes["column"]]
		if visual_grid.shape[visual_axes["column"]] ==  1 and wrap:
			facet_columns = min(wrap_columns, visual_grid.shape[visual_axes["row"]])
			wrapped = True
		elif visual_grid.shape[visual_axes["row"]] ==  1 and wrap:
			facet_columns = min(wrap_columns, visual_grid.shape[visual_axes["column"]])
			wrapped = True
		else:
			wrapped = False
			facet_columns = visual_grid.shape[visual_axes["column"]]
		facet_rows = int(math.ceil(facets/facet_columns))
		logger.debug("facet_rows: %r", facet_rows)
		logger.debug("facet_columns: %r", facet_columns)
			#if visual_grid.shape[visual_axes["row"]] > 1: # and not wrap:
			#	#facet_row_axis = axis["what"]
			#	facet_columns = visual_grid.shape[visual_axes["column"]]
			#else:
			#	facet_columns = min(wrap_columns, facets)
		#if grid.shape[axis["plot"]] > 1:#  and not facet:

		# this loop could be done using axis arguments everywhere
		#assert len(normalize_axis) == 1, "currently only 1 normalization axis supported"
		grid = visual_grid * 1.
		fgrid = visual_grid * 1.
		ngrid = visual_grid * 1.
		#colorgrid = np.zeros(ngrid.shape + (4,), float)
		#print "norma", normalize_axis, visual_grid.shape[visual_axes[visual[normalize_axis]]]
		vmins = _expand(vmin, visual_grid.shape[visual_axes[visual[normalize_axis]]], type=list)
		vmaxs = _expand(vmax, visual_grid.shape[visual_axes[visual[normalize_axis]]], type=list)
		#for name in normalize_axis:
		visual_grid
		if smooth_pre:
			grid = vaex.grids.gf(grid, smooth_pre)
		if 1:
			axis = visual_axes[visual[normalize_axis]]
			for i in range(visual_grid.shape[axis]):
				item = [slice(None, None, None), ] * len(visual_grid.shape)
				item[axis] = i
				item = tuple(item)
				f = _parse_f(fs[i])
				fgrid.__setitem__(item, f(grid.__getitem__(item)))
				#print vmins[i], vmaxs[i]
				if vmins[i] is not None and vmaxs[i] is not None:
					nsubgrid = fgrid.__getitem__(item) * 1
					nsubgrid -= vmins[i]
					nsubgrid /= (vmaxs[i]-vmins[i])
					nsubgrid = np.clip(nsubgrid, 0, 1)
				else:
					nsubgrid, vmin, vmax = n(fgrid.__getitem__(item))
					vmins[i] = vmin
					vmaxs[i] = vmax
				#print "    ", vmins[i], vmaxs[i]
				ngrid.__setitem__(item, nsubgrid)

		if 0: # TODO: above should be like the code below, with custom vmin and vmax
			grid = visual_grid[i]
			f = _parse_f(fs[i])
			fgrid = f(grid)
			finite_mask = np.isfinite(grid)
			finite_mask = np.any(finite_mask, axis=0)
			if vmin is not None and vmax is not None:
				ngrid = fgrid * 1
				ngrid -= vmin
				ngrid /= (vmax-vmin)
				ngrid = np.clip(ngrid, 0, 1)
			else:
				ngrid, vmin, vmax = n(fgrid)
				#vmin, vmax = np.nanmin(fgrid), np.nanmax(fgrid)
		# every 'what', should have its own colorbar, check if what corresponds to
		# rows or columns in facets, if so, do a colorbar per row or per column


		rows, columns = int(math.ceil(facets / float(facet_columns))), facet_columns
		colorbar_location = "individual"
		if visual["what"] == "row" and visual_grid.shape[1] == facet_columns:
			colorbar_location = "per_row"
		if visual["what"] == "column" and visual_grid.shape[0] == facet_rows:
			colorbar_location = "per_column"
		#values = np.linspace(facet_limits[0], facet_limits[1], facet_count+1)
		logger.debug("rows: %r, columns: %r", rows, columns)
		import matplotlib.gridspec as gridspec
		column_scale = 1
		row_scale = 1
		row_offset = 0
		if facets > 1:
			if colorbar_location == "per_row":
				column_scale = 4
				gs = gridspec.GridSpec(rows, columns*column_scale+1)
			elif colorbar_location == "per_column":
				row_offset = 1
				row_scale = 4
				gs = gridspec.GridSpec(rows*row_scale+1, columns)
			else:
				gs = gridspec.GridSpec(rows, columns)
		facet_index = 0
		fs = _expand(f, len(whats))
		colormaps = _expand(colormap, len(whats))

		# row
		for i in range(visual_grid.shape[0]):
			# column
			for j in range(visual_grid.shape[1]):
				if colorbar and colorbar_location == "per_column" and i == 0:
					norm = matplotlib.colors.Normalize(vmins[j], vmaxs[j])
					sm = matplotlib.cm.ScalarMappable(norm, colormaps[j])
					sm.set_array(1) # make matplotlib happy (strange behavious)
					if facets > 1:
						ax = pylab.subplot(gs[0, j])
						colorbar = fig.colorbar(sm, cax=ax, orientation="horizontal")
					else:
						colorbar = fig.colorbar(sm)
					if "what" in labels:
						label = labels["what"][j]
						if facets > 1:
							colorbar.ax.set_title(label)
						else:
							colorbar.ax.set_ylabel(label)

				if colorbar and colorbar_location == "per_row" and j == 0:
					norm = matplotlib.colors.Normalize(vmins[i], vmaxs[i])
					sm = matplotlib.cm.ScalarMappable(norm, colormaps[i])
					sm.set_array(1) # make matplotlib happy (strange behavious)
					if facets > 1:
						ax = pylab.subplot(gs[i, -1])
						colorbar = fig.colorbar(sm, cax=ax)
					else:
						colorbar = fig.colorbar(sm)
					label = labels["what"][i]
					colorbar.ax.set_ylabel(label)

				rgrid = ngrid[i,j] * 1.
				#print rgrid.shape
				for k in range(rgrid.shape[0]):
					for l in range(rgrid.shape[0]):
						if smooth_post is not None:
							rgrid[k,l] = vaex.grids.gf(rgrid, smooth_post)
				if visual["what"] == "column":
					what_index = j
				elif visual["what"] == "row":
					what_index = i
				else:
					what_index = 0


				if visual[normalize_axis] == "column":
					normalize_index = j
				elif visual[normalize_axis] == "row":
					normalize_index = i
				else:
					normalize_index = 0
				for r in reduce:
					r = _parse_reduction(r, colormaps[what_index], [])
					rgrid = r(rgrid)

				finite_mask = np.isfinite(ngrid[i,j])
				rgrid[~finite_mask,3] = 0
				row = facet_index / facet_columns
				column = facet_index % facet_columns

				if colorbar and colorbar_location == "individual":
					#visual_grid.shape[visual_axes[visual[normalize_axis]]]
					norm = matplotlib.colors.Normalize(vmins[normalize_index], vmaxs[normalize_index])
					sm = matplotlib.cm.ScalarMappable(norm, colormaps[what_index])
					sm.set_array(1) # make matplotlib happy (strange behavious)
					if facets > 1:
						ax = pylab.subplot(gs[row, column])
						colorbar = fig.colorbar(sm, ax=ax)
					else:
						colorbar = fig.colorbar(sm)
					label = labels["what"][what_index]
					colorbar.ax.set_ylabel(label)


				if facets > 1:
					ax = pylab.subplot(gs[row_offset + row * row_scale:row_offset + (row+1) * row_scale, column*column_scale:(column+1)*column_scale])
				else:
					ax = pylab.gca()
				axes.append(ax)
				logger.debug("rgrid: %r", rgrid.shape)
				plot_rgrid = rgrid
				assert plot_rgrid.shape[1] == 1, "no layers supported yet"
				plot_rgrid = plot_rgrid[:,0]
				if plot_rgrid.shape[0] > 1:
					plot_rgrid = vaex.image.fade(plot_rgrid[::-1])
				else:
					plot_rgrid = plot_rgrid[0]
				extend = None
				if visual["subspace"] == "row":
					subplot_index = i
				elif visual["subspace"] == "column":
					subplot_index = j
				else:
					subplot_index = 0
				extend = np.array(xlimits[subplot_index][-2:]).flatten()
				#	extend = np.array(xlimits[i]).flatten()
				logger.debug("plot rgrid: %r", plot_rgrid.shape)
				plot_rgrid = np.transpose(plot_rgrid, (1,0,2))
				im = ax.imshow(plot_rgrid, extent=extend.tolist(), origin="lower", aspect=aspect, interpolation=interpolation)
				#v1, v2 = values[i], values[i+1]
				def label(index, label, expression):
					if label and _issequence(label):
						return label[i]
					else:
						return self.label(expression)
				# we don't need titles when we have a colorbar
				if (visual_reverse["row"] != "what") or not colorbar:
					labelsxy = labels.get(visual_reverse["row"])
					has_title = False
					if isinstance(labelsxy, tuple):
						labelsx, labelsy = labelsxy
						pylab.xlabel(labelsx[i])
						pylab.ylabel(labelsy[i])
					elif labelsxy is not None:
						ax.set_title(labelsxy[i])
						has_title = True
					#print visual_reverse["row"], visual_reverse["column"], labels.get(visual_reverse["row"]), labels.get(visual_reverse["column"])
				if (visual_reverse["column"] != "what")  or not colorbar:
					labelsxy = labels.get(visual_reverse["column"])
					if isinstance(labelsxy, tuple):
						labelsx, labelsy = labelsxy
						pylab.xlabel(labelsx[j])
						pylab.ylabel(labelsy[j])
					elif labelsxy is not None and not has_title:
						ax.set_title(labelsxy[j])
						pass
				facet_index += 1
		if title:
			fig.suptitle(title, fontsize="x-large")
		if tight_layout:
			if title:
				pylab.tight_layout(rect=[0, 0.03, 1, 0.95])
			else:
				pylab.tight_layout()
		if hardcopy:
			pylab.savefig(hardcopy)
		if show:
			pylab.show()
		if return_extra:
			return im, grid, fgrid, ngrid, rgrid, rgba8
		else:
			return im
		#colorbar = None
		#return im, colorbar

	def plot1d(self, x=None, what="count(*)", grid=None, shape=64, facet=None, limits=None, figsize=None, f="identity", n=None, normalize_axis=None,
		xlabel=None, ylabel=None, label=None,
		selection=None, show=False, tight_layout=True, hardcopy=None,
			   **kwargs):
		"""

		:param x: Expression to bin in the x direction
		:param what: What to plot, count(*) will show a N-d histogram, mean('x'), the mean of the x column, sum('x') the sum
		:param grid:
		:param grid: if the binning is done before by yourself, you can pass it
		:param facet: Expression to produce facetted plots ( facet='x:0,1,12' will produce 12 plots with x in a range between 0 and 1)
		:param limits: list of [xmin, xmax], or a description such as 'minmax', '99%'
		:param figsize: (x, y) tuple passed to pylab.figure for setting the figure size
		:param f: transform values by: 'identity' does nothing 'log' or 'log10' will show the log of the value
		:param n: normalization function, currently only 'normalize' is supported, or None for no normalization
		:param normalize_axis: which axes to normalize on, None means normalize by the global maximum.
		:param normalize_axis:
		:param xlabel: String for label on x axis (may contain latex)
		:param ylabel: Same for y axis
		:param: tight_layout: call pylab.tight_layout or not
		:param kwargs: extra argument passed to pylab.plot
		:return:
		"""



		import pylab
		f = _parse_f(f)
		n = _parse_n(n)
		if type(shape) == int:
			shape = (shape,)
		binby = []
		for expression in [x]:
			if expression is not None:
				binby = [expression] + binby
		limits = self.limits(binby, limits)
		if figsize is not None:
			pylab.figure(num=None, figsize=figsize, dpi=80, facecolor='w', edgecolor='k')
		fig = pylab.gcf()
		import re
		if facet is not None:
			match = re.match("(.*):(.*),(.*),(.*)", facet)
			if match:
				groups = match.groups()
				import ast
				facet_expression = groups[0]
				facet_limits = [ast.literal_eval(groups[1]), ast.literal_eval(groups[2])]
				facet_count = ast.literal_eval(groups[3])
				limits.append(facet_limits)
				binby.append(facet_expression)
				shape = (facet_count,) + shape
			else:
				raise ValueError("Could not understand 'facet' argument %r, expected something in form: 'column:-1,10:5'" % facet)

		if grid is None:
			if what:
				what = what.strip()
				index = what.index("(")
				import re
				groups = re.match("(.*)\((.*)\)", what).groups()
				if groups and len(groups) == 2:
					function = groups[0]
					arguments = groups[1].strip()
					functions = ["mean", "sum"]
					if function in functions:
						grid = getattr(self, function)(arguments, binby, limits=limits, shape=shape, selection=selection)
					elif function == "count" and arguments == "*":
						grid = self.count(binby=binby, shape=shape, limits=limits, selection=selection)
					elif function == "cumulative" and arguments == "*":
						# TODO: comulative should also include the tails outside limits
						grid = self.count(binby=binby, shape=shape, limits=limits, selection=selection)
						grid = np.cumsum(grid)
					else:
						raise ValueError("Could not understand method: %s, expected one of %r'" % (function, functions))
				else:
					raise ValueError("Could not understand 'what' argument %r, expected something in form: 'count(*)', 'mean(x)'" % what)
			else:
				grid = self.histogram(binby, size=shape, limits=limits, selection=selection)
		fgrid = f(grid)
		if n is not None:
			#ngrid = n(fgrid, axis=normalize_axis)
			ngrid = fgrid / fgrid.sum()
		else:
			ngrid = fgrid
			#reductions = [_parse_reduction(r, colormap, colors) for r in reduce]
			#rgrid = ngrid * 1.
			#for r in reduce:
			#	r = _parse_reduction(r, colormap, colors)
			#	rgrid = r(rgrid)
			#grid = self.reduce(grid, )
		xmin, xmax = limits[-1]
		if facet:
			N = len(grid[-1])
		else:
			N = len(grid)
		xar = np.arange(N) / (N-1.0) * (xmax-xmin) + xmin
		if facet:
			import math
			rows, columns = int(math.ceil(facet_count / 4.)), 4
			values = np.linspace(facet_limits[0], facet_limits[1], facet_count+1)
			for i in range(facet_count):
				ax = pylab.subplot(rows, columns, i+1)
				value = ax.plot(xar, ngrid[i], drawstyle="steps", label=label or x, **kwargs)
				v1, v2 = values[i], values[i+1]
				pylab.xlabel(xlabel or x)
				pylab.ylabel(ylabel or what)
				ax.set_title("%3f <= %s < %3f" % (v1, facet_expression, v2))
				#pylab.show()
		else:
			#im = pylab.imshow(rgrid, extent=np.array(limits[:2]).flatten(), origin="lower", aspect=aspect)
			pylab.xlabel(xlabel or self.label(x))
			pylab.ylabel(ylabel or what)
			value = pylab.plot(xar, ngrid, drawstyle="steps", label=label or x, **kwargs)
		if tight_layout:
			pylab.tight_layout()
		if hardcopy:
			pylab.savefig(hardcopy)
		if show:
			pylab.show()
		return value
		#N = len(grid)
		#xmin, xmax = limits[0]
		#return pylab.plot(np.arange(N) / (N-1.0) * (xmax-xmin) + xmin, f(grid,), drawstyle="steps", **kwargs)
		#pylab.ylim(-1, 6)

	@property
	def stat(self):
		class StatList(object):
			pass
		statslist = StatList()
		for name in self.get_column_names(virtual=True):
			class Stats(object):
				pass
			stats = Stats()
			for f in _functions_statistics_1d:
				fname = f.__name__
				p = property(lambda self, name=name, f=f, ds=self: f(ds, name))
				setattr(Stats, fname, p)
			setattr(statslist, name, stats)
		return statslist

	@property
	def col(self):
		"""Gives direct access to the data as numpy-like arrays.

		Convenient when working with ipython in combination with small datasets, since this gives tab-completion

		Columns can be accesed by there names, which are attributes. The attribues are currently strings, so you cannot
		do computations with them

		:Example:
		>>> ds = vx.example()
		>>> ds.plot(ds.col.x, ds.col.y)

		"""
		class ColumnList(object):
			pass
		data = ColumnList()
		for name in self.get_column_names(virtual=True):
			setattr(data, name, name)
		return data


	def close_files(self):
		"""Close any possible open file handles, the dataset will not be in a usable state afterwards"""
		pass

	def byte_size(self, selection=False):
		"""Return the size in bytes the whole dataset requires (or the selection), respecting the active_fraction"""
		bytes_per_row = 0
		for column in list(self.get_column_names()):
			dtype = self.dtype(column)
			bytes_per_row += dtype.itemsize
		return bytes_per_row * self.count(selection=selection)


	def dtype(self, expression):
		if expression in self.columns.keys():
			return self.columns[expression].dtype
		else:
			return np.zeros(1, dtype=np.float64).dtype

	def label(self, expression, unit=None, output_unit=None, format="latex_inline"):
		label = expression
		unit = unit or self.unit(expression)
		try: # if we can convert the unit, use that for the labeling
			if output_unit and unit: # avoid unnecessary error msg'es
				output_unit.to(unit)
				unit = output_unit
		except:
			logger.exception("unit error")
		if unit is not None:
			label = "%s (%s)" % (label, unit.to_string('latex_inline')  )
		return label

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
			unit_or_quantity = eval(expression, expression_namespace, UnitScope(self))
			return unit_or_quantity.unit if hasattr(unit_or_quantity, "unit") else unit_or_quantity
		except:
			#logger.exception("error evaluating unit expression: %s", expression)
			# astropy doesn't add units, so we try with a quatiti
			try:
				return eval(expression, expression_namespace, UnitScope(self, 1.)).unit
			except:
				#logger.exception("error evaluating unit expression: %s", expression)
				return default

	def ucd_find(self, ucds, exclude=[]):
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
		if isinstance(ucds, six.string_types):
			ucds = [ucds]
		if len(ucds) == 1:
			ucd = ucds[0]
			if ucd[0] == "^": # we want it to start with
				ucd = ucd[1:]
				columns = [name for name in self.get_column_names(virtual=True) if self.ucds.get(name, "").startswith(ucd) and name not in exclude]
			else:
				columns = [name for name in self.get_column_names(virtual=True) if ucd in self.ucds.get(name, "") and name not in exclude]
			return None if len(columns) == 0 else columns[0]
		else:
			columns = [self.ucd_find([ucd], exclude=exclude) for ucd in ucds]
			return None if None in columns else columns

	def selection_favorite_add(self, name, selection_name="default"):
		selection = self.get_selection(name=selection_name)
		if selection:
			self.favorite_selections[name] = selection
			self.selections_favorite_store()
		else:
			raise ValueError("no selection exists")

	def selection_favorite_remove(self, name):
		del self.favorite_selections[name]
		self.selections_favorite_store()

	def selection_favorite_apply(self, name, selection_name="default", executor=None):
		self.set_selection(self.favorite_selections[name], name=selection_name, executor=executor)

	def selections_favorite_store(self):
		path = os.path.join(self.get_private_dir(create=True), "favorite_selection.yaml")
		selections = collections.OrderedDict([(key,value.to_dict()) for key,value in self.favorite_selections.items()])
		vaex.utils.write_json_or_yaml(path, selections)

	def selections_favorite_load(self):
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
			name = os.path.abspath(self.path).replace("/", "_")[:250]  # should not be too long for most os'es
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

	def combinations(self, expressions_list=None, dimension=2, exclude=None, **kwargs):
		"""Generate a Subspaces object, based on a custom list of expressions or all possible combinations based on
		dimension

		:param expressions_list: list of list of expressions, where the inner list defines the subspace
		:param dimensions: if given, generates a subspace with all possible combinations for that dimension
		:param exclude: list of
		"""
		if dimension is not None:
			expressions_list = list(itertools.combinations(self.get_column_names(), dimension))
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
		return expressions_list


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

	def evaluate_selection_mask(self, name="default", i1=None, i2=None, selection=None):
		i1 = i1 or 0
		i2 = i2 or len(self)
		scope = _BlockScopeSelection(self, i1, i2, selection)
		return scope.evaluate(name)

		#if _is_string(selection):




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

	def select(self, boolean_expression, mode="replace", name="default"):
		"""Select rows based on the boolean_expression, if there was a previous selection, the mode is taken into account.

		if boolean_expression is None, remove the selection, has_selection() will returns false

		Note that per dataset, only one selection is possible.

		:param str boolean_expression: boolean expression, such as 'x < 0', '(x < 0) || (y > -10)' or None to remove the selection
		:param str mode: boolean operation to perform with the previous selection, "replace", "and", "or", "xor", "subtract"
		:return: None
		"""
		raise NotImplementedError

	def add_column(self, name, f_or_array):
		"""Add an in memory array as a column"""
		if isinstance(f_or_array, np.ndarray):
			self.columns[name] = f_or_array
			self.column_names.append(name)
		else:
			raise ValueError("functions not yet implemented")

	def add_virtual_column_bearing(self, name, lon1, lat1, lon2, lat2):
		lon1 = "(pickup_longitude * pi / 180)"
		lon2 = "(dropoff_longitude * pi / 180)"
		lat1 = "(pickup_latitude * pi / 180)"
		lat2 = "(dropoff_latitude * pi / 180)"
		p1 = lat1
		p2 = lat2
		l1 = lon1
		l2 = lon2
		# from http://www.movable-type.co.uk/scripts/latlong.html
		expr = "arctan2(sin({l2}-{l1}) * cos({p2}), cos({p1})*sin({p2}) - sin({p1})*cos({p2})*cos({l2}-{l1}))" \
			.format(**locals())
		self.add_virtual_column("bearing", expr)

	def add_virtual_columns_distance_from_parallax(self, parallax, distance_name, parallax_uncertainty=None, uncertainty_postfix="_uncertainty"):
		unit = self.unit(parallax)
		#if unit:
			#convert = unit.to(astropy.units.mas)
			#	distance_expression = "%f/(%s)" % (convert, parallax)
			#else:
		distance_expression = "1/%s" % (parallax)
		self.ucds[distance_name] = "pos.distance"
		self.descriptions[distance_name] = "Derived from parallax (%s)" % parallax
		if unit:
			if unit == astropy.units.milliarcsecond:
				self.units[distance_name] = astropy.units.kpc
			if unit == astropy.units.arcsecond:
				self.units[distance_name] = astropy.units.parsec
		self.add_virtual_column(distance_name, distance_expression)
		if parallax_uncertainty:
			"""
			y = 1/x
			sigma_y**2 = (1/x**2)**2 sigma_x**2
			sigma_y = (1/x**2) sigma_x
			sigma_y = y**2 sigma_x
			sigma_y/y = (1/x) sigma_x
			"""
			name = distance_name + uncertainty_postfix
			distance_uncertainty_expression = "{parallax_uncertainty}/({parallax})**2".format(**locals())
			self.add_virtual_column(name, distance_uncertainty_expression)
			self.descriptions[name] = "Uncertainty on parallax (%s)" % parallax
			self.ucds[name] = "stat.error;pos.distance"



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


	def _covariance_matrix_guess(self, columns, full=False):
		all_column_names = self.get_column_names(virtual=True)
		def _guess(x, y):
			if x == y:
				postfixes = ["_error", "_uncertainty", "e", "_e"]
				prefixes = ["e", "e_"]
				for postfix in postfixes:
					if x + postfix in all_column_names:
						return x+postfix
				for prefix in prefixes:
					if prefix + x in all_column_names:
						return prefix + x
				if full:
					raise ValueError("No uncertainty found for %r" % x)
			else:

				postfixes = ["_cov", "_covariance"]
				for postfix in postfixes:
					if x +"_" + y + postfix in all_column_names:
						return x +"_" + y + postfix
					if y +"_" + x + postfix in all_column_names:
						return y +"_" + x + postfix
				postfixes = ["_correlation", "_corr"]
				for postfix in postfixes:
					if x +"_" + y + postfix in all_column_names:
						return x +"_" + y + postfix + " * " +  _guess(x, x) + " * " +  _guess(y, y)
					if y +"_" + x + postfix in all_column_names:
						return y +"_" + x + postfix  + " * " +  _guess(y, y) + " * " +  _guess(x, x)
				if full:
					raise ValueError("No covariance or correlation found for %r and %r" % (x, y))
			return ""
		N = len(columns)
		cov_matrix = [[""] * N for i in range(N)]
		for i in range(N):
			for j in range(N):
				cov = _guess(columns[i], columns[j])
				if i == j and cov:
					cov += "**2" # square the diagnal
				cov_matrix[i][j] = cov
		return cov_matrix

	def add_virtual_columns_cartesian_to_polar(self, x="x", y="y", radius_out="r_polar", azimuth_out="phi_polar",
												 cov_matrix_x_y=None,
												 covariance_postfix="_covariance",
												 uncertainty_postfix="_uncertainty",
												 radians=False):
		if radians:
			to_degrees = ""
		else:
			to_degrees = "*180/pi"
		self.add_virtual_column(radius_out, "sqrt({x}**2 + {y}**2)".format(**locals()))
		self.add_virtual_column(azimuth_out, "arctan2({y}, {x}){to_degrees}".format(**locals()))
		if cov_matrix_x_y:
			# function and it's jacobian
			# f_obs(x, y) = [r, phi] = (..., ....)
			J = [["-{x}/{radius_out}", "-{y}/{radius_out}"],
				 ["-{y}/({x}**2+{y}**2){to_degrees}", "{x}/({x}**2+{y}**2){to_degrees}"]]
			if cov_matrix_x_y in ["full", "auto"]:
				names = [x, y]
				cov_matrix_x_y = self._covariance_matrix_guess(names, full=cov_matrix_x_y=="full")

			cov_matrix_r_phi = [[""] * 2 for i in range(2)]
			for i in range(2):
				for j in range(2):
					for k in range(2):
						for l in range(2):
							sigma = cov_matrix_x_y[k][l]
							if sigma and J[i][k] and J[j][l]:
								cov_matrix_r_phi[i][j] += "+(%s)*(%s)*(%s)" % (J[i][k], sigma, J[j][l])

			names = [radius_out, azimuth_out]
			for i in range(2):
				for j in range(i+1):
					sigma = cov_matrix_r_phi[i][j].format(**locals())
					if i != j:
						self.add_virtual_column(names[i]+"_" + names[j]+covariance_postfix, sigma)
					else:
						self.add_virtual_column(names[i]+uncertainty_postfix, "sqrt(%s)" % sigma)

	def add_virtual_columns_cartesian_velocities_to_polar(self, x="x", y="y", vx="vx", radius_polar=None, vy="vy", vr_out="vr_polar", vazimuth_out="vphi_polar",
												 cov_matrix_x_y_vx_vy=None,
												 covariance_postfix="_covariance",
												 uncertainty_postfix="_uncertainty"):
		if radius_polar is None:
			radius_polar = "sqrt(({x})**2 + ({y})**2)".format(**locals())
		self.add_virtual_column(vr_out,       "(({x})*({vx})+({y})*({vy}))/{radius_polar}".format(**locals()))
		self.add_virtual_column(vazimuth_out, "(({x})*({vy})-({y})*({vx}))/{radius_polar}".format(**locals()))

		if cov_matrix_x_y_vx_vy:
			# function and it's jacobian
			# f_obs(x, y, vx, vy) = [vr, vphi] = ( (x*vx+y*vy) /r , (x*vy - y * vx)/r )

			J = [[
					"-({x}*{vr_out})/({radius_polar})**2 + {vx}/({radius_polar})",
					"-({y}*{vr_out})/({radius_polar})**2 + {vy}/({radius_polar})",
					"{x}/({radius_polar})",
					"{y}/({radius_polar})"
				], [
					"-({x}*{vazimuth_out})/({radius_polar})**2 + {vy}/({radius_polar})",
					"-({y}*{vazimuth_out})/({radius_polar})**2 - {vx}/({radius_polar})",
					"-{y}/({radius_polar})",
					" {x}/({radius_polar})"
				 ]]
			if cov_matrix_x_y_vx_vy in ["full", "auto"]:
				names = [x, y, vx, vy]
				cov_matrix_x_y_vx_vy = self._covariance_matrix_guess(names, full=cov_matrix_x_y_vx_vy=="full")

			cov_matrix_vr_vphi = [[""] * 2 for i in range(2)]
			for i in range(2):
				for j in range(2):
					for k in range(4):
						for l in range(4):
							sigma = cov_matrix_x_y_vx_vy[k][l]
							if sigma and J[i][k] and J[j][l]:
								cov_matrix_vr_vphi[i][j] += "+(%s)*(%s)*(%s)" % (J[i][k], sigma, J[j][l])

			names = [vr_out, vazimuth_out]
			for i in range(2):
				for j in range(i+1):
					sigma = cov_matrix_vr_vphi[i][j].format(**locals())
					if i != j:
						self.add_virtual_column(names[i]+"_" + names[j]+covariance_postfix, sigma)
					else:
						self.add_virtual_column(names[i]+uncertainty_postfix, "sqrt(%s)" % sigma)



	def add_virtual_columns_proper_motion_eq2gal(self, long_in="ra", lat_in="dec", pm_long="pm_ra", pm_lat="pm_dec", pm_long_out="pm_l", pm_lat_out="pm_b",
												 cov_matrix_alpha_delta_pma_pmd=None,
												 covariance_postfix="_covariance",
												 uncertainty_postfix="_uncertainty",
												 name_prefix="__proper_motion_eq2gal", radians=False):
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
		long_in_original = long_in
		lat_in_original  =  lat_in
		if not radians:
			long_in = "pi/180.*%s" % long_in
			lat_in = "pi/180.*%s" % lat_in
			to_radians = "*pi/180" # used for the derivatives
		else:
			to_radians = ""
		c1 = name_prefix + "_C1"
		c2 = name_prefix + "_C2"
		self.add_variable("right_ascension_galactic_pole", np.radians(192.85).item(), overwrite=False)
		self.add_variable("declination_galactic_pole", np.radians(27.12).item(), overwrite=False)
		self.add_virtual_column(c1, "sin(declination_galactic_pole) * cos({lat_in}) - cos(declination_galactic_pole)*sin({lat_in})*cos({long_in}-right_ascension_galactic_pole)".format(**locals()))
		self.add_virtual_column(c2, "cos(declination_galactic_pole) * sin({long_in}-right_ascension_galactic_pole)".format(**locals()))
		self.add_virtual_column(pm_long_out, "({c1} * {pm_long} + {c2} * {pm_lat})/sqrt({c1}**2+{c2}**2)".format(**locals()))
		self.add_virtual_column(pm_lat_out, "(-{c2} * {pm_long} + {c1} * {pm_lat})/sqrt({c1}**2+{c2}**2)".format(**locals()))
		if cov_matrix_alpha_delta_pma_pmd:
			# function and it's jacobian
			# f(long, lat, pm_long, pm_lat) = [pm_long, pm_lat, c1, c2] = [pm_long, pm_lat, ..., ...]
			J = [ [None, None, "1", None],
                  [None, None, None, "1"],
				  [                                                    "cos(declination_galactic_pole)*sin({lat_in})*sin({long_in}-right_ascension_galactic_pole){to_radians}",
				     "-sin(declination_galactic_pole) * sin({lat_in}){to_radians} - cos(declination_galactic_pole)*cos({lat_in})*cos({long_in}-right_ascension_galactic_pole){to_radians}",
					 None, None],
				  ["cos(declination_galactic_pole)*cos({long_in}-right_ascension_galactic_pole){to_radians}", None, None, None],
			  ]

			if cov_matrix_alpha_delta_pma_pmd in ["full", "auto"]:
				names = [long_in_original, lat_in_original, pm_long, pm_lat]
				cov_matrix_alpha_delta_pma_pmd = self._covariance_matrix_guess(names, full=cov_matrix_alpha_delta_pma_pmd=="full")

			cov_matrix_pm_long_pm_lat_c1_c2 = [[""] * 4 for i in range(4)]
			for i in range(4):
				for j in range(4):
					for k in range(4):
						for l in range(4):
							sigma = cov_matrix_alpha_delta_pma_pmd[k][l]
							if sigma and J[i][k] and J[j][l]:
								cov_matrix_pm_long_pm_lat_c1_c2[i][j] += "+(%s)*(%s)*(%s)" % (J[i][k], sigma, J[j][l])

			cov_matrix_pml_pmb = [[""] * 2 for i in range(2)]

			# function and it's jacobian
			# f(pm_long, pm_lat, c1, c2) = [pm_l, pm_b] = [..., ...]
			J = [
				[" ({c1}                               )/sqrt({c1}**2+{c2}**2)",
				 " (                    {c2}           )/sqrt({c1}**2+{c2}**2)",
				 "( {c2} *  {pm_long} - {c1} * {pm_lat})/    ({c1}**2+{c2}**2)**(3./2)*{c2}",
				 "(-{c2} *  {pm_long} + {c1} * {pm_lat})/    ({c1}**2+{c2}**2)**(3./2)*{c1}"],
				["(-{c2}                               )/sqrt({c1}**2+{c2}**2)",
				 " (                    {c1}           )/sqrt({c1}**2+{c2}**2)",
				 "({c1} * {pm_long} + {c2} * {pm_lat})/      ({c1}**2+{c2}**2)**(3./2)*{c2}",
				 "-({c1} * {pm_long} + {c2} * {pm_lat})/     ({c1}**2+{c2}**2)**(3./2)*{c1}"]
			]
			for i in range(2):
				for j in range(2):
					for k in range(4):
						for l in range(4):
							sigma = cov_matrix_pm_long_pm_lat_c1_c2[k][l]
							if sigma and J[i][k] != "0" and J[j][l] != "0":
								cov_matrix_pml_pmb[i][j] += "+(%s)*(%s)*(%s)" % (J[i][k], sigma, J[j][l])
			names = [pm_long_out, pm_lat_out]
			#cnames = ["c1", "c2"]
			for i in range(2):
				for j in range(i+1):
					sigma = cov_matrix_pml_pmb[i][j].format(**locals())
					if i != j:
						self.add_virtual_column(names[i]+"_" + names[j]+covariance_postfix, sigma)
					else:
						self.add_virtual_column(names[i]+uncertainty_postfix, "sqrt(%s)" % sigma)
						#sigma = cov_matrix_pm_long_pm_lat_c1_c2[i+2][j+2].format(**locals())
						#self.add_virtual_column(cnames[i]+uncertainty_postfix, "sqrt(%s)" % sigma)
						#sigma = cov_matrix_vr_vl_vb[i][j].format(**locals())

	def add_virtual_columns_proper_motion2vperpendicular(self, distance="distance", pm_long="pm_l", pm_lat="pm_b",
														   vl="vl", vb="vb",
														   cov_matrix_distance_pm_long_pm_lat=None,
														   uncertainty_postfix="_uncertainty", covariance_postfix="_covariance",
														   radians=False):
		k = 4.74057
		self.add_variable("k", k, overwrite=False)
		self.add_virtual_column(vl, "k*{pm_long}*{distance}".format(**locals()))
		self.add_virtual_column(vb, "k* {pm_lat}*{distance}".format(**locals()))
		if cov_matrix_distance_pm_long_pm_lat:
			# function and it's jacobian
			# f_obs(distance, pm_long, pm_lat) = [v_long, v_lat] = (k * pm_long * distance, k * pm_lat * distance)
			J = [["k * {pm_long}",  "k * {distance}", ""],
				 ["k * {pm_lat}",                 "", "k * {distance}"]]
			if cov_matrix_distance_pm_long_pm_lat in ["full", "auto"]:
				names = [distance, pm_long, pm_lat]
				cov_matrix_distance_pm_long_pm_lat = self._covariance_matrix_guess(names, full=cov_matrix_distance_pm_long_pm_lat=="full")


			cov_matrix_vl_vb = [[""] * 2 for i in range(2)]
			for i in range(2):
				for j in range(2):
					for k in range(3):
						for l in range(3):
							sigma = cov_matrix_distance_pm_long_pm_lat[k][l]
							if sigma and J[i][k] and J[j][l]:
								cov_matrix_vl_vb[i][j] += "+(%s)*(%s)*(%s)" % (J[i][k], sigma, J[j][l])

			names = [vl, vb]
			for i in range(2):
				for j in range(i+1):
					sigma = cov_matrix_vl_vb[i][j].format(**locals())
					if i != j:
						self.add_virtual_column(names[i]+"_" + names[j]+covariance_postfix, sigma)
					else:
						self.add_virtual_column(names[i]+uncertainty_postfix, "sqrt(%s)" % sigma)

	def add_virtual_columns_lbrvr_proper_motion2vcartesian(self, long_in="l", lat_in="b", distance="distance", pm_long="pm_l", pm_lat="pm_b",
														   vr="vr", vx="vx", vy="vy", vz="vz",
														   cov_matrix_vr_distance_pm_long_pm_lat=None,
														   uncertainty_postfix="_uncertainty", covariance_postfix="_covariance",
														   name_prefix="__lbvr_proper_motion2vcartesian", center_v=(0,0,0), center_v_name="solar_motion", radians=False):
		"""Convert radial velocity and galactic proper motions (and positions) to cartesian velocities wrt the center_v
		Based on http://adsabs.harvard.edu/abs/1987AJ.....93..864J

		v_long = k * pm_long * distance
		v_lat  = k * pm_lat * distance
		v_obs = [vr, v_long, v_lat]

		v = T * v_obs  + v_ref

		var_vr = sigma_vr
		var_v_long = (k * distance * sigma_pm_l)**2 + (k * pm_long * sigma_distance)**2 + 2 k * pm_long * k *distance* covar_distance_long
		covar_v_long_lat = k * distance

		f_obs(distance, pm_long, pm_lat) = [v_long, v_lat] = (k * pm_long * distance, k * pm_lat * distance)
		J = [[ k * pm_long,  k * distance,                     0],
			  [k * pm_lat ,             0, k* distance]]
		\Sigma_obs = [

		(k * distance * sigma_pm_l)**2 + (k * pm_long * sigma_distance)**2 + 2 k * pm_long * k *distance* covar_distance_long
		var_v_long = (k /pi * sigma_pm_l)**2 + (k * / pm**2 pm_long * sigma_distance)**2


		var_v_lat  = idem
		\Sigma_obs = [[var_vr,             0,        0],
				  [     0,   var_v_long, covar_vlb],
				  [     0,    covar_vlb, var_v_lat]]
		e_v = T \Sigma T.T
		\Sigma_v_ij = T_ik \Sigma_obs_kl T.T_lj
		\Sigma_v_ij = T_ik \Sigma_obs_kl T_jl

		A_ij = B_ik * C_kj


		e_v_r**2 = var_vr
		e_v_x**2 = T[1,1] * var_v_long T[1,1] + T[2,1] * covar_vlb * T[1,2]

		f_v(l, b, distance, vr_r, pm_1, pm_2) = T(3,6) * v

		m = 3, n = 6
		v0 =

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
		self.add_variable("k", k, overwrite=False)
		A = [["cos({a})*cos({d})", "-sin({a})", "-cos({a})*sin({d})"],
			 ["sin({a})*cos({d})",  "cos({a})", "-sin({a})*sin({d})"],
			 [         "sin({d})",         "0",           "cos({d})"]]
		a = long_in
		d = lat_in
		if not radians:
			a = "pi/180.*%s" % a
			d = "pi/180.*%s" % d
		for i in range(3):
			for j in range(3):
				A[i][j] = A[i][j].format(**locals())
		if 0: # used for testing
			self.add_virtual_column("vl", "k*{pm_long}*{distance}".format(**locals()))
			self.add_virtual_column("vb", "k* {pm_lat}*{distance}".format(**locals()))
		self.add_virtual_columns_matrix3d(vr, "k*{pm_long}*{distance}".format(**locals()), "k*{pm_lat}*{distance}".format(**locals()), name_prefix +vx, name_prefix +vy, name_prefix +vz, \
										  A, name_prefix+"_matrix", matrix_is_expression=True)
		self.add_variable(center_v_name, center_v)
		self.add_virtual_column(vx, "%s + %s[0]" % (name_prefix +vx, center_v_name))
		self.add_virtual_column(vy, "%s + %s[1]" % (name_prefix +vy, center_v_name))
		self.add_virtual_column(vz, "%s + %s[2]" % (name_prefix +vz, center_v_name))

		if cov_matrix_vr_distance_pm_long_pm_lat:
			# function and it's jacobian
			# f_obs(vr, distance, pm_long, pm_lat) = [vr, v_long, v_lat] = (vr, k * pm_long * distance, k * pm_lat * distance)
			J = [ ["1", "", "", ""],
				 ["", "k * {pm_long}",  "k * {distance}", ""],
				 ["", "k * {pm_lat}",                 "", "k * {distance}"]]

			if cov_matrix_vr_distance_pm_long_pm_lat in ["full", "auto"]:
				names = [vr, distance, pm_long, pm_lat]
				cov_matrix_vr_distance_pm_long_pm_lat = self._covariance_matrix_guess(names, full=cov_matrix_vr_distance_pm_long_pm_lat=="full")

			cov_matrix_vr_vl_vb = [[""] * 3 for i in range(3)]
			for i in range(3):
				for j in range(3):
					for k in range(4):
						for l in range(4):
							sigma = cov_matrix_vr_distance_pm_long_pm_lat[k][l]
							if sigma and J[i][k] and J[j][l]:
								cov_matrix_vr_vl_vb[i][j] += "+(%s)*(%s)*(%s)" % (J[i][k], sigma, J[j][l])

			cov_matrix_vx_vy_vz = [[""] * 3 for i in range(3)]

			# here A is the Jacobian
			for i in range(3):
				for j in range(3):
					for k in range(3):
						for l in range(3):
							sigma = cov_matrix_vr_vl_vb[k][l]
							if sigma and A[i][k] != "0" and A[j][l] != "0":
								cov_matrix_vx_vy_vz[i][j] += "+(%s)*(%s)*(%s)" % (A[i][k], sigma, A[j][l])
			vnames = [vx, vy, vz]
			vrlb_names = ["vr", "vl", "vb"]
			for i in range(3):
				for j in range(i+1):
					sigma = cov_matrix_vx_vy_vz[i][j].format(**locals())
					if i != j:
						self.add_virtual_column(vnames[i]+"_" + vnames[j]+covariance_postfix, sigma)
					else:
						self.add_virtual_column(vnames[i]+uncertainty_postfix, "sqrt(%s)" % sigma)
						#sigma = cov_matrix_vr_vl_vb[i][j].format(**locals())
						#self.add_virtual_column(vrlb_names[i]+uncertainty_postfix, "sqrt(%s)" % sigma)


			#self.add_virtual_column(vx, x)



	def add_virtual_columns_celestial(self, long_in, lat_in, long_out, lat_out, input=None, output=None, name_prefix="__celestial", radians=False):
		import kapteyn.celestial as c
		input = input if input is not None else c.equatorial
		output = output if output is not None else c.galactic
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
		#self.add_virtual_columns_spherical_to_cartesian(long_in, lat_in, None, x_in, y_in, z_in, cov_matrix_alpha_delta=)
		self.add_virtual_columns_matrix3d(x_in, y_in, z_in, x_out, y_out, z_out, \
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


	def add_virtual_columns_spherical_to_cartesian(self, alpha, delta, distance, xname="x", yname="y", zname="z",
												   cov_matrix_alpha_delta_distance=None,
												   covariance_postfix="_covariance",
												   uncertainty_postfix="_uncertainty",
												   center=None, center_name="solar_position", radians=True):
		alpha_original = alpha
		delta_original = delta
		if not radians:
			alpha = "pi/180.*%s" % alpha
			delta = "pi/180.*%s" % delta
			to_radians = "*pi/180" # used for the derivatives
		else:
			to_radians = ""
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
		if cov_matrix_alpha_delta_distance:
			# function and it's jacobian
			# f_obs(alpha, delta, distance) = [x, y, z] = (cos(alpha) * cos(delta) * distance,
			# 												sin(alpha) * cos(delta) * distance,
			# 												sin(delta) * distance)
			J = [ ["-sin({alpha})*cos({delta})*{distance}{to_radians}", "-cos({alpha})*sin({delta})*{distance}{to_radians}", "cos({alpha})*cos({delta})"],
				  [" cos({alpha})*cos({delta})*{distance}{to_radians}", "-sin({alpha})*sin({delta})*{distance}{to_radians}", "sin({alpha})*cos({delta})"],
				 [                              None,                     "cos({delta})*{distance}{to_radians}",              "sin({delta})"]]

			if cov_matrix_alpha_delta_distance in ["full", "auto"]:
				names = [alpha_original, delta_original, distance]
				cov_matrix_alpha_delta_distance = self._covariance_matrix_guess(names, full=cov_matrix_alpha_delta_distance=="full")

			cov_matrix_xyz = [[""] * 3 for i in range(3)]
			for i in range(3):
				for j in range(3):
					for k in range(3):
						for l in range(3):
							sigma = cov_matrix_alpha_delta_distance[k][l]
							if sigma and J[i][k] and J[j][l]:
								cov_matrix_xyz[i][j] += "+(%s)*(%s)*(%s)" % (J[i][k], sigma, J[j][l])
							#if sigma and J[k][i] and J[l][j]:
							#	cov_matrix_xyz[i][j] += "+(%s)*(%s)*(%s)" % (J[k][i], sigma, J[l][j])

			names = [xname, yname, zname]
			for i in range(3):
				for j in range(i+1):
					sigma = cov_matrix_xyz[i][j].format(**locals())
					if i != j:
						self.add_virtual_column(names[i]+"_" + names[j]+covariance_postfix, sigma)
					else:
						self.add_virtual_column(names[i]+uncertainty_postfix, "sqrt(%s)" % sigma)

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

	def add_variable(self, name, expression, overwrite=True):
		"""Add a variable column to the dataset

		:param: str name: name of virtual varible
		:param: expression: expression for the variable

		Variable may refer to other variables, and virtual columns and expression may refer to variables

		:Example:
		>>> dataset.add_variable("center")
		>>> dataset.add_virtual_column("x_prime", "x-center")
		>>> dataset.select("x_prime < 0")
		"""
		if overwrite or name not in self.variables:
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

	def get_column_names(self, virtual=False, hidden=False, strings=False):
		"""Return a list of column names


		:param virtual: If True, also return virtual columns
		:param hidden: If True, also return hidden columns
		:rtype: list of str
 		"""
		return list([name for name in self.column_names if strings or self.dtype(name).type != np.string_]) \
			   + ([key for key in self.virtual_columns.keys() if (hidden or (not key.startswith("__")))] if virtual else [])

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
			self._index_start = 0
			self._index_end = self._length
			self.signal_active_fraction_changed.emit(self, value)

	def get_active_range(self):
		return self._index_start, self._index_end
	def set_active_range(self, i1, i2):
		"""Sets the active_fraction, set picked row to None, and remove selection

		TODO: we may be able to keep the selection, if we keep the expression, and also the picked row
		"""
		logger.debug("set active range to: %r", (i1, i2))
		self._active_fraction = (i2-i1) / float(self.full_length())
		#self._fraction_length = int(self._length * self._active_fraction)
		self._index_start = i1
		self._index_end = i2
		self.select(None)
		self.set_current_row(None)
		self._length = i2-i1
		self.signal_active_fraction_changed.emit(self, self._active_fraction)


	def get_selection(self, name="default"):
		"""Get the current selection object (mostly for internal use atm)"""
		selection_history = self.selection_histories[name]
		index = self.selection_history_indices[name]
		if index == -1:
			return None
		else:
			return selection_history[index]

	def selection_undo(self, name="default", executor=None):
		"""Undo selection, for the name"""
		logger.debug("undo")
		executor = executor or self.executor
		assert self.selection_can_undo(name=name)
		selection_history = self.selection_histories[name]
		index = self.selection_history_indices[name]
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
		self.selection_history_indices[name] -= 1
		logger.debug("undo: selection history is %r, index is %r", selection_history, self.selection_history_indices[name])
		return result


	def selection_redo(self, name="default", executor=None):
		"""Redo selection, for the name"""
		logger.debug("redo")
		executor = executor or self.executor
		assert self.selection_can_redo(name=name)
		selection_history = self.selection_histories[name]
		index = self.selection_history_indices[name]
		next = selection_history[index+1]
		if self.is_local():
			result = next.execute(executor=executor)
		else:
			self.signal_selection_changed.emit(self)
			result = vaex.promise.Promise.fulfilled(None)
		self.selection_history_indices[name] += 1
		logger.debug("redo: selection history is %r, index is %r", selection_history, index)
		return result

	def selection_can_undo(self, name="default"):
		"""Can selection name be undone?"""
		return self.selection_history_indices[name] > -1

	def selection_can_redo(self, name="default"):
		"""Can selection name be redone?"""
		return (self.selection_history_indices[name] + 1) < len(self.selection_histories[name])

	def select(self, boolean_expression, mode="replace", name="default", executor=None):
		"""Perform a selection, defined by the boolean expression, and combined with the previous selection using the given mode

		Selections are recorded in a history tree, per name, undo/redo can be done for them seperately

		:param str boolean_expression: Any valid column expression, with comparison operators
		:param str mode: Possible boolean operator: replace/and/or/xor/subtract
		:param str name: history tree or selection 'slot' to use
		:param executor:
		:return:
		"""
		if boolean_expression is None and not self.has_selection(name=name):
			pass # we don't want to pollute the history with many None selections
			self.signal_selection_changed.emit(self) # TODO: unittest want to know, does this make sense?
		else:
			def create(current):
				return SelectionExpression(self, boolean_expression, current, mode) if boolean_expression else None
			return self._selection(create, name)

	def select_nothing(self, name="default"):
		"""Select nothing"""
		logger.debug("selecting nothing")
		self.select(None, name=name)
	#self.signal_selection_changed.emit(self)

	def select_rectangle(self, expression_x, expression_y, limits, mode="replace"):
		(x1, x2), (y1, y2) = limits
		xmin, xmax = min(x1, x2), max(x1, x2)
		ymin, ymax = min(y1, y2), max(y1, y2)
		args = (expression_x, xmin, expression_x, xmax, expression_y, ymin, expression_y, ymax)
		expression = "((%s) >= %f) & ((%s) <= %f) & ((%s) >= %f) & ((%s) <= %f)" % args
		self.select(expression, mode=mode)

	def select_lasso(self, expression_x, expression_y, xsequence, ysequence, mode="replace", name="default", executor=None):
		"""For performance reasons, a lasso selection is handled differently.

		:param str expression_x: Name/expression for the x coordinate
		:param str expression_y: Name/expression for the y coordinate
		:param xsequence: list of x numbers defining the lasso, together with y
		:param ysequence:
		:param str mode: Possible boolean operator: replace/and/or/xor/subtract
		:param str name:
		:param executor:
		:return:
		"""


		def create(current):
			return SelectionLasso(self, expression_x, expression_y, xsequence, ysequence, current, mode)
		return self._selection(create, name, executor=executor)

	def select_inverse(self, name="default", executor=None):
		"""Invert the selection, i.e. what is selected will not be, and vice versa

		:param str name:
		:param executor:
		:return:
		"""


		def create(current):
			return SelectionInvert(self, current)
		return self._selection(create, name, executor=executor)

	def set_selection(self, selection, name="default", executor=None):
		"""Sets the selection object

		:param selection: Selection object
		:param name: selection 'slot'
		:param executor:
		:return:
		"""
		def create(current):
			return selection
		return self._selection(create, name, executor=executor, execute_fully=True)


	def _selection(self, create_selection, name, executor=None, execute_fully=False):
		"""select_lasso and select almost share the same code"""
		selection_history = self.selection_histories[name]
		previous_index = self.selection_history_indices[name]
		current = selection_history[previous_index] if selection_history else None
		selection = create_selection(current)
		executor = executor or self.executor
		selection_history.append(selection)
		self.selection_history_indices[name] += 1
		# clip any redo history
		del selection_history[self.selection_history_indices[name]:-1]
		if 0:
			if self.is_local():
				if selection:
					#result = selection.execute(executor=executor, execute_fully=execute_fully)
					result = vaex.promise.Promise.fulfilled(None)
					self.signal_selection_changed.emit(self)
				else:
					result = vaex.promise.Promise.fulfilled(None)
					self.signal_selection_changed.emit(self)
			else:
				self.signal_selection_changed.emit(self)
				result = vaex.promise.Promise.fulfilled(None)
		self.signal_selection_changed.emit(self)
		result = vaex.promise.Promise.fulfilled(None)
		logger.debug("select selection history is %r, index is %r", selection_history, self.selection_history_indices[name])
		return result

	def has_selection(self, name="default"):
		"""Returns True of there is a selection"""
		return self.get_selection(name) != None


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
		self._selection_mask_caches = collections.defaultdict(dict)


	@property
	def data(self):
		"""Gives direct access to the data as numpy arrays.

		Convenient when working with IPython in combination with small datasets, since this gives tab-completion.
		Only real columns (i.e. no virtual) columns can be accessed, for getting the data from virtual columns, use
		Dataset.evalulate(...)

		Columns can be accesed by there names, which are attributes. The attribues are of type numpy.ndarray

		:Example:
		>>> ds = vx.example()
		>>> r = np.sqrt(ds.data.x**2 + ds.data.y**2)

		"""
		class Datas(object):
			pass

		datas = Datas()
		for name, array in self.columns.items():
			setattr(datas, name, array)
		return datas

	def shallow_copy(self, virtual=True, variables=True):
		"""Creates a (shallow) copy of the dataset

		It will link to the same data, but will have its own state, e.g. virtual columns, variables, selection etc

		"""
		dataset = DatasetLocal(self.name, self.path, self.column_names)
		dataset.columns.update(self.columns)
		dataset._full_length = self._full_length
		dataset._index_end = self._full_length
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

	def echo(self, arg): return arg

	def __getitem__(self, arg):
		"""Alias for call, to mimic Pandas a bit

		:Example:
		>> ds["Lz"]
		>> ds["Lz", "E"]
		>> ds[ds.names.Lz]

		"""
		if isinstance(arg, tuple):
			return self(*arg)
		else:
			return self(arg)

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

	def export_hdf5(self, path, column_names=None, byteorder="=", shuffle=False, selection=False, progress=None, virtual=False):
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
		vaex.export.export_hdf5(self, path, column_names, byteorder, shuffle, selection, progress=progress, virtual=virtual)

	def export_fits(self, path, column_names=None, shuffle=False, selection=False, progress=None, virtual=False):
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
		vaex.export.export_fits(self, path, column_names, shuffle, selection, progress=progress, virtual=virtual)

	def _needs_copy(self, column_name):
		return not \
			((column_name in self.column_names  \
			and not isinstance(self.columns[column_name], vaex.dataset._ColumnConcatenatedLazy)\
			and not isinstance(self.columns[column_name], vaex.file.other.DatasetTap.TapColumn)\
			and self.columns[column_name].dtype.type==np.float64 \
			and self.columns[column_name].strides[0] == 8 \
			and column_name not in self.virtual_columns) or self.dtype(column_name).kind == 'S')
				#and False:

	def selected_length(self, selection="default"):
		"""The local implementation of :func:`Dataset.selected_length`"""
		return int(self.count(selection=selection).item())
			#np.sum(self.mask) if self.has_selection() else None



	def _set_mask(self, mask):
		self.mask = mask
		self._has_selection = mask is not None
		self.signal_selection_changed.emit(self)


class _ColumnConcatenatedLazy(object):
	def __init__(self, datasets, column_name):
		self.datasets = datasets
		self.column_name = column_name
		dtypes = [dataset.columns[self.column_name].dtype for dataset in datasets]
		# np.datetime64 and find_common_type don't mix very well
		if all([dtype.type == np.datetime64 for dtype in dtypes]):
			self.dtype = dtypes[0]
		else:
			if all([dtype == dtypes[0] for dtype in dtypes]): # find common types doesn't always behave well
				self.dtype = dtypes[0]
			else:
				self.dtype = np.find_common_type(dtypes, [])
			logger.debug("common type for %r is %r", dtypes, self.dtype)
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
		for column_name in first.get_column_names(strings=True):
			if all([column_name in dataset.get_column_names(strings=True) for dataset in tail]):
				self.column_names.append(column_name)
		self.columns = {}
		for column_name in self.get_column_names(strings=True):
			self.columns[column_name] = _ColumnConcatenatedLazy(datasets, column_name)

		for name in list(first.virtual_columns.keys()):
			if all([first.virtual_columns[name] == dataset.virtual_columns.get(name, None) for dataset in tail]):
				self.virtual_columns[name] = first.virtual_columns[name]
		for dataset in datasets:
			for name, value in list(dataset.variables.items()):
				self.set_variable(name, value)


		self._full_length = sum(ds.full_length() for ds in self.datasets)
		self._length = self.full_length()
		self._index_end = self._full_length

def _is_dtype_ok(dtype):
	return dtype.type in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float32, np.float64] or\
		dtype.type == np.string_

def _is_array_type_ok(array):
	return _is_dtype_ok(array.dtype)

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
		assert _is_array_type_ok(data), "dtype not supported: %r, %r" % (data.dtype, data.dtype.type)
		self.column_names.append(name)
		self.columns[name] = data
		#self._length = len(data)
		if self._full_length is None:
			self._full_length = len(data)
			self._index_end = self._full_length
		else:
			assert self.full_length() == len(data), "columns should be of equal length, length should be %d, while it is %d" % ( self.full_length(), len(data))
		self._length = int(round(self.full_length() * self._active_fraction))
		#self.set_active_fraction(self._active_fraction)






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


