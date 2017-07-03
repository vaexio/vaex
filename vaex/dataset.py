# -*- coding: utf-8 -*-
from __future__ import division, print_function

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

def _normalize_selection_name(name):
	if name is True:
		return "default"
	elif name is False:
		return None
	else:
		return name

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
		import matplotlib.cm
		cmap = matplotlib.cm.get_cmap(colormap)
		def f(grid):
			masked_grid = np.ma.masked_invalid(grid) # convert inf/nan to a mask so that mpl colors bad values correcty
			return cmap(masked_grid)
		return f
	elif name.startswith("stack.color"):
		def f(grid, colors=colors, colormap=colormap):
			import matplotlib.cm
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
		blocks = [as_flat_float(block) for block in blocks]
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

def as_flat_float(a):
	if a.dtype.type==np.float64 and a.strides[0] == 8:
		return a
	else:
		return a.astype(np.float64, copy=False)


class TaskStatistic(Task):
	def __init__(self, dataset, expressions, shape, limits, masked=False, weight=None, op=OP_ADD1, selection=None, edges=False):
		if not isinstance(expressions, (tuple, list)):
			expressions = [expressions]
		# edges include everything outside at index 1 and -1, and nan's at index 0, so we add 3 to each dimension
		self.shape = tuple([k +3 if edges else k for k in _expand_shape(shape, len(expressions))])
		self.limits = limits
		self.weight = weight
		self.selection_waslist, [self.selections,] = vaex.utils.listify(selection)
		self.op = op
		self.edges = edges
		Task.__init__(self, dataset, expressions, name="statisticNd")
		self.dtype = np.float64
		self.masked = masked

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
		return "<%s(dataset=%r, expressions=%r, shape=%r, limits=%r, weight=%r, selections=%r, op=%r)> instance at 0x%x" % (name, self.dataset, self.expressions, self.shape, self.limits, self.weight, self.selections, self.op, id(self))

	def map(self, thread_index, i1, i2, *blocks):
		class Info(object):
			pass
		info = Info()
		info.i1 = i1
		info.i2 = i2
		info.first = i1 == 0
		info.last = i2 == len(self.dataset)
		info.size = i2-i1

		masks = [block.mask for block in blocks if np.ma.isMaskedArray(block)]
		blocks = [block.data if np.ma.isMaskedArray(block) else block for block in blocks]
		mask = None
		if masks:
			mask = masks[0].copy()
			for other in masks[1:]:
				mask |= other
			blocks = [block[~mask] for block in blocks]

		blocks = [as_flat_float(block) for block in blocks]

		this_thread_grid = self.grid[thread_index]
		for i, selection in enumerate(self.selections):
			if selection:
				selection_mask = self.dataset.evaluate_selection_mask(selection, i1=i1, i2=i2) # TODO
				if selection_mask is None:
					raise ValueError("performing operation on selection while no selection present")
				if mask is not None:
					selection_mask = selection_mask[~mask]
				selection_blocks = [block[selection_mask] for block in blocks]
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
						this_thread_grid[i][0] += np.sum(selection_mask)
					else:
						this_thread_grid[i][0] += i2-i1
				else:
					raise ValueError("Nothing to compute for OP %s" % self.op.code)

			blocks = list(blocks) # histogramNd wants blocks to be a list
			vaex.vaexfast.statisticNd(selection_blocks, subblock_weight, this_thread_grid[i], self.minima, self.maxima, self.op.code, self.edges)
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





import vaex.events
import cgi

# mutex for numexpr (is not thread save)
ne_lock = threading.Lock()

class UnitScope(object):
	def __init__(self, dataset, value=None):
		self.dataset = dataset
		self.value = value

	def __getitem__(self, variable):
		import astropy.units
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
	def __init__(self, dataset, i1, i2, mask=None, **variables):
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
		self.mask = mask if mask is not None else slice(None, None, None)

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
			if variable in self.values:
				return self.values[variable]
			elif variable in self.dataset.get_column_names(strings=True):
				if self.dataset._needs_copy(variable):
					#self._ensure_buffer(variable)
					#self.values[variable] = self.buffers[variable] = self.dataset.columns[variable][self.i1:self.i2].astype(np.float64)
					#Previously we casted anything to .astype(np.float64), this led to rounding off of int64, when exporting
					self.values[variable] = self.dataset.columns[variable][self.i1:self.i2][self.mask]
				else:
					self.values[variable] = self.dataset.columns[variable][self.i1:self.i2][self.mask]
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

	def __getitem__(self, variable):
		#logger.debug("getitem for selection: %s", variable)
		try:
			selection = self.selection
			if selection is None and self.dataset.has_selection(variable):
				selection = self.dataset.get_selection(variable)
			#logger.debug("selection for %r: %s %r", variable, selection, self.dataset.selection_histories)
			key = (self.i1, self.i2)
			if selection:
				cache = self.dataset._selection_mask_caches[variable]
				#logger.debug("selection cache: %r" % cache)
				selection_in_cache, mask = cache.get(key, (None, None))
				#logger.debug("mask for %r is %r", variable, mask)
				if selection_in_cache == selection:
					return mask
				#logger.debug("was not cached")
				if variable in self.dataset.variables:
					return self.dataset.variables[variable]
				mask = selection.evaluate(variable, self.i1, self.i2)
				#logger.debug("put selection in mask with key %r" % (key,))
				cache[key] = selection, mask
				return mask
			else:
					if variable in expression_namespace:
						return expression_namespace[variable]
					elif variable in self.dataset.get_column_names(strings=True):
						return self.dataset.columns[variable][self.i1:self.i2]
					elif variable in self.dataset.variables:
						return self.dataset.variables[variable]
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
		current_mask = self.dataset.evaluate_selection_mask(self.boolean_expression, i1, i2).astype(np.bool)
		if previous_mask is None:
			logger.debug("setting mask")
			mask = current_mask
		else:
			logger.debug("combining previous mask with current mask using op %r", self.mode)
			mode_function = _select_functions[self.mode]
			mask = mode_function(previous_mask, current_mask)
		return mask


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
		blockx = as_flat_float(blockx)
		blocky = as_flat_float(blocky)
		vaex.vaexfast.pnpoly(x, y, blockx, blocky, current_mask, meanx, meany, radius)
		if previous_mask is None:
			logger.debug("setting mask")
			mask = current_mask
		else:
			logger.debug("combining previous mask with current mask using op %r", self.mode)
			mode_function = _select_functions[self.mode]
			mask = mode_function(previous_mask, current_mask)
		return mask


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
rad2deg
deg2rad
minimum
maximum
""".strip().split()]
expression_namespace = {}
for name, numpy_name in function_mapping:
	if not hasattr(np, numpy_name):
		raise SystemError("numpy does not have: %s" % numpy_name)
	else:
		expression_namespace[name] = getattr(np, numpy_name)
def dayofweek(x):
	import pandas as pd
	x = x.astype("<M8[ns]")
	return pd.Series(x).dt.dayofweek.values.astype(np.float64)
expression_namespace["dayofweek"] = dayofweek
def hourofday(x):
	import pandas as pd
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
_doc_snippets["selection"] = """Name of selection to use (or True for the 'default'), or all the data (when selection is None or False), or a list of selections"""
_doc_snippets["async"] = """Do not return the result, but a proxy for asynchronous calculations (currently only for internal use)"""
_doc_snippets["progress"] = """A callable that takes one argument (a floating point value between 0 and 1) indicating the progress, calculations are cancelled when this callable returns False"""
_doc_snippets["expression_limits"] = _doc_snippets["expression"]
_doc_snippets["grid"] = """If grid is given, instead if compuation a statistic given by what, use this Nd-numpy array instead, this is often useful when a custom computation/statistic is calculated, but you still want to use the plotting machinery."""
_doc_snippets["edges"] = """Currently for internal use only (it includes nan's and values outside the limits at borders, nan and 0, smaller than at 1, and larger at -1"""

_doc_snippets["healpix_expression"] = """Expression which maps to a healpix index, for the Gaia catalogue this is for instance 'source_id/34359738368', other catalogues may simply have a healpix column."""
_doc_snippets["healpix_max_level"] = """The healpix level associated to the healpix_expression, for Gaia this is 12"""
_doc_snippets["healpix_level"] = """The healpix level to use for the binning, this defines the size of the first dimension of the grid."""

_doc_snippets["return_stat_scalar"] = """Numpy array with the given shape, or a scalar when no binby argument is given, with the statistic"""
_doc_snippets["return_limits"] = """List in the form [[xmin, xmax], [ymin, ymax], .... ,[zmin, zmax]] or [xmin, xmax] when expression is not a list"""
_doc_snippets["cov_matrix"] = """List all convariance values as a double list of expressions, or "full" to guess all entries (which gives an error when values are not found), or "auto" to guess, but allow for missing values"""

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

	def bins(self, expression, limits, shape=default_shape, edges=True):
		vmin, vmax = limits
		if edges:
			bins = np.ogrid[limits[0]:limits[1]:(shape+1)*1j]
			return bins
		else:
			dx = (limits[1] - limits[0])/shape
			bins = np.ogrid[limits[0]:limits[1]-dx:(shape)*1j]
			return bins + dx/2

	def nearest_bin(self, value, limits, shape):
		bins = self.bins('', limits=limits, edges=False, shape=shape)
		index = np.argmin(np.abs(bins-value))
		print(bins, value, index)
		return index

	@docsubst
	def count(self, expression=None, binby=[], limits=None, shape=default_shape, selection=False, async=False, edges=False, progress=None):
		"""Count the number of non-NaN values (or all, if expression is None or "*")

		Examples:


		>>> ds.count()
		330000.0
		>>> ds.count("*")
		330000.0
		>>> ds.count("*", binby=["x"], shape=4)
		array([  10925.,  155427.,  152007.,   10748.])

		:param expression: Expression or column for which to count non-missing values, or None or '*' for counting the rows
		:param binby: {binby}
		:param limits: {limits}
		:param shape: {shape}
		:param selection: {selection}
		:param async: {async}
		:param progress: {progress}
		:param edges: {edges}
		:return: {return_stat_scalar}
		"""
		logger.debug("count(%r, binby=%r, limits=%r)", expression, binby, limits)
		@delayed
		def calculate(expression, limits):
			if expression in ["*", None]:
				#if not binby: # if we have nothing to iterate over, the statisticNd code won't do anything
				#\3	return np.array([self.length(selection=selection)], dtype=float)
				#else:
				task = TaskStatistic(self, binby, shape, limits, op=OP_ADD1, selection=selection, edges=edges)
			else:
				task = TaskStatistic(self, binby, shape, limits, weight=expression, op=OP_COUNT, selection=selection, edges=edges)
			self.executor.schedule(task)
			progressbar.add_task(task, "count for %s" % expression)
			return task
		@delayed
		def finish(*stats_args):
			stats = np.array(stats_args)
			counts = stats[...,0]
			return vaex.utils.unlistify(waslist, counts)
		waslist, [expressions,] = vaex.utils.listify(expression)
		progressbar = vaex.utils.progressbars(progress)
		limits = self.limits(binby, limits, async=True)
		stats = [calculate(expression, limits) for expression in expressions]
		var = finish(*stats)
		return self._async(async, var)

	@docsubst
	@stat_1d
	def mean(self, expression, binby=[], limits=None, shape=default_shape, selection=False, async=False, progress=None):
		"""Calculate the mean for expression, possibly on a grid defined by binby.

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
		:param progress: {progress}
		:return: {return_stat_scalar}
		"""
		logger.debug("mean of %r, with binby=%r, limits=%r, shape=%r, selection=%r, async=%r", expression, binby, limits, shape, selection, async)
		@delayed
		def calculate(expression, limits):
			task = TaskStatistic(self, binby, shape, limits, weight=expression, op=OP_ADD_WEIGHT_MOMENTS_01, selection=selection)
			self.executor.schedule(task)
			progressbar.add_task(task, "mean for %s" % expression)
			return task
		@delayed
		def finish(*stats_args):
			stats = np.array(stats_args)
			counts = stats[...,0]
			with np.errstate(divide='ignore', invalid='ignore'):
				mean = stats[...,1] / counts
			return vaex.utils.unlistify(waslist, mean)
		waslist, [expressions,] = vaex.utils.listify(expression)
		progressbar = vaex.utils.progressbars(progress)
		limits = self.limits(binby, limits, async=True)
		stats = [calculate(expression, limits) for expression in expressions]
		var = finish(*stats)
		return self._async(async, var)

	@docsubst
	@stat_1d
	def sum(self, expression, binby=[], limits=None, shape=default_shape, selection=False, async=False, progress=None):
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
		:param progress: {progress}
		:return: {return_stat_scalar}
		"""
		@delayed
		def calculate(expression, limits):
			task = TaskStatistic(self, binby, shape, limits, weight=expression, op=OP_ADD_WEIGHT_MOMENTS_01, selection=selection)
			self.executor.schedule(task)
			progressbar.add_task(task, "sum for %s" % expression)
			return task
		@delayed
		def finish(*stats_args):
			stats = np.array(stats_args)
			sum = stats[...,1]
			return vaex.utils.unlistify(waslist, sum)
		waslist, [expressions,] = vaex.utils.listify(expression)
		progressbar = vaex.utils.progressbars(progress)
		limits = self.limits(binby, limits, async=True)
		stats = [calculate(expression, limits) for expression in expressions]
		s = finish(*stats)
		return self._async(async, s)

	@docsubst
	@stat_1d
	def std(self, expression, binby=[], limits=None, shape=default_shape, selection=False, async=False, progress=None):
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
		:param progress: {progress}
		:return: {return_stat_scalar}
		"""
		@delayed
		def finish(var):
			return var**0.5
		return self._async(async, finish(self.var(expression, binby=binby, limits=limits, shape=shape, selection=selection, async=True, progress=progress)))

	@docsubst
	@stat_1d
	def var(self, expression, binby=[], limits=None, shape=default_shape, selection=False, async=False, progress=None):
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
		:param progress: {progress}
		:return: {return_stat_scalar}
		"""
		@delayed
		def calculate(expression, limits):
			task = TaskStatistic(self, binby, shape, limits, weight=expression, op=OP_ADD_WEIGHT_MOMENTS_012, selection=selection)
			progressbar.add_task(task, "var for %s" % expression)
			self.executor.schedule(task)
			return task
		@delayed
		def finish(*stats_args):
			stats = np.array(stats_args)
			counts = stats[...,0]
			with np.errstate(divide='ignore'):
				with np.errstate(divide='ignore', invalid='ignore'): # these are fine, we are ok with nan's in vaex
					mean = stats[...,1] / counts
					raw_moments2 = stats[...,2] / counts
			variance = (raw_moments2-mean**2)
			return vaex.utils.unlistify(waslist, variance)
		waslist, [expressions,] = vaex.utils.listify(expression)
		progressbar = vaex.utils.progressbars(progress)
		limits = self.limits(binby, limits, async=True)
		stats = [calculate(expression, limits) for expression in expressions]
		var = finish(*stats)
		return self._async(async, var)


	@docsubst
	def covar(self, x, y, binby=[], limits=None, shape=default_shape, selection=False, async=False, progress=None):
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
		:param progress: {progress}
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
			results = []
			for x, y in zip(xlist, ylist):
				mx = self.mean(x, binby=binby, limits=limits, shape=shape, selection=selection, async=True, progress=progressbar)
				my = self.mean(y, binby=binby, limits=limits, shape=shape, selection=selection, async=True, progress=progressbar)
				cxy = self.mean("(%s)*(%s)" % (x, y), binby=binby, limits=limits, shape=shape, selection=selection,
						  async=True, progress=progressbar)
				results.append(cov(mx, my, cxy))
			return results

		progressbar = vaex.utils.progressbars(progress)
		covars = calculate(limits)
		@delayed
		def finish(covars):
			value = np.array(vaex.utils.unlistify(waslist, covars))
			return value
		return self._async(async, finish(delayed_list(covars)))

	@docsubst
	def correlation(self, x, y=None, binby=[], limits=None, shape=default_shape, sort=False, sort_key=np.abs, selection=False, async=False, progress=None):
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
		:param progress: {progress}
		:return: {return_stat_scalar}
		"""
		@delayed
		def corr(cov):
			with np.errstate(divide='ignore', invalid='ignore'): # these are fine, we are ok with nan's in vaex
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
			results = []
			for x, y in zip(xlist, ylist):
				task = self.cov(x, y, binby=binby, limits=limits, shape=shape, selection=selection, async=True,
							 progress=progressbar)
				results.append(corr(task))
			return results

		progressbar = vaex.utils.progressbars(progress)
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
	def cov(self, x, y=None, binby=[], limits=None, shape=default_shape, selection=False, async=False, progress=None):
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
		progressbar = vaex.utils.progressbars(progress)
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
			means = [self.mean(expression, binby=binby, limits=limits, shape=shape, selection=selection, async=True, progress=progressbar) for expression in expressions]
			vars  = [self.var (expression, binby=binby, limits=limits, shape=shape, selection=selection, async=True, progress=progressbar) for expression in expressions]
			raw_mixed = []
			for i in range(N):
				for j in range(i+1):
					if i != j:
						raw_mixed.append(self.mean("(%s)*(%s)" % (expressions[i], expressions[j]), binby=binby, limits=limits, shape=shape, selection=selection, async=True, progress=progressbar))
			return calculate_matrix(delayed_list(means), delayed_list(vars), delayed_list(raw_mixed))

		covars = calculate(limits)
		return self._async(async, covars)

	@docsubst
	@stat_1d
	def minmax(self, expression, binby=[], limits=None, shape=default_shape, selection=False, async=False, progress=None):
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
		:param progress: {progress}
		:return: {return_stat_scalar}, the last dimension is of shape (2)
		"""
		@delayed
		def calculate(expression, limits):
			task =  TaskStatistic(self, binby, shape, limits, weight=expression, op=OP_MIN_MAX, selection=selection)
			self.executor.schedule(task)
			progressbar.add_task(task, "minmax for %s" % expression)
			return task
		@delayed
		def finish(*minmax_list):
			value = vaex.utils.unlistify(waslist, np.array(minmax_list))
			return value
		waslist, [expressions,] = vaex.utils.listify(expression)
		progressbar = vaex.utils.progressbars(progress, name="minmaxes" )
		limits = self.limits(binby, limits, selection=selection, async=True)
		tasks = [calculate(expression, limits) for expression in expressions]
		result = finish(*tasks)
		return self._async(async, result)

	@docsubst
	@stat_1d
	def min(self, expression, binby=[], limits=None, shape=default_shape, selection=False, async=False, progress=None):
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
		:param progress: {progress}
		:return: {return_stat_scalar}, the last dimension is of shape (2)
		"""
		@delayed
		def finish(result):
			return result[...,0]
		return self._async(async, finish(self.minmax(expression, binby=binby, limits=limits, shape=shape, selection=selection, async=async, progress=progress)))

	@docsubst
	@stat_1d
	def max(self, expression, binby=[], limits=None, shape=default_shape, selection=False, async=False, progress=None):
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
		:param progress: {progress}
		:return: {return_stat_scalar}, the last dimension is of shape (2)
		"""
		@delayed
		def finish(result):
			return result[...,1]
		return self._async(async, finish(self.minmax(expression, binby=binby, limits=limits, shape=shape, selection=selection, async=async, progress=progress)))

	@docsubst
	@stat_1d
	def median_approx(self, expression, percentage=50., binby=[], limits=None, shape=default_shape, percentile_shape=256, percentile_limits="minmax", selection=False, async=False):
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
		return self.percentile_approx(expression, 50, binby=binby, limits=limits, shape=shape, percentile_shape=percentile_shape, percentile_limits=percentile_limits, selection=selection, async=async)

	@docsubst
	def percentile_approx(self, expression, percentage=50., binby=[], limits=None, shape=default_shape, percentile_shape=1024, percentile_limits="minmax", selection=False, async=False):
		"""Calculate the percentile given by percentage, possible on a grid defined by binby

		NOTE: this value is approximated by calculating the cumulative distribution on a grid defined by
		percentile_shape and percentile_limits


		>>> ds.percentile_approx("x", 10), ds.percentile_approx("x", 90)
		(array([-8.3220355]), array([ 7.92080358]))
		>>> ds.percentile_approx("x", 50, binby="x", shape=5, limits=[-10, 10])
		array([[-7.56462982],
			   [-3.61036641],
			   [-0.01296306],
			   [ 3.56697863],
			   [ 7.45838367]])



		0:1:0.1
		1:1:0.2
		2:1:0.3
		3:1:0.4
		4:1:0.5

		5:1:0.6
		6:1:0.7
		7:1:0.8
		8:1:0.9
		9:1:1.0


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
		waslist, [expressions,] = vaex.utils.listify(expression)
		if not isinstance(binby, (tuple, list)):
			binby = [binby]
		else:
			binby = binby
		@delayed
		def calculate(expression, shape, limits):
			#task =  TaskStatistic(self, [expression] + binby, shape, limits, op=OP_ADD1, selection=selection)
			#self.executor.schedule(task)
			#return task
			return self.count(binby=list(binby) + [expression], shape=shape, limits=limits, selection=selection, async=True, edges=True)
		@delayed
		def finish(percentile_limits, counts_list):
			print(">>>", percentile_limits)
			results = []
			for i, counts in enumerate(counts_list):
				print(percentile_limits)
				#print(counts, counts.shape)
				#counts = counts[...,0]
				#print("percentile_limits", percentile_limits)
				#print("counts=", counts)
				#print("counts shape=", counts.shape)
				# F is the 'cumulative distribution'
				if 1:
					# remove the nan and boundary edges from the first dimension,
					#print("sums", counts.sum(), counts[2:-1].sum())
					#import pdb
					#pdb.set_trace()
					nonnans = list([slice(2, -1, None) for k in range(len(counts.shape)-1)])
					nonnans.append(slice(1,None, None)) # we're gonna get rid only of the nan's, and keep the overflow edges
					#print(counts.shape)
					#print("nonnans", nonnans)
					cumulative_grid = np.cumsum(counts.__getitem__(nonnans), -1) # convert to cumulative grid

					# keep the nans in the last dim, since we need it for the total counts
					nonnans = list([slice(2, -1, None) for k in range(len(counts.shape)-1)])
					nonnans.append(slice(0,None, None))
					totalcounts =  np.sum(counts.__getitem__(nonnans), -1)
					empty = totalcounts == 0

					original_shape = counts.shape
					shape = cumulative_grid.shape# + (original_shape[-1] - 1,) #

					#print("original_shape", original_shape)
					#print("shape", shape)
					counts = np.sum(counts, -1)
					edges_floor = np.zeros(shape[:-1] + (2,), dtype=np.int64)
					edges_ceil = np.zeros(shape[:-1] + (2,), dtype=np.int64)
					# if we have an off  # of elements, say, N=3, the center is at i=1=(N-1)/2
					# if we have an even # of elements, say, N=4, the center is between i=1=(N-2)/2 and i=2=(N/2)
					#index = (shape[-1] -1-3) * percentage/100. # the -3 is for the edges
					values = np.array((totalcounts+1) * percentage/100.) # make sure it's an ndarray
					values[empty] = 0
					floor_values = np.array(np.floor(values))
					ceil_values = np.array(np.ceil(values))
					#print("totalcounts", repr(totalcounts))
					#print("values       ", repr(values))
					#print("floor_values ", repr(floor_values))
					#print("ceil_values ", repr(ceil_values))
					#print(totalcounts, percentage, values)
					#print(">>", floor_values[8], cumulative_grid[8])
					#print(cumulative_grid.strides)
					#print(edges.shape)
					#print([k.shape for k in (cumulative_grid, values, edges)])
					#print(cumulative_grid, repr(values), edges)
					#print("shapes", [k.shape for k in [cumulative_grid, floor_values, edges_floor]])
					#print("strides", [k.strides for k in [cumulative_grid, floor_values, edges_floor]])
					#print("cumulative_grid", cumulative_grid)
					vaex.vaexfast.grid_find_edges(cumulative_grid, floor_values, edges_floor)
					vaex.vaexfast.grid_find_edges(cumulative_grid, ceil_values, edges_ceil)
					#print(floor_values)

					def index_choose(a, indices):
						# alternative to np.choise, which doesn't like the last dim to be >= 32
						out = np.zeros(a.shape[:-1])
						for i in np.ndindex(out.shape):
							out[i] = a[i+(indices[i],)]
						return out
					def calculate_x(edges, values):
						left, right = edges[...,0], edges[...,1]
						#left_value, right_value = cumulative_grid[...,left], cumulative_grid[...,right]
						#print(left.shape, cumulative_grid.shape)
						#import pdb
						#pdb.set_trace()
						left_value = index_choose(cumulative_grid, left)
						right_value = index_choose(cumulative_grid, right)
						#print("edges_left", edges)
						#print("edges_right", edges)
						#print("left_value", left_value)
						#print("right_value", right_value)
						u = np.array((values - left_value)/(right_value - left_value))
						#u[left==right] = 0
						#print("percentile_limits", percentile_limits)
						#print("shape", shape)
						# TODO: should it really be -3? not -2
						xleft, xright = percentile_limits[i][0] + (left-0.5)  * (percentile_limits[i][1] - percentile_limits[i][0]) / (shape[-1]-3),\
										percentile_limits[i][0] + (right-0.5) * (percentile_limits[i][1] - percentile_limits[i][0]) / (shape[-1]-3)
						x = xleft + (xright - xleft) * u #/2
						#print("u ", u)
						#print("xleft", xleft)
						#print("xright", xright)
						#print("# left", self.count(selection="x < %f" % xleft))
						#print("# right ", self.count(selection="x < %f" % xright))
						#print("x", x)
						return x

					x1 = calculate_x(edges_floor, floor_values)
					x2 = calculate_x(edges_ceil, ceil_values)
					u = values - floor_values
					x = x1 + (x2 - x1) * u
					#print("x1, x2, u, x", x1, x2, u, x)




					results.append(x)
					continue
				#counts[-1] += 1
				#print("shape", shape)
				#F = np.cumsum(counts, axis=-1).reshape((1,) + shape)
				F /= len(self) #np.max(F, axis=(-1))
				if 1:
					#for i in range(len(counts)):
					#	print(i, F[0,i])
					#print(counts.shape, F.shape)
					#print("counts", counts, counts.dtype)
					#print(F)
					y = np.zeros((1,) + shape[:-1])
					vaex.vaexfast.grid_interpolate(F, y, percentage/100.)
					#print("fraction", percentage/100.)
					#print("y = ", y, y * shape[0], shape[0])
					#print("y >>> ", y * (percentile_limits[i][1] - percentile_limits[i][0]) + percentile_limits[i][0])
					#return 0
					#print(percentile_limits)
					#dx = (percentile_limits[i][1] - percentile_limits[i][0]) / (shape[0]-1)
					medians.append(y[0] * (percentile_limits[i][1] - percentile_limits[i][0]) + percentile_limits[i][0])
				else:
					# we'll fill empty values with nan later on..
					ok = F[...,-1] > 0
					#F /= np.max(F, axis=(0))
					#print(F[-1])
					# find indices around 0.5 for each bin
					i2 = np.apply_along_axis(lambda x: x.searchsorted(percentage/100., side='left'), axis = -1, arr = F)
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
			#medians = np.array(medians)
			#value = np.array(vaex.utils.unlistify(waslist, medians))
			#return medians
			return results

		shape = _expand_shape(shape, len(binby))
		percentile_shapes = _expand_shape(percentile_shape, len(expressions))
		if percentile_limits:
			percentile_limits = _expand_limits(percentile_limits, len(expressions))
		limits = self.limits(binby, limits, selection=selection, async=True)
		percentile_limits = self.limits(expressions, percentile_limits, selection=selection, async=True)
		@delayed
		def calculation(limits, percentile_limits):
			#print(">>>", expressions, percentile_limits)
			#print(percentile_limits[0], list(percentile_limits[0]))
			#print(list(np.array(limits).tolist()) + list(percentile_limits[0]))
			#print("limits", limits, expressions, percentile_limits, ">>", list(limits) + [list(percentile_limits[0]))
			tasks = [calculate(expression, tuple(shape) + (percentile_shape, ), list(limits) + [list(percentile_limit)])
					 for    percentile_shape,  percentile_limit, expression
					 in zip(percentile_shapes, percentile_limits, expressions)]
			return finish(percentile_limits, delayed_args(*tasks))
			#return tasks
		result = calculation(limits, percentile_limits)
		@delayed
		def finish2(grid):
			value = vaex.utils.unlistify(waslist, np.array(grid))
			return value
		return self._async(async, finish2(result))

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
		>>> ds.percentile_approx("x", 5), ds.percentile_approx("x", 95)
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
							limits =  self.limits_percentage(expression, number, async=False)
						elif type in ["%s", "%square", "percentsquare"]:
							limits =  self.limits_percentage(expression, number, square=True, async=True)
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


	def plot_widget(self, x, y, z=None, grid=None, shape=256, limits=None, what="count(*)", figsize=None,
			 f="identity", figure_key=None, fig=None, axes=None, xlabel=None, ylabel=None, title=None,
			 show=True, selection=[None, True], colormap="afmhot", grid_limits=None, normalize="normalize",
			 grid_before=None,
			 what_kwargs={}, type="default",
			 scales=None, tool_select=False, bq_cleanup=True,
			 backend="bqplot",
			 **kwargs):
		import vaex.notebook.plot
		backend = vaex.notebook.plot.create_backend(backend)
		cls = vaex.notebook.plot.get_type(type)
		plot2d = cls(backend=backend, dataset=self, x=x, y=y, z=z, grid=grid, shape=shape, limits=limits, what=what,
				f=f, figure_key=figure_key, fig=fig,
				selection=selection, grid_before=grid_before,
				grid_limits=grid_limits, normalize=normalize, colormap=colormap, what_kwargs=what_kwargs, **kwargs)
		if show:
			plot2d.show()
		return plot2d
	def plot_bq(self, x, y, grid=None, shape=256, limits=None, what="count(*)", figsize=None,
			 f="identity", figure_key=None, fig=None, axes=None, xlabel=None, ylabel=None, title=None,
			 show=True, selection=[None, True], colormap="afmhot", grid_limits=None, normalize="normalize",
			 grid_before=None,
			 what_kwargs={}, type="default",
			 scales=None, tool_select=False, bq_cleanup=True,
			 **kwargs):
		import vaex.ext.bqplot
		cls = vaex.ext.bqplot.get_class(type)
		plot2d = cls(dataset=self, x=x, y=y, grid=grid, shape=shape, limits=limits, what=what,
				f=f, figure_key=figure_key, fig=fig,
				selection=selection, grid_before=grid_before,
				grid_limits=grid_limits, normalize=normalize, colormap=colormap, what_kwargs=what_kwargs, **kwargs)
		if show:
			plot2d.show()
		return plot2d

	#"""Use bqplot to create an interactive plot, this method is subject to change, it is currently a tech demo"""
		#subspace = self(x, y)
		#return subspace.plot_bq(grid, size, limits, square, center, weight, figsize, aspect, f, fig, axes, xlabel, ylabel, title,
		#						group_by, group_limits, group_colors, group_labels, group_count, cmap, scales, tool_select, bq_cleanup, **kwargs)


	def healpix_count(self, expression=None, healpix_expression=None, healpix_max_level=12, healpix_level=8, binby=None, limits=None, shape=default_shape, async=False, progress=None, selection=None):
		"""Count non missing value for expression on an array which represents healpix data.

		:param expression: Expression or column for which to count non-missing values, or None or '*' for counting the rows
		:param healpix_expression: {healpix_max_level}
		:param healpix_max_level: {healpix_max_level}
		:param healpix_level: {healpix_level}
		:param binby: {binby}, these dimension follow the first healpix dimension.
		:param limits: {limits}
		:param shape: {shape}
		:param selection: {selection}
		:param async: {async}
		:param progress: {progress}
		:return:
		"""
		#if binby is None:
		import healpy as hp
		if healpix_expression is None:
			if self.ucds.get("source_id", None) == 'meta.id;meta.main': # we now assume we have gaia data
				healpix_expression = "source_id/34359738368"

		if healpix_expression is None:
			raise ValueError("no healpix_expression given, and was unable to guess")

		reduce_level = healpix_max_level - healpix_level
		NSIDE = 2**healpix_level
		nmax = hp.nside2npix(NSIDE)
		scaling = 4**reduce_level
		expr = "%s/%s" % (healpix_expression, scaling)
		binby = [expr] + ([] if binby is None else _ensure_list(binby))
		shape = (nmax,) + _expand_shape(shape, len(binby)-1)
		epsilon = 1./scaling/2
		limits = [[-epsilon, nmax-epsilon]] + ([] if limits is None else limits)
		return self.count(expression, binby=binby, limits=limits, shape=shape, async=async, progress=progress, selection=selection)

	def healpix_plot(self, healpix_expression="source_id/34359738368", healpix_max_level=12, healpix_level=8, what="count(*)", selection=None,
					 grid=None,
					 healpix_input="equatorial", healpix_output="galactic", f=None,
					 colormap="afmhot", grid_limits=None, image_size =800, nest=True,
					 figsize=None, interactive=False,title="", smooth=None, show=False, colorbar=True,
 					 rotation=(0,0,0)):
		"""

		:param healpix_expression: {healpix_max_level}
		:param healpix_max_level: {healpix_max_level}
		:param healpix_level: {healpix_level}
		:param what: {what}
		:param selection: {selection}
		:param grid: {grid}
		:param healpix_input: Specificy if the healpix index is in "equatorial", "galactic" or "ecliptic".
		:param healpix_output: Plot in "equatorial", "galactic" or "ecliptic".
		:param f: function to apply to the data
		:param colormap: matplotlib colormap
		:param grid_limits: Optional sequence [minvalue, maxvalue] that determine the min and max value that map to the colormap (values below and above these are clipped to the the min/max). (default is [min(f(grid)), max(f(grid)))
		:param image_size: size for the image that healpy uses for rendering
		:param nest: If the healpix data is in nested (True) or ring (False)
		:param figsize: If given, modify the matplotlib figure size. Example (14,9)
		:param interactive: (Experimental, uses healpy.mollzoom is True)
		:param title: Title of figure
		:param smooth: apply gaussian smoothing, in degrees
		:param show: Call matplotlib's show (True) or not (False, defaut)
		:param rotation: Rotatate the plot, in format (lon, lat, psi) such that (lon, lat) is the center, and rotate on the screen by angle psi. All angles are degrees.
		:return:
		"""
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
			epsilon = 1. / scaling / 2
			grid = self._stat(what=what, binby="%s/%s" % (healpix_expression, scaling), limits=[-epsilon, nmax-epsilon], shape=nmax, selection=selection)
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
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			f(fgrid, unit=what_label, rot=rotation, nest=nest ,title=title, coord=[coord_map[healpix_input], coord_map[healpix_output]], cmap=colormap, hold=True, xsize=image_size,
					min=grid_min, max=grid_max, cbar=colorbar)#, min=6-1, max=8.7-1)
		if show:
			plt.show()

	@docsubst
	@stat_1d
	def _stat(self, what="count(*)", what_kwargs={}, binby=[], limits=None, shape=default_shape, selection=False, async=False, progress=None):
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
						grid = getattr(self, function)(arguments, binby=binby, limits=limits, shape=shape,
								   selection=selections, progress=progress, async=async)
					elif function == "count":
						grid = self.count(arguments, binby, shape=shape, limits=limits, selection=selections,
								  progress=progress, async=async)
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
		s = finish(delayed_list(grids))
		return self._async(async, s)

	def scatter(self, x, y, xerr=None, yerr=None, s_expr=None, c_expr=None, selection=None, length_limit=50000, length_check=True, label=None, xlabel=None, ylabel=None, errorbar_kwargs={}, **kwargs):
		"""Convenience wrapper around pylab.scatter when for working with small datasets or selections

		:param x: Expression for x axis
		:param y: Idem for y
		:param s_expr: When given, use if for the s (size) argument of pylab.scatter
		:param c_expr: When given, use if for the c (color) argument of pylab.scatter
		:param selection: Single selection expression, or None
		:param length_limit: maximum number of rows it will plot
		:param length_check: should we do the maximum row check or not?
		:param xlabel: label for x axis, if None .label(x) is used
		:param ylabel: label for y axis, if None .label(y) is used
		:param errorbar_kwargs: extra dict with arguments passed to plt.errorbar
		:param kwargs: extra arguments passed to pylab.scatter
		:return:
		"""
		import pylab as plt
		if length_check:
			count = self.count(selection=selection)
			if count > length_limit:
				raise ValueError("the number of rows (%d) is above the limit (%d), pass length_check=False, or increase length_limit" % (count, length_limit))
		x_values = self.evaluate(x, selection=selection)
		y_values = self.evaluate(y, selection=selection)
		if s_expr:
			kwargs["s"] = self.evaluate(s_expr, selection=selection)
		if c_expr:
			kwargs["c"] = self.evaluate(c_expr, selection=selection)
		plt.xlabel(xlabel or self.label(x))
		plt.ylabel(ylabel or self.label(y))
		s = plt.scatter(x_values, y_values, **kwargs)
		if label:
			label_values = self.evaluate(label)
			for i, label_value in enumerate(label_values):
				plt.annotate(label_value, (x_values[i],y_values[i]))
		xerr_values = None
		yerr_values = None
		if xerr is not None:
			if _issequence(xerr):
				assert len(xerr) == 2, "if xerr is a sequence it should be of length 2"
				xerr_values = [self.evaluate(xerr[0], selection=selection), self.evaluate(xerr[1], selection=selection)]
			else:
				xerr_values = self.evaluate(xerr, selection=selection)
		if yerr is not None:
			if _issequence(yerr):
				assert len(yerr) == 2, "if yerr is a sequence it should be of length 2"
				yerr_values = [self.evaluate(yerr[0], selection=selection), self.evaluate(yerr[1], selection=selection)]
			else:
				yerr_values = self.evaluate(yerr, selection=selection)
		if xerr_values is not None or yerr_values is not None:
			plt.errorbar(x_values, y_values, yerr=yerr_values, xerr=xerr_values, **errorbar_kwargs)
		return s

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
		invert = lambda x: dict((v, k) for k, v in x.items())
		#visual_default_reverse = invert(visual_default)
		#visual_ = visual_default
		#visual = dict(visual) # copy for modification
		# add entries to avoid mapping multiple times to the same axis
		free_visual_axes = list(visual_default.keys())
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

		#normalize_axis = _ensure_list(normalize_axis)

		fs = _expand(f, total_grid.shape[grid_axes[normalize_axis]])
		#assert len(vwhat)
		#labels["y"] = ylabels
		what_labels = []
		if grid is None:
			grid_of_grids = []
			for i, (binby, limits) in enumerate(zip(x, xlimits)):
				grid_of_grids.append([])
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
						functions = ["mean", "sum", "std", "var", "correlation", "covar", "min", "max", "median_approx"]
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
							grid = getattr(self, function)(arguments, binby=binby, limits=limits, shape=shape, selection=selections, async=True)
						elif function == "count":
							grid = self.count(arguments, binby, shape=shape, limits=limits, selection=selections, async=True)
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
					grid_of_grids[-1].append(grid)
			self.executor.execute()
			for i, (binby, limits) in enumerate(zip(x, xlimits)):
				for j, what in enumerate(whats):
					grid = grid_of_grids[i][j].get()
					total_grid[i,j,:,:] = grid[:,None,...]
			labels["what"] = what_labels
		else:
			dims_left = 6-len(grid.shape)
			total_grid = np.broadcast_to(grid, (1,) * dims_left + grid.shape)

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

		#visual_grid = np.moveaxis(total_grid, move.keys(), move.values())
		# np.moveaxis is in np 1.11 only?, use transpose
		axes = [None] * len(move)
		for key, value in move.items():
			axes[value] = key
		visual_grid = np.transpose(total_grid, axes)

		logger.debug("grid shape: %r", total_grid.shape)
		logger.debug("visual: %r", visual.items())
		logger.debug("move: %r", move)
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
				with np.errstate(divide='ignore', invalid='ignore'): # these are fine, we are ok with nan's in vaex
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


				row = facet_index // facet_columns
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

	def plot3d(self, x, y, z, vx=None, vy=None, vz=None, vwhat=None, limits=None, grid=None, what="count(*)", shape=128, selection=[None, True], f=None,
			   vcount_limits=None,
			   smooth_pre=None, smooth_post=None, grid_limits=None, normalize="normalize", colormap="afmhot",
			   figure_key=None, fig=None,
			   lighting=True, level=[0.1, 0.5, 0.9], opacity=[0.01, 0.05, 0.1], level_width=0.1,
			   show=True, **kwargs):
		"""Use at own risk, requires ipyvolume"""
		import vaex.ext.ipyvolume
		#vaex.ext.ipyvolume.
		cls = vaex.ext.ipyvolume.PlotDefault
		plot3d = cls(dataset=self, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz,
					 grid=grid, shape=shape, limits=limits, what=what,
				f=f, figure_key=figure_key, fig=fig,
				selection=selection, smooth_pre=smooth_pre, smooth_post=smooth_post,
				grid_limits=grid_limits, vcount_limits=vcount_limits, normalize=normalize, colormap=colormap, **kwargs)
		if show:
			plot3d.show()
		return plot3d

		import ipyvolume.pylab as p3
		import ipyvolume.examples
		binby = [x,y,z]
		limits = self.limits(binby, limits)
		print(limits)
		p3.clear()
		volume_data = self._stat(what=what, binby=binby, limits=limits, shape=shape, selection=selection)
		if smooth_pre:
			volume_data = grid = vaex.grids.gf(volume_data, smooth_pre)
		f = _parse_f(f)
		volume_data = f(volume_data)
		if smooth_post:
			volume_data = grid = vaex.grids.gf(volume_data, smooth_post)
		p3.volshow(volume_data.T, lighting=lighting, level=level, opacity=opacity, level_width=level_width)

		#p3.xyzlabel(self.label(x), self.label(y), self.label(z))
		if vx and vy and vz:
			# TODO: we want to count vx vy and vz... should we multiply or add them?
			vcount = self.count(vx, binby=[x, y, z], limits=limits, shape=vshape, selection=selection)
			vx = self.mean(vx, binby=[x, y, z], limits=limits, shape=vshape, selection=selection)
			vy = self.mean(vy, binby=[x, y, z], limits=limits, shape=vshape, selection=selection)
			vz = self.mean(vz, binby=[x, y, z], limits=limits, shape=vshape, selection=selection)
			ok = np.isfinite(vx) & np.isfinite(vy) & np.isfinite(vz) & (vcount > vmin)
			x, y, z = ipyvolume.examples.xyz(16, limits=limits, sparse=False, centers=True)
			v1d = [k[ok] for k in [x,y,z,vx,vy,vz]]
			q = p3.quiver(*v1d, size=vsize, color=vcolor)
		p3.xlim(*limits[0])
		p3.ylim(*limits[1])
		p3.zlim(*limits[2])


		if show:
			p3.show()
		return p3.gcc()


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
					functions = ["mean", "sum", "std"]
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
		xar = np.arange(N+1) / (N-0.) * (xmax-xmin) + xmin
		if facet:
			import math
			rows, columns = int(math.ceil(facet_count / 4.)), 4
			values = np.linspace(facet_limits[0], facet_limits[1], facet_count+1)
			for i in range(facet_count):
				ax = pylab.subplot(rows, columns, i+1)
				value = ax.plot(xar, ngrid[i], drawstyle="steps-mid", label=label or x, **kwargs)
				v1, v2 = values[i], values[i+1]
				pylab.xlabel(xlabel or x)
				pylab.ylabel(ylabel or what)
				ax.set_title("%3f <= %s < %3f" % (v1, facet_expression, v2))
				#pylab.show()
		else:
			#im = pylab.imshow(rgrid, extent=np.array(limits[:2]).flatten(), origin="lower", aspect=aspect)
			pylab.xlabel(xlabel or self.label(x))
			pylab.ylabel(ylabel or what)
			#print(xar, ngrid)
			# repeat the first element, that's how plot/steps likes it..
			g = np.concatenate([ngrid[0:1], ngrid])
			value = pylab.plot(xar, g, drawstyle="steps-pre", label=label or x, **kwargs)
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
		for name in self.get_column_names(virtual=True, strings=True):
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
			unit = unit_or_quantity.unit if hasattr(unit_or_quantity, "unit") else unit_or_quantity
			return unit if isinstance(unit, astropy.units.Unit) else None
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
			name = os.path.abspath(self.path).replace(os.path.sep, "_")[:250]  # should not be too long for most os'es
			name = name.replace(":", "_") # for windows drive names
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
		import astropy.units
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
		import astropy.units
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
		import vaex.legacy
		return vaex.legacy.Subspaces([self(*expressions, **kwargs) for expressions in expressions_list])

	def combinations(self, expressions_list=None, dimension=2, exclude=None, **kwargs):
		"""Generate a list of combinations for the possible expressions for the given dimension

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

	def set_variable(self, name, expression_or_value, write=True):
		"""Set the variable to an expression or value defined by expression_or_value

		:Example:
		>>> ds.set_variable("a", 2.)
		>>> ds.set_variable("b", "a**2")
		>>> ds.get_variable("b")
		'a**2'
		>>> ds.evaluate_variable("b")
		4.0

		:param name: Name of the variable
		:param write: write variable to meta file
		:param expression: value or expression
		"""
		self.variables[name] = expression_or_value
		if write:
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




	def evaluate(self, expression, i1=None, i2=None, out=None, selection=None):
		"""Evaluate an expression, and return a numpy array with the results for the full column or a part of it.

		Note that this is not how vaex should be used, since it means a copy of the data needs to fit in memory.

		To get partial results, use i1 and i2/

		:param str expression: Name/expression to evaluate
		:param int i1: Start row index, default is the start (0)
		:param int i2: End row index, default is the length of the dataset
		:param ndarray out: Output array, to which the result may be written (may be used to reuse an array, or write to
		a memory mapped array)
		:param selection: selection to apply
		:return:
		"""
		raise NotImplementedError

	@docsubst
	def to_items(self, column_names=None, selection=None, strings=True, virtual=False):
		"""Return a list of [(column_name, ndarray), ...)] pairs where the ndarray corresponds to the evaluated data

		:param column_names: list of column names, to export, when None Dataset.get_column_names(strings=strings, virtual=virtual) is used
		:param selection: {selection}
		:param strings: argument passed to Dataset.get_column_names when column_names is None
		:param virtual: argument passed to Dataset.get_column_names when column_names is None
		:return: list of (name, ndarray) pairs
		"""
		items = []
		for name in column_names or self.get_column_names(strings=strings, virtual=virtual):
			items.append((name, self.evaluate(name, selection=selection)))
		return items

	@docsubst
	def to_dict(self, column_names=None, selection=None, strings=True, virtual=False):
		"""Return a dict containing the ndarray corresponding to the evaluated data

		:param column_names: list of column names, to export, when None Dataset.get_column_names(strings=strings, virtual=virtual) is used
		:param selection: {selection}
		:param strings: argument passed to Dataset.get_column_names when column_names is None
		:param virtual: argument passed to Dataset.get_column_names when column_names is None
		:return: dict
		"""
		return dict(self.to_items(column_names=column_names, selection=selection, strings=strings, virtual=virtual))

	@docsubst
	def to_copy(self, column_names=None, selection=None, strings=True, virtual=False):
		"""Return a copy of the Dataset, if selection is None, it does not copy the data, it just has a reference

		:param column_names: list of column names, to copy, when None Dataset.get_column_names(strings=strings, virtual=virtual) is used
		:param selection: {selection}
		:param strings: argument passed to Dataset.get_column_names when column_names is None
		:param virtual: argument passed to Dataset.get_column_names when column_names is None
		:return: dict
		"""
		ds = vaex.from_items(*self.to_items(column_names=column_names, selection=selection, strings=strings, virtual=virtual))
		ds.copy_metadata(self)
		return ds

	def copy_metadata(self, other):
		for name in self.get_column_names(strings=True):
			if name in other.units:
				self.units[name] = other.units[name]
			if name in other.descriptions:
				self.descriptions[name] = other.descriptions[name]
			if name in other.ucds:
				self.ucds[name] = other.ucds[name]
		self.description = other.description

	@docsubst
	def to_pandas_df(self, column_names=None, selection=None, strings=True, virtual=False, index_name=None):
		"""Return a pandas DataFrame containing the ndarray corresponding to the evaluated data

		 If index is given, that column is used for the index of the dataframe.

		 :Example:
		 >>> df = ds.to_pandas_df(["x", "y", "z"])
		 >>> ds_copy = vx.from_pandas(df)

		:param column_names: list of column names, to export, when None Dataset.get_column_names(strings=strings, virtual=virtual) is used
		:param selection: {selection}
		:param strings: argument passed to Dataset.get_column_names when column_names is None
		:param virtual: argument passed to Dataset.get_column_names when column_names is None
		:param index_column: if this column is given it is used for the index of the DataFrame
		:return: pandas.DataFrame object
		"""
		import pandas as pd
		data = self.to_dict(column_names=column_names, selection=selection, strings=strings, virtual=virtual)
		if index_name is not None:
			if index_name in data:
				index = data.pop(index_name)
			else:
				index = self.evaluate(index_name, selection=selection)
		else:
			index = None
		df = pd.DataFrame(data=data, index=index)
		if index is not None:
			df.index.name = index_name
		return df

	@docsubst
	def to_astropy_table(self, column_names=None, selection=None, strings=True, virtual=False, index=None):
		"""Returns a astropy table object containing the ndarrays corresponding to the evaluated data

		:param column_names: list of column names, to export, when None Dataset.get_column_names(strings=strings, virtual=virtual) is used
		:param selection: {selection}
		:param strings: argument passed to Dataset.get_column_names when column_names is None
		:param virtual: argument passed to Dataset.get_column_names when column_names is None
		:param index: if this column is given it is used for the index of the DataFrame
		:return: astropy.table.Table object
		"""
		from astropy.table import Table, Column, MaskedColumn
		meta = dict()
		meta["name"] = self.name
		meta["description"] = self.description

		table = Table(meta=meta)
		for name, data in self.to_items(column_names=column_names, selection=selection, strings=strings, virtual=virtual):
			meta = dict()
			if name in self.ucds:
				meta["ucd"] = self.ucds[name]
			if np.ma.isMaskedArray(data):
				cls = MaskedColumn
			else:
				cls = Column
			table[name] = cls(data, unit=self.unit(name), description=self.descriptions.get(name), meta=meta)
		return table


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
			if name not in self.column_names:
				self.column_names.append(name)
		else:
			raise ValueError("functions not yet implemented")

	def rename_column(self, name, new_name):
		"""Renames a column, not this is only the in memory name, this will not be reflected on disk"""
		data = self.columns[name]
		del self.columns[name]
		self.column_names[self.column_names.index(name)] = new_name
		self.columns[new_name] = data
		for d in [self.ucds, self.units, self.descriptions]:
			if name in d:
				d[new_name] = d[name]
				del d[name]

	def add_column_healpix(self, name="healpix", longitude="ra", latitude="dec", degrees=True, healpix_order=12, nest=True):
		"""Add a healpix (in memory) column based on a longitude and latitude

		:param name: Name of column
		:param longitude: longitude expression
		:param latitude: latitude expression  (astronomical convenction latitude=90 is north pole)
		:param degrees: If lon/lat are in degrees (default) or radians.
		:param healpix_order: healpix order, >= 0
		:param nest: Nested healpix (default) or ring.
		"""
		import healpy as hp
		if degrees:
			scale = "*pi/180"
		else:
			scale = ""
		# TODO: multithread this
		phi = self.evaluate("(%s)%s" % (longitude, scale))
		theta = self.evaluate("pi/2-(%s)%s" % (latitude, scale))
		hp_index = hp.ang2pix(hp.order2nside(healpix_order), theta, phi, nest=nest)
		self.add_column("healpix", hp_index)

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

	def add_virtual_columns_distance_from_parallax(self, parallax="parallax", distance_name="distance", parallax_uncertainty=None, uncertainty_postfix="_uncertainty"):
		"""Convert parallax to distance (i.e. 1/parallax)

		:param parallax: expression for the parallax, e.g. "parallax"
		:param distance_name: name for the virtual column of the distance, e.g. "distance"
		:param parallax_uncertainty: expression for the uncertainty on the parallax, e.g. "parallax_error"
		:param uncertainty_postfix: distance_name + uncertainty_postfix is the name for the virtual column, e.g. "distance_uncertainty" by default
		:return:
		"""
		"""


		"""
		import astropy.units
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

	def add_virtual_columns_eq2ecl(self, long_in="ra", lat_in="dec", long_out="lambda_", lat_out="beta", input=None, output=None, name_prefix="__celestial_eq2ecl", radians=False):
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

	def add_virtual_columns_eq2gal(self, long_in="ra", lat_in="dec", long_out="l", lat_out="b", input=None, output=None, name_prefix="__celestial_eq2gal", radians=False):
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

	@docsubst
	def add_virtual_columns_cartesian_to_polar(self, x="x", y="y", radius_out="r_polar", azimuth_out="phi_polar",
												 cov_matrix_x_y=None,
												 covariance_postfix="_covariance",
												 uncertainty_postfix="_uncertainty",
												 radians=False):
		"""Convert cartesian to polar coordinates

		:param x: expression for x
		:param y: expression for y
		:param radius_out: name for the virtual column for the radius
		:param azimuth_out: name for the virtual column for the azimuth angle
		:param cov_matrix_x_y: {cov_matrix}
		:param covariance_postfix:
		:param uncertainty_postfix:
		:param radians: if True, azimuth is in radians, defaults to degrees
		:return:
		"""
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

	def add_virtual_columns_cartesian_velocities_to_spherical(self, x="x", y="y", z="z", vx="vx", vy="vy", vz="vz", vr="vr", vlong="vlong", vlat="vlat", distance=None):
		"""Concert velocities from a cartesian to a spherical coordinate system

		TODO: errors

		:param x: name of x column (input)
		:param y:         y
		:param z:         z
		:param vx:       vx
		:param vy:       vy
		:param vz:       vz
		:param vr: name of the column for the radial velocity in the r direction (output)
		:param vlong: name of the column for the velocity component in the longitude direction  (output)
		:param vlat: name of the column for the velocity component in the latitude direction, positive points to the north pole (output)
		:param distance: Expression for distance, if not given defaults to sqrt(x**2+y**2+z**2), but if this column already exists, passing this expression may lead to a better performance
		:return:
		"""
		#see http://www.astrosurf.com/jephem/library/li110spherCart_en.htm
		if distance is None:
			distance = "sqrt({x}**2+{y}**2+{z}**2)".format(**locals())
		self.add_virtual_column(vr, "({x}*{vx}+{y}*{vy}+{z}*{vz})/{distance}".format(**locals()))
		self.add_virtual_column(vlong, "-({vx}*{y}-{x}*{vy})/sqrt({x}**2+{y}**2)".format(**locals()))
		self.add_virtual_column(vlat, "-({z}*({x}*{vx}+{y}*{vy}) - ({x}**2+{y}**2)*{vz})/( {distance}*sqrt({x}**2+{y}**2) )".format(**locals()))


	def add_virtual_columns_cartesian_velocities_to_pmvr(self, x="x", y="y", z="z", vx="vx", vy="vy", vz="vz", vr="vr", pm_long="pm_long", pm_lat="pm_lat", distance=None):
		"""Concert velocities from a cartesian system to proper motions and radial velocities

		TODO: errors

		:param x: name of x column (input)
		:param y:         y
		:param z:         z
		:param vx:       vx
		:param vy:       vy
		:param vz:       vz
		:param vr: name of the column for the radial velocity in the r direction (output)
		:param pm_long: name of the column for the proper motion component in the longitude direction  (output)
		:param pm_lat: name of the column for the proper motion component in the latitude direction, positive points to the north pole (output)
		:param distance: Expression for distance, if not given defaults to sqrt(x**2+y**2+z**2), but if this column already exists, passing this expression may lead to a better performance
		:return:
		"""
		if distance is None:
			distance = "sqrt({x}**2+{y}**2+{z}**2)".format(**locals())
		k = 4.74057
		self.add_variable("k", k, overwrite=False)
		self.add_virtual_column(vr, "({x}*{vx}+{y}*{vy}+{z}*{vz})/{distance}".format(**locals()))
		self.add_virtual_column(pm_long, "-({vx}*{y}-{x}*{vy})/sqrt({x}**2+{y}**2)/{distance}/k".format(**locals()))
		self.add_virtual_column(pm_lat, "-({z}*({x}*{vx}+{y}*{vy}) - ({x}**2+{y}**2)*{vz})/( ({x}**2+{y}**2+{z}**2) * sqrt({x}**2+{y}**2) )/k".format(**locals()))


	def add_virtual_columns_cartesian_velocities_to_polar(self, x="x", y="y", vx="vx", radius_polar=None, vy="vy", vr_out="vr_polar", vazimuth_out="vphi_polar",
												 cov_matrix_x_y_vx_vy=None,
												 covariance_postfix="_covariance",
												 uncertainty_postfix="_uncertainty"):
		"""Convert cartesian to polar velocities.

		:param x:
		:param y:
		:param vx:
		:param radius_polar: Optional expression for the radius, may lead to a better performance when given.
		:param vy:
		:param vr_out:
		:param vazimuth_out:
		:param cov_matrix_x_y_vx_vy:
		:param covariance_postfix:
		:param uncertainty_postfix:
		:return:
		"""
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
		"""Convert proper motion to perpendicular velocities.

		:param distance:
		:param pm_long:
		:param pm_lat:
		:param vl:
		:param vb:
		:param cov_matrix_distance_pm_long_pm_lat:
		:param uncertainty_postfix:
		:param covariance_postfix:
		:param radians:
		:return:
		"""
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
				self.set_variable(matrix_name +"_%d%d" % (i,j), matrix[i,j].item())
		self.virtual_columns[xnew] = "{m}_00 * {x} + {m}_01 * {y}".format(**locals())
		self.virtual_columns[ynew] = "{m}_10 * {x} + {m}_11 * {y}".format(**locals())


	def add_virtual_columns_spherical_to_cartesian(self, alpha, delta, distance, xname="x", yname="y", zname="z",
												   cov_matrix_alpha_delta_distance=None,
												   covariance_postfix="_covariance",
												   uncertainty_postfix="_uncertainty",
												   center=None, center_name="solar_position", radians=False):
		"""Convert spherical to cartesian coordinates.



		:param alpha:
		:param delta: polar angle, ranging from the -90 (south pole) to 90 (north pole)
		:param distance: radial distance, determines the units of x, y and z
		:param xname:
		:param yname:
		:param zname:
		:param cov_matrix_alpha_delta_distance:
		:param covariance_postfix:
		:param uncertainty_postfix:
		:param center:
		:param center_name:
		:param radians:
		:return:
		"""
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

	def add_virtual_columns_cartesian_to_spherical(self, x="x", y="y", z="z", alpha="l", delta="b", distance="distance", radians=False, center=None, center_name="solar_position"):
		"""Convert cartesian to spherical coordinates.



		:param x:
		:param y:
		:param z:
		:param alpha:
		:param delta: name for polar angle, ranges from -90 to 90 (or -pi to pi when radians is True).
		:param distance:
		:param radians:
		:param center:
		:param center_name:
		:return:
		"""
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

	def add_virtual_columns_projection_gnomic(self, alpha, delta, alpha0=0, delta0=0, x="x", y="y", radians=False, postfix=""):
		if not radians:
			alpha = "pi/180.*%s" % alpha
			delta = "pi/180.*%s" % delta
			alpha0 = alpha0 * np.pi/180
			delta0 = delta0 * np.pi/180
		transform = "" if radians else "*180./pi"
		# aliases
		ra = alpha
		dec = delta
		ra_center = alpha0
		dec_center = delta0
		gnomic_denominator = 'sin({dec_center}) * tan({dec}) + cos({dec_center}) * cos({ra} - {ra_center})'.format(**locals())
		denominator_name = 'gnomic_denominator' + postfix
		xi = 'sin({ra} - {ra_center})/{denominator_name}{transform}'.format(**locals())
		eta = '(cos({dec_center}) * tan({dec}) - sin({dec_center}) * cos({ra} - {ra_center}))/{denominator_name}{transform}'.format(**locals())
		self.add_virtual_column(denominator_name, gnomic_denominator)
		self.add_virtual_column(x, xi)
		self.add_virtual_column(y, eta)
		#return xi, eta


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

		Example:
		>>> dataset.add_virtual_column("r", "sqrt(x**2 + y**2 + z**2)")
		>>> dataset.select("r < 10")

		:param: str name: name of virtual column
		:param: expression: expression for the column
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

	def info(self, description=True):
		from IPython import display
		self._output_css()
		display.display(display.HTML(self._info(description=description)))

	def _info(self, description=True):
		parts = ["""<div><h2>{}</h2> {:,} rows</div>""".format(self.name, len(self))]
		if hasattr(self, 'path'):
			parts += ["""<div>path: <i>%s</i></div>""" % (self.path)]
		if self.description:
			parts += ["""<div>Description: {}</div>""".format(self.description)]
		parts += ["<h2>Columns:</h2>"]
		parts += ["<table class='table-striped'>"]
		parts += ["<thead><tr>"]
		for header in "column type unit description expression".split():
			if description or header != "description":
				parts += ["<th>%s</th>" % header]
		parts += ["</tr></thead>"]
		for name in self.get_column_names(virtual=True, strings=True):
			parts += ["<tr>"]
			parts += ["<td>%s</td>" % name]
			virtual = name not in self.column_names
			if name in self.column_names:
				type = self.dtype(name).name
			else:
				type = "</i>virtual column</i>"
			parts += ["<td>%s</td>" % type]
			units = self.unit(name)
			units = units.to_string("latex_inline") if units else ""
			parts += ["<td>%s</td>" % units]
			if description:
				parts += ["<td ><pre>%s</pre></td>" % self.descriptions.get(name, "")]
			if virtual:
				parts += ["<td><code>%s</code></td>" % self.virtual_columns[name]]
			else:
				parts += ["<td></td>"]
			parts += ["</tr>"]
		parts += "</table>"

		parts += ["<h2>Variables:</h2>"]
		parts += ["<table class='table-striped'>"]
		parts += ["<thead><tr>"]
		for header in "variable type unit description expression".split():
			if description or header != "description":
				parts += ["<th>%s</th>" % header]
		parts += ["</tr></thead>"]
		for name in self.variables.keys():
			parts += ["<tr>"]
			parts += ["<td>%s</td>" % name]
			type = self.dtype(name).name
			parts += ["<td>%s</td>" % type]
			units = self.unit(name)
			units = units.to_string("latex_inline") if units else ""
			parts += ["<td>%s</td>" % units]
			if description:
				parts += ["<td ><pre>%s</pre></td>" % self.descriptions.get(name, "")]
			parts += ["<td><code>%s</code></td>" % (self.variables[name], )]
			parts += ["</tr>"]
		parts += "</table>"

		return "".join(parts)+ "<h2>Data:</h2>" +self._head_and_tail()

	def head(self, n=10):
		self.cat(i1=0, i2=min(len(self), n))

	def tail(self, n=10):
		N = len(self)
		self.cat(i1=max(0, N-n), i2=min(len(self), N))

	def _head_and_tail(self, n=10):
		N = len(self)
		if N <= n*2:
			return self._as_html_table(0, N)
		else:
			return self._as_html_table(0, n, N-n, N)
	def head_and_tail(self, n=10):
		from IPython import display
		display.display(display.HTML(self._head_and_tail(n)))

	def cat(self, i1, i2):
		from IPython import display
		html = self._as_html_table(i1, i2)
		display.display(display.HTML(html))

	def _as_html_table(self, i1, i2, j1=None, j2=None):
		parts = [] #"""<div>%s (length=%d)</div>""" % (self.name, len(self))]
		parts += ["<table class='table-striped'>"]

		column_names = self.get_column_names(virtual=True, strings=True)
		parts += ["<thead><tr>"]
		for name in ["#"] + column_names:
			parts += ["<th>%s</th>" % name]
		parts += ["</tr></thead>"]
		def table_part(k1, k2, parts):
			data_parts = {}
			N = k2-k1
			for name in column_names:
				try:
					data_parts[name] = self.evaluate(name, i1=k1, i2=k2)
				except:
					data_parts[name] = ["error"] * (N)
			for i in range(k2-k1):
				parts += ["<tr>"]
				parts += ["<td>%r</td>" % (i+k1)]
				for name in column_names:
					parts += ["<td>%r</td>" % data_parts[name][i]]
				parts += ["</tr>"]
			return parts
		parts = table_part(i1, i2, parts)
		if j1 is not None and j2 is not None:
			for i in range(len(column_names)+1):
				parts += ["<td>...</td>"]
			parts = table_part(j1, j2, parts)
		parts += "</table>"
		html = "".join(parts)
		return html

	def _output_css(self):
		css = """.vaex-description pre {
		  max-width : 450px;
		  white-space : nowrap;
		  overflow : hidden;
		  text-overflow: ellipsis;
		}

		.vex-description pre:hover {
		  max-width : initial;
		  white-space: pre;
		}"""
		from IPython import display
		style = "<style>%s</style>" % css
		display.display(display.HTML(style))

	def _repr_html_(self):
		"""Representation for Jupyter"""
		self._output_css()
		return self._info()


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
		name = _normalize_selection_name(name)
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
		self.selection_history_indices[name] -= 1
		self.signal_selection_changed.emit(self)
		logger.debug("undo: selection history is %r, index is %r", selection_history, self.selection_history_indices[name])


	def selection_redo(self, name="default", executor=None):
		"""Redo selection, for the name"""
		logger.debug("redo")
		executor = executor or self.executor
		assert self.selection_can_redo(name=name)
		selection_history = self.selection_histories[name]
		index = self.selection_history_indices[name]
		next = selection_history[index+1]
		self.selection_history_indices[name] += 1
		self.signal_selection_changed.emit(self)
		logger.debug("redo: selection history is %r, index is %r", selection_history, index)

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
			self._selection(create, name)

	def select_nothing(self, name="default"):
		"""Select nothing"""
		logger.debug("selecting nothing")
		self.select(None, name=name)
	#self.signal_selection_changed.emit(self)

	def select_rectangle(self, x, y, limits, mode="replace"):
		"""Select a 2d rectangular box in the space given by x and y, bounds by limits

		Example:
		>>> ds.select_box('x', 'y', [(0, 10), (0, 1)])

		:param x: expression for the x space
		:param y: expression fo the y space
		:param limits: sequence of shape [(x1, x2), (y1, y2)]
		:param mode:
		:return:
		"""
		self.select_box([x, y], limits, mode=mode)

	def select_box(self, spaces, limits, mode="replace"):
		"""Select a n-dimensional rectangular box bounded by limits

		The following examples are equivalent:
		>>> ds.select_box(['x', 'y'], [(0, 10), (0, 1)])
		>>> ds.select_rectangle('x', 'y', [(0, 10), (0, 1)])
		:param spaces: list of expressions
		:param limits: sequence of shape [(x1, x2), (y1, y2)]
		:param mode:
		:return:
		"""
		sorted_limits = [(min(l), max(l)) for l in limits]
		expressions = ["((%s) >= %f) & ((%s) <= %f)" % (expression, lmin, expression, lmax) for\
					   (expression, (lmin, lmax)) in zip(spaces, sorted_limits)]
		self.select("&".join(expressions), mode=mode)

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
		self._selection(create, name, executor=executor)

	def select_inverse(self, name="default", executor=None):
		"""Invert the selection, i.e. what is selected will not be, and vice versa

		:param str name:
		:param executor:
		:return:
		"""


		def create(current):
			return SelectionInvert(self, current)
		self._selection(create, name, executor=executor)

	def set_selection(self, selection, name="default", executor=None):
		"""Sets the selection object

		:param selection: Selection object
		:param name: selection 'slot'
		:param executor:
		:return:
		"""
		def create(current):
			return selection
		self._selection(create, name, executor=executor, execute_fully=True)


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
		import vaex.legacy
		return vaex.legacy.SubspaceLocal(self, expressions, kwargs.get("executor") or self.executor, async=kwargs.get("async", False))

	def echo(self, arg): return arg

	def __getitem__(self, item):
		"""Alias to dataset.to_copy(column_names), to mimic Pandas a bit

		:Example:
		>> ds["Lz"]
		>> ds["Lz", "E"]
		>> ds[ds.names.Lz]

		"""
		#print(arg)
		if _issequence(item):
			return self.to_copy(column_names=item)
		else:
			return self.to_copy(column_names=[item])

	def __iter__(self):
		"""Iterator over the column names (for the moment non-virtual and non-strings only)"""
		return iter(list(self.get_column_names(virtual=False, strings=False)))

	def __array__(self, dtype=None):
		"""Casts the dataset to a numpy array

		Example:
		>>> ar = np.array(ds)
		"""
		if dtype is None:
			dtype = np.float64
		chunks = []
		for name in self.get_column_names():
			if not np.can_cast(self.dtype(name), dtype):
				if self.dtype(name) != dtype:
					raise ValueError("Cannot cast %r (of type %r) to %r" % (name, self.dtype(name), dtype))
			else:
				chunks.append(self.evaluate(name))
		return np.array(chunks, dtype=dtype).T



	def _hstack(self, other, prefix=None):
		"""Join the columns of the other dataset to this one, assuming the ordering is the same"""
		assert len(self) == len(other), "does not make sense to horizontally stack datasets with different lengths"
		for name in other.get_column_names():
			if prefix:
				new_name = prefix + name
			else:
				new_name = name
			self.add_column(new_name, other.columns[name])


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

	def evaluate(self, expression, i1=None, i2=None, out=None, selection=None):
		"""The local implementation of :func:`Dataset.evaluate`"""
		i1 = i1 or 0
		i2 = i2 or len(self)
		mask = self.evaluate_selection_mask(selection, i1, i2) if selection is not None else None
		scope = _BlockScope(self, i1, i2, mask=mask, **self.variables)
		#	value = value[mask]
		if out is not None:
			scope.buffers[expression] = out
		value = scope.evaluate(expression)
		return value

	def compare(self, other, report_missing=True, report_difference=False, show=10, orderby=None, column_names=None):
		"""Compare two datasets and report their difference, use with care for large datasets"""
		if column_names is None:
			column_names = self.get_column_names(strings=True)
			for other_column_name in other.get_column_names(strings=True):
				if other_column_name not in column_names:
					column_names.append(other_column_name)
		different_values = []
		missing = []
		type_mismatch = []
		meta_mismatch = []
		assert len(self) == len(other)
		if orderby:
			index1 = np.argsort(self.columns[orderby])
			index2 = np.argsort(other.columns[orderby])
		for column_name in column_names:
			if column_name not in self.get_column_names(strings=True):
				missing.append(column_name)
				if report_missing:
					print("%s missing from this dataset" % column_name)
			elif column_name not in other.get_column_names(strings=True):
				missing.append(column_name)
				if report_missing:
					print("%s missing from other dataset" % column_name)
			else:
				ucd1 = self.ucds.get(column_name)
				ucd2 = other.ucds.get(column_name)
				if ucd1 != ucd2:
					print("ucd mismatch : %r vs %r for %s" % (ucd1, ucd2, column_name))
					meta_mismatch.append(column_name)
				unit1 = self.units.get(column_name)
				unit2 = other.units.get(column_name)
				if unit1 != unit2:
					print("unit mismatch : %r vs %r for %s" % (unit1, unit2, column_name))
					meta_mismatch.append(column_name)
				if self.dtype(column_name).type != other.dtype(column_name).type:
					print("different dtypes: %s vs %s for %s" % (self.dtype(column_name), other.dtype(column_name), column_name))
					type_mismatch.append(column_name)
				else:
					a = self.columns[column_name]
					b = other.columns[column_name]
					if orderby:
						a = a[index1]
						b = b[index2]
					def normalize(ar):
						if ar.dtype.kind == "f" and hasattr(ar, "mask"):
							mask = ar.mask
							ar = ar.copy()
							ar[mask] = np.nan
						if ar.dtype.kind in "SU":
							if hasattr(ar, "mask"):
								data = ar.data
							else:
								data = ar
							values = [value.strip() for value in data.tolist()]
							if hasattr(ar, "mask"):
								ar = np.ma.masked_array(values, ar.mask)
							else:
								ar = np.array(values)
						return ar
					def equal_mask(a, b):
						a = normalize(a)
						b = normalize(b)
						boolean_mask = (a == b)
						if self.dtype(column_name).kind == 'f': # floats with nan won't equal itself, i.e. NaN != NaN
							boolean_mask |= (np.isnan(a) & np.isnan(b))
						return boolean_mask
					boolean_mask = equal_mask(a, b)
					all_equal = np.all(boolean_mask)
					if not all_equal:
						count = np.sum(~boolean_mask)
						print("%s does not match for both datasets, %d rows are diffent out of %d" % (column_name, count, len(self)))
						different_values.append(column_name)
						if report_difference:
							indices = np.arange(len(self))[~boolean_mask]
							values1 = self.columns[column_name][~boolean_mask]
							values2 = other.columns[column_name][~boolean_mask]
							print("\tshowing difference for the first 10")
							for i in range(min(len(values1), show)):
								try:
									diff = values1[i] - values2[i]
								except:
									diff = "does not exists"
								print("%s[%d] == %s != %s other.%s[%d] (diff = %s)" % (column_name, indices[i], values1[i], values2[i], column_name, indices[i], diff))
		return different_values, missing, type_mismatch, meta_mismatch


	def _join(self, key, other, key_other, column_names=None, prefix=None):
		N = len(self)
		N_other = len(other)
		if column_names is None:
			column_names = other.get_column_names()
		key = self.evaluate(key)
		key_other = other.evaluate(key_other)
		index = dict(zip(key, range(N)))
		index_other = dict(zip(key_other, range(N_other)))

		from_indices = np.zeros(N, dtype=np.int64)
		to_indices = np.zeros(N, dtype=np.int64)
		for i in range(N):
			to_indices[i] = index[key[i]]
			from_indices[i] = index_other[key[i]]
			# try:
			# 	tmass_index = tmass_id_to_index[xmatch.data.original_ext_source_id[i]+b" "]
			# except:
			# 	tmass_index = tmass_id_to_index[xmatch.data.original_ext_source_id[i]]

		for column_name in column_names:
			dtype = other.dtype(column_name)
			data = np.zeros(N)
			if data.dtype.kind == "f":
				data[:] = np.nan
			else:
				raise ValueError("types other than float not yet supported (%r)" % dtype)
			data[to_indices] = other.evaluate(column_name)[from_indices]
			if prefix:
				new_name = prefix + column_name
			else:
				new_name = column_name
			self.add_column(new_name, data)


	def export_hdf5(self, path, column_names=None, byteorder="=", shuffle=False, selection=False, progress=None, virtual=False, sort=None, ascending=True):
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
		:param str sort: expression used for sorting the output
		:param bool ascending: sort ascending (True) or descending
		:return:
		"""
		import vaex.export
		vaex.export.export_hdf5(self, path, column_names, byteorder, shuffle, selection, progress=progress, virtual=virtual, sort=sort, ascending=ascending)

	def export_fits(self, path, column_names=None, shuffle=False, selection=False, progress=None, virtual=False, sort=None, ascending=True):
		"""Exports the dataset to a fits file that is compatible with TOPCAT colfits format

		:param DatasetLocal dataset: dataset to export
		:param str path: path for file
		:param lis[str] column_names: list of column names to export or None for all columns
		:param bool shuffle: export rows in random order
		:param bool selection: export selection or not
		:param progress: progress callback that gets a progress fraction as argument and should return True to continue,
			or a default progress bar when progress=True
		:param: bool virtual: When True, export virtual columns
		:param str sort: expression used for sorting the output
		:param bool ascending: sort ascending (True) or descending
		:return:
		"""
		import vaex.export
		vaex.export.export_fits(self, path, column_names, shuffle, selection, progress=progress, virtual=virtual, sort=sort, ascending=ascending)

	def _needs_copy(self, column_name):
		import vaex.file.other
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
		for dataset in datasets[:1]:
			for name, value in list(dataset.variables.items()):
				if name not in self.variables:
					self.set_variable(name, value, write=False)
		self.write_virtual_meta()

		self._full_length = sum(ds.full_length() for ds in self.datasets)
		self._length = self.full_length()
		self._index_end = self._full_length

def _is_dtype_ok(dtype):
	return dtype.type in [np.bool_, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float32, np.float64] or\
		dtype.type == np.string_ or dtype.type == np.unicode_

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
		super(DatasetArrays, self).add_column(name, data)
		#self._length = len(data)
		if self._full_length is None:
			self._full_length = len(data)
			self._index_end = self._full_length
		else:
			assert self.full_length() == len(data), "columns should be of equal length, length should be %d, while it is %d" % ( self.full_length(), len(data))
		self._length = int(round(self.full_length() * self._active_fraction))
		#self.set_active_fraction(self._active_fraction)
