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
import re
from functools import reduce
import threading
import six
import vaex.utils
# import vaex.image
import numpy as np
import concurrent.futures
import numbers

from vaex.utils import Timer
import vaex.events
# import vaex.ui.undo
import vaex.grids
import vaex.multithreading
import vaex.promise
import vaex.execution
import vaex.expresso
import logging
import vaex.kld
from . import selections, tasks, scopes
from .expression import expression_namespace
from .delayed import delayed, delayed_args, delayed_list
from .column import Column, ColumnIndexed, ColumnSparse, ColumnString, ColumnConcatenatedLazy, str_type
from .array_types import to_numpy, to_array_type
import vaex.events

# py2/p3 compatibility
try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse

_DEBUG = os.environ.get('VAEX_DEBUG', False)  # extra sanify checks that might hit performance

DEFAULT_REPR_FORMAT = 'plain'
FILTER_SELECTION_NAME = '__filter__'

sys_is_le = sys.byteorder == 'little'

logger = logging.getLogger("vaex")
lock = threading.Lock()
default_shape = 128
# executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
# executor = vaex.execution.default_executor

def _len(o):
    return o.__len__()


def _requires(name):
    def wrap(*args, **kwargs):
        raise RuntimeError('this function is wrapped by a placeholder, you probably want to install vaex-' + name)
    return wrap

from .utils import (_ensure_strings_from_expressions,
    _ensure_string_from_expression,
    _ensure_list,
    _is_limit,
    _isnumber,
    _issequence,
    _is_string,
    _parse_reduction,
    _parse_n,
    _normalize_selection_name,
    _normalize,
    _parse_f,
    _expand,
    _expand_shape,
    _expand_limits,
    as_flat_float,
    as_flat_array,
    _split_and_combine_mask)

main_executor = None  # vaex.execution.Executor(vaex.multithreading.pool)
from vaex.execution import Executor


def get_main_executor():
    global main_executor
    if main_executor is None:
        main_executor = vaex.execution.ExecutorLocal(vaex.multithreading.get_main_pool())
    return main_executor


# we import after function_mapping is defined
from .expression import Expression


_doc_snippets = {}
_doc_snippets["expression"] = "expression or list of expressions, e.g. df.x, 'x', or ['x, 'y']"
_doc_snippets["expression_one"] = "expression in the form of a string, e.g. 'x' or 'x+y' or vaex expression object, e.g. df.x or df.x+df.y "
_doc_snippets["expression_single"] = "if previous argument is not a list, this argument should be given"
_doc_snippets["binby"] = "List of expressions for constructing a binned grid"
_doc_snippets["limits"] = """description for the min and max values for the expressions, e.g. 'minmax' (default), '99.7%', [0, 10], or a list of, e.g. [[0, 10], [0, 20], 'minmax']"""
_doc_snippets["shape"] = """shape for the array where the statistic is calculated on, if only an integer is given, it is used for all dimensions, e.g. shape=128, shape=[128, 256]"""
_doc_snippets["percentile_limits"] = """description for the min and max values to use for the cumulative histogram, should currently only be 'minmax'"""
_doc_snippets["percentile_shape"] = """shape for the array where the cumulative histogram is calculated on, integer type"""
_doc_snippets["selection"] = """Name of selection to use (or True for the 'default'), or all the data (when selection is None or False), or a list of selections"""
_doc_snippets["selection1"] = """Name of selection to use (or True for the 'default'), or all the data (when selection is None or False)"""
_doc_snippets["delay"] = """Do not return the result, but a proxy for delayhronous calculations (currently only for internal use)"""
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
_doc_snippets['propagate_uncertainties'] = """If true, will propagate errors for the new virtual columns, see :meth:`propagate_uncertainties` for details"""
_doc_snippets['note_copy'] = '.. note:: Note that no copy of the underlying data is made, only a view/reference is made.'
_doc_snippets['note_filter'] = '.. note:: Note that filtering will be ignored (since they may change), you may want to consider running :meth:`extract` first.'
_doc_snippets['inplace'] = 'Make modifications to self or return a new DataFrame'
_doc_snippets['return_shallow_copy'] = 'Returns a new DataFrame with a shallow copy/view of the underlying data'
_doc_snippets['chunk_size'] = 'Return an iterator with cuts of the object in lenght of this size'
_doc_snippets['chunk_size_export'] = 'Number of rows to be written to disk in a single iteration'
_doc_snippets['evaluate_parallel'] = 'Evaluate the (virtual) columns in parallel'
_doc_snippets['array_type'] = 'Type of output array, possible values are None/"numpy" (ndarray), "xarray" for a xarray.DataArray, or "list" for a Python list'


def docsubst(f):
    if f.__doc__:
        f.__doc__ = f.__doc__.format(**_doc_snippets)
    return f


_functions_statistics_1d = []


def stat_1d(f):
    _functions_statistics_1d.append(f)
    return f

def _hidden(meth):
    """Mark a method as hidden"""
    meth.__hidden__ = True
    return meth

class DataFrame(object):
    """All local or remote datasets are encapsulated in this class, which provides a pandas
    like API to your dataset.

    Each DataFrame (df) has a number of columns, and a number of rows, the length of the DataFrame.

    All DataFrames have multiple 'selection', and all calculations are done on the whole DataFrame (default)
    or for the selection. The following example shows how to use the selection.

    >>> df.select("x < 0")
    >>> df.sum(df.y, selection=True)
    >>> df.sum(df.y, selection=[df.x < 0, df.x > 0])

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
        self.signal_column_changed = vaex.events.Signal("a column changed")  # (df, column_name, change_type=["add", "remove", "change"])
        self.signal_variable_changed = vaex.events.Signal("a variable changed")

        self.variables = {}
        self.virtual_columns = {}
        # we also store the virtual columns as expressions, for performance reasons
        # the expression object can cache the ast, making renaming/rewriting faster
        self._virtual_expressions = {}
        self.functions = {}
        self._length_original = None
        self._length_unfiltered = None
        self._cached_filtered_length = None
        self._active_fraction = 1
        self._current_row = None
        self._index_start = 0
        self._index_end = None

        self.description = None
        self.ucds = {}
        self.units = {}
        self.descriptions = {}
        self._dtypes_override = {}

        self.favorite_selections = {}

        self.mask = None  # a bitmask for the selection does not work for server side

        # maps from name to list of Selection objets
        self.selection_histories = collections.defaultdict(list)
        # after an undo, the last one in the history list is not the active one, -1 means no selection
        self.selection_history_indices = collections.defaultdict(lambda: -1)
        assert self.filtered is False
        self._auto_fraction = False

        self._sparse_matrices = {}  # record which sparse columns belong to which sparse matrix

        self._categories = {}
        self._selection_mask_caches = collections.defaultdict(dict)
        self._selection_masks = {}  # maps to vaex.superutils.Mask object
        self._renamed_columns = []
        self._column_aliases = {}  # maps from invalid variable names to valid ones

        # weak refs of expression that we keep to rewrite expressions
        self._expressions = []

        self.local = threading.local()
        # a check to avoid nested aggregator calls, which make stack traces very difficult
        # like the ExecutorLocal.local.executing, this needs to be thread local
        self.local._aggregator_nest_count = 0

        self._task_aggs = {}
        self._binners = {}
        self._grids = {}

    def __getattr__(self, name):
        # will support the hidden methods
        if name in self.__hidden__:
            return self.__hidden__[name].__get__(self)
        else:
            return object.__getattribute__(self, name)

    @property
    def func(self):
        class Functions(object):
            pass

        functions = Functions()
        for name, value in expression_namespace.items():
            # f = vaex.expression.FunctionBuiltin(self, name)
            def closure(name=name, value=value):
                local_name = name
                def wrap(*args, **kwargs):
                    def myrepr(k):
                        if isinstance(k, Expression):
                            return str(k)
                        elif isinstance(k, np.ndarray) and k.ndim == 0:
                            # to support numpy scalars
                            return myrepr(k.item())
                        elif isinstance(k, np.ndarray):
                            # to support numpy arrays
                            var = self.add_variable('arg_numpy_array', k, unique=True)
                            return var
                        elif isinstance(k, list):
                            # to support numpy scalars
                            return '[' + ', '.join(myrepr(i) for i in k) + ']'
                        else:
                            return repr(k)
                    arg_string = ", ".join([myrepr(k) for k in args] + ['{}={}'.format(name, myrepr(value)) for name, value in kwargs.items()])
                    expression = "{}({})".format(local_name, arg_string)
                    return vaex.expression.Expression(self, expression)
                return wrap
            f = closure()
            try:
                f = functools.wraps(value)(f)
            except AttributeError:
                pass # python2 quicks.. ?
            setattr(functions, name, f)
        for name, value in self.functions.items():
            setattr(functions, name, value)

        return functions

    @_hidden
    @vaex.utils.deprecated('use is_category')
    def iscategory(self, column):
        return self.is_category(column)

    def is_datetime(self, expression):
        dtype = self.data_type(expression)
        return dtype != str_type and dtype.kind == 'M'

    def is_category(self, column):
        """Returns true if column is a category."""
        column = _ensure_string_from_expression(column)
        return column in self._categories

    def category_labels(self, column):
        column = _ensure_string_from_expression(column)
        return self._categories[column]['labels']

    def category_values(self, column):
        column = _ensure_string_from_expression(column)
        return self._categories[column]['values']

    def category_count(self, column):
        column = _ensure_string_from_expression(column)
        return self._categories[column]['N']

    def category_offset(self, column):
        column = _ensure_string_from_expression(column)
        return self._categories[column]['min_value']

    def execute(self):
        '''Execute all delayed jobs.'''
        from .asyncio import just_run
        just_run(self.execute_async())

    async def execute_async(self):
        '''Async version of execute'''
        # no need to clear _task_aggs anymore, since they will be removed for the executors' task list
        await self.executor.execute_async()

    @property
    def filtered(self):
        return self.has_selection(FILTER_SELECTION_NAME)

    def map_reduce(self, map, reduce, arguments, progress=False, delay=False, info=False, to_numpy=True, ignore_filter=False, pre_filter=False, name='map reduce (custom)', selection=None):
        # def map_wrapper(*blocks):
        task = tasks.TaskMapReduce(self, arguments, map, reduce, info=info, to_numpy=to_numpy, ignore_filter=ignore_filter, selection=selection, pre_filter=pre_filter)
        progressbar = vaex.utils.progressbars(progress)
        progressbar.add_task(task, name)
        self.executor.schedule(task)
        return self._delay(delay, task)

    def apply(self, f, arguments=None, dtype=None, delay=False, vectorize=False):
        """Apply a function on a per row basis across the entire DataFrame.

        Example:

        >>> import vaex
        >>> df = vaex.example()
        >>> def func(x, y):
        ...     return (x+y)/(x-y)
        ...
        >>> df.apply(func, arguments=[df.x, df.y])
        Expression = lambda_function(x, y)
        Length: 330,000 dtype: float64 (expression)
        -------------------------------------------
             0  -0.460789
             1    3.90038
             2  -0.642851
             3   0.685768
             4  -0.543357


        :param f: The function to be applied
        :param arguments: List of arguments to be passed on to the function f.
        :return: A function that is lazily evaluated.
        """
        assert arguments is not None, 'for now, you need to supply arguments'
        import types
        if isinstance(f, types.LambdaType):
            name = 'lambda_function'
        else:
            name = f.__name__
        if not vectorize:
            f = vaex.expression.FunctionToScalar(f)
        lazy_function = self.add_function(name, f, unique=True)
        arguments = _ensure_strings_from_expressions(arguments)
        return lazy_function(*arguments)

    def nop(self, expression, progress=False, delay=False):
        """Evaluates expression, and drop the result, usefull for benchmarking, since vaex is usually lazy"""
        expression = _ensure_string_from_expression(expression)
        def map(ar):
            pass
        def reduce(a, b):
            pass
        return self.map_reduce(map, reduce, [expression], delay=delay, progress=progress, name='nop', to_numpy=False)

    def _set(self, expression, progress=False, selection=None, delay=False):
        column = _ensure_string_from_expression(expression)
        columns = [column]
        from .hash import ordered_set_type_from_dtype
        from vaex.column import _to_string_sequence

        transient = self[str(expression)].transient or self.filtered or self.is_masked(expression)
        if self.data_type(expression) == str_type and not transient:
            # string is a special case, only ColumnString are not transient
            ar = self.columns[str(expression)]
            if not isinstance(ar, ColumnString):
                transient = True

        dtype = self.data_type(column)
        ordered_set_type = ordered_set_type_from_dtype(dtype, transient)
        sets = [None] * self.executor.thread_pool.nthreads
        def map(thread_index, i1, i2, ar):
            if sets[thread_index] is None:
                sets[thread_index] = ordered_set_type()
            if dtype == str_type:
                previous_ar = ar
                ar = _to_string_sequence(ar)
                if not transient:
                    assert ar is previous_ar.string_sequence
            if np.ma.isMaskedArray(ar):
                mask = np.ma.getmaskarray(ar)
                sets[thread_index].update(ar, mask)
            else:
                sets[thread_index].update(ar)
        def reduce(a, b):
            pass
        self.map_reduce(map, reduce, columns, delay=delay, name='set', info=True, to_numpy=False, selection=selection)
        sets = [k for k in sets if k is not None]
        set0 = sets[0]
        for other in sets[1:]:
            set0.merge(other)
        return set0

    def _index(self, expression, progress=False, delay=False):
        column = _ensure_string_from_expression(expression)
        column = self._column_aliases.get(column, column)
        columns = [column]
        from .hash import index_type_from_dtype
        from vaex.column import _to_string_sequence

        transient = self[column].transient or self.filtered or self.is_masked(column)
        if self.data_type(expression) == str_type and not transient:
            # string is a special case, only ColumnString are not transient
            ar = self.columns[str(self[column].expand())]
            if not isinstance(ar, ColumnString):
                transient = True

        dtype = self.data_type(column)
        index_type = index_type_from_dtype(dtype, transient)
        index_list = [None] * self.executor.thread_pool.nthreads
        def map(thread_index, i1, i2, ar):
            if index_list[thread_index] is None:
                index_list[thread_index] = index_type()
            if dtype == str_type:
                previous_ar = ar
                ar = _to_string_sequence(ar)
                if not transient:
                    assert ar is previous_ar.string_sequence
            if np.ma.isMaskedArray(ar):
                mask = np.ma.getmaskarray(ar)
                index_list[thread_index].update(ar, mask, i1)
            else:
                index_list[thread_index].update(ar, i1)
        def reduce(a, b):
            pass
        self.map_reduce(map, reduce, columns, delay=delay, name='index', info=True, to_numpy=False)
        index_list = [k for k in index_list if k is not None]
        index0 = index_list[0]
        for other in index_list[1:]:
            index0.merge(other)
        return index0

    def unique(self, expression, return_inverse=False, dropna=False, dropnan=False, dropmissing=False, progress=False, selection=None, delay=False):
        if dropna:
            dropnan = True
            dropmissing = True
        expression = _ensure_string_from_expression(expression)
        ordered_set = self._set(expression, progress=progress, selection=selection)
        transient = True
        if return_inverse:
            # inverse type can be smaller, depending on length of set
            inverse = np.zeros(self._length_unfiltered, dtype=np.int64)
            dtype = self.data_type(expression)
            from vaex.column import _to_string_sequence
            def map(thread_index, i1, i2, ar):
                if dtype == str_type:
                    previous_ar = ar
                    ar = _to_string_sequence(ar)
                    if not transient:
                        assert ar is previous_ar.string_sequence
                # TODO: what about masked values?
                inverse[i1:i2:] = ordered_set.map_ordinal(ar)
            def reduce(a, b):
                pass
            self.map_reduce(map, reduce, [expression], delay=delay, name='unique_return_inverse', info=True, to_numpy=False, selection=selection)
        keys = ordered_set.keys()
        if not dropnan:
            if ordered_set.has_nan:
                keys = [np.nan] + keys
        if not dropmissing:
            if ordered_set.has_null:
                keys = [np.ma.core.MaskedConstant()] + keys
        keys = np.asarray(keys)
        if return_inverse:
            return keys, inverse
        else:
            return keys

    @docsubst
    def mutual_information(self, x, y=None, mi_limits=None, mi_shape=256, binby=[], limits=None, shape=default_shape, sort=False, selection=False, delay=False):
        """Estimate the mutual information between and x and y on a grid with shape mi_shape and mi_limits, possibly on a grid defined by binby.

        If sort is True, the mutual information is returned in sorted (descending) order and the list of expressions is returned in the same order.

        Example:

        >>> df.mutual_information("x", "y")
        array(0.1511814526380327)
        >>> df.mutual_information([["x", "y"], ["x", "z"], ["E", "Lz"]])
        array([ 0.15118145,  0.18439181,  1.07067379])
        >>> df.mutual_information([["x", "y"], ["x", "z"], ["E", "Lz"]], sort=True)
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
        :param delay: {delay}
        :return: {return_stat_scalar},
        """
        if y is None:
            waslist, [x, ] = vaex.utils.listify(x)
        else:
            waslist, [x, y] = vaex.utils.listify(x, y)
            x = list(zip(x, y))
            if mi_limits:
                mi_limits = [mi_limits]
        # print("x, mi_limits", x, mi_limits)
        limits = self.limits(binby, limits, delay=True)
        # print("$"*80)
        mi_limits = self.limits(x, mi_limits, delay=True)
        # print("@"*80)

        @delayed
        def calculate(counts):
            # TODO: mutual information doesn't take axis arguments, so ugly solution for now
            counts = counts.astype(np.float64)
            fullshape = _expand_shape(shape, len(binby))
            out = np.zeros((fullshape), dtype=float)
            if len(fullshape) == 0:
                out = vaex.kld.mutual_information(counts)
                # print("count> ", np.sum(counts))
            elif len(fullshape) == 1:
                for i in range(fullshape[0]):
                    out[i] = vaex.kld.mutual_information(counts[..., i])
                    # print("counti> ", np.sum(counts[...,i]))
                # print("countt> ", np.sum(counts))
            elif len(fullshape) == 2:
                for i in range(fullshape[0]):
                    for j in range(fullshape[1]):
                        out[i, j] = vaex.kld.mutual_information(counts[..., i, j])
            elif len(fullshape) == 3:
                for i in range(fullshape[0]):
                    for j in range(fullshape[1]):
                        for k in range(fullshape[2]):
                            out[i, j, k] = vaex.kld.mutual_information(counts[..., i, j, k])
            else:
                raise ValueError("binby with dim > 3 is not yet supported")
            return out

        @delayed
        def has_limits(limits, mi_limits):
            if not _issequence(binby):
                limits = [list(limits)]
            values = []
            for expressions, expression_limits in zip(x, mi_limits):
                # print("mi for", expressions, expression_limits)
                # total_shape =  _expand_shape(mi_shape, len(expressions)) + _expand_shape(shape, len(binby))
                total_shape = _expand_shape(mi_shape, len(expressions)) + _expand_shape(shape, len(binby))
                # print("expressions", expressions)
                # print("total_shape", total_shape)
                # print("limits", limits,expression_limits)
                # print("limits>", list(limits) + list(expression_limits))
                counts = self.count(binby=list(expressions) + list(binby), limits=list(expression_limits) + list(limits),
                                    shape=total_shape, delay=True, selection=selection)
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
        return self._delay(delay, values)

    def bin_edges(self, expression, limits, shape=default_shape):
        return self.bins(expression, limits, shape=shape, edges=True)

    def bin_centers(self, expression, limits, shape=default_shape):
        return self.bins(expression, limits, shape=shape, edges=False)

    def bins(self, expression, limits, shape=default_shape, edges=True):
        vmin, vmax = limits
        if edges:
            bins = np.ogrid[limits[0]:limits[1]:(shape + 1) * 1j]
            return bins
        else:
            dx = (limits[1] - limits[0]) / shape
            bins = np.ogrid[limits[0]:limits[1] - dx:(shape) * 1j]
            return bins + dx / 2

    def nearest_bin(self, value, limits, shape):
        bins = self.bins('', limits=limits, edges=False, shape=shape)
        index = np.argmin(np.abs(bins - value))
        print(bins, value, index)
        return index

    def _compute_agg(self, name, expression, binby=[], limits=None, shape=default_shape, selection=False, delay=False, edges=False, progress=None, extra_expressions=None, array_type=None):
        logger.debug("aggregate %s(%r, binby=%r, limits=%r)", name, expression, binby, limits)
        expression = _ensure_strings_from_expressions(expression)
        if extra_expressions:
            extra_expressions = _ensure_strings_from_expressions(extra_expressions)
        expression_waslist, [expressions, ] = vaex.utils.listify(expression)
        expressions = [self._column_aliases.get(k, k) for k in expressions]
        import traceback
        trace = ''.join(traceback.format_stack())
        for expression in expressions:
            if expression and expression != "*":
                self.validate_expression(expression)
        if not hasattr(self.local, '_aggregator_nest_count'):
            self.local._aggregator_nest_count = 0
        if self.local._aggregator_nest_count != 0:
            raise RuntimeError("nested aggregator call: \nlast trace:\n%s\ncurrent trace:\n%s" % (self.local.last_trace, trace))
        else:
            self.local.last_trace = trace
        # Instead of 'expression is not None', we would like to have 'not virtual'
        # but in agg.py we do some casting, which results in calling .dtype(..) with a non-column
        # expression even though all expressions passed here are column references
        # virtual = [k for k in expressions if k and k not in self.columns]
        if self.filtered and expression is not None:
            # When our dataframe is filtered, and we have expressions, we may end up calling
            # df.dtype(..) which in turn may call df.evaluate(..) which in turn needs to have
            # the filter cache filled in order to compute the first non-missing row. This last
            # item could call df.count() again, leading to nested aggregators, which we do not
            # support. df.dtype() needs to call evaluate with filtering enabled since we consider
            # it invalid that expressions are evaluate with filtered data. Sklearn for instance may
            # give errors when evaluated with NaN's present.
            # TODO: GET RID OF THIS
            len(self) # fill caches and masks
            # pass
        grid = self._create_grid(binby, limits, shape, selection=selection, delay=True)
        @delayed
        def compute(expression, grid, selection, edges, progressbar):
            self.local._aggregator_nest_count += 1
            try:
                if expression in ["*", None]:
                    agg = vaex.agg.aggregates[name](selection=selection, edges=edges)
                else:
                    if extra_expressions:
                        agg = vaex.agg.aggregates[name](expression, *extra_expressions, selection=selection, edges=edges)
                    else:
                        agg = vaex.agg.aggregates[name](expression, selection=selection, edges=edges)
                task, new_task = self._get_task_agg(grid)
                agg_subtask = agg.add_operations(task)
                if new_task:
                    # it is important we schedule the task after we add an operation
                    # otherwise the task will fail to executor (has to have >= 1 operation)
                    self.executor.schedule(task)

                progressbar.add_task(task, "%s for %s" % (name, expression))
                @delayed
                def finish(counts):
                    return np.asarray(counts)
                return finish(agg_subtask)
            finally:
                self.local._aggregator_nest_count -= 1
        @delayed
        def finish(grid, *counts):
            if array_type == 'xarray':
                binners = grid.binners
                dims = [binner.expression for binner in binners]
                if expression_waslist:
                    dims = ['expression'] + dims

                def to_coord(binner):
                    name = type(binner).__name__
                    if name.startswith('BinnerOrdinal_'):
                        return self.category_labels(binner.expression)
                    elif name.startswith('BinnerScalar_'):
                        return self.bin_centers(binner.expression, [binner.vmin, binner.vmax], binner.bins)
                coords = [to_coord(binner) for binner in binners]
                if expression_waslist:
                    coords = [expressions] + coords
                    counts = np.asarray(counts)
                else:
                    counts = counts[0]
                import xarray
                return xarray.DataArray(counts, dims=dims, coords=coords)
            elif array_type == 'list':
                return vaex.utils.unlistify(expression_waslist, counts).tolist()
            elif array_type in [None, 'numpy']:
                return np.asarray(vaex.utils.unlistify(expression_waslist, counts))
            else:
                raise RuntimeError(f'Unknown array_type {format}')
        progressbar = vaex.utils.progressbars(progress)
        stats = [compute(expression, grid, selection=selection, edges=edges, progressbar=progressbar) for expression in expressions]
        var = finish(grid, *stats)
        return self._delay(delay, var)

    @docsubst
    def count(self, expression=None, binby=[], limits=None, shape=default_shape, selection=False, delay=False, edges=False, progress=None, array_type=None):
        """Count the number of non-NaN values (or all, if expression is None or "*").

        Example:

        >>> df.count()
        330000
        >>> df.count("*")
        330000.0
        >>> df.count("*", binby=["x"], shape=4)
        array([  10925.,  155427.,  152007.,   10748.])

        :param expression: Expression or column for which to count non-missing values, or None or '*' for counting the rows
        :param binby: {binby}
        :param limits: {limits}
        :param shape: {shape}
        :param selection: {selection}
        :param delay: {delay}
        :param progress: {progress}
        :param edges: {edges}
        :param array_type: {array_type}
        :return: {return_stat_scalar}
        """
        return self._compute_agg('count', expression, binby, limits, shape, selection, delay, edges, progress, array_type=array_type)

    @delayed
    def _first_calculation(self, expression, order_expression, binby, limits, shape, selection, edges, progressbar):
        if shape:
            limits, shapes = limits
        else:
            limits, shapes = limits, shape
        task = tasks.TaskStatistic(self, binby, shapes, limits, weights=[expression, order_expression], op=tasks.OP_FIRST, selection=selection, edges=edges)
        self.executor.schedule(task)
        progressbar.add_task(task, "count for %s" % expression)
        @delayed
        def finish(counts):
            counts = np.array(counts)
            return counts
        return finish(task)

    @docsubst
    def first(self, expression, order_expression, binby=[], limits=None, shape=default_shape, selection=False, delay=False, edges=False, progress=None, array_type=None):
        """Return the first element of a binned `expression`, where the values each bin are sorted by `order_expression`.

        Example:

        >>> import vaex
        >>> df = vaex.example()
        >>> df.first(df.x, df.y, shape=8)
        >>> df.first(df.x, df.y, shape=8, binby=[df.y])
        >>> df.first(df.x, df.y, shape=8, binby=[df.y])
        array([-4.81883764, 11.65378   ,  9.70084476, -7.3025589 ,  4.84954977,
                8.47446537, -5.73602629, 10.18783   ])

        :param expression: The value to be placed in the bin.
        :param order_expression: Order the values in the bins by this expression.
        :param binby: {binby}
        :param limits: {limits}
        :param shape: {shape}
        :param selection: {selection}
        :param delay: {delay}
        :param progress: {progress}
        :param edges: {edges}
        :param array_type: {array_type}
        :return: Ndarray containing the first elements.
        :rtype: numpy.array
        """
        return self._compute_agg('first', expression, binby, limits, shape, selection, delay, edges, progress, extra_expressions=[order_expression], array_type=array_type)
        logger.debug("count(%r, binby=%r, limits=%r)", expression, binby, limits)
        logger.debug("count(%r, binby=%r, limits=%r)", expression, binby, limits)
        expression = _ensure_strings_from_expressions(expression)
        order_expression = _ensure_string_from_expression(order_expression)
        binby = _ensure_strings_from_expressions(binby)
        waslist, [expressions,] = vaex.utils.listify(expression)
        @delayed
        def finish(*counts):
            counts = np.asarray(counts)
            return vaex.utils.unlistify(waslist, counts)
        progressbar = vaex.utils.progressbars(progress)
        limits = self.limits(binby, limits, delay=True, shape=shape)
        stats = [self._first_calculation(expression, order_expression, binby=binby, limits=limits, shape=shape, selection=selection, edges=edges, progressbar=progressbar) for expression in expressions]
        var = finish(*stats)
        return self._delay(delay, var)

    @docsubst
    @stat_1d
    def mean(self, expression, binby=[], limits=None, shape=default_shape, selection=False, delay=False, progress=None, edges=False, array_type=None):
        """Calculate the mean for expression, possibly on a grid defined by binby.

        Example:

        >>> df.mean("x")
        -0.067131491264005971
        >>> df.mean("(x**2+y**2)**0.5", binby="E", shape=4)
        array([  2.43483742,   4.41840721,   8.26742458,  15.53846476])

        :param expression: {expression}
        :param binby: {binby}
        :param limits: {limits}
        :param shape: {shape}
        :param selection: {selection}
        :param delay: {delay}
        :param progress: {progress}
        :param array_type: {array_type}
        :return: {return_stat_scalar}
        """
        return self._compute_agg('mean', expression, binby, limits, shape, selection, delay, edges, progress, array_type=array_type)
        logger.debug("mean of %r, with binby=%r, limits=%r, shape=%r, selection=%r, delay=%r", expression, binby, limits, shape, selection, delay)
        expression = _ensure_strings_from_expressions(expression)
        selection = _ensure_strings_from_expressions(selection)
        binby = _ensure_strings_from_expressions(binby)

        @delayed
        def calculate(expression, limits):
            task = tasks.TaskStatistic(self, binby, shape, limits, weight=expression, op=tasks.OP_ADD_WEIGHT_MOMENTS_01, selection=selection)
            self.executor.schedule(task)
            progressbar.add_task(task, "mean for %s" % expression)
            return task

        @delayed
        def finish(*stats_args):
            stats = np.array(stats_args)
            counts = stats[..., 0]
            with np.errstate(divide='ignore', invalid='ignore'):
                mean = stats[..., 1] / counts
            return vaex.utils.unlistify(waslist, mean)
        waslist, [expressions, ] = vaex.utils.listify(expression)
        progressbar = vaex.utils.progressbars(progress)
        limits = self.limits(binby, limits, delay=True)
        stats = [calculate(expression, limits) for expression in expressions]
        var = finish(*stats)
        return self._delay(delay, var)

    @delayed
    def _sum_calculation(self, expression, binby, limits, shape, selection, progressbar):
        task = tasks.TaskStatistic(self, binby, shape, limits, weight=expression, op=tasks.OP_ADD_WEIGHT_MOMENTS_01, selection=selection)
        self.executor.schedule(task)
        progressbar.add_task(task, "sum for %s" % expression)
        @delayed
        def finish(sum_grid):
            stats = np.array(sum_grid)
            return stats[...,1]
        return finish(task)

    @docsubst
    @stat_1d
    def sum(self, expression, binby=[], limits=None, shape=default_shape, selection=False, delay=False, progress=None, edges=False, array_type=None):
        """Calculate the sum for the given expression, possible on a grid defined by binby

        Example:

        >>> df.sum("L")
        304054882.49378014
        >>> df.sum("L", binby="E", shape=4)
        array([  8.83517994e+06,   5.92217598e+07,   9.55218726e+07,
                         1.40008776e+08])

        :param expression: {expression}
        :param binby: {binby}
        :param limits: {limits}
        :param shape: {shape}
        :param selection: {selection}
        :param delay: {delay}
        :param progress: {progress}
        :param array_type: {array_type}
        :return: {return_stat_scalar}
        """
        return self._compute_agg('sum', expression, binby, limits, shape, selection, delay, edges, progress, array_type=array_type)
        @delayed
        def finish(*sums):
            return vaex.utils.unlistify(waslist, sums)
        expression = _ensure_strings_from_expressions(expression)
        binby = _ensure_strings_from_expressions(binby)
        waslist, [expressions, ] = vaex.utils.listify(expression)
        progressbar = vaex.utils.progressbars(progress)
        limits = self.limits(binby, limits, delay=True)
        # stats = [calculate(expression, limits) for expression in expressions]
        sums = [self._sum_calculation(expression, binby=binby, limits=limits, shape=shape, selection=selection, progressbar=progressbar) for expression in expressions]
        s = finish(*sums)
        return self._delay(delay, s)

    @docsubst
    @stat_1d
    def std(self, expression, binby=[], limits=None, shape=default_shape, selection=False, delay=False, progress=None, array_type=None):
        """Calculate the standard deviation for the given expression, possible on a grid defined by binby


        >>> df.std("vz")
        110.31773397535071
        >>> df.std("vz", binby=["(x**2+y**2)**0.5"], shape=4)
        array([ 123.57954851,   85.35190177,   61.14345748,   38.0740619 ])

        :param expression: {expression}
        :param binby: {binby}
        :param limits: {limits}
        :param shape: {shape}
        :param selection: {selection}
        :param delay: {delay}
        :param progress: {progress}
        :param array_type: {array_type}
        :return: {return_stat_scalar}
        """
        @delayed
        def finish(var):
            return var**0.5
        return self._delay(delay, finish(self.var(expression, binby=binby, limits=limits, shape=shape, selection=selection, delay=True, progress=progress)))

    @docsubst
    @stat_1d
    def var(self, expression, binby=[], limits=None, shape=default_shape, selection=False, delay=False, progress=None, array_type=None):
        """Calculate the sample variance for the given expression, possible on a grid defined by binby

        Example:

        >>> df.var("vz")
        12170.002429456246
        >>> df.var("vz", binby=["(x**2+y**2)**0.5"], shape=4)
        array([ 15271.90481083,   7284.94713504,   3738.52239232,   1449.63418988])
        >>> df.var("vz", binby=["(x**2+y**2)**0.5"], shape=4)**0.5
        array([ 123.57954851,   85.35190177,   61.14345748,   38.0740619 ])
        >>> df.std("vz", binby=["(x**2+y**2)**0.5"], shape=4)
        array([ 123.57954851,   85.35190177,   61.14345748,   38.0740619 ])

        :param expression: {expression}
        :param binby: {binby}
        :param limits: {limits}
        :param shape: {shape}
        :param selection: {selection}
        :param delay: {delay}
        :param progress: {progress}
        :param array_type: {array_type}
        :return: {return_stat_scalar}
        """
        edges = False
        return self._compute_agg('var', expression, binby, limits, shape, selection, delay, edges, progress, array_type=array_type)
        expression = _ensure_strings_from_expressions(expression)
        @delayed
        def calculate(expression, limits):
            task = tasks.TaskStatistic(self, binby, shape, limits, weight=expression, op=tasks.OP_ADD_WEIGHT_MOMENTS_012, selection=selection)
            progressbar.add_task(task, "var for %s" % expression)
            self.executor.schedule(task)
            return task

        @delayed
        def finish(*stats_args):
            stats = np.array(stats_args)
            counts = stats[..., 0]
            with np.errstate(divide='ignore'):
                with np.errstate(divide='ignore', invalid='ignore'):  # these are fine, we are ok with nan's in vaex
                    mean = stats[..., 1] / counts
                    raw_moments2 = stats[..., 2] / counts
            variance = (raw_moments2 - mean**2)
            return vaex.utils.unlistify(waslist, variance)
        binby = _ensure_strings_from_expressions(binby)
        waslist, [expressions, ] = vaex.utils.listify(expression)
        progressbar = vaex.utils.progressbars(progress)
        limits = self.limits(binby, limits, delay=True)
        stats = [calculate(expression, limits) for expression in expressions]
        var = finish(*stats)
        return self._delay(delay, var)

    @docsubst
    def covar(self, x, y, binby=[], limits=None, shape=default_shape, selection=False, delay=False, progress=None):
        """Calculate the covariance cov[x,y] between x and y, possibly on a grid defined by binby.

        Example:

        >>> df.covar("x**2+y**2+z**2", "-log(-E+1)")
        array(52.69461456005138)
        >>> df.covar("x**2+y**2+z**2", "-log(-E+1)")/(df.std("x**2+y**2+z**2") * df.std("-log(-E+1)"))
        0.63666373822156686
        >>> df.covar("x**2+y**2+z**2", "-log(-E+1)", binby="Lz", shape=4)
        array([ 10.17387143,  51.94954078,  51.24902796,  20.2163929 ])



        :param x: {expression}
        :param y: {expression}
        :param binby: {binby}
        :param limits: {limits}
        :param shape: {shape}
        :param selection: {selection}
        :param delay: {delay}
        :param progress: {progress}
        :return: {return_stat_scalar}
        """
        @delayed
        def cov(mean_x, mean_y, mean_xy):
            return mean_xy - mean_x * mean_y

        waslist, [xlist, ylist] = vaex.utils.listify(x, y)
        # print("limits", limits)
        limits = self.limits(binby, limits, selection=selection, delay=True)
        # print("limits", limits)

        @delayed
        def calculate(limits):
            results = []
            for x, y in zip(xlist, ylist):
                mx = self.mean(x, binby=binby, limits=limits, shape=shape, selection=selection, delay=True, progress=progressbar)
                my = self.mean(y, binby=binby, limits=limits, shape=shape, selection=selection, delay=True, progress=progressbar)
                cxy = self.mean("(%s)*(%s)" % (x, y), binby=binby, limits=limits, shape=shape, selection=selection,
                                delay=True, progress=progressbar)
                results.append(cov(mx, my, cxy))
            return results

        progressbar = vaex.utils.progressbars(progress)
        covars = calculate(limits)

        @delayed
        def finish(covars):
            value = np.array(vaex.utils.unlistify(waslist, covars))
            return value
        return self._delay(delay, finish(delayed_list(covars)))

    @docsubst
    def correlation(self, x, y=None, binby=[], limits=None, shape=default_shape, sort=False, sort_key=np.abs, selection=False, delay=False, progress=None):
        """Calculate the correlation coefficient cov[x,y]/(std[x]*std[y]) between x and y, possibly on a grid defined by binby.

        Example:

        >>> df.correlation("x**2+y**2+z**2", "-log(-E+1)")
        array(0.6366637382215669)
        >>> df.correlation("x**2+y**2+z**2", "-log(-E+1)", binby="Lz", shape=4)
        array([ 0.40594394,  0.69868851,  0.61394099,  0.65266318])

        :param x: {expression}
        :param y: {expression}
        :param binby: {binby}
        :param limits: {limits}
        :param shape: {shape}
        :param selection: {selection}
        :param delay: {delay}
        :param progress: {progress}
        :return: {return_stat_scalar}
        """
        @delayed
        def corr(cov):
            with np.errstate(divide='ignore', invalid='ignore'):  # these are fine, we are ok with nan's in vaex
                return cov[..., 0, 1] / (cov[..., 0, 0] * cov[..., 1, 1])**0.5

        if y is None:
            if not isinstance(x, (tuple, list)):
                raise ValueError("if y not given, x is expected to be a list or tuple, not %r" % x)
            if _issequence(x) and not _issequence(x[0]) and len(x) == 2:
                x = [x]
            if not(_issequence(x) and all([_issequence(k) and len(k) == 2 for k in x])):
                raise ValueError("if y not given, x is expected to be a list of lists with length 2, not %r" % x)
            # waslist, [xlist,ylist] = vaex.utils.listify(*x)
            waslist = True
            xlist, ylist = zip(*x)
            # print xlist, ylist
        else:
            waslist, [xlist, ylist] = vaex.utils.listify(x, y)
        xlist = _ensure_strings_from_expressions(xlist)
        ylist = _ensure_strings_from_expressions(ylist)
        limits = self.limits(binby, limits, selection=selection, delay=True)

        @delayed
        def echo(limits):
            logger.debug(">>>>>>>>: %r %r", limits, np.array(limits).shape)
        echo(limits)

        @delayed
        def calculate(limits):
            results = []
            for x, y in zip(xlist, ylist):
                task = self.cov(x, y, binby=binby, limits=limits, shape=shape, selection=selection, delay=True,
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
        return self._delay(delay, finish(delayed_list(correlations)))

    @docsubst
    def cov(self, x, y=None, binby=[], limits=None, shape=default_shape, selection=False, delay=False, progress=None):
        """Calculate the covariance matrix for x and y or more expressions, possibly on a grid defined by binby.

        Either x and y are expressions, e.g.:

        >>> df.cov("x", "y")

        Or only the x argument is given with a list of expressions, e.g.:

        >>> df.cov(["x, "y, "z"])

        Example:

        >>> df.cov("x", "y")
        array([[ 53.54521742,  -3.8123135 ],
        [ -3.8123135 ,  60.62257881]])
        >>> df.cov(["x", "y", "z"])
        array([[ 53.54521742,  -3.8123135 ,  -0.98260511],
        [ -3.8123135 ,  60.62257881,   1.21381057],
        [ -0.98260511,   1.21381057,  25.55517638]])

        >>> df.cov("x", "y", binby="E", shape=2)
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
        :param delay: {delay}
        :return: {return_stat_scalar}, the last dimensions are of shape (2,2)
        """
        selection = _ensure_strings_from_expressions(selection)
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
        limits = self.limits(binby, limits, selection=selection, delay=True)

        @delayed
        def calculate(expressions, limits):
            # print('limits', limits)
            task = tasks.TaskStatistic(self, binby, shape, limits, weights=expressions, op=tasks.OP_COV, selection=selection)
            self.executor.schedule(task)
            progressbar.add_task(task, "covariance values for %r" % expressions)
            return task

        @delayed
        def finish(values):
            N = len(expressions)
            counts = values[..., :N]
            sums = values[..., N:2 * N]
            with np.errstate(divide='ignore', invalid='ignore'):
                means = sums / counts
            # matrix of means * means.T
            meansxy = means[..., None] * means[..., None, :]

            counts = values[..., 2 * N:2 * N + N**2]
            sums = values[..., 2 * N + N**2:]
            shape = counts.shape[:-1] + (N, N)
            counts = counts.reshape(shape)
            sums = sums.reshape(shape)
            with np.errstate(divide='ignore', invalid='ignore'):
                moments2 = sums / counts
            cov_matrix = moments2 - meansxy
            return cov_matrix
        progressbar = vaex.utils.progressbars(progress)
        values = calculate(expressions, limits)
        cov_matrix = finish(values)
        return self._delay(delay, cov_matrix)

    @docsubst
    @stat_1d
    def minmax(self, expression, binby=[], limits=None, shape=default_shape, selection=False, delay=False, progress=None):
        """Calculate the minimum and maximum for expressions, possibly on a grid defined by binby.


        Example:

        >>> df.minmax("x")
        array([-128.293991,  271.365997])
        >>> df.minmax(["x", "y"])
        array([[-128.293991 ,  271.365997 ],
                   [ -71.5523682,  146.465836 ]])
        >>> df.minmax("x", binby="x", shape=5, limits=[-10, 10])
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
        :param delay: {delay}
        :param progress: {progress}
        :return: {return_stat_scalar}, the last dimension is of shape (2)
        """
        # vmin  = self._compute_agg('min', expression, binby, limits, shape, selection, delay, edges, progress)
        # vmax =  self._compute_agg('max', expression, binby, limits, shape, selection, delay, edges, progress)
        @delayed
        def finish(*minmax_list):
            value = vaex.utils.unlistify(waslist, np.array(minmax_list))
            value = value.astype(dtype0)
            return value

        @delayed
        def calculate(expression, limits):
            task = tasks.TaskStatistic(self, binby, shape, limits, weight=expression, op=tasks.OP_MIN_MAX, selection=selection)
            self.executor.schedule(task)
            progressbar.add_task(task, "minmax for %s" % expression)
            return task
        @delayed
        def finish(*minmax_list):
            value = vaex.utils.unlistify(waslist, np.array(minmax_list))
            value = value.astype(dtype0)
            return value
        expression = _ensure_strings_from_expressions(expression)
        binby = _ensure_strings_from_expressions(binby)
        waslist, [expressions, ] = vaex.utils.listify(expression)
        expressions = [self._column_aliases.get(k, k) for k in expressions]
        dtypes = [self.data_type(expr) for expr in expressions]
        dtype0 = dtypes[0]
        if not all([k.kind == dtype0.kind for k in dtypes]):
            raise ValueError("cannot mix datetime and non-datetime expressions")
        progressbar = vaex.utils.progressbars(progress, name="minmaxes")
        limits = self.limits(binby, limits, selection=selection, delay=True)
        all_tasks = [calculate(expression, limits) for expression in expressions]
        result = finish(*all_tasks)
        return self._delay(delay, result)

    @docsubst
    @stat_1d
    def min(self, expression, binby=[], limits=None, shape=default_shape, selection=False, delay=False, progress=None, edges=False, array_type=None):
        """Calculate the minimum for given expressions, possibly on a grid defined by binby.


        Example:

        >>> df.min("x")
        array(-128.293991)
        >>> df.min(["x", "y"])
        array([-128.293991 ,  -71.5523682])
        >>> df.min("x", binby="x", shape=5, limits=[-10, 10])
        array([-9.99919128, -5.99972439, -1.99991322,  2.0000093 ,  6.0004878 ])

        :param expression: {expression}
        :param binby: {binby}
        :param limits: {limits}
        :param shape: {shape}
        :param selection: {selection}
        :param delay: {delay}
        :param progress: {progress}
        :param array_type: {array_type}
        :return: {return_stat_scalar}, the last dimension is of shape (2)
        """
        return self._compute_agg('min', expression, binby, limits, shape, selection, delay, edges, progress, array_type=array_type)
        @delayed
        def finish(result):
            return result[..., 0]
        return self._delay(delay, finish(self.minmax(expression, binby=binby, limits=limits, shape=shape, selection=selection, delay=delay, progress=progress)))

    @docsubst
    @stat_1d
    def max(self, expression, binby=[], limits=None, shape=default_shape, selection=False, delay=False, progress=None, edges=False, array_type=None):
        """Calculate the maximum for given expressions, possibly on a grid defined by binby.


        Example:

        >>> df.max("x")
        array(271.365997)
        >>> df.max(["x", "y"])
        array([ 271.365997,  146.465836])
        >>> df.max("x", binby="x", shape=5, limits=[-10, 10])
        array([-6.00010443, -2.00002384,  1.99998057,  5.99983597,  9.99984646])

        :param expression: {expression}
        :param binby: {binby}
        :param limits: {limits}
        :param shape: {shape}
        :param selection: {selection}
        :param delay: {delay}
        :param progress: {progress}
        :param array_type: {array_type}
        :return: {return_stat_scalar}, the last dimension is of shape (2)
        """
        return self._compute_agg('max', expression, binby, limits, shape, selection, delay, edges, progress, array_type=array_type)
        @delayed
        def finish(result):
            return result[..., 1]
        return self._delay(delay, finish(self.minmax(expression, binby=binby, limits=limits, shape=shape, selection=selection, delay=delay, progress=progress)))

    @docsubst
    @stat_1d
    def median_approx(self, expression, percentage=50., binby=[], limits=None, shape=default_shape, percentile_shape=256, percentile_limits="minmax", selection=False, delay=False):
        """Calculate the median, possibly on a grid defined by binby.

        NOTE: this value is approximated by calculating the cumulative distribution on a grid defined by
        percentile_shape and percentile_limits


        :param expression: {expression}
        :param binby: {binby}
        :param limits: {limits}
        :param shape: {shape}
        :param percentile_limits: {percentile_limits}
        :param percentile_shape: {percentile_shape}
        :param selection: {selection}
        :param delay: {delay}
        :return: {return_stat_scalar}
        """
        return self.percentile_approx(expression, 50, binby=binby, limits=limits, shape=shape, percentile_shape=percentile_shape, percentile_limits=percentile_limits, selection=selection, delay=delay)

    @docsubst
    def percentile_approx(self, expression, percentage=50., binby=[], limits=None, shape=default_shape, percentile_shape=1024, percentile_limits="minmax", selection=False, delay=False):
        """Calculate the percentile given by percentage, possibly on a grid defined by binby.

        NOTE: this value is approximated by calculating the cumulative distribution on a grid defined by
        percentile_shape and percentile_limits.


        Example:

        >>> df.percentile_approx("x", 10), df.percentile_approx("x", 90)
        (array([-8.3220355]), array([ 7.92080358]))
        >>> df.percentile_approx("x", 50, binby="x", shape=5, limits=[-10, 10])
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
        :param delay: {delay}
        :return: {return_stat_scalar}
        """
        waslist, [expressions, ] = vaex.utils.listify(expression)
        if not isinstance(binby, (tuple, list)):
            binby = [binby]
        else:
            binby = binby

        @delayed
        def calculate(expression, shape, limits):
            # task =  TaskStatistic(self, [expression] + binby, shape, limits, op=OP_ADD1, selection=selection)
            # self.executor.schedule(task)
            # return task
            return self.count(binby=list(binby) + [expression], shape=shape, limits=limits, selection=selection, delay=True, edges=True)

        @delayed
        def finish(percentile_limits, counts_list):
            results = []
            for i, counts in enumerate(counts_list):
                counts = counts.astype(np.float)
                # remove the nan and boundary edges from the first dimension,
                nonnans = list([slice(2, -1, None) for k in range(len(counts.shape) - 1)])
                nonnans.append(slice(1, None, None))  # we're gonna get rid only of the nan's, and keep the overflow edges
                nonnans = tuple(nonnans)
                cumulative_grid = np.cumsum(counts.__getitem__(nonnans), -1)  # convert to cumulative grid

                totalcounts = np.sum(counts.__getitem__(nonnans), -1)
                empty = totalcounts == 0

                original_shape = counts.shape
                shape = cumulative_grid.shape  # + (original_shape[-1] - 1,) #

                counts = np.sum(counts, -1)
                edges_floor = np.zeros(shape[:-1] + (2,), dtype=np.int64)
                edges_ceil = np.zeros(shape[:-1] + (2,), dtype=np.int64)
                # if we have an off  # of elements, say, N=3, the center is at i=1=(N-1)/2
                # if we have an even # of elements, say, N=4, the center is between i=1=(N-2)/2 and i=2=(N/2)
                # index = (shape[-1] -1-3) * percentage/100. # the -3 is for the edges
                waslist_percentage, [percentages, ] = vaex.utils.listify(percentage)
                percentiles = []
                for p in percentages:
                    values = np.array((totalcounts + 1) * p / 100.)  # make sure it's an ndarray
                    values[empty] = 0
                    floor_values = np.array(np.floor(values))
                    ceil_values = np.array(np.ceil(values))
                    vaex.vaexfast.grid_find_edges(cumulative_grid, floor_values, edges_floor)
                    vaex.vaexfast.grid_find_edges(cumulative_grid, ceil_values, edges_ceil)

                    def index_choose(a, indices):
                        # alternative to np.choise, which doesn't like the last dim to be >= 32
                        # print(a, indices)
                        out = np.zeros(a.shape[:-1])
                        # print(out.shape)
                        for i in np.ndindex(out.shape):
                            # print(i, indices[i])
                            out[i] = a[i + (indices[i],)]
                        return out

                    def calculate_x(edges, values):
                        left, right = edges[..., 0], edges[..., 1]
                        left_value = index_choose(cumulative_grid, left)
                        right_value = index_choose(cumulative_grid, right)
                        with np.errstate(divide='ignore', invalid='ignore'):
                            u = np.array((values - left_value) / (right_value - left_value))
                            # TODO: should it really be -3? not -2
                            xleft, xright = percentile_limits[i][0] + (left - 0.5) * (percentile_limits[i][1] - percentile_limits[i][0]) / (shape[-1] - 3),\
                                percentile_limits[i][0] + (right - 0.5) * (percentile_limits[i][1] - percentile_limits[i][0]) / (shape[-1] - 3)
                            x = xleft + (xright - xleft) * u  # /2
                        return x

                    x1 = calculate_x(edges_floor, floor_values)
                    x2 = calculate_x(edges_ceil, ceil_values)
                    u = values - floor_values
                    x = x1 + (x2 - x1) * u
                    percentiles.append(x)
                percentile = vaex.utils.unlistify(waslist_percentage, np.array(percentiles))
                results.append(percentile)

            return results

        shape = _expand_shape(shape, len(binby))
        percentile_shapes = _expand_shape(percentile_shape, len(expressions))
        if percentile_limits:
            percentile_limits = _expand_limits(percentile_limits, len(expressions))
        limits = self.limits(binby, limits, selection=selection, delay=True)
        percentile_limits = self.limits(expressions, percentile_limits, selection=selection, delay=True)

        @delayed
        def calculation(limits, percentile_limits):
            # print(">>>", expressions, percentile_limits)
            # print(percentile_limits[0], list(percentile_limits[0]))
            # print(list(np.array(limits).tolist()) + list(percentile_limits[0]))
            # print("limits", limits, expressions, percentile_limits, ">>", list(limits) + [list(percentile_limits[0]))
            tasks = [calculate(expression, tuple(shape) + (percentile_shape, ), list(limits) + [list(percentile_limit)])
                     for percentile_shape, percentile_limit, expression
                     in zip(percentile_shapes, percentile_limits, expressions)]
            return finish(percentile_limits, delayed_args(*tasks))
            # return tasks
        result = calculation(limits, percentile_limits)

        @delayed
        def finish2(grid):
            value = vaex.utils.unlistify(waslist, np.array(grid))
            return value
        return self._delay(delay, finish2(result))

    def _use_delay(self, delay):
        return delay == True

    def _delay(self, delay, task, progressbar=False):
        if task.isRejected:
            task.get()
        if delay:
            return task
        else:
            self.execute()
            return task.get()

    @docsubst
    def limits_percentage(self, expression, percentage=99.73, square=False, selection=False, delay=False):
        """Calculate the [min, max] range for expression, containing approximately a percentage of the data as defined
        by percentage.

        The range is symmetric around the median, i.e., for a percentage of 90, this gives the same results as:

        Example:

        >>> df.limits_percentage("x", 90)
        array([-12.35081376,  12.14858052]
        >>> df.percentile_approx("x", 5), df.percentile_approx("x", 95)
        (array([-12.36813152]), array([ 12.13275818]))

        NOTE: this value is approximated by calculating the cumulative distribution on a grid.
        NOTE 2: The values above are not exactly the same, since percentile and limits_percentage do not share the same code

        :param expression: {expression_limits}
        :param float percentage: Value between 0 and 100
        :param delay: {delay}
        :return: {return_limits}
        """
        import scipy
        logger.info("limits_percentage for %r, with percentage=%r", expression, percentage)
        waslist, [expressions, ] = vaex.utils.listify(expression)
        limits = []
        for expr in expressions:
            limits_minmax = self.minmax(expr, selection=selection)
            vmin, vmax = limits_minmax
            size = 1024 * 16
            counts = self.count(binby=expr, shape=size, limits=limits_minmax, selection=selection)
            cumcounts = np.concatenate([[0], np.cumsum(counts)])
            cumcounts = cumcounts / cumcounts.max()
            # TODO: this is crude.. see the details!
            f = (1 - percentage / 100.) / 2
            x = np.linspace(vmin, vmax, size + 1)
            l = scipy.interp([f, 1 - f], cumcounts, x)
            limits.append(l)
        return vaex.utils.unlistify(waslist, limits)

    @docsubst
    def limits(self, expression, value=None, square=False, selection=None, delay=False, shape=None):
        """Calculate the [min, max] range for expression, as described by value, which is 'minmax' by default.

        If value is a list of the form [minvalue, maxvalue], it is simply returned, this is for convenience when using mixed
        forms.

        Example:

        >>> import vaex
        >>> df = vaex.example()
        >>> df.limits("x")
        array([-128.293991,  271.365997])
        >>> df.limits("x", "99.7%")
        array([-28.86381927,  28.9261226 ])
        >>> df.limits(["x", "y"])
        (array([-128.293991,  271.365997]), array([ -71.5523682,  146.465836 ]))
        >>> df.limits(["x", "y"], "99.7%")
        (array([-28.86381927,  28.9261226 ]), array([-28.60476934,  28.96535249]))
        >>> df.limits(["x", "y"], ["minmax", "90%"])
        (array([-128.293991,  271.365997]), array([-13.37438402,  13.4224423 ]))
        >>> df.limits(["x", "y"], ["minmax", [0, 10]])
        (array([-128.293991,  271.365997]), [0, 10])

        :param expression: {expression_limits}
        :param value: {limits}
        :param selection: {selection}
        :param delay: {delay}
        :return: {return_limits}
        """
        if expression == []:
            return [] if shape is None else ([], [])
        waslist, [expressions, ] = vaex.utils.listify(expression)
        expressions = _ensure_strings_from_expressions(expressions)
        selection = _ensure_strings_from_expressions(selection)
        if value is None:
            value = "minmax"
        if _is_limit(value) or not _issequence(value):
            values = (value,) * len(expressions)
        else:
            values = value

        initial_expressions, initial_values = expressions, values
        expression_values = dict()
        expression_shapes = dict()
        for i, (expression, value) in enumerate(zip(expressions, values)):
            if _issequence(expression):
                expressions = expression
                nested = True
            else:
                expressions = [expression]
                nested = False
            if _is_limit(value) or not _issequence(value):
                values = (value,) * len(expressions)
            else:
                values = value
            for j, (expression, value) in enumerate(zip(expressions, values)):
                if shape is not None:
                    if _issequence(shape):
                        shapes = shape
                    else:
                        shapes = (shape, ) * (len(expressions) if nested else len(initial_expressions))

                shape_index = j if nested else i

                if not _is_limit(value):
                    expression_values[(expression, value)] = None
                if self.is_category(expression):
                    N = self._categories[_ensure_string_from_expression(expression)]['N']
                    expression_shapes[expression] = min(N, shapes[shape_index] if shape is not None else default_shape)
                else:
                    expression_shapes[expression] = shapes[shape_index] if shape is not None else default_shape

        limits_list = []
        for expression, value in expression_values.keys():
            if self.is_category(expression):
                N = self._categories[_ensure_string_from_expression(expression)]['N']
                limits = [-0.5, N-0.5]
            else:
                if isinstance(value, six.string_types):
                    if value == "minmax":
                        limits = self.minmax(expression, selection=selection, delay=True)
                    else:
                        match = re.match(r"([\d.]*)(\D*)", value)
                        if match is None:
                            raise ValueError("do not understand limit specifier %r, examples are 90%, 3sigma")
                        else:
                            number, type = match.groups()
                            import ast
                            number = ast.literal_eval(number)
                            type = type.strip()
                            if type in ["s", "sigma"]:
                                limits = self.limits_sigma(number)
                            elif type in ["ss", "sigmasquare"]:
                                limits = self.limits_sigma(number, square=True)
                            elif type in ["%", "percent"]:
                                limits = self.limits_percentage(expression, number, selection=selection, delay=False)
                            elif type in ["%s", "%square", "percentsquare"]:
                                limits = self.limits_percentage(expression, number, selection=selection, square=True, delay=True)
                elif value is None:
                    limits = self.minmax(expression, selection=selection, delay=True)
                else:
                    limits = value
            limits_list.append(limits)
            if limits is None:
                raise ValueError("limit %r not understood" % value)
            expression_values[(expression, value)] = limits

        limits_list = delayed_args(*limits_list)

        @delayed
        def finish(limits_list):
            # print("##### 2)", expression_values.keys())
            limits_outer = []
            shapes_list = []
            for expression, value in zip(initial_expressions, initial_values):
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
                # print("expressions 3)", expressions)
                # print("values      3)", values)
                limits = []
                shapes = []
                for expression, value in zip(expressions, values):
                    if not _is_limit(value):
                        value = expression_values[(expression, value)]
                        if not _is_limit(value):
                            # print(">>> value", value)
                            value = value.get()
                    limits.append(value)
                    shapes.append(expression_shapes[expression])
                    # if not _is_limit(value): # if a
                    #   #value = tuple(value) # list is not hashable
                    #   expression_values[(expression, value)] = expression_values[(expression, value)].get()
                    # else:
                    #   #value = tuple(value) # list is not hashable
                    #   expression_values[(expression, value)] = ()
                if waslist2:
                    limits_outer.append(limits)
                    shapes_list.append(shapes)
                else:
                    limits_outer.append(limits[0])
                    shapes_list.append(shapes[0])
            # logger.debug(">>>>>>>> complete list of limits: %r %r", limits_list, np.array(limits_list).shape)

            # print("limits", limits_outer)
            if shape:
                return vaex.utils.unlistify(waslist, limits_outer), vaex.utils.unlistify(waslist, shapes_list)
            else:
                return vaex.utils.unlistify(waslist, limits_outer)
        return self._delay(delay, finish(limits_list))

    def mode(self, expression, binby=[], limits=None, shape=256, mode_shape=64, mode_limits=None, progressbar=False, selection=None):
        """Calculate/estimate the mode."""
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
            centers = np.linspace(pmin, pmax, mode_shape + 1)[:-1]  # ignore last bin
            centers += (centers[1] - centers[0]) / 2  # and move half a bin to the right

            modes = centers[indices]
            ok = counts.sum(axis=0) > 0
            modes[~ok] = np.nan
            return modes

    @vaex.utils.deprecated('use df.widget.heatmap')
    def plot_widget(self, x, y, limits=None, f="identity", **kwargs):
        return self.widget.heatmap(x, y, limits=limits, transform=f, **kwargs)

    @vaex.utils.deprecated('use plot_widget')
    def plot_bq(self, x, y, grid=None, shape=256, limits=None, what="count(*)", figsize=None,
                f="identity", figure_key=None, fig=None, axes=None, xlabel=None, ylabel=None, title=None,
                show=True, selection=[None, True], colormap="afmhot", grid_limits=None, normalize="normalize",
                grid_before=None,
                what_kwargs={}, type="default",
                scales=None, tool_select=False, bq_cleanup=True,
                **kwargs):
        import vaex.ext.bqplot
        cls = vaex.ext.bqplot.get_class(type)
        plot2d = cls(df=self, x=x, y=y, grid=grid, shape=shape, limits=limits, what=what,
                     f=f, figure_key=figure_key, fig=fig,
                     selection=selection, grid_before=grid_before,
                     grid_limits=grid_limits, normalize=normalize, colormap=colormap, what_kwargs=what_kwargs, **kwargs)
        if show:
            plot2d.show()
        return plot2d

    # @_hidden
    def healpix_count(self, expression=None, healpix_expression=None, healpix_max_level=12, healpix_level=8, binby=None, limits=None, shape=default_shape, delay=False, progress=None, selection=None):
        """Count non missing value for expression on an array which represents healpix data.

        :param expression: Expression or column for which to count non-missing values, or None or '*' for counting the rows
        :param healpix_expression: {healpix_max_level}
        :param healpix_max_level: {healpix_max_level}
        :param healpix_level: {healpix_level}
        :param binby: {binby}, these dimension follow the first healpix dimension.
        :param limits: {limits}
        :param shape: {shape}
        :param selection: {selection}
        :param delay: {delay}
        :param progress: {progress}
        :return:
        """
        # if binby is None:
        import healpy as hp
        if healpix_expression is None:
            if self.ucds.get("source_id", None) == 'meta.id;meta.main':  # we now assume we have gaia data
                healpix_expression = "source_id/34359738368"

        if healpix_expression is None:
            raise ValueError("no healpix_expression given, and was unable to guess")

        reduce_level = healpix_max_level - healpix_level
        NSIDE = 2**healpix_level
        nmax = hp.nside2npix(NSIDE)
        scaling = 4**reduce_level
        expr = "%s/%s" % (healpix_expression, scaling)
        binby = [expr] + ([] if binby is None else _ensure_list(binby))
        shape = (nmax,) + _expand_shape(shape, len(binby) - 1)
        epsilon = 1. / scaling / 2
        limits = [[-epsilon, nmax - epsilon]] + ([] if limits is None else limits)
        return self.count(expression, binby=binby, limits=limits, shape=shape, delay=delay, progress=progress, selection=selection)

    # @_hidden
    def healpix_plot(self, healpix_expression="source_id/34359738368", healpix_max_level=12, healpix_level=8, what="count(*)", selection=None,
                     grid=None,
                     healpix_input="equatorial", healpix_output="galactic", f=None,
                     colormap="afmhot", grid_limits=None, image_size=800, nest=True,
                     figsize=None, interactive=False, title="", smooth=None, show=False, colorbar=True,
                     rotation=(0, 0, 0), **kwargs):
        """Viz data in 2d using a healpix column.

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
        # plot_level = healpix_level #healpix_max_level-reduce_level
        import healpy as hp
        import pylab as plt
        if grid is None:
            reduce_level = healpix_max_level - healpix_level
            NSIDE = 2**healpix_level
            nmax = hp.nside2npix(NSIDE)
            # print nmax, np.sqrt(nmax)
            scaling = 4**reduce_level
            # print nmax
            epsilon = 1. / scaling / 2
            grid = self._stat(what=what, binby="%s/%s" % (healpix_expression, scaling), limits=[-epsilon, nmax - epsilon], shape=nmax, selection=selection)
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
            # grid[np.isnan(grid)] = np.nanmean(grid)
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
            coord = coord_map[healpix_input], coord_map[healpix_output]
            if coord_map[healpix_input] == coord_map[healpix_output]:
                coord = None
            f(fgrid, unit=what_label, rot=rotation, nest=nest, title=title, coord=coord,
              cmap=colormap, hold=True, xsize=image_size, min=grid_min, max=grid_max, cbar=colorbar, **kwargs)
        if show:
            plt.show()

    @docsubst
    @stat_1d
    def _stat(self, what="count(*)", what_kwargs={}, binby=[], limits=None, shape=default_shape, selection=False, delay=False, progress=None):
        waslist_what, [whats, ] = vaex.utils.listify(what)
        limits = self.limits(binby, limits, delay=True)
        waslist_selection, [selections] = vaex.utils.listify(selection)
        binby = _ensure_list(binby)

        what_labels = []
        shape = _expand_shape(shape, len(binby))
        total_grid = np.zeros((len(whats), len(selections)) + shape, dtype=float)

        @delayed
        def copy_grids(grids):
            total_grid[index] = grid

        @delayed
        def get_whats(limits):
            grids = []
            for j, what in enumerate(whats):
                what = what.strip()
                index = what.index("(")
                groups = re.match(r"(.*)\((.*)\)", what).groups()
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
                                                       selection=selections, progress=progress, delay=delay)
                    elif function == "count":
                        grid = self.count(arguments, binby, shape=shape, limits=limits, selection=selections,
                                          progress=progress, delay=delay)
                    else:
                        raise ValueError("Could not understand method: %s, expected one of %r'" % (function, functions))
                    # what_labels.append(what_label)
                    grids.append(grid)

            # else:
            #   raise ValueError("Could not understand 'what' argument %r, expected something in form: 'count(*)', 'mean(x)'" % what)
            return grids
        grids = get_whats(limits)
        # print grids
        # grids = delayed_args(*grids)

        @delayed
        def finish(grids):
            for i, grid in enumerate(grids):
                total_grid[i] = grid
            return total_grid[slice(None, None, None) if waslist_what else 0, slice(None, None, None) if waslist_selection else 0]
        s = finish(delayed_list(grids))
        return self._delay(delay, s)

    plot = _requires('viz')
    plot1d = _requires('viz')
    scatter = _requires('viz')

    def plot3d(self, x, y, z, vx=None, vy=None, vz=None, vwhat=None, limits=None, grid=None, what="count(*)", shape=128, selection=[None, True], f=None,
               vcount_limits=None,
               smooth_pre=None, smooth_post=None, grid_limits=None, normalize="normalize", colormap="afmhot",
               figure_key=None, fig=None,
               lighting=True, level=[0.1, 0.5, 0.9], opacity=[0.01, 0.05, 0.1], level_width=0.1,
               show=True, **kwargs):
        """Use at own risk, requires ipyvolume"""
        import vaex.ext.ipyvolume
        # vaex.ext.ipyvolume.
        cls = vaex.ext.ipyvolume.PlotDefault
        plot3d = cls(df=self, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz,
                     grid=grid, shape=shape, limits=limits, what=what,
                     f=f, figure_key=figure_key, fig=fig,
                     selection=selection, smooth_pre=smooth_pre, smooth_post=smooth_post,
                     grid_limits=grid_limits, vcount_limits=vcount_limits, normalize=normalize, colormap=colormap, **kwargs)
        if show:
            plot3d.show()
        return plot3d

    @property
    def col(self):
        """Gives direct access to the columns only (useful for tab completion).

        Convenient when working with ipython in combination with small DataFrames, since this gives tab-completion.

        Columns can be accessed by their names, which are attributes. The attributes are currently expressions, so you can
        do computations with them.

        Example

        >>> ds = vaex.example()
        >>> df.plot(df.col.x, df.col.y)

        """
        class ColumnList(object):
            pass
        data = ColumnList()
        for name in self.get_column_names():
            expression = getattr(self, name, None)
            if not isinstance(expression, Expression):
                expression = Expression(self, name)
            setattr(data, name, expression)
        return data

    def close_files(self):
        """Close any possible open file handles, the DataFrame will not be in a usable state afterwards."""
        pass

    def byte_size(self, selection=False, virtual=False):
        """Return the size in bytes the whole DataFrame requires (or the selection), respecting the active_fraction."""
        bytes_per_row = 0
        N = self.count(selection=selection)
        extra = 0
        for column in list(self.get_column_names(virtual=virtual)):
            dtype = self.data_type(column)
            dtype_internal = self.data_type(column, internal=True)
            #if dtype in [str_type, str] and dtype_internal.kind == 'O':
            if isinstance(self.columns[column], ColumnString):
                # TODO: document or fix this
                # is it too expensive to calculate this exactly?
                extra += self.columns[column].nbytes
            else:
                bytes_per_row += dtype_internal.itemsize
                if np.ma.isMaskedArray(self.columns[column]):
                    bytes_per_row += 1
        return bytes_per_row * self.count(selection=selection) + extra

    @property
    def nbytes(self):
        """Alias for `df.byte_size()`, see :meth:`DataFrame.byte_size`."""
        return self.byte_size()

    def _shape_of(self, expression, filtered=True, check_alias=True):
        if check_alias:
            if str(expression) in self._column_aliases:
                expression = self._column_aliases[str(expression)]  # translate the alias name into the real name
        sample = self.evaluate(expression, 0, 1, filtered=False, internal=True, parallel=False)
        rows = len(self) if filtered else self.length_unfiltered()
        return (rows,) + sample.shape[1:]

    def data_type(self, expression, internal=False, check_alias=True):
        """Return the numpy dtype for the given expression, if not a column, the first row will be evaluated to get the dtype."""
        expression = _ensure_string_from_expression(expression)
        if check_alias:
            if expression in self._column_aliases:
                expression = self._column_aliases[expression]  # translate the alias name into the real name
        if expression in self._dtypes_override:
            return self._dtypes_override[expression]
        if expression in self.variables:
            return np.float64(1).dtype
        elif self.is_local() and expression in self.columns.keys():
            column = self.columns[expression]
            if hasattr(column, 'dtype'):
                dtype = column.dtype
            else:
                data = column[0:1]
                dtype = data.dtype
        else:
            try:
                data = self.evaluate(expression, 0, 1, filtered=False, internal=True, parallel=False)
            except:
                data = self.evaluate(expression, 0, 1, filtered=True, internal=True, parallel=False)
            dtype = data.dtype
        if not internal:
            if dtype != str_type:
                if dtype.kind in 'US':
                    return str_type
        return dtype

    @property
    def dtypes(self):
        """Gives a Pandas series object containing all numpy dtypes of all columns (except hidden)."""
        from pandas import Series
        return Series({column_name:self.data_type(column_name) for column_name in self.get_column_names()})

    def is_masked(self, column):
        '''Return if a column is a masked (numpy.ma) column.'''
        column = _ensure_string_from_expression(column)
        if column in self.columns:
            column = self.columns[column]
            if isinstance(column, np.ndarray):
                return np.ma.isMaskedArray(column)
            else:
                # in case the column is not a numpy array, we take a small slice
                # which should return a numpy array
                return np.ma.isMaskedArray(column[0:1])
        else:
            ar = self.evaluate(column, i1=0, i2=1, parallel=False)
            if isinstance(ar, np.ndarray) and np.ma.isMaskedArray(ar):
                return True
        return False

    def label(self, expression, unit=None, output_unit=None, format="latex_inline"):
        label = expression
        unit = unit or self.unit(expression)
        try:  # if we can convert the unit, use that for the labeling
            if output_unit and unit:  # avoid unnecessary error msg'es
                output_unit.to(unit)
                unit = output_unit
        except:
            logger.exception("unit error")
        if unit is not None:
            label = "%s (%s)" % (label, unit.to_string('latex_inline'))
        return label

    def unit(self, expression, default=None):
        """Returns the unit (an astropy.unit.Units object) for the expression.

        Example

        >>> import vaex
        >>> ds = vaex.example()
        >>> df.unit("x")
        Unit("kpc")
        >>> df.unit("x*L")
        Unit("km kpc2 / s")


        :param expression: Expression, which can be a column name
        :param default: if no unit is known, it will return this
        :return: The resulting unit of the expression
        :rtype: astropy.units.Unit
        """
        expression = _ensure_string_from_expression(expression)
        try:
            # if an expression like pi * <some_expr> it will evaluate to a quantity instead of a unit
            unit_or_quantity = eval(expression, expression_namespace, scopes.UnitScope(self))
            unit = unit_or_quantity.unit if hasattr(unit_or_quantity, "unit") else unit_or_quantity
            return unit if isinstance(unit, astropy.units.Unit) else None
        except:
            # logger.exception("error evaluating unit expression: %s", expression)
            # astropy doesn't add units, so we try with a quatiti
            try:
                return eval(expression, expression_namespace, scopes.UnitScope(self, 1.)).unit
            except:
                # logger.exception("error evaluating unit expression: %s", expression)
                return default

    def ucd_find(self, ucds, exclude=[]):
        """Find a set of columns (names) which have the ucd, or part of the ucd.

        Prefixed with a ^, it will only match the first part of the ucd.

        Example

        >>> df.ucd_find('pos.eq.ra', 'pos.eq.dec')
        ['RA', 'DEC']
        >>> df.ucd_find('pos.eq.ra', 'doesnotexist')
        >>> df.ucds[df.ucd_find('pos.eq.ra')]
        'pos.eq.ra;meta.main'
        >>> df.ucd_find('meta.main')]
        'dec'
        >>> df.ucd_find('^meta.main')]
        """
        if isinstance(ucds, six.string_types):
            ucds = [ucds]
        if len(ucds) == 1:
            ucd = ucds[0]
            if ucd[0] == "^":  # we want it to start with
                ucd = ucd[1:]
                columns = [name for name in self.get_column_names() if self.ucds.get(name, "").startswith(ucd) and name not in exclude]
            else:
                columns = [name for name in self.get_column_names() if ucd in self.ucds.get(name, "") and name not in exclude]
            return None if len(columns) == 0 else columns[0]
        else:
            columns = [self.ucd_find([ucd], exclude=exclude) for ucd in ucds]
            return None if None in columns else columns

    @vaex.utils.deprecated('Will most likely disappear or move')
    @_hidden
    def selection_favorite_add(self, name, selection_name="default"):
        selection = self.get_selection(name=selection_name)
        if selection:
            self.favorite_selections[name] = selection
            self.selections_favorite_store()
        else:
            raise ValueError("no selection exists")

    @vaex.utils.deprecated('Will most likely disappear or move')
    @_hidden
    def selection_favorite_remove(self, name):
        del self.favorite_selections[name]
        self.selections_favorite_store()

    @vaex.utils.deprecated('Will most likely disappear or move')
    @_hidden
    def selection_favorite_apply(self, name, selection_name="default", executor=None):
        self.set_selection(self.favorite_selections[name], name=selection_name, executor=executor)

    @vaex.utils.deprecated('Will most likely disappear or move')
    @_hidden
    def selections_favorite_store(self):
        path = os.path.join(self.get_private_dir(create=True), "favorite_selection.yaml")
        selections = collections.OrderedDict([(key, value.to_dict()) for key, value in self.favorite_selections.items()])
        vaex.utils.write_json_or_yaml(path, selections)

    @vaex.utils.deprecated('Will most likely disappear or move')
    @_hidden
    def selections_favorite_load(self):
        try:
            path = os.path.join(self.get_private_dir(create=True), "favorite_selection.yaml")
            if os.path.exists(path):
                selections_dict = vaex.utils.read_json_or_yaml(path)
                for key, value in selections_dict.items():
                    self.favorite_selections[key] = selections.selection_from_dict(self, value)
        except:
            logger.exception("non fatal error")

    def get_private_dir(self, create=False):
        """Each DataFrame has a directory where files are stored for metadata etc.

        Example

        >>> import vaex
        >>> ds = vaex.example()
        >>> vaex.get_private_dir()
        '/Users/users/breddels/.vaex/dfs/_Users_users_breddels_vaex-testing_data_helmi-dezeeuw-2000-10p.hdf5'

        :param bool create: is True, it will create the directory if it does not exist
        """
        if self.is_local():
            name = os.path.abspath(self.path).replace(os.path.sep, "_")[:250]  # should not be too long for most os'es
            name = name.replace(":", "_")  # for windows drive names
        else:
            server = self.server
            name = "%s_%s_%s_%s" % (server.hostname, server.port, server.base_path.replace("/", "_"), self.name)
        dir = os.path.join(vaex.utils.get_private_dir(), "dfs", name)
        if create and not os.path.exists(dir):
            os.makedirs(dir)
        return dir

    def state_get(self):
        """Return the internal state of the DataFrame in a dictionary

        Example:

        >>> import vaex
        >>> df = vaex.from_scalars(x=1, y=2)
        >>> df['r'] = (df.x**2 + df.y**2)**0.5
        >>> df.state_get()
        {'active_range': [0, 1],
        'column_names': ['x', 'y', 'r'],
        'description': None,
        'descriptions': {},
        'functions': {},
        'renamed_columns': [],
        'selections': {'__filter__': None},
        'ucds': {},
        'units': {},
        'variables': {},
        'virtual_columns': {'r': '(((x ** 2) + (y ** 2)) ** 0.5)'}}
        """

        virtual_names = list(self.virtual_columns.keys()) + list(self.variables.keys())
        units = {key: str(value) for key, value in self.units.items()}
        ucds = {key: value for key, value in self.ucds.items() if key in virtual_names}
        descriptions = {key: value for key, value in self.descriptions.items()}
        import vaex.serialize

        def check(key, value):
            if not vaex.serialize.can_serialize(value.f):
                warnings.warn('Cannot serialize function for virtual column {} (use vaex.serialize.register)'.format(key))
                return False
            return True

        def clean(value):
            return vaex.serialize.to_dict(value.f)
        functions = {key: clean(value) for key, value in self.functions.items() if check(key, value)}
        virtual_columns = {key: value for key, value in self.virtual_columns.items()}
        selections = {name: self.get_selection(name) for name, history in self.selection_histories.items()}
        selections = {name: selection.to_dict() if selection is not None else None for name, selection in selections.items()}
    # if selection is not None}
        state = dict(virtual_columns=virtual_columns,
                     column_names=self.column_names,
                     renamed_columns=self._renamed_columns,
                     variables=self.variables,
                     functions=functions,
                     selections=selections,
                     ucds=ucds,
                     units=units,
                     descriptions=descriptions,
                     description=self.description,
                     active_range=[self._index_start, self._index_end],
                     column_aliases=self._column_aliases)
        return state

    def state_set(self, state, use_active_range=False, trusted=True):
        """Sets the internal state of the df

        Example:

        >>> import vaex
        >>> df = vaex.from_scalars(x=1, y=2)
        >>> df
          #    x    y        r
          0    1    2  2.23607
        >>> df['r'] = (df.x**2 + df.y**2)**0.5
        >>> state = df.state_get()
        >>> state
        {'active_range': [0, 1],
        'column_names': ['x', 'y', 'r'],
        'description': None,
        'descriptions': {},
        'functions': {},
        'renamed_columns': [],
        'selections': {'__filter__': None},
        'ucds': {},
        'units': {},
        'variables': {},
        'virtual_columns': {'r': '(((x ** 2) + (y ** 2)) ** 0.5)'}}
        >>> df2 = vaex.from_scalars(x=3, y=4)
        >>> df2.state_set(state)  # now the virtual functions are 'copied'
        >>> df2
          #    x    y    r
          0    3    4    5

        :param state: dict as returned by :meth:`DataFrame.state_get`.
        :param bool use_active_range: Whether to use the active range or not.
        """
        self.description = state['description']
        if use_active_range:
            self._index_start, self._index_end = state['active_range']
        self._length_unfiltered = self._index_end - self._index_start
        if 'renamed_columns' in state:
            for old, new in state['renamed_columns']:
                self._rename(old, new)
        for name, value in state['functions'].items():
            self.add_function(name, vaex.serialize.from_dict(value, trusted=trusted))
        if 'column_names' in state:
            # we clear all columns, and add them later on, since otherwise self[name] = ... will try
            # to rename the columns (which is unsupported for remote dfs)
            self.column_names = []
            self.virtual_columns = {}
            for name, value in state['virtual_columns'].items():
                self[name] = self._expr(value)
                # self._save_assign_expression(name)
            self.column_names = list(state['column_names'])
            for name in self.column_names:
                self._save_assign_expression(name)
        else:
            # old behaviour
            self.virtual_columns = {}
            for name, value in state['virtual_columns'].items():
                self[name] = self._expr(value)
        self.variables = state['variables']
        self._column_aliases = state.get('column_aliases', {})
        import astropy  # TODO: make this dep optional?
        units = {key: astropy.units.Unit(value) for key, value in state["units"].items()}
        self.units.update(units)
        for name, selection_dict in state['selections'].items():
            # TODO: make selection use the vaex.serialize framework
            if selection_dict is None:
                selection = None
            else:
                selection = selections.selection_from_dict(selection_dict)
            self.set_selection(selection, name=name)

    def state_write(self, f):
        """Write the internal state to a json or yaml file (see :meth:`DataFrame.state_get`)

        Example

        >>> import vaex
        >>> df = vaex.from_scalars(x=1, y=2)
        >>> df['r'] = (df.x**2 + df.y**2)**0.5
        >>> df.state_write('state.json')
        >>> print(open('state.json').read())
        {
        "virtual_columns": {
            "r": "(((x ** 2) + (y ** 2)) ** 0.5)"
        },
        "column_names": [
            "x",
            "y",
            "r"
        ],
        "renamed_columns": [],
        "variables": {
            "pi": 3.141592653589793,
            "e": 2.718281828459045,
            "km_in_au": 149597870.7,
            "seconds_per_year": 31557600
        },
        "functions": {},
        "selections": {
            "__filter__": null
        },
        "ucds": {},
        "units": {},
        "descriptions": {},
        "description": null,
        "active_range": [
            0,
            1
        ]
        }
        >>> df.state_write('state.yaml')
        >>> print(open('state.yaml').read())
        active_range:
        - 0
        - 1
        column_names:
        - x
        - y
        - r
        description: null
        descriptions: {}
        functions: {}
        renamed_columns: []
        selections:
        __filter__: null
        ucds: {}
        units: {}
        variables:
        pi: 3.141592653589793
        e: 2.718281828459045
        km_in_au: 149597870.7
        seconds_per_year: 31557600
        virtual_columns:
        r: (((x ** 2) + (y ** 2)) ** 0.5)

        :param str f: filename (ending in .json or .yaml)
        """
        vaex.utils.write_json_or_yaml(f, self.state_get())

    def state_load(self, f, use_active_range=False):
        """Load a state previously stored by :meth:`DataFrame.state_write`, see also :meth:`DataFrame.state_set`."""
        state = vaex.utils.read_json_or_yaml(f)
        self.state_set(state, use_active_range=use_active_range)

    def remove_virtual_meta(self):
        """Removes the file with the virtual column etc, it does not change the current virtual columns etc."""
        dir = self.get_private_dir(create=True)
        path = os.path.join(dir, "virtual_meta.yaml")
        try:
            if os.path.exists(path):
                os.remove(path)
            if not os.listdir(dir):
                os.rmdir(dir)
        except:
            logger.exception("error while trying to remove %s or %s", path, dir)
    # def remove_meta(self):
    #   path = os.path.join(self.get_private_dir(create=True), "meta.yaml")
    #   os.remove(path)

    @_hidden
    def write_virtual_meta(self):
        """Writes virtual columns, variables and their ucd,description and units.

        The default implementation is to write this to a file called virtual_meta.yaml in the directory defined by
        :func:`DataFrame.get_private_dir`. Other implementation may store this in the DataFrame file itself.

        This method is called after virtual columns or variables are added. Upon opening a file, :func:`DataFrame.update_virtual_meta`
        is called, so that the information is not lost between sessions.

        Note: opening a DataFrame twice may result in corruption of this file.

        """
        path = os.path.join(self.get_private_dir(create=True), "virtual_meta.yaml")
        virtual_names = list(self.virtual_columns.keys()) + list(self.variables.keys())
        units = {key: str(value) for key, value in self.units.items() if key in virtual_names}
        ucds = {key: value for key, value in self.ucds.items() if key in virtual_names}
        descriptions = {key: value for key, value in self.descriptions.items() if key in virtual_names}
        meta_info = dict(virtual_columns=self.virtual_columns,
                         variables=self.variables,
                         ucds=ucds, units=units, descriptions=descriptions)
        vaex.utils.write_json_or_yaml(path, meta_info)

    @_hidden
    def update_virtual_meta(self):
        """Will read back the virtual column etc, written by :func:`DataFrame.write_virtual_meta`. This will be done when opening a DataFrame."""
        import astropy.units
        try:
            path = os.path.join(self.get_private_dir(create=False), "virtual_meta.yaml")
            if os.path.exists(path):
                meta_info = vaex.utils.read_json_or_yaml(path)
                if 'virtual_columns' not in meta_info:
                    return
                self.virtual_columns.update(meta_info["virtual_columns"])
                self.variables.update(meta_info["variables"])
                self.ucds.update(meta_info["ucds"])
                self.descriptions.update(meta_info["descriptions"])
                units = {key: astropy.units.Unit(value) for key, value in meta_info["units"].items()}
                self.units.update(units)
        except:
            logger.exception("non fatal error")

    @_hidden
    def write_meta(self):
        """Writes all meta data, ucd,description and units

        The default implementation is to write this to a file called meta.yaml in the directory defined by
        :func:`DataFrame.get_private_dir`. Other implementation may store this in the DataFrame file itself.
        (For instance the vaex hdf5 implementation does this)

        This method is called after virtual columns or variables are added. Upon opening a file, :func:`DataFrame.update_meta`
        is called, so that the information is not lost between sessions.

        Note: opening a DataFrame twice may result in corruption of this file.

        """
        # raise NotImplementedError
        path = os.path.join(self.get_private_dir(create=True), "meta.yaml")
        units = {key: str(value) for key, value in self.units.items()}
        meta_info = dict(description=self.description,
                         ucds=self.ucds, units=units, descriptions=self.descriptions,
                         )
        vaex.utils.write_json_or_yaml(path, meta_info)

    @_hidden
    def update_meta(self):
        """Will read back the ucd, descriptions, units etc, written by :func:`DataFrame.write_meta`. This will be done when opening a DataFrame."""
        import astropy.units
        try:
            path = os.path.join(self.get_private_dir(create=False), "meta.yaml")
            if os.path.exists(path):
                meta_info = vaex.utils.read_json_or_yaml(path)
                self.description = meta_info["description"]
                self.ucds.update(meta_info["ucds"])
                self.descriptions.update(meta_info["descriptions"])
                # self.virtual_columns.update(meta_info["virtual_columns"])
                # self.variables.update(meta_info["variables"])
                units = {key: astropy.units.Unit(value) for key, value in meta_info["units"].items()}
                self.units.update(units)
        except:
            logger.exception("non fatal error, but could read/understand %s", path)

    def is_local(self):
        """Returns True if the DataFrame is local, False when a DataFrame is remote."""
        raise NotImplementedError

    def get_auto_fraction(self):
        return self._auto_fraction

    def set_auto_fraction(self, enabled):
        self._auto_fraction = enabled

    @classmethod
    def can_open(cls, path, *args, **kwargs):
        # """Tests if this class can open the file given by path"""
        return False

    @classmethod
    def get_options(cls, path):
        return []

    @classmethod
    def option_to_args(cls, option):
        return []

    def combinations(self, expressions_list=None, dimension=2, exclude=None, **kwargs):
        """Generate a list of combinations for the possible expressions for the given dimension.

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
                        # $#expressions = set(expressions)
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

    def set_variable(self, name, expression_or_value, write=True):
        """Set the variable to an expression or value defined by expression_or_value.

        Example

        >>> df.set_variable("a", 2.)
        >>> df.set_variable("b", "a**2")
        >>> df.get_variable("b")
        'a**2'
        >>> df.evaluate_variable("b")
        4.0

        :param name: Name of the variable
        :param write: write variable to meta file
        :param expression: value or expression
        """
        self.variables[name] = expression_or_value
        # if write:
        #   self.write_virtual_meta()

    def get_variable(self, name):
        """Returns the variable given by name, it will not evaluate it.

        For evaluation, see :func:`DataFrame.evaluate_variable`, see also :func:`DataFrame.set_variable`

        """
        return self.variables[name]

    def evaluate_variable(self, name):
        """Evaluates the variable given by name."""
        if isinstance(self.variables[name], six.string_types):
            # TODO: this does not allow more than one level deep variable, like a depends on b, b on c, c is a const
            value = eval(self.variables[name], expression_namespace, self.variables)
            return value
        else:
            return self.variables[name]

    def _evaluate_selection_mask(self, name="default", i1=None, i2=None, selection=None, cache=False, filter_mask=None):
        """Internal use, ignores the filter"""
        i1 = i1 or 0
        i2 = i2 or len(self)
        scope = scopes._BlockScopeSelection(self, i1, i2, selection, cache=cache, filter_mask=filter_mask)
        return vaex.utils.unmask_selection_mask(scope.evaluate(name))

    def evaluate_selection_mask(self, name="default", i1=None, i2=None, selection=None, cache=False, filtered=True, pre_filtered=True):
        i1 = i1 or 0
        i2 = i2 or self.length_unfiltered()
        if isinstance(name, vaex.expression.Expression):
            # make sure if we get passed an expression, it is converted to a string
            # otherwise the name != <sth> will evaluate to an Expression object
            name = str(name)
        if name in [None, False] and self.filtered and filtered:
            scope_global = scopes._BlockScopeSelection(self, i1, i2, None, cache=cache)
            mask_global = scope_global.evaluate(FILTER_SELECTION_NAME)
            return vaex.utils.unmask_selection_mask(mask_global)
        elif self.filtered and filtered and name != FILTER_SELECTION_NAME:
            scope_global = scopes._BlockScopeSelection(self, i1, i2, None, cache=cache)
            mask_global = scope_global.evaluate(FILTER_SELECTION_NAME)
            if pre_filtered:
                scope = scopes._BlockScopeSelection(self, i1, i2, selection, filter_mask=vaex.utils.unmask_selection_mask(mask_global))
                mask = scope.evaluate(name)
                return vaex.utils.unmask_selection_mask(mask)
            else:  # only used in legacy.py?
                scope = scopes._BlockScopeSelection(self, i1, i2, selection)
                mask = scope.evaluate(name)
                return vaex.utils.unmask_selection_mask(mask & mask_global)
        else:
            if name in [None, False]:
                # # in this case we can
                return np.full(i2-i1, True)
            scope = scopes._BlockScopeSelection(self, i1, i2, selection, cache=cache)
            return vaex.utils.unmask_selection_mask(scope.evaluate(name))

        # if _is_string(selection):

    def evaluate(self, expression, i1=None, i2=None, out=None, selection=None, filtered=True, internal=None, parallel=True, chunk_size=None):
        """Evaluate an expression, and return a numpy array with the results for the full column or a part of it.

        Note that this is not how vaex should be used, since it means a copy of the data needs to fit in memory.

        To get partial results, use i1 and i2

        :param str expression: Name/expression to evaluate
        :param int i1: Start row index, default is the start (0)
        :param int i2: End row index, default is the length of the DataFrame
        :param ndarray out: Output array, to which the result may be written (may be used to reuse an array, or write to
            a memory mapped array)
        :param selection: selection to apply
        :return:
        """
        if chunk_size is not None:
            return self.evaluate_iterator(expression, s1=i1, s2=i2, out=out, selection=selection, filtered=filtered, internal=internal, parallel=parallel, chunk_size=chunk_size)
        else:
            return self._evaluate_implementation(expression, i1=i1, i2=i2, out=out, selection=selection, filtered=filtered, internal=internal, parallel=parallel, chunk_size=chunk_size)

    def evaluate_iterator(self, expression, s1=None, s2=None, out=None, selection=None, filtered=True, internal=None, parallel=True, chunk_size=None, prefetch=True):
        """Generator to efficiently evaluate expressions in chunks (number of rows).

        See :func:`DataFrame.evaluate` for other arguments.

        Example:

        >>> import vaex
        >>> df = vaex.example()
        >>> for i1, i2, chunk in df.evaluate_iterator(df.x, chunk_size=100_000):
        ...     print(f"Total of {i1} to {i2} = {chunk.sum()}")
        ...
        Total of 0 to 100000 = -7460.610158279056
        Total of 100000 to 200000 = -4964.85827154921
        Total of 200000 to 300000 = -7303.271340043915
        Total of 300000 to 330000 = -2424.65234724951

        :param prefetch: Prefetch/compute the next chunk in parallel while the current value is yielded/returned.
        """
        offset = 0
        import concurrent.futures
        if not prefetch:
            # this is the simple implementation
            for l1, l2, i1, i2 in self._unfiltered_chunk_slices(chunk_size):
                yield l1, l2, self._evaluate_implementation(expression, i1=i1, i2=i2, out=out, selection=selection, filtered=filtered, internal=internal, parallel=parallel, raw=True)
        # But this implementation is faster if the main thread work is single threaded
        else:
            with concurrent.futures.ThreadPoolExecutor(1) as executor:
                iter = self._unfiltered_chunk_slices(chunk_size)
                def f(i1, i2):
                    return self._evaluate_implementation(expression, i1=i1, i2=i2, out=out, selection=selection, filtered=filtered, internal=internal, parallel=parallel, raw=True)
                previous_l1, previous_l2, previous_i1, previous_i2 = next(iter)
                # we submit the 1st job
                previous = executor.submit(f, previous_i1, previous_i2)
                for l1, l2, i1, i2 in iter:
                    # and we submit the next job before returning the previous, so they run in parallel
                    # but make sure the previous is done
                    previous_chunk = previous.result()
                    current = executor.submit(f, i1, i2)
                    yield previous_l1, previous_l2, previous_chunk
                    previous = current
                    previous_l1, previous_l2 = l1, l2
                previous_chunk = previous.result()
                yield previous_l1, previous_l2, previous_chunk

    @docsubst
    def to_items(self, column_names=None, selection=None, strings=True, virtual=True, parallel=True, chunk_size=None, array_type=None):
        """Return a list of [(column_name, ndarray), ...)] pairs where the ndarray corresponds to the evaluated data

        :param column_names: list of column names, to export, when None DataFrame.get_column_names(strings=strings, virtual=virtual) is used
        :param selection: {selection}
        :param strings: argument passed to DataFrame.get_column_names when column_names is None
        :param virtual: argument passed to DataFrame.get_column_names when column_names is None
        :param parallel: {evaluate_parallel}
        :param chunk_size: {chunk_size}
        :param array_type: {array_type}
        :return: list of (name, ndarray) pairs or iterator of
        """
        column_names = column_names or self.get_column_names(strings=strings, virtual=virtual)
        if chunk_size is not None:
            def iterator():
                for i1, i2, chunks in self.evaluate_iterator(column_names, selection=selection, parallel=parallel, chunk_size=chunk_size):
                    yield i1, i2, list(zip(column_names, [to_array_type(chunk, array_type) for chunk in chunks]))
            return iterator()
        else:
            return list(zip(column_names, [to_array_type(chunk, array_type) for chunk in self.evaluate(column_names, selection=selection, parallel=parallel)]))

    @docsubst
    def to_arrays(self, column_names=None, selection=None, strings=True, virtual=True, parallel=True, chunk_size=None, array_type=None):
        """Return a list of ndarrays

        :param column_names: list of column names, to export, when None DataFrame.get_column_names(strings=strings, virtual=virtual) is used
        :param selection: {selection}
        :param strings: argument passed to DataFrame.get_column_names when column_names is None
        :param virtual: argument passed to DataFrame.get_column_names when column_names is None
        :param parallel: {evaluate_parallel}
        :param chunk_size: {chunk_size}
        :param array_type: {array_type}
        :return: list of arrays
        """
        column_names = column_names or self.get_column_names(strings=strings, virtual=virtual)
        if chunk_size is not None:
            def iterator():
                for i1, i2, chunks in self.evaluate_iterator(column_names, selection=selection, parallel=parallel, chunk_size=chunk_size):
                    yield i1, i2, [to_array_type(chunk, array_type) for chunk in chunks]
            return iterator()
        return [to_array_type(chunk, array_type) for chunk in self.evaluate(column_names, selection=selection, parallel=parallel)]

    @docsubst
    def to_dict(self, column_names=None, selection=None, strings=True, virtual=True, parallel=True, chunk_size=None, array_type=None):
        """Return a dict containing the ndarray corresponding to the evaluated data

        :param column_names: list of column names, to export, when None DataFrame.get_column_names(strings=strings, virtual=virtual) is used
        :param selection: {selection}
        :param strings: argument passed to DataFrame.get_column_names when column_names is None
        :param virtual: argument passed to DataFrame.get_column_names when column_names is None
        :param parallel: {evaluate_parallel}
        :param chunk_size: {chunk_size}
        :param array_type: {array_type}
        :return: dict
        """
        column_names = column_names or self.get_column_names(strings=strings, virtual=virtual)
        if chunk_size is not None:
            def iterator():
                for i1, i2, chunks in self.evaluate_iterator(column_names, selection=selection, parallel=parallel, chunk_size=chunk_size):
                    yield i1, i2, dict(list(zip(column_names, [to_array_type(chunk, array_type) for chunk in chunks])))
            return iterator()
        return dict(list(zip(column_names, [to_array_type(chunk, array_type) for chunk in self.evaluate(column_names, selection=selection, parallel=parallel)])))

    @docsubst
    def to_copy(self, column_names=None, selection=None, strings=True, virtual=True, selections=True):
        """Return a copy of the DataFrame, if selection is None, it does not copy the data, it just has a reference

        :param column_names: list of column names, to copy, when None DataFrame.get_column_names(strings=strings, virtual=virtual) is used
        :param selection: {selection}
        :param strings: argument passed to DataFrame.get_column_names when column_names is None
        :param virtual: argument passed to DataFrame.get_column_names when column_names is None
        :param selections: copy selections to a new DataFrame
        :return: dict
        """
        if column_names:
            column_names = _ensure_strings_from_expressions(column_names)
        df = vaex.from_items(*self.to_items(column_names=column_names, selection=selection, strings=strings, virtual=False))
        if virtual:
            for name, value in self.virtual_columns.items():
                df.add_virtual_column(name, value)
        if selections:
            # the filter selection does not need copying
            for key, value in self.selection_histories.items():
                if key != FILTER_SELECTION_NAME:
                    df.selection_histories[key] = list(value)
            for key, value in self.selection_history_indices.items():
                if key != FILTER_SELECTION_NAME:
                    df.selection_history_indices[key] = value
        df.functions.update(self.functions)
        df.copy_metadata(self)
        return df

    def copy_metadata(self, other):
        for name in self.get_column_names(strings=True):
            if name in other.units:
                self.units[name] = other.units[name]
            if name in other.descriptions:
                self.descriptions[name] = other.descriptions[name]
            if name in other.ucds:
                self.ucds[name] = other.ucds[name]
        self._column_aliases = dict(other._column_aliases)
        self.description = other.description

    @docsubst
    def to_pandas_df(self, column_names=None, selection=None, strings=True, virtual=True, index_name=None, parallel=True, chunk_size=None):
        """Return a pandas DataFrame containing the ndarray corresponding to the evaluated data

         If index is given, that column is used for the index of the dataframe.

         Example

         >>> df_pandas = df.to_pandas_df(["x", "y", "z"])
         >>> df_copy = vaex.from_pandas(df_pandas)

        :param column_names: list of column names, to export, when None DataFrame.get_column_names(strings=strings, virtual=virtual) is used
        :param selection: {selection}
        :param strings: argument passed to DataFrame.get_column_names when column_names is None
        :param virtual: argument passed to DataFrame.get_column_names when column_names is None
        :param index_column: if this column is given it is used for the index of the DataFrame
        :param parallel: {evaluate_parallel}
        :param chunk_size: {chunk_size}
        :return: pandas.DataFrame object or iterator of
        """
        import pandas as pd
        column_names = column_names or self.get_column_names(strings=strings, virtual=virtual)
        if index_name not in column_names and index_name is not None:
            column_names = column_names + [index_name]

        def create_pdf(data):
            if index_name is not None:
                index = data.pop(index_name)
            else:
                index = None
            df = pd.DataFrame(data=data, index=index)
            if index is not None:
                df.index.name = index_name
            return df
        if chunk_size is not None:
            def iterator():
                for i1, i2, chunks in self.evaluate_iterator(column_names, selection=selection, parallel=parallel, chunk_size=chunk_size):
                    yield i1, i2, create_pdf(dict(zip(column_names, chunks)))
            return iterator()
        else:
            return create_pdf(self.to_dict(column_names=column_names, selection=selection, parallel=parallel))

    @docsubst
    def to_arrow_table(self, column_names=None, selection=None, strings=True, virtual=True, parallel=True, chunk_size=None):
        """Returns an arrow Table object containing the arrays corresponding to the evaluated data

        :param column_names: list of column names, to export, when None DataFrame.get_column_names(strings=strings, virtual=virtual) is used
        :param selection: {selection}
        :param strings: argument passed to DataFrame.get_column_names when column_names is None
        :param virtual: argument passed to DataFrame.get_column_names when column_names is None
        :param parallel: {evaluate_parallel}
        :param chunk_size: {chunk_size}
        :return: pyarrow.Table object or iterator of
        """
        from vaex_arrow.convert import arrow_table_from_vaex_df, arrow_array_from_numpy_array
        import pyarrow as pa
        column_names = column_names or self.get_column_names(strings=strings, virtual=virtual)
        if chunk_size is not None:
            def iterator():
                for i1, i2, chunks in self.evaluate_iterator(column_names, selection=selection, parallel=parallel, chunk_size=chunk_size):
                    chunks = list(map(arrow_array_from_numpy_array, chunks))
                    yield i1, i2, pa.Table.from_arrays(chunks, column_names)
            return iterator()
        else:
            chunks = self.evaluate(column_names, selection=selection, parallel=parallel)
            chunks = list(map(arrow_array_from_numpy_array, chunks))
            return pa.Table.from_arrays(chunks, column_names)

    @docsubst
    def to_astropy_table(self, column_names=None, selection=None, strings=True, virtual=True, index=None, parallel=True):
        """Returns a astropy table object containing the ndarrays corresponding to the evaluated data

        :param column_names: list of column names, to export, when None DataFrame.get_column_names(strings=strings, virtual=virtual) is used
        :param selection: {selection}
        :param strings: argument passed to DataFrame.get_column_names when column_names is None
        :param virtual: argument passed to DataFrame.get_column_names when column_names is None
        :param index: if this column is given it is used for the index of the DataFrame
        :return: astropy.table.Table object
        """
        from astropy.table import Table, Column, MaskedColumn
        meta = dict()
        meta["name"] = self.name
        meta["description"] = self.description

        table = Table(meta=meta)
        for name, data in self.to_items(column_names=column_names, selection=selection, strings=strings, virtual=virtual, parallel=parallel):
            if self.data_type(name) == str_type:  # for astropy we convert it to unicode, it seems to ignore object type
                data = np.array(data).astype('U')
            meta = dict()
            if name in self.ucds:
                meta["ucd"] = self.ucds[name]
            if np.ma.isMaskedArray(data):
                cls = MaskedColumn
            else:
                cls = Column
            table[name] = cls(data, unit=self.unit(name), description=self.descriptions.get(name), meta=meta)
        return table

    def to_dask_array(self, chunks="auto"):
        """Lazily expose the DataFrame as a dask.array

        Example

        >>> df = vaex.example()
        >>> A = df[['x', 'y', 'z']].to_dask_array()
        >>> A
        dask.array<vaex-df-1f048b40-10ec-11ea-9553, shape=(330000, 3), dtype=float64, chunksize=(330000, 3), chunktype=numpy.ndarray>
        >>> A+1
        dask.array<add, shape=(330000, 3), dtype=float64, chunksize=(330000, 3), chunktype=numpy.ndarray>

        :param chunks: How to chunk the array, similar to :func:`dask.array.from_array`.
        :return: :class:`dask.array.Array` object.
        """
        import dask.array as da
        import uuid
        dtype = self._dtype
        chunks = da.core.normalize_chunks(chunks, shape=self.shape, dtype=dtype)
        name = 'vaex-df-%s' % str(uuid.uuid1())
        def getitem(df, item):
            return np.array(df.__getitem__(item).to_arrays(parallel=False)).T
        dsk = da.core.getem(name, chunks, getitem=getitem, shape=self.shape, dtype=dtype)
        dsk[name] = self
        return da.Array(dsk, name, chunks, dtype=dtype)

    def validate_expression(self, expression):
        """Validate an expression (may throw Exceptions)"""
        # return self.evaluate(expression, 0, 2)
        if str(expression) in self.virtual_columns:
            return
        if self.is_local() and str(expression) in self.columns:
            return
        vars = set(self.get_names(hidden=True, alias=False))
        funcs = set(expression_namespace.keys())  | set(self.functions.keys())
        try:
            return vaex.expresso.validate_expression(expression, vars, funcs)
        except NameError as e:
            raise NameError(str(e)) from None

    def _block_scope(self, i1, i2):
        variables = {key: self.evaluate_variable(key) for key in self.variables.keys()}
        return scopes._BlockScope(self, i1, i2, **variables)

    def select(self, boolean_expression, mode="replace", name="default"):
        """Select rows based on the boolean_expression, if there was a previous selection, the mode is taken into account.

        if boolean_expression is None, remove the selection, has_selection() will returns false

        Note that per DataFrame, multiple selections are possible, and one filter (see :func:`DataFrame.select`).

        :param str boolean_expression: boolean expression, such as 'x < 0', '(x < 0) || (y > -10)' or None to remove the selection
        :param str mode: boolean operation to perform with the previous selection, "replace", "and", "or", "xor", "subtract"
        :return: None
        """
        raise NotImplementedError

    def add_column(self, name, f_or_array, dtype=None):
        """Add an in memory array as a column."""
        column_position = len(self.column_names)
        if name in self.get_column_names():
            column_position = self.column_names.index(name)
            renamed = '__' +vaex.utils.find_valid_name(name, used=self.get_column_names())
            self._rename(name, renamed)

        if isinstance(f_or_array, (np.ndarray, Column)):
            data = ar = f_or_array
            # it can be None when we have an 'empty' DataFrameArrays
            if self._length_original is None:
                self._length_unfiltered = _len(data)
                self._length_original = _len(data)
                self._index_end = self._length_unfiltered
            if _len(ar) != self.length_original():
                if self.filtered:
                    # give a better warning to avoid confusion
                    if len(self) == len(ar):
                        raise ValueError("Array is of length %s, while the length of the DataFrame is %s due to the filtering, the (unfiltered) length is %s." % (len(ar), len(self), self.length_unfiltered()))
                raise ValueError("array is of length %s, while the length of the DataFrame is %s" % (len(ar), self.length_original()))
            # assert self.length_unfiltered() == len(data), "columns should be of equal length, length should be %d, while it is %d" % ( self.length_unfiltered(), len(data))
            valid_name = vaex.utils.find_valid_name(name, used=self.get_column_names(hidden=True))
            if name != valid_name:
                self._column_aliases[name] = valid_name
            ar = f_or_array
            if dtype is not None:
                self._dtypes_override[valid_name] = dtype
            else:
                if isinstance(ar, np.ndarray) and ar.dtype.kind == 'O':
                    ar_data = ar
                    if np.ma.isMaskedArray(ar):
                        ar_data = ar.data

                    try:
                        # "k != k" is a way to detect NaN's and NaT's
                        types = list({type(k) for k in ar_data if k is not None and k == k})
                    except ValueError:
                        # If there is an array value in the column, Numpy throws a ValueError
                        # "The truth value of an array with more than one element is ambiguous".
                        # We don't handle this by default as it is a bit slower.
                        def is_missing(k):
                            if k is None:
                                return True
                            try:
                                # a way to detect NaN's and NaT
                                return not (k == k)
                            except ValueError:
                                # if a value is an array, this will fail, and it is a non-missing
                                return False
                        types = list({type(k) for k in ar_data if k is not is_missing(k)})

                    if len(types) == 1 and issubclass(types[0], six.string_types):
                        self._dtypes_override[valid_name] = str_type
                    if len(types) == 0:  # can only be if all nan right?
                        ar = ar.astype(np.float64)
            self.columns[valid_name] = ar
            if valid_name not in self.column_names:
                self.column_names.insert(column_position, valid_name)
        else:
            raise ValueError("functions not yet implemented")
        self._save_assign_expression(valid_name, Expression(self, valid_name))

    def _sparse_matrix(self, column):
        column = _ensure_string_from_expression(column)
        return self._sparse_matrices.get(column)

    def add_columns(self, names, columns):
        from scipy.sparse import csc_matrix, csr_matrix
        if isinstance(columns, csr_matrix):
            if len(names) != columns.shape[1]:
                raise ValueError('number of columns ({}) does not match number of column names ({})'.format(columns.shape[1], len(names)))
            for i, name in enumerate(names):
                valid_name = vaex.utils.find_valid_name(name, used=self.get_column_names(hidden=True))
                if name != valid_name:
                    self._column_aliases[name] = valid_name
                self.columns[valid_name] = ColumnSparse(columns, i)
                self.column_names.append(valid_name)
                self._sparse_matrices[valid_name] = columns
                self._save_assign_expression(valid_name, Expression(self, valid_name))
        else:
            raise ValueError('only scipy.sparse.csr_matrix is supported')

    def _save_assign_expression(self, name, expression=None):
        obj = getattr(self, name, None)
        # it's ok to set it if it does not exist, or we overwrite an older expression
        if obj is None or isinstance(obj, Expression):
            if expression is None:
                expression = Expression(self, name)
            if isinstance(expression, six.string_types):
                expression = Expression(self, expression)
            setattr(self, name, expression)

    @_hidden
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


    @_hidden
    def add_virtual_columns_matrix3d(self, x, y, z, xnew, ynew, znew, matrix, matrix_name='deprecated', matrix_is_expression=False, translation=[0, 0, 0], propagate_uncertainties=False):
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
        m = matrix
        x, y, z = self._expr(x, y, z)

        self[xnew] = m[0][0] * x + m[0][1] * y + m[0][2] * z + translation[0]
        self[ynew] = m[1][0] * x + m[1][1] * y + m[1][2] * z + translation[1]
        self[znew] = m[2][0] * x + m[2][1] * y + m[2][2] * z + translation[2]

        if propagate_uncertainties:
            self.propagate_uncertainties([self[xnew], self[ynew], self[znew]], [x, y, z])

    # wrap these with an informative msg
    # add_virtual_columns_eq2ecl = _requires('astro')
    # add_virtual_columns_eq2gal = _requires('astro')
    # add_virtual_columns_distance_from_parallax = _requires('astro')
    # add_virtual_columns_cartesian_velocities_to_pmvr = _requires('astro')
    # add_virtual_columns_proper_motion_eq2gal = _requires('astro')
    # add_virtual_columns_lbrvr_proper_motion2vcartesian = _requires('astro')
    # add_virtual_columns_equatorial_to_galactic_cartesian = _requires('astro')
    # add_virtual_columns_celestial = _requires('astro')
    # add_virtual_columns_proper_motion2vperpendicular = _requires('astro')

    def _covariance_matrix_guess(self, columns, full=False, as_expression=False):
        all_column_names = self.get_column_names()
        columns = _ensure_strings_from_expressions(columns)

        def _guess(x, y):
            if x == y:
                postfixes = ["_error", "_uncertainty", "e", "_e"]
                prefixes = ["e", "e_"]
                for postfix in postfixes:
                    if x + postfix in all_column_names:
                        return x + postfix
                for prefix in prefixes:
                    if prefix + x in all_column_names:
                        return prefix + x
                if full:
                    raise ValueError("No uncertainty found for %r" % x)
            else:

                postfixes = ["_cov", "_covariance"]
                for postfix in postfixes:
                    if x + "_" + y + postfix in all_column_names:
                        return x + "_" + y + postfix
                    if y + "_" + x + postfix in all_column_names:
                        return y + "_" + x + postfix
                postfixes = ["_correlation", "_corr"]
                for postfix in postfixes:
                    if x + "_" + y + postfix in all_column_names:
                        return x + "_" + y + postfix + " * " + _guess(x, x) + " * " + _guess(y, y)
                    if y + "_" + x + postfix in all_column_names:
                        return y + "_" + x + postfix + " * " + _guess(y, y) + " * " + _guess(x, x)
                if full:
                    raise ValueError("No covariance or correlation found for %r and %r" % (x, y))
            return "0"
        N = len(columns)
        cov_matrix = [[""] * N for i in range(N)]
        for i in range(N):
            for j in range(N):
                cov = _guess(columns[i], columns[j])
                if i == j and cov:
                    cov += "**2"  # square the diagnal
                cov_matrix[i][j] = cov
        if as_expression:
            return [[self[k] for k in row] for row in cov_matrix]
        else:
            return cov_matrix

    def _jacobian(self, expressions, variables):
        expressions = _ensure_strings_from_expressions(expressions)
        return [[self[expression].expand(stop=[var]).derivative(var) for var in variables] for expression in expressions]

    def propagate_uncertainties(self, columns, depending_variables=None, cov_matrix='auto',
                                covariance_format="{}_{}_covariance",
                                uncertainty_format="{}_uncertainty"):
        """Propagates uncertainties (full covariance matrix) for a set of virtual columns.

        Covariance matrix of the depending variables is guessed by finding columns prefixed by "e"
        or `"e_"` or postfixed by "_error", "_uncertainty", "e" and `"_e"`.
        Off diagonals (covariance or correlation) by postfixes with "_correlation" or "_corr" for
        correlation or "_covariance" or "_cov" for covariances.
        (Note that x_y_cov = x_e * y_e * x_y_correlation.)


        Example

        >>> df = vaex.from_scalars(x=1, y=2, e_x=0.1, e_y=0.2)
        >>> df["u"] = df.x + df.y
        >>> df["v"] = np.log10(df.x)
        >>> df.propagate_uncertainties([df.u, df.v])
        >>> df.u_uncertainty, df.v_uncertainty

        :param columns: list of columns for which to calculate the covariance matrix.
        :param depending_variables: If not given, it is found out automatically, otherwise a list of columns which have uncertainties.
        :param cov_matrix: List of list with expressions giving the covariance matrix, in the same order as depending_variables. If 'full' or 'auto',
            the covariance matrix for the depending_variables will be guessed, where 'full' gives an error if an entry was not found.
        """

        names = _ensure_strings_from_expressions(columns)
        virtual_columns = self._expr(*columns, always_list=True)

        if depending_variables is None:
            depending_variables = set()
            for expression in virtual_columns:
                depending_variables |= expression.expand().variables()
            depending_variables = list(sorted(list(depending_variables)))

        fs = [self[self.virtual_columns[name]] for name in names]
        jacobian = self._jacobian(fs, depending_variables)
        m = len(fs)
        n = len(depending_variables)

        # n x n matrix
        cov_matrix = self._covariance_matrix_guess(depending_variables, full=cov_matrix == "full", as_expression=True)

        # empty m x m matrix
        cov_matrix_out = [[self['0'] for __ in range(m)] for __ in range(m)]
        for i in range(m):
            for j in range(m):
                for k in range(n):
                    for l in range(n):
                        if jacobian[i][k].expression == '0' or jacobian[j][l].expression == '0' or cov_matrix[k][l].expression == '0':
                            pass
                        else:
                            cov_matrix_out[i][j] = cov_matrix_out[i][j] + jacobian[i][k] * cov_matrix[k][l] * jacobian[j][l]
        for i in range(m):
            for j in range(i + 1):
                sigma = cov_matrix_out[i][j]
                sigma = self._expr(vaex.expresso.simplify(_ensure_string_from_expression(sigma)))
                if i != j:
                    self.add_virtual_column(covariance_format.format(names[i], names[j]), sigma)
                else:
                    self.add_virtual_column(uncertainty_format.format(names[i]), np.sqrt(sigma))

    @_hidden
    def add_virtual_columns_cartesian_to_polar(self, x="x", y="y", radius_out="r_polar", azimuth_out="phi_polar",
                                               propagate_uncertainties=False,
                                               radians=False):
        kwargs = dict(**locals())
        del kwargs['self']
        return self.geo.cartesian_to_polar(inplace=True, **kwargs)

    @_hidden
    def add_virtual_columns_cartesian_velocities_to_spherical(self, x="x", y="y", z="z", vx="vx", vy="vy", vz="vz", vr="vr", vlong="vlong", vlat="vlat", distance=None):
        kwargs = dict(**locals())
        del kwargs['self']
        return self.geo.velocity_cartesian2spherical(inplace=True, **kwargs)

    def _expr(self, *expressions, **kwargs):
        always_list = kwargs.pop('always_list', False)
        return Expression(self, expressions[0]) if len(expressions) == 1 and not always_list else [Expression(self, k) for k in expressions]

    @_hidden
    def add_virtual_columns_cartesian_velocities_to_polar(self, x="x", y="y", vx="vx", radius_polar=None, vy="vy", vr_out="vr_polar", vazimuth_out="vphi_polar",
                                                          propagate_uncertainties=False,):
        kwargs = dict(**locals())
        del kwargs['self']
        return self.geo.velocity_cartesian2polar(inplace=True, **kwargs)

    @_hidden
    def add_virtual_columns_polar_velocities_to_cartesian(self, x='x', y='y', azimuth=None, vr='vr_polar', vazimuth='vphi_polar', vx_out='vx', vy_out='vy', propagate_uncertainties=False):
        kwargs = dict(**locals())
        del kwargs['self']
        return self.geo.velocity_polar2cartesian(inplace=True, **kwargs)

    @_hidden
    def add_virtual_columns_rotation(self, x, y, xnew, ynew, angle_degrees, propagate_uncertainties=False):
        kwargs = dict(**locals())
        del kwargs['self']
        return self.geo.rotation_2d(inplace=True, **kwargs)

    @docsubst
    @_hidden
    def add_virtual_columns_spherical_to_cartesian(self, alpha, delta, distance, xname="x", yname="y", zname="z",
                                                   propagate_uncertainties=False,
                                                   center=[0, 0, 0], radians=False):
        kwargs = dict(**locals())
        del kwargs['self']
        return self.geo.spherical2cartesian(inplace=True, **kwargs)

    @_hidden
    def add_virtual_columns_cartesian_to_spherical(self, x="x", y="y", z="z", alpha="l", delta="b", distance="distance", radians=False, center=None, center_name="solar_position"):
        kwargs = dict(**locals())
        del kwargs['self']
        return self.geo.cartesian2spherical(inplace=True, **kwargs)

    @_hidden
    def add_virtual_columns_aitoff(self, alpha, delta, x, y, radians=True):
        kwargs = dict(**locals())
        del kwargs['self']
        return self.geo.project_aitoff(inplace=True, **kwargs)

    @_hidden
    def add_virtual_columns_projection_gnomic(self, alpha, delta, alpha0=0, delta0=0, x="x", y="y", radians=False, postfix=""):
        kwargs = dict(**locals())
        del kwargs['self']
        return self.geo.project_gnomic(inplace=True, **kwargs)

    def add_function(self, name, f, unique=False):
        name = vaex.utils.find_valid_name(name, used=[] if not unique else self.functions.keys())
        function = vaex.expression.Function(self, name, f)
        self.functions[name] = function
        return function

    def add_virtual_column(self, name, expression, unique=False):
        """Add a virtual column to the DataFrame.

        Example:

        >>> df.add_virtual_column("r", "sqrt(x**2 + y**2 + z**2)")
        >>> df.select("r < 10")

        :param: str name: name of virtual column
        :param: expression: expression for the column
        :param str unique: if name is already used, make it unique by adding a postfix, e.g. _1, or _2
        """
        type = "change" if name in self.virtual_columns else "add"
        if isinstance(expression, Expression):
            if expression.df is not self:
                expression = expression.copy(self)
        column_position = len(self.column_names)
        # if the current name is an existing column name....
        real_name = self._column_aliases.get(name, name)
        if name in self.get_column_names(hidden=True):
            column_position = self.column_names.index(real_name)
            renamed = vaex.utils.find_valid_name('__' +name, used=self.get_column_names(hidden=True))
            # we rewrite all existing expressions (including the passed down expression argument)
            self._rename(real_name, renamed)
        expression = _ensure_string_from_expression(expression)

        if vaex.utils.find_valid_name(name) != name:
            # if we have to rewrite the name, we need to make it unique
            unique = True
        valid_name = vaex.utils.find_valid_name(name, used=[] if not unique else self.get_column_names(hidden=True))
        if name != valid_name:
            self._column_aliases[name] = valid_name

        self.virtual_columns[valid_name] = expression
        self._virtual_expressions[valid_name] = Expression(self, expression)
        if name not in self.column_names:
            self.column_names.insert(column_position, valid_name)
        self._save_assign_expression(valid_name)
        self.signal_column_changed.emit(self, valid_name, "add")
        # self.write_virtual_meta()

    def rename(self, name, new_name, unique=False):
        """Renames a column or variable, and rewrite expressions such that they refer to the new name"""
        if name == new_name:
            return
        if name in self._column_aliases:
            # nobody should refer to an alias, so we can just rename it
            self._column_aliases[new_name] = self._column_aliases.pop(name)
            return
        new_name = vaex.utils.find_valid_name(new_name, used=[] if not unique else self.get_column_names(hidden=True))
        self._rename(name, new_name, rename_meta_data=True)
        return new_name

    def _rename(self, old, new, rename_meta_data=False):
        if old in self._dtypes_override:
            self._dtypes_override[new] = self._dtypes_override.pop(old)
        is_variable = False
        if old in self.variables:
            self.variables[new] = self.variables.pop(old)
            is_variable = True
        elif old in self.virtual_columns:
            self.virtual_columns[new] = self.virtual_columns.pop(old)
            self._virtual_expressions[new] = self._virtual_expressions.pop(old)
        elif self.is_local() and old in self.columns:
            # we only have to do this locally
            # if we don't do this locally, we still store this info
            # in self._renamed_columns, so it will happen at the server
            self.columns[new] = self.columns.pop(old)
        if rename_meta_data:
            for d in [self.ucds, self.units, self.descriptions]:
                if old in d:
                    d[new] = d[old]
                    del d[old]
        for key, value in self.selection_histories.items():
            self.selection_histories[key] = list([k if k is None else k._rename(self, old, new) for k in value])
        if not is_variable:
            if new not in self.virtual_columns:
                self._renamed_columns.append((old, new))
            self.column_names[self.column_names.index(old)] = new
            if hasattr(self, old):
                try:
                    if isinstance(getattr(self, old), Expression):
                        delattr(self, old)
                except:
                    pass
            self._save_assign_expression(new)
        existing_expressions = [k() for k in self._expressions]
        existing_expressions = [k for k in existing_expressions if k is not None]
        for expression in existing_expressions:
            expression._rename(old, new, inplace=True)
        self.virtual_columns = {k:self._virtual_expressions[k].expression for k, v in self.virtual_columns.items()}


    def delete_virtual_column(self, name):
        """Deletes a virtual column from a DataFrame."""
        del self.virtual_columns[name]
        del self._virtual_expressions[name]
        self.signal_column_changed.emit(self, name, "delete")
        # self.write_virtual_meta()

    def add_variable(self, name, expression, overwrite=True, unique=True):
        """Add a variable to a DataFrame.

        A variable may refer to other variables, and virtual columns and expression may refer to variables.

        Example

        >>> df.add_variable('center', 0)
        >>> df.add_virtual_column('x_prime', 'x-center')
        >>> df.select('x_prime < 0')

        :param: str name: name of virtual varible
        :param: expression: expression for the variable
        """
        if unique or overwrite or name not in self.variables:
            existing_names = self.get_column_names(virtual=False) + list(self.variables.keys())
            name = vaex.utils.find_valid_name(name, used=[] if not unique else existing_names)
            self.variables[name] = expression
            self.signal_variable_changed.emit(self, name, "add")
            if unique:
                return name

    def delete_variable(self, name):
        """Deletes a variable from a DataFrame."""
        del self.variables[name]
        self.signal_variable_changed.emit(self, name, "delete")
        # self.write_virtual_meta()

    def info(self, description=True):
        from IPython import display
        self._output_css()
        display.display(display.HTML(self._info(description=description)))

    def _info(self, description=True):
        parts = ["""<div><h2>{}</h2> <b>rows</b>: {:,}</div>""".format(self.name, len(self))]
        if hasattr(self, 'path'):
            parts += ["""<div><b>path</b>: <i>%s</i></div>""" % (self.path)]
        if self.description:
            parts += ["""<div><b>Description</b>: {}</div>""".format(self.description)]
        parts += ["<h2>Columns:</h2>"]
        parts += ["<table class='table-striped'>"]
        parts += ["<thead><tr>"]
        for header in "column type unit description expression".split():
            if description or header != "description":
                parts += ["<th>%s</th>" % header]
        parts += ["</tr></thead>"]
        for name in self.get_column_names():
            column_name = self._column_aliases.get(name, name)
            parts += ["<tr>"]
            parts += ["<td>%s</td>" % name]
            virtual = column_name in self.virtual_columns
            if not virtual:
                dtype = str(self.data_type(name)) if self.data_type(name) != str else 'str'
            else:
                dtype = "</i>virtual column</i>"
            parts += ["<td>%s</td>" % dtype]
            units = self.unit(name)
            units = units.to_string("latex_inline") if units else ""
            parts += ["<td>%s</td>" % units]
            if description:
                parts += ["<td ><pre>%s</pre></td>" % self.descriptions.get(name, "")]
            if virtual:
                parts += ["<td><code>%s</code></td>" % self.virtual_columns[column_name]]
            else:
                parts += ["<td></td>"]
            parts += ["</tr>"]
        parts += "</table>"

        ignore_list = 'pi e km_in_au seconds_per_year'.split()
        variable_names = [name for name in self.variables.keys() if name not in ignore_list]
        if variable_names:
            parts += ["<h2>Variables:</h2>"]
            parts += ["<table class='table-striped'>"]
            parts += ["<thead><tr>"]
            for header in "variable type unit description expression".split():
                if description or header != "description":
                    parts += ["<th>%s</th>" % header]
            parts += ["</tr></thead>"]
            for name in variable_names:
                parts += ["<tr>"]
                parts += ["<td>%s</td>" % name]
                type = self.data_type(name).name
                parts += ["<td>%s</td>" % type]
                units = self.unit(name)
                units = units.to_string("latex_inline") if units else ""
                parts += ["<td>%s</td>" % units]
                if description:
                    parts += ["<td ><pre>%s</pre></td>" % self.descriptions.get(name, "")]
                parts += ["<td><code>%s</code></td>" % (self.variables[name], )]
                parts += ["</tr>"]
            parts += "</table>"

        return "".join(parts) + "<h2>Data:</h2>" + self._head_and_tail_table()

    def head(self, n=10):
        """Return a shallow copy a DataFrame with the first n rows."""
        return self[:min(n, len(self))]

    def tail(self, n=10):
        """Return a shallow copy a DataFrame with the last n rows."""
        N = len(self)
        # self.cat(i1=max(0, N-n), i2=min(len(self), N))
        return self[max(0, N - n):min(len(self), N)]

    def _head_and_tail_table(self, n=5, format='html'):
        N = _len(self)
        if N <= n * 2:
            return self._as_table(0, N, format=format)
        else:
            return self._as_table(0, n, N - n, N, format=format)

    def head_and_tail_print(self, n=5):
        """Display the first and last n elements of a DataFrame."""
        from IPython import display
        display.display(display.HTML(self._head_and_tail_table(n)))

    def describe(self, strings=True, virtual=True, selection=None):
        """Give a description of the DataFrame.

        >>> import vaex
        >>> df = vaex.example()[['x', 'y', 'z']]
        >>> df.describe()
                         x          y          z
        dtype      float64    float64    float64
        count       330000     330000     330000
        missing          0          0          0
        mean    -0.0671315 -0.0535899  0.0169582
        std        7.31746    7.78605    5.05521
        min       -128.294   -71.5524   -44.3342
        max        271.366    146.466    50.7185
        >>> df.describe(selection=df.x > 0)
                           x         y          z
        dtype        float64   float64    float64
        count         164060    164060     164060
        missing       165940    165940     165940
        mean         5.13572 -0.486786 -0.0868073
        std          5.18701   7.61621    5.02831
        min      1.51635e-05  -71.5524   -44.3342
        max          271.366   78.0724    40.2191

        :param bool strings: Describe string columns or not
        :param bool virtual: Describe virtual columns or not
        :param selection: Optional selection to use.
        :return: Pandas dataframe

        """
        import pandas as pd
        N = len(self)
        columns = {}
        for feature in self.get_column_names(strings=strings, virtual=virtual)[:]:
            dtype = str(self.data_type(feature)) if self.data_type(feature) != str else 'str'
            if self.data_type(feature) == str_type or self.data_type(feature).kind in ['S', 'U']:
                count = self.count(feature, selection=selection, delay=True)
                self.execute()
                count = count.get()
                columns[feature] = ((dtype, count, N-count, '--', '--', '--', '--'))
            elif self.data_type(feature).kind == 'O':
                # this will also properly count NaN-like objects like NaT
                count_na = self[feature].isna().astype('int').sum(delay=True)
                self.execute()
                count_na = count_na.get()
                columns[feature] = ((dtype, N-count_na, count_na, '--', '--', '--', '--'))
            else:
                is_datetime = self.is_datetime(feature)
                mean = self.mean(feature, selection=selection, delay=True)
                std = self.std(feature, selection=selection, delay=True)
                minmax = self.minmax(feature, selection=selection, delay=True)
                if is_datetime:  # this path tests using isna, which test for nat
                    count_na = self[feature].isna().astype('int').sum(delay=True)
                else:
                    count = self.count(feature, selection=selection, delay=True)
                self.execute()
                if is_datetime:
                    count_na, mean, std, minmax = count_na.get(), mean.get(), std.get(), minmax.get()
                    count = N - int(count_na)
                else:
                    count, mean, std, minmax = count.get(), mean.get(), std.get(), minmax.get()
                    count = int(count)
                columns[feature] = ((dtype, count, N-count, mean, std, minmax[0], minmax[1]))
        return pd.DataFrame(data=columns, index=['dtype', 'count', 'NA', 'mean', 'std', 'min', 'max'])

    def cat(self, i1, i2, format='html'):
        """Display the DataFrame from row i1 till i2

        For format, see https://pypi.org/project/tabulate/

        :param int i1: Start row
        :param int i2: End row.
        :param str format: Format to use, e.g. 'html', 'plain', 'latex'
        """
        from IPython import display
        if format == 'html':
            output = self._as_html_table(i1, i2)
            display.display(display.HTML(output))
        else:
            output = self._as_table(i1, i2, format=format)
            print(output)

    def _as_table(self, i1, i2, j1=None, j2=None, format='html'):
        from .formatting import _format_value
        parts = []  # """<div>%s (length=%d)</div>""" % (self.name, len(self))]
        parts += ["<table class='table-striped'>"]

        aliases_reverse = {value: key for key, value in self._column_aliases.items()}
        # we need to get the underlying names since we use df.evaluate
        column_names = self.get_column_names(alias=False)
        values_list = []
        values_list.append(['#', []])
        # parts += ["<thead><tr>"]
        for name in column_names:
            values_list.append([aliases_reverse.get(name, name), []])
            # parts += ["<th>%s</th>" % name]
        # parts += ["</tr></thead>"]

        def table_part(k1, k2, parts):
            N = k2 - k1
            # slicing will invoke .extract which will make the evaluation
            # much quicker
            df = self[k1:k2]
            try:
                values = dict(zip(column_names, df.evaluate(column_names)))
            except:
                values = {}
                for i, name in enumerate(column_names):
                    try:
                        values[name] = df.evaluate(name)
                    except:
                        values[name] = ["error"] * (N)
                        logger.exception('error evaluating: %s at rows %i-%i' % (name, k1, k2))
            for i in range(k2 - k1):
                # parts += ["<tr>"]
                # parts += ["<td><i style='opacity: 0.6'>{:,}</i></td>".format(i + k1)]
                if format == 'html':
                    value = "<i style='opacity: 0.6'>{:,}</i>".format(i + k1)
                else:
                    value = "{:,}".format(i + k1)
                values_list[0][1].append(value)
                for j, name in enumerate(column_names):
                    value = values[name][i]
                    value = _format_value(value)
                    values_list[j+1][1].append(value)
                # parts += ["</tr>"]
            # return values_list
        if i2 - i1 > 0:
            parts = table_part(i1, i2, parts)
            if j1 is not None and j2 is not None:
                values_list[0][1].append('...')
                for i in range(len(column_names)):
                    # parts += ["<td>...</td>"]
                    values_list[i+1][1].append('...')

                # parts = table_part(j1, j2, parts)
                table_part(j1, j2, parts)
        else:
            for header, values in values_list:
                values.append(None)
        # parts += "</table>"
        # html = "".join(parts)
        # return html
        values_list = dict(values_list)
        # print(values_list)
        import tabulate
        table_text = str(tabulate.tabulate(values_list, headers="keys", tablefmt=format))
        if tabulate.__version__ == '0.8.7':
            # Tabulate 0.8.7 escapes html :()
            table_text = table_text.replace('&lt;i style=&#x27;opacity: 0.6&#x27;&gt;', "<i style='opacity: 0.6'>")
            table_text = table_text.replace('&lt;/i&gt;', "</i>")
        if i2 - i1 == 0:
            if self._length_unfiltered != len(self):
                footer_text = 'No rows to display (because of filtering).'
            else:
                footer_text = 'No rows to display.'
            if format == 'html':
                table_text += f'<i>{footer_text}</i>'
            if format == 'plain':
                table_text += f'\n{footer_text}'
        return table_text

    def _as_html_table(self, i1, i2, j1=None, j2=None):
        # TODO: this method can be replaced by _as_table
        from .formatting import _format_value
        parts = []  # """<div>%s (length=%d)</div>""" % (self.name, len(self))]
        parts += ["<table class='table-striped'>"]

        column_names = self.get_column_names()
        parts += ["<thead><tr>"]
        for name in ["#"] + column_names:
            parts += ["<th>%s</th>" % name]
        parts += ["</tr></thead>"]

        def table_part(k1, k2, parts):
            data_parts = {}
            N = k2 - k1
            for name in column_names:
                try:
                    data_parts[name] = self.evaluate(name, i1=k1, i2=k2)
                except:
                    data_parts[name] = ["error"] * (N)
                    logger.exception('error evaluating: %s at rows %i-%i' % (name, k1, k2))
            for i in range(k2 - k1):
                parts += ["<tr>"]
                parts += ["<td><i style='opacity: 0.6'>{:,}</i></td>".format(i + k1)]
                for name in column_names:
                    value = data_parts[name][i]
                    value = _format_value(value)
                    parts += ["<td>%r</td>" % value]
                parts += ["</tr>"]
            return parts
        parts = table_part(i1, i2, parts)
        if j1 is not None and j2 is not None:
            for i in range(len(column_names) + 1):
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

    def _repr_mimebundle_(self, include=None, exclude=None, **kwargs):
        # TODO: optimize, since we use the same data in both versions
        # TODO: include latex version
        return {'text/html':self._head_and_tail_table(format='html'), 'text/plain': self._head_and_tail_table(format='plain')}

    def _repr_html_(self):
        """Representation for Jupyter."""
        self._output_css()
        return self._head_and_tail_table()

    def __str__(self):
        return self._head_and_tail_table(format='plain')

    if not _DEBUG:
        def __repr__(self):
            return self._head_and_tail_table(format='plain')

    def __current_sequence_index(self):
        """TODO"""
        return 0

    def has_current_row(self):
        """Returns True/False if there currently is a picked row."""
        return self._current_row is not None

    def get_current_row(self):
        """Individual rows can be 'picked', this is the index (integer) of the current row, or None there is nothing picked."""
        return self._current_row

    def set_current_row(self, value):
        """Set the current row, and emit the signal signal_pick."""
        if (value is not None) and ((value < 0) or (value >= len(self))):
            raise IndexError("index %d out of range [0,%d]" % (value, len(self)))
        self._current_row = value
        self.signal_pick.emit(self, value)

    def __has_snapshots(self):
        # currenly disabled
        return False

    def column_count(self, hidden=False):
        """Returns the number of columns (including virtual columns).

        :param bool hidden: If True, include hidden columns in the tally
        :returns: Number of columns in the DataFrame
        """
        return len(self.get_column_names(hidden=hidden))

    def get_names(self, hidden=False, alias=True):
        """Return a list of column names and variable names."""
        names = self.get_column_names(hidden=hidden, alias=alias)
        return names + [k for k in self.variables.keys() if not hidden or not k.startswith('__')]

    def get_column_names(self, virtual=True, strings=True, hidden=False, regex=None, alias=True):
        """Return a list of column names

        Example:

        >>> import vaex
        >>> df = vaex.from_scalars(x=1, x2=2, y=3, s='string')
        >>> df['r'] = (df.x**2 + df.y**2)**2
        >>> df.get_column_names()
        ['x', 'x2', 'y', 's', 'r']
        >>> df.get_column_names(virtual=False)
        ['x', 'x2', 'y', 's']
        >>> df.get_column_names(regex='x.*')
        ['x', 'x2']

        :param virtual: If False, skip virtual columns
        :param hidden: If False, skip hidden columns
        :param strings: If False, skip string columns
        :param regex: Only return column names matching the (optional) regular expression
        :param alias: Return the alias (True) or internal name (False).
        :rtype: list of str
        """
        aliases_reverse = {value: key for key, value in self._column_aliases.items()} if alias else {}
        def column_filter(name):
            '''Return True if column with specified name should be returned'''
            if regex and not re.match(regex, name):
                return False
            if not virtual and name in self.virtual_columns:
                return False
            if not strings and (self.data_type(name) == str_type or self.data_type(name).type == np.string_):
                return False
            if not hidden and name.startswith('__'):
                return False
            return True
        if hidden and virtual and regex is None and not alias:
            return list(self.column_names)  # quick path
        if not hidden and virtual and regex is None and not alias:
            return [k for k in self.column_names if not k.startswith('__')]  # also a quick path
        return [aliases_reverse.get(name, name) for name in self.column_names if column_filter(name)]

    def __bool__(self):
        return True  # we are always true :) otherwise Python might call __len__, which can be expensive

    def __len__(self):
        """Returns the number of rows in the DataFrame (filtering applied)."""
        if not self.filtered:
            return self._length_unfiltered
        else:
            if self._cached_filtered_length is None:
               self. _cached_filtered_length = int(self.count())
            return self._cached_filtered_length

    def selected_length(self):
        """Returns the number of rows that are selected."""
        raise NotImplementedError

    def length_original(self):
        """the full length of the DataFrame, independent what active_fraction is, or filtering. This is the real length of the underlying ndarrays."""
        return self._length_original

    def length_unfiltered(self):
        """The length of the arrays that should be considered (respecting active range), but without filtering."""
        return self._length_unfiltered

    def active_length(self):
        return self._length_unfiltered

    def get_active_fraction(self):
        """Value in the range (0, 1], to work only with a subset of rows.
        """
        return self._active_fraction

    def set_active_fraction(self, value):
        """Sets the active_fraction, set picked row to None, and remove selection.

        TODO: we may be able to keep the selection, if we keep the expression, and also the picked row
        """
        if value != self._active_fraction:
            self._active_fraction = value
            # self._fraction_length = int(self._length * self._active_fraction)
            self.select(None)
            self.set_current_row(None)
            self._length_unfiltered = int(round(self._length_original * self._active_fraction))
            self._cached_filtered_length = None
            self._index_start = 0
            self._index_end = self._length_unfiltered
            self.signal_active_fraction_changed.emit(self, value)

    def get_active_range(self):
        return self._index_start, self._index_end

    def set_active_range(self, i1, i2):
        """Sets the active_fraction, set picked row to None, and remove selection.

        TODO: we may be able to keep the selection, if we keep the expression, and also the picked row
        """
        logger.debug("set active range to: %r", (i1, i2))
        self._active_fraction = (i2 - i1) / float(self.length_original())
        # self._fraction_length = int(self._length * self._active_fraction)
        self._index_start = i1
        self._index_end = i2
        self.select(None)
        self.set_current_row(None)
        self._length_unfiltered = i2 - i1
        self._cached_filtered_length = None
        self.signal_active_fraction_changed.emit(self, self._active_fraction)

    @docsubst
    def trim(self, inplace=False):
        '''Return a DataFrame, where all columns are 'trimmed' by the active range.

        For the returned DataFrame, df.get_active_range() returns (0, df.length_original()).

        {note_copy}

        :param inplace: {inplace}
        :rtype: DataFrame
        '''
        df = self if inplace else self.copy()
        if self._index_start == 0 and self._index_end == self._length_original:
            return df
        for name in df.columns.keys():
            column = df.columns.get(name)
            if column is not None:
                if self._index_start == 0 and len(column) == self._index_end:
                    pass  # we already assigned it in .copy
                else:
                    if isinstance(column, np.ndarray):  # real array
                        df.columns[name] = column[self._index_start:self._index_end]
                    else:
                        df.columns[name] = column.trim(self._index_start, self._index_end)

        df._length_original = self.length_unfiltered()
        df._length_unfiltered = df._length_original
        df._cached_filtered_length = None
        df._index_start = 0
        df._index_end = df._length_original
        df._active_fraction = 1
        # trim should be cheap, we don't invalidate the cache unless it is
        # really trimmed
        if self._index_start != 0 or self._index_end != self._length_original:
            df._invalidate_selection_cache()
        return df

    @docsubst
    def take(self, indices, filtered=True, dropfilter=True):
        '''Returns a DataFrame containing only rows indexed by indices

        {note_copy}

        Example:

        >>> import vaex, numpy as np
        >>> df = vaex.from_arrays(s=np.array(['a', 'b', 'c', 'd']), x=np.arange(1,5))
        >>> df.take([0,2])
         #  s      x
         0  a      1
         1  c      3

        :param indices: sequence (list or numpy array) with row numbers
        :param filtered: (for internal use) The indices refer to the filtered data.
        :param dropfilter: (for internal use) Drop the filter, set to False when
            indices refer to unfiltered, but may contain rows that still need to be filtered out.
        :return: DataFrame which is a shallow copy of the original data.
        :rtype: DataFrame
        '''
        df_trimmed = self.trim()
        df = df_trimmed.copy()
        # if the columns in ds already have a ColumnIndex
        # we could do, direct_indices = df.column['bla'].indices[indices]
        # which should be shared among multiple ColumnIndex'es, so we store
        # them in this dict
        direct_indices_map = {}
        indices = np.asarray(indices)
        if df.filtered and filtered:
            # we translate the indices that refer to filters row indices to
            # indices of the unfiltered row indices
            df.count() # make sure the mask is filled
            max_index = indices.max()
            mask = df._selection_masks[FILTER_SELECTION_NAME]
            filtered_indices = mask.first(max_index+1)
            indices = filtered_indices[indices]
        for name, column in df.columns.items():
            if column is not None:
                # we optimize this somewhere, so we don't do multiple
                # levels of indirection
                df.columns[name] = ColumnIndexed.index(df_trimmed, column, name, indices, direct_indices_map)
        df._length_original = len(indices)
        df._length_unfiltered = df._length_original
        df._cached_filtered_length = None
        df._index_start = 0
        df._index_end = df._length_original
        if dropfilter:
            # if the indices refer to the filtered rows, we can discard the
            # filter in the final dataframe
            df.set_selection(None, name=FILTER_SELECTION_NAME)
        # if we will not drop the filter, we will have to invalidate the cache
        # since it refers to the previous dataframe rows
        # TODO perf: we could instead of dropping the cache, take out the rows we need
        df._invalidate_selection_cache()
        return df

    @docsubst
    def extract(self):
        '''Return a DataFrame containing only the filtered rows.

        {note_copy}

        The resulting DataFrame may be more efficient to work with when the original DataFrame is
        heavily filtered (contains just a small number of rows).

        If no filtering is applied, it returns a trimmed view.
        For the returned df, len(df) == df.length_original() == df.length_unfiltered()

        :rtype: DataFrame
        '''
        trimmed = self.trim()
        if trimmed.filtered:
            self.count()  # make sure the mask is filled
            mask = self._selection_masks[FILTER_SELECTION_NAME]
            indices = mask.first(len(self))
            assert len(indices) == len(self)
            return self.take(indices, filtered=False)
        else:
            return trimmed

    @docsubst
    def sample(self, n=None, frac=None, replace=False, weights=None, random_state=None):
        '''Returns a DataFrame with a random set of rows

        {note_copy}

        Provide either n or frac.

        Example:

        >>> import vaex, numpy as np
        >>> df = vaex.from_arrays(s=np.array(['a', 'b', 'c', 'd']), x=np.arange(1,5))
        >>> df
          #  s      x
          0  a      1
          1  b      2
          2  c      3
          3  d      4
        >>> df.sample(n=2, random_state=42) # 2 random rows, fixed seed
          #  s      x
          0  b      2
          1  d      4
        >>> df.sample(frac=1, random_state=42) # 'shuffling'
          #  s      x
          0  c      3
          1  a      1
          2  d      4
          3  b      2
        >>> df.sample(frac=1, replace=True, random_state=42) # useful for bootstrap (may contain repeated samples)
          #  s      x
          0  d      4
          1  a      1
          2  a      1
          3  d      4

        :param int n: number of samples to take (default 1 if frac is None)
        :param float frac: fractional number of takes to take
        :param bool replace: If true, a row may be drawn multiple times
        :param str or expression weights: (unnormalized) probability that a row can be drawn
        :param int or RandomState: seed or RandomState for reproducability, when None a random seed it chosen
        :return: {return_shallow_copy}
        :rtype: DataFrame
        '''
        self = self.extract()
        if type(random_state) == int or random_state is None:
            random_state = np.random.RandomState(seed=random_state)
        if n is None and frac is None:
            n = 1
        elif frac is not None:
            n = int(round(frac * len(self)))
        weights_values = None
        if weights is not None:
            weights_values = self.evaluate(weights)
            weights_values = weights_values / self.sum(weights)
        indices = random_state.choice(len(self), n, replace=replace, p=weights_values)
        return self.take(indices)

    @docsubst
    @vaex.utils.gen_to_list
    def split_random(self, frac, random_state=None):
        '''Returns a list containing random portions of the DataFrame.

        {note_copy}

        Example:

        >>> import vaex, import numpy as np
        >>> np.random.seed(111)
        >>> df = vaex.from_arrays(x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> for dfs in df.split_random(frac=0.3, random_state=42):
        ...     print(dfs.x.values)
        ...
        [8 1 5]
        [0 7 2 9 4 3 6]
        >>> for split in df.split_random(frac=[0.2, 0.3, 0.5], random_state=42):
        ...     print(dfs.x.values)
        [8 1]
        [5 0 7]
        [2 9 4 3 6]

        :param int/list frac: If int will split the DataFrame in two portions, the first of which will have size as specified by this parameter. If list, the generator will generate as many portions as elements in the list, where each element defines the relative fraction of that portion.
        :param int random_state: (default, None) Random number seed for reproducibility.
        :return: A list of DataFrames.
        :rtype: list
        '''
        self = self.extract()
        if type(random_state) == int or random_state is None:
            random_state = np.random.RandomState(seed=random_state)
        indices = random_state.choice(len(self), len(self), replace=False)
        return self.take(indices).split(frac)

    @docsubst
    @vaex.utils.gen_to_list
    def split(self, frac):
        '''Returns a list containing ordered subsets of the DataFrame.

        {note_copy}

        Example:

        >>> import vaex
        >>> df = vaex.from_arrays(x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> for dfs in df.split(frac=0.3):
        ...     print(dfs.x.values)
        ...
        [0 1 3]
        [3 4 5 6 7 8 9]
        >>> for split in df.split(frac=[0.2, 0.3, 0.5]):
        ...     print(dfs.x.values)
        [0 1]
        [2 3 4]
        [5 6 7 8 9]

        :param int/list frac: If int will split the DataFrame in two portions, the first of which will have size as specified by this parameter. If list, the generator will generate as many portions as elements in the list, where each element defines the relative fraction of that portion.
        :return: A list of DataFrames.
        :rtype: list
        '''
        self = self.extract()
        if _issequence(frac):
            # make sure it is normalized
            total = sum(frac)
            frac = [k / total for k in frac]
        else:
            assert frac <= 1, "fraction should be <= 1"
            frac = [frac, 1 - frac]
        offsets = np.round(np.cumsum(frac) * len(self)).astype(np.int64)
        start = 0
        for offset in offsets:
            yield self[start:offset]
            start = offset

    @docsubst
    def sort(self, by, ascending=True, kind='quicksort'):
        '''Return a sorted DataFrame, sorted by the expression 'by'

        The kind keyword is ignored if doing multi-key sorting.

        {note_copy}

        {note_filter}

        Example:

        >>> import vaex, numpy as np
        >>> df = vaex.from_arrays(s=np.array(['a', 'b', 'c', 'd']), x=np.arange(1,5))
        >>> df['y'] = (df.x-1.8)**2
        >>> df
          #  s      x     y
          0  a      1  0.64
          1  b      2  0.04
          2  c      3  1.44
          3  d      4  4.84
        >>> df.sort('y', ascending=False)  # Note: passing '(x-1.8)**2' gives the same result
          #  s      x     y
          0  d      4  4.84
          1  c      3  1.44
          2  a      1  0.64
          3  b      2  0.04

        :param str or expression by: expression to sort by
        :param bool ascending: ascending (default, True) or descending (False)
        :param str kind: kind of algorithm to use (passed to numpy.argsort)
        '''
        self = self.trim()
        if not isinstance(by, list):
            values = self.evaluate(by)
            indices = np.argsort(values, kind=kind)
        if isinstance(by, (list, tuple)):
            by = _ensure_strings_from_expressions(by)[::-1]
            values = self.evaluate(by)
            indices = np.lexsort(values)
        if not ascending:
            indices = indices[::-1].copy()  # this may be used a lot, so copy for performance
        return self.take(indices)

    @docsubst
    def fillna(self, value, column_names=None, prefix='__original_', inplace=False):
        '''Return a DataFrame, where missing values/NaN are filled with 'value'.

        The original columns will be renamed, and by default they will be hidden columns. No data is lost.

        {note_copy}

        {note_filter}

        Example:

        >>> import vaex
        >>> import numpy as np
        >>> x = np.array([3, 1, np.nan, 10, np.nan])
        >>> df = vaex.from_arrays(x=x)
        >>> df_filled = df.fillna(value=-1, column_names=['x'])
        >>> df_filled
          #    x
          0    3
          1    1
          2   -1
          3   10
          4   -1

        :param float value: The value to use for filling nan or masked values.
        :param bool fill_na: If True, fill np.nan values with `value`.
        :param bool fill_masked: If True, fill masked values with `values`.
        :param list column_names: List of column names in which to fill missing values.
        :param str prefix: The prefix to give the original columns.
        :param inplace: {inplace}
        '''
        df = self.trim(inplace=inplace)
        column_names = column_names or list(self)
        for name in column_names:
            column = df.columns.get(name)
            df[name] = df.func.fillna(df[name], value)
        return df

    def materialize(self, virtual_column, inplace=False):
        '''Returns a new DataFrame where the virtual column is turned into an in memory numpy array.

        Example:

        >>> x = np.arange(1,4)
        >>> y = np.arange(2,5)
        >>> df = vaex.from_arrays(x=x, y=y)
        >>> df['r'] = (df.x**2 + df.y**2)**0.5 # 'r' is a virtual column (computed on the fly)
        >>> df = df.materialize('r')  # now 'r' is a 'real' column (i.e. a numpy array)

        :param inplace: {inplace}
        '''
        df = self.trim(inplace=inplace)
        virtual_column = _ensure_string_from_expression(virtual_column)
        if virtual_column not in df.virtual_columns:
            raise KeyError('Virtual column not found: %r' % virtual_column)
        ar = df.evaluate(virtual_column, filtered=False)
        del df[virtual_column]
        df.add_column(virtual_column, ar)
        return df

    def get_selection(self, name="default"):
        """Get the current selection object (mostly for internal use atm)."""
        name = _normalize_selection_name(name)
        selection_history = self.selection_histories[name]
        index = self.selection_history_indices[name]
        if index == -1:
            return None
        else:
            return selection_history[index]

    def selection_undo(self, name="default", executor=None):
        """Undo selection, for the name."""
        logger.debug("undo")
        executor = executor or self.executor
        assert self.selection_can_undo(name=name)
        selection_history = self.selection_histories[name]
        index = self.selection_history_indices[name]
        self.selection_history_indices[name] -= 1
        self.signal_selection_changed.emit(self, name)
        logger.debug("undo: selection history is %r, index is %r", selection_history, self.selection_history_indices[name])

    def selection_redo(self, name="default", executor=None):
        """Redo selection, for the name."""
        logger.debug("redo")
        executor = executor or self.executor
        assert self.selection_can_redo(name=name)
        selection_history = self.selection_histories[name]
        index = self.selection_history_indices[name]
        next = selection_history[index + 1]
        self.selection_history_indices[name] += 1
        self.signal_selection_changed.emit(self, name)
        logger.debug("redo: selection history is %r, index is %r", selection_history, index)

    def selection_can_undo(self, name="default"):
        """Can selection name be undone?"""
        return self.selection_history_indices[name] > -1

    def selection_can_redo(self, name="default"):
        """Can selection name be redone?"""
        return (self.selection_history_indices[name] + 1) < len(self.selection_histories[name])

    def select(self, boolean_expression, mode="replace", name="default", executor=None):
        """Perform a selection, defined by the boolean expression, and combined with the previous selection using the given mode.

        Selections are recorded in a history tree, per name, undo/redo can be done for them separately.

        :param str boolean_expression: Any valid column expression, with comparison operators
        :param str mode: Possible boolean operator: replace/and/or/xor/subtract
        :param str name: history tree or selection 'slot' to use
        :param executor:
        :return:
        """
        boolean_expression = _ensure_string_from_expression(boolean_expression)
        if boolean_expression is None and not self.has_selection(name=name):
            pass  # we don't want to pollute the history with many None selections
            self.signal_selection_changed.emit(self, name)  # TODO: unittest want to know, does this make sense?
        else:
            def create(current):
                return selections.SelectionExpression(boolean_expression, current, mode) if boolean_expression else None
            self._selection(create, name)

    def select_non_missing(self, drop_nan=True, drop_masked=True, column_names=None, mode="replace", name="default"):
        """Create a selection that selects rows having non missing values for all columns in column_names.

        The name reflects Pandas, no rows are really dropped, but a mask is kept to keep track of the selection

        :param drop_nan: drop rows when there is a NaN in any of the columns (will only affect float values)
        :param drop_masked: drop rows when there is a masked value in any of the columns
        :param column_names: The columns to consider, default: all (real, non-virtual) columns
        :param str mode: Possible boolean operator: replace/and/or/xor/subtract
        :param str name: history tree or selection 'slot' to use
        :return:
        """
        column_names = column_names or self.get_column_names(virtual=False)

        def create(current):
            return selections.SelectionDropNa(drop_nan, drop_masked, column_names, current, mode)
        self._selection(create, name)

    def dropmissing(self, column_names=None):
        """Create a shallow copy of a DataFrame, with filtering set using ismissing.

        :param column_names: The columns to consider, default: all (real, non-virtual) columns
        :rtype: DataFrame
        """
        return self._filter_all(self.func.ismissing, column_names)

    def dropnan(self, column_names=None):
        """Create a shallow copy of a DataFrame, with filtering set using isnan.

        :param column_names: The columns to consider, default: all (real, non-virtual) columns
        :rtype: DataFrame
        """
        return self._filter_all(self.func.isnan, column_names)

    def dropna(self, column_names=None):
        """Create a shallow copy of a DataFrame, with filtering set using isna.

        :param column_names: The columns to consider, default: all (real, non-virtual) columns
        :rtype: DataFrame
        """
        return self._filter_all(self.func.isna, column_names)

    def _filter_all(self, f, column_names=None):
        copy = self.copy()
        column_names = column_names or self.get_column_names(virtual=False)
        expression = f(self[column_names[0]])
        for column in column_names[1:]:
            expression = expression | f(self[column])
        copy.select(~expression, name=FILTER_SELECTION_NAME, mode='and')
        return copy

    def select_nothing(self, name="default"):
        """Select nothing."""
        logger.debug("selecting nothing")
        self.select(None, name=name)
        self.signal_selection_changed.emit(self, name)

    def select_rectangle(self, x, y, limits, mode="replace", name="default"):
        """Select a 2d rectangular box in the space given by x and y, bounded by limits.

        Example:

        >>> df.select_box('x', 'y', [(0, 10), (0, 1)])

        :param x: expression for the x space
        :param y: expression fo the y space
        :param limits: sequence of shape [(x1, x2), (y1, y2)]
        :param mode:
        """
        self.select_box([x, y], limits, mode=mode, name=name)

    def select_box(self, spaces, limits, mode="replace", name="default"):
        """Select a n-dimensional rectangular box bounded by limits.

        The following examples are equivalent:

        >>> df.select_box(['x', 'y'], [(0, 10), (0, 1)])
        >>> df.select_rectangle('x', 'y', [(0, 10), (0, 1)])

        :param spaces: list of expressions
        :param limits: sequence of shape [(x1, x2), (y1, y2)]
        :param mode:
        :param name:
        :return:
        """
        sorted_limits = [(min(l), max(l)) for l in limits]
        expressions = ["((%s) >= %f) & ((%s) <= %f)" % (expression, lmin, expression, lmax) for
                       (expression, (lmin, lmax)) in zip(spaces, sorted_limits)]
        self.select("&".join(expressions), mode=mode, name=name)

    def select_circle(self, x, y, xc, yc, r, mode="replace", name="default", inclusive=True):
        """
        Select a circular region centred on xc, yc, with a radius of r.

        Example:

        >>> df.select_circle('x','y',2,3,1)

        :param x: expression for the x space
        :param y: expression for the y space
        :param xc: location of the centre of the circle in x
        :param yc: location of the centre of the circle in y
        :param r: the radius of the circle
        :param name: name of the selection
        :param mode:
        :return:
        """

        # expr = "({x}-{xc})**2 + ({y}-{yc})**2 <={r}**2".format(**locals())
        if inclusive:
            expr = (self[x] - xc)**2 + (self[y] - yc)**2 <= r**2
        else:
            expr = (self[x] - xc)**2 + (self[y] - yc)**2 < r**2

        self.select(boolean_expression=expr, mode=mode, name=name)

    def select_ellipse(self, x, y, xc, yc, width, height, angle=0, mode="replace", name="default", radians=False, inclusive=True):
        """
        Select an elliptical region centred on xc, yc, with a certain width, height
        and angle.

        Example:

        >>> df.select_ellipse('x','y', 2, -1, 5,1, 30, name='my_ellipse')

        :param x: expression for the x space
        :param y: expression for the y space
        :param xc: location of the centre of the ellipse in x
        :param yc: location of the centre of the ellipse in y
        :param width: the width of the ellipse (diameter)
        :param height: the width of the ellipse (diameter)
        :param angle: (degrees) orientation of the ellipse, counter-clockwise
                      measured from the y axis
        :param name: name of the selection
        :param mode:
        :return:

        """

        # Computing the properties of the ellipse prior to selection
        if radians:
            pass
        else:
            alpha = np.deg2rad(angle)
        xr = width / 2
        yr = height / 2
        r = max(xr, yr)
        a = xr / r
        b = yr / r

        expr = "(({x}-{xc})*cos({alpha})+({y}-{yc})*sin({alpha}))**2/{a}**2 + (({x}-{xc})*sin({alpha})-({y}-{yc})*cos({alpha}))**2/{b}**2 <= {r}**2".format(**locals())

        if inclusive:
            expr = ((self[x] - xc) * np.cos(alpha) + (self[y] - yc) * np.sin(alpha))**2 / a**2 + ((self[x] - xc) * np.sin(alpha) - (self[y] - yc) * np.cos(alpha))**2 / b**2 <= r**2
        else:
            expr = ((self[x] - xc) * np.cos(alpha) + (self[y] - yc) * np.sin(alpha))**2 / a**2 + ((self[x] - xc) * np.sin(alpha) - (self[y] - yc) * np.cos(alpha))**2 / b**2 < r**2

        self.select(boolean_expression=expr, mode=mode, name=name)

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
            return selections.SelectionLasso(expression_x, expression_y, xsequence, ysequence, current, mode)
        self._selection(create, name, executor=executor)

    def select_inverse(self, name="default", executor=None):
        """Invert the selection, i.e. what is selected will not be, and vice versa

        :param str name:
        :param executor:
        :return:
        """

        def create(current):
            return selections.SelectionInvert(current)
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
        self.signal_selection_changed.emit(self, name)
        result = vaex.promise.Promise.fulfilled(None)
        logger.debug("select selection history is %r, index is %r", selection_history, self.selection_history_indices[name])
        return result

    def has_selection(self, name="default"):
        """Returns True if there is a selection with the given name."""
        return self.get_selection(name) is not None

    def __setitem__(self, name, value):
        '''Convenient way to add a virtual column / expression to this DataFrame.

        Example:

        >>> import vaex, numpy as np
        >>> df = vaex.example()
        >>> df['r'] = np.sqrt(df.x**2 + df.y**2 + df.z**2)
        >>> df.r
        <vaex.expression.Expression(expressions='r')> instance at 0x121687e80 values=[2.9655450396553587, 5.77829281049018, 6.99079603950256, 9.431842752707537, 0.8825613121347967 ... (total 330000 values) ... 7.453831761514681, 15.398412491068198, 8.864250273925633, 17.601047186042507, 14.540181524970293]
        '''

        if isinstance(name, six.string_types):
            if isinstance(value, (np.ndarray, Column)):
                self.add_column(name, value)
            else:
                self.add_virtual_column(name, value)
        else:
            raise TypeError('__setitem__ only takes strings as arguments, not {}'.format(type(name)))

    def drop_filter(self, inplace=False):
        """Removes all filters from the DataFrame"""
        df = self if inplace else self.copy()
        df.select_nothing(name=FILTER_SELECTION_NAME)
        df._invalidate_caches()
        return df

    def filter(self, expression, mode="and"):
        """General version of df[<boolean expression>] to modify the filter applied to the DataFrame.

        See :func:`DataFrame.select` for usage of selection.

        Note that using `df = df[<boolean expression>]`, one can only narrow the filter (i.e. only less rows
        can be selected). Using the filter method, and a different boolean mode (e.g. "or") one can actually
        cause more rows to be selected. This differs greatly from numpy and pandas for instance, which can only
        narrow the filter.

        Example:

        >>> import vaex
        >>> import numpy as np
        >>> x = np.arange(10)
        >>> df = vaex.from_arrays(x=x, y=x**2)
        >>> df
        #    x    y
        0    0    0
        1    1    1
        2    2    4
        3    3    9
        4    4   16
        5    5   25
        6    6   36
        7    7   49
        8    8   64
        9    9   81
        >>> dff = df[df.x<=2]
        >>> dff
        #    x    y
        0    0    0
        1    1    1
        2    2    4
        >>> dff = dff.filter(dff.x >=7, mode="or")
        >>> dff
        #    x    y
        0    0    0
        1    1    1
        2    2    4
        3    7   49
        4    8   64
        5    9   81
        """
        df = self.copy()
        df.select(expression, name=FILTER_SELECTION_NAME, mode=mode)
        df._cached_filtered_length = None  # invalide cached length
        # WARNING: this is a special case where we create a new filter
        # the cache mask chunks still hold references to views on the old
        # mask, and this new mask will be filled when required
        df._selection_masks[FILTER_SELECTION_NAME] = vaex.superutils.Mask(df._length_unfiltered)
        return df

    def __getitem__(self, item):
        """Convenient way to get expressions, (shallow) copies of a few columns, or to apply filtering.

        Example:

        >>> df['Lz']  # the expression 'Lz
        >>> df['Lz/2'] # the expression 'Lz/2'
        >>> df[["Lz", "E"]] # a shallow copy with just two columns
        >>> df[df.Lz < 0]  # a shallow copy with the filter Lz < 0 applied

        """
        if isinstance(item, int):
            names = self.get_column_names()
            return [self.evaluate(name, item, item+1)[0] for name in names]
        elif isinstance(item, six.string_types):
            if item in self._column_aliases:
                item = self._column_aliases[item]  # translate the alias name into the real name
            if hasattr(self, item) and isinstance(getattr(self, item), Expression):
                return getattr(self, item)
            # if item in self.virtual_columns:
            #   return Expression(self, self.virtual_columns[item])
            # if item in self._virtual_expressions:
            #     return self._virtual_expressions[item]
            self.validate_expression(item)
            return Expression(self, item)  # TODO we'd like to return the same expression if possible
        elif isinstance(item, Expression):
            expression = item.expression
            return self.filter(expression)
        elif isinstance(item, (tuple, list)):
            df = self
            if isinstance(item[0], slice):
                df = df[item[0]]
            if len(item) > 1:
                if isinstance(item[1], int):
                    name = self.get_column_names()[item[1]]
                    return df[name]
                elif isinstance(item[1], slice):
                    names = self.get_column_names().__getitem__(item[1])
                    return df[names]
            for expression in item:
                if expression in self._column_aliases:
                    expression = self._column_aliases[expression]
                self.validate_expression(expression)
            df = self.copy(column_names=item)
            return df
        elif isinstance(item, slice):
            start, stop, step = item.start, item.stop, item.step
            start = start or 0
            stop = stop or len(self)
            if start < 0:
                start = len(self)+start
            if stop < 0:
                stop = len(self)+stop
            stop = min(stop, len(self))
            assert step in [None, 1]
            if self.filtered:
                count_check = self.count()  # fill caches and masks
                mask = self._selection_masks[FILTER_SELECTION_NAME]
                start, stop = mask.indices(start, stop-1) # -1 since it is inclusive
                assert start != -1
                assert stop != -1
                stop = stop+1  # +1 to make it inclusive
            df = self.trim()
            df.set_active_range(start, stop)
            return df.trim()

    def __delitem__(self, item):
        '''Removes a (virtual) column from the DataFrame.

        Note: this does not check if the column is used in a virtual expression or in the filter\
            and may lead to issues. It is safer to use :meth:`drop`.
        '''
        if isinstance(item, Expression):
            name = item.expression
        else:
            name = item
        if name in self.columns:
            del self.columns[name]
            self.column_names.remove(name)
        elif name in self.virtual_columns:
            del self.virtual_columns[name]
            del self._virtual_expressions[name]
            self.column_names.remove(name)
        else:
            raise KeyError('no such column or virtual_columns named %r' % name)
        self.signal_column_changed.emit(self, name, "delete")
        if hasattr(self, name):
            try:
                if isinstance(getattr(self, name), Expression):
                    delattr(self, name)
            except:
                pass

    @docsubst
    def drop(self, columns, inplace=False, check=True):
        """Drop columns (or a single column).

        :param columns: List of columns or a single column name
        :param inplace: {inplace}
        :param check: When true, it will check if the column is used in virtual columns or the filter, and hide it instead.
        """
        columns = _ensure_list(columns)
        columns = _ensure_strings_from_expressions(columns)
        df = self if inplace else self.copy()
        depending_columns = df._depending_columns(columns_exclude=columns)
        for column in columns:
            if check and column in depending_columns:
                df._hide_column(column)
            else:
                del df[column]
        return df

    def _hide_column(self, column):
        '''Hides a column by prefixing the name with \'__\''''
        column = _ensure_string_from_expression(column)
        new_name = self._find_valid_name('__' + column)
        self._rename(column, new_name)
        return new_name

    def _find_valid_name(self, initial_name):
        '''Finds a non-colliding name by optional postfixing'''
        return vaex.utils.find_valid_name(initial_name, used=self.get_column_names(hidden=True))

    def _depending_columns(self, columns=None, columns_exclude=None, check_filter=True):
        '''Find all depending column for a set of column (default all), minus the excluded ones'''
        columns = set(columns or self.get_column_names(hidden=True))
        if columns_exclude:
            columns -= set(columns_exclude)
        depending_columns = set()
        for column in columns:
            expression = self._expr(column)
            depending_columns |= expression.variables()
        depending_columns -= set(columns)
        if check_filter:
            if self.filtered:
                selection = self.get_selection(FILTER_SELECTION_NAME)
                depending_columns |= selection._depending_columns(self)
        return depending_columns

    def iterrows(self):
        columns = self.get_column_names()
        for i in range(len(self)):
            yield i, {key: self.evaluate(key, i, i+1)[0] for key in columns}
            #return self[i]

    def __iter__(self):
        """Iterator over the column names."""
        return iter(list(self.get_column_names()))

    def _root_nodes(self):
        """Returns a list of string which are the virtual columns that are not used in any other virtual column."""
        # these lists (~used as ordered set) keep track of leafes and root nodes
        # root nodes
        root_nodes = []
        leafes = []
        def walk(node):
            # this function recursively walks the expression graph
            if isinstance(node, six.string_types):
                # we end up at a leaf
                leafes.append(node)
                if node in root_nodes:  # so it cannot be a root node
                    root_nodes.remove(node)
            else:
                node_repr, fname, fobj, deps = node
                if node_repr in self.virtual_columns:
                    # we encountered a virtual column, similar behaviour as leaf
                    leafes.append(node_repr)
                    if node_repr in root_nodes:
                        root_nodes.remove(node_repr)
                # resursive part
                for dep in deps:
                    walk(dep)
        for column in self.virtual_columns.keys():
            if column not in leafes:
                root_nodes.append(column)
            node = self[column]._graph()
            # we don't do the virtual column itself, just it's depedencies
            node_repr, fname, fobj, deps = node
            for dep in deps:
                walk(dep)
        return root_nodes

    def _graphviz(self, dot=None):
        """Return a graphviz.Digraph object with a graph of all virtual columns"""
        from graphviz import Digraph
        dot = dot or Digraph(comment='whole dataframe')
        root_nodes = self._root_nodes()
        for column in root_nodes:
            self[column]._graphviz(dot=dot)
        return dot


    def _get_task_agg(self, grid):
        new_task = False
        with self.executor.lock:
            # if we did not create a task yet for this grid, or it was already scheduled for execution
            if grid not in self._task_aggs or self._task_aggs[grid] not in self.executor.tasks:
                # we create a new one
                self._task_aggs[grid] = vaex.tasks.TaskAggregations(self, grid)
                new_task = True
        return self._task_aggs[grid], new_task

    @docsubst
    @stat_1d
    def _agg(self, aggregator, grid, selection=False, delay=False, progress=None):
        """

        :param selection: {selection}
        :param delay: {delay}
        :param progress: {progress}
        :return: {return_stat_scalar}
        """
        task_agg, new_task = self._get_task_agg(grid)
        sub_task = aggregator.add_operations(task_agg)
        if new_task:
            self.executor.schedule(task_agg)
        return self._delay(delay, sub_task)

    def _binner(self, expression, limits=None, shape=None, selection=None, delay=False):
        expression = str(expression)
        if limits is not None and not isinstance(limits, (tuple, str)):
            limits = tuple(limits)
        key = (expression, limits, shape)
        if key not in self._binners:
            if expression in self._categories:
                N = self._categories[expression]['N']
                min_value = self._categories[expression]['min_value']
                binner = self._binner_ordinal(expression, N, min_value)
                self._binners[key] = vaex.promise.Promise.fulfilled(binner)
            else:
                self._binners[key] = vaex.promise.Promise()
                @delayed
                def create_binner(limits):
                    return self._binner_scalar(expression, limits, shape)
                self._binners[key] = create_binner(self.limits(expression, limits, selection=selection, delay=True))
        return self._delay(delay, self._binners[key])

    def _grid(self, binners):
        key = tuple(binners)
        if key in self._grids:
            return self._grids[key]
        else:
            self._grids[key] = grid = vaex.superagg.Grid(binners)
            return grid

    def _binner_scalar(self, expression, limits, shape):
        type = vaex.utils.find_type_from_dtype(vaex.superagg, "BinnerScalar_", self.data_type(expression))
        vmin, vmax = limits
        return type(expression, vmin, vmax, shape)

    def _binner_ordinal(self, expression, ordinal_count, min_value=0):
        expression = _ensure_string_from_expression(expression)
        type = vaex.utils.find_type_from_dtype(vaex.superagg, "BinnerOrdinal_", self.data_type(expression))
        return type(expression, ordinal_count, min_value)

    def _create_grid(self, binby, limits, shape, selection=None, delay=False):
        if isinstance(binby, (list, tuple)):
            binbys = binby
        else:
            binbys = [binby]
        binbys = _ensure_strings_from_expressions(binbys)
        for expression in binbys:
            if expression:
                self.validate_expression(expression)
        binners = []
        if len(binbys):
            limits = _expand_limits(limits, len(binbys))
        else:
            limits = []
        shapes = _expand_shape(shape, len(binbys))
        for binby, limits1, shape in zip(binbys, limits, shapes):
            binners.append(self._binner(binby, limits1, shape, selection, delay=True))
        @delayed
        def finish(*binners):
            return self._grid(binners)
        return self._delay(delay, finish(*binners))

DataFrame.__hidden__ = {}
hidden = [name for name, func in vars(DataFrame).items() if getattr(func, '__hidden__', False)]
for name in hidden:
    DataFrame.__hidden__[name] = getattr(DataFrame, name)
    delattr(DataFrame, name)
del hidden



class DataFrameLocal(DataFrame):
    """Base class for DataFrames that work with local file/data"""

    def __init__(self, name, path, column_names):
        super(DataFrameLocal, self).__init__(name, column_names)
        self.path = path
        self.mask = None
        self.columns = {}

    def _readonly(self, inplace=False):
        # make arrays read only if possib;e
        df = self if inplace else self.copy()
        for key, ar in self.columns.items():
            if isinstance(ar, np.ndarray):
                df.columns[key] = ar = ar.view() # make new object so we don't modify others
                ar.flags['WRITEABLE'] = False
        return df

    @docsubst
    def categorize(self, column, min_value=0, max_value=None, labels=None, inplace=False):
        """Mark column as categorical.

        This may help speed up calculations using integer columns between a range of [min_value, max_value].

        If max_value is not given, the [min_value and max_value] are calcuated from the data.

        Example:

        >>> import vaex
        >>> df = vaex.from_arrays(year=[2012, 2015, 2019], weekday=[0, 4, 6])
        >>> df.categorize('year', min_value=2020, max_value=2019)
        >>> df.categorize('weekday', labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

        :param column: column to assume is categorical.
        :param labels: labels to associate to the values between min_value and max_value
        :param min_value: minimum integer value (if max_value is not given, this is calculated)
        :param max_value: maximum integer value (if max_value is not given, this is calculated)
        :param labels: Labels to associate to each value, list(range(min_value, max_value+1)) by default
        :param inplace: {inplace}
        """
        df = self if inplace else self.copy()
        column = _ensure_string_from_expression(column)
        if max_value is not None:
            labels = list(range(min_value, max_value+1))
            N = len(labels)
        else:
            vmin, vmax = df.minmax(column)
            if labels is None:
                N = int(vmax + 1)
                labels = list(range(vmin, vmax+1))
                min_value = vmin
            else:
                min_value = vmin
            if (vmax - vmin) >= len(labels):
                raise ValueError('value of {} found, which is larger than number of labels {}'.format(vmax, len(labels)))
        df._categories[column] = dict(labels=labels, N=len(labels), min_value=min_value)
        return df

    def ordinal_encode(self, column, values=None, inplace=False):
        """Encode column as ordinal values and mark it as categorical.

        The existing column is renamed to a hidden column and replaced by a numerical columns
        with values between [0, len(values)-1].
        """
        column = _ensure_string_from_expression(column)
        df = self if inplace else self.copy()
        # for the codes, we need to work on the unfiltered dataset, since the filter
        # may change, and we also cannot add an array that is smaller in length
        df_unfiltered = df.copy()
        # maybe we need some filter manipulation methods
        df_unfiltered.select_nothing(name=FILTER_SELECTION_NAME)
        df_unfiltered._length_unfiltered = df._length_original
        df_unfiltered.set_active_range(0, df._length_original)
        # codes point to the index of found_values
        # meaning: found_values[codes[0]] == ds[column].values[0]
        found_values, codes = df_unfiltered.unique(column, return_inverse=True)
        if values is None:
            values = found_values
        else:
            # we have specified which values we should support, anything
            # not found will be masked
            translation = np.zeros(len(found_values), dtype=np.uint64)
            # mark values that are in the column, but not in values with a special value
            missing_value = len(found_values)
            for i, found_value in enumerate(found_values):
                try:
                    found_value = found_value.decode('ascii')
                except:
                    pass
                if found_value not in values:  # not present, we need a missing value
                    translation[i] = missing_value
                else:
                    translation[i] = values.index(found_value)
            codes = translation[codes]
            if missing_value in translation:
                # all special values will be marked as missing
                codes = np.ma.masked_array(codes, codes==missing_value)

        original_column = df.rename(column, '__original_' + column, unique=True)
        df.add_column(column, codes)
        df._categories[column] = dict(labels=values, N=len(values), min_value=0)
        return df

    # for backward compatibility
    label_encode = _hidden(vaex.utils.deprecated('use is_category')(ordinal_encode))

    @property
    def data(self):
        """Gives direct access to the data as numpy arrays.

        Convenient when working with IPython in combination with small DataFrames, since this gives tab-completion.
        Only real columns (i.e. no virtual) columns can be accessed, for getting the data from virtual columns, use
        DataFrame.evaluate(...).

        Columns can be accessed by their names, which are attributes. The attributes are of type numpy.ndarray.

        Example:

        >>> df = vaex.example()
        >>> r = np.sqrt(df.data.x**2 + df.data.y**2)

        """
        class Datas(object):
            pass

        datas = Datas()
        for name, array in self.columns.items():
            setattr(datas, name, array)
        return datas

    def copy(self, column_names=None, virtual=True):
        df = DataFrameArrays()
        df._length_unfiltered = self._length_unfiltered
        df._length_original = self._length_original
        df._cached_filtered_length = self._cached_filtered_length
        df._index_end = self._index_end
        df._index_start = self._index_start
        df._active_fraction = self._active_fraction
        df._renamed_columns = list(self._renamed_columns)
        df.units.update(self.units)
        df.variables.update(self.variables)  # we add all, could maybe only copy used
        df._categories.update(self._categories)
        if column_names is None:
            column_names = self.get_column_names(hidden=True, alias=False)
            df._column_aliases = dict(self._column_aliases)
        else:
            df._column_aliases = {alias: real_name for alias, real_name in self._column_aliases.items()}
            column_names = [self._column_aliases.get(k, k) for k in column_names]
        all_column_names = self.get_column_names(hidden=True, alias=False)

        # put in the selections (thus filters) in place
        # so drop moves instead of really dropping it
        df.functions.update(self.functions)
        for key, value in self.selection_histories.items():
            # TODO: selection_histories begin a defaultdict always gives
            # us the filtered selection, so check if we really have a
            # selection
            if self.get_selection(key):
                df.selection_histories[key] = list(value)
                # the filter should never be modified, so we can share a reference
                # except when we add filter on filter using
                # df = df[df.x>0]
                # df = df[df.x < 10]
                # in that case we make a copy in __getitem__
                if key == FILTER_SELECTION_NAME:
                    df._selection_masks[key] = self._selection_masks[key]
                else:
                    df._selection_masks[key] = vaex.superutils.Mask(df._length_original)
                # and make sure the mask is consistent with the cache chunks
                np.asarray(df._selection_masks[key])[:] = np.asarray(self._selection_masks[key])
        for key, value in self.selection_history_indices.items():
            if self.get_selection(key):
                df.selection_history_indices[key] = value
                # we can also copy the caches, which prevents recomputations of selections
                df._selection_mask_caches[key] = collections.defaultdict(dict)
                df._selection_mask_caches[key].update(self._selection_mask_caches[key])

        if 1:
            # print("-----", column_names)
            depending = set()
            added = set()
            for name in column_names:
                # print("add", name)
                added.add(name)
                if name in self.columns:
                    column = self.columns[name]
                    df.add_column(name, column, dtype=self._dtypes_override.get(name))
                    if isinstance(column, ColumnSparse):
                        df._sparse_matrices[name] = self._sparse_matrices[name]
                elif name in self.virtual_columns:
                    if virtual:  # TODO: check if the ast is cached
                        df.add_virtual_column(name, self.virtual_columns[name])
                        deps = [key for key, value in df._virtual_expressions[name].ast_names.items()]
                        # print("add virtual", name, df._virtual_expressions[name].expression, deps)
                        depending.update(deps)
                else:
                    # this might be an expression, create a valid name
                    real_column_name = df._column_aliases.get(name, name)
                    valid_name = vaex.utils.find_valid_name(name)
                    self.validate_expression(real_column_name)
                    # add the expression
                    df[valid_name] = df._expr(real_column_name)
                    # then get the dependencies
                    deps = [key for key, value in df._virtual_expressions[valid_name].ast_names.items()]
                    depending.update(deps)
                # print(depending, "after add")
            # depending |= column_names
            # print(depending)
            # print(depending, "before filter")
            if self.filtered:
                selection = self.get_selection(FILTER_SELECTION_NAME)
                depending |= selection._depending_columns(self)
            depending.difference_update(added)  # remove already added
            # print(depending, "after filter")
            # return depending_columns

            hide = []

            while depending:
                new_depending = set()
                for name in depending:
                    added.add(name)
                    if name in self.columns:
                        # print("add column", name)
                        df.add_column(name, self.columns[name], dtype=self._dtypes_override.get(name))
                        # print("and hide it")
                        # df._hide_column(name)
                        hide.append(name)
                    elif name in self.virtual_columns:
                        if virtual:  # TODO: check if the ast is cached
                            df.add_virtual_column(name, self.virtual_columns[name])
                            deps = [key for key, value in self._virtual_expressions[name].ast_names.items()]
                            new_depending.update(deps)
                        # df._hide_column(name)
                        hide.append(name)
                    elif name in self.variables:
                        # if must be a variables?
                        # TODO: what if the variable depends on other variables
                        # we already add all variables
                        # df.add_variable(name, self.variables[name])
                        pass

                # print("new_depending", new_depending)
                new_depending.difference_update(added)
                depending = new_depending
            for name in hide:
                df._hide_column(name)

        else:
            # we copy all columns, but drop the ones that are not wanted
            # this makes sure that needed columns are hidden instead
            def add_columns(columns):
                for name in columns:
                    if name in self.columns:
                        df.add_column(name, self.columns[name], dtype=self._dtypes_override.get(name))
                    elif name in self.virtual_columns:
                        if virtual:
                            df.add_virtual_column(name, self.virtual_columns[name])
                    else:
                        # this might be an expression, create a valid name
                        expression = name
                        name = vaex.utils.find_valid_name(name)
                        df[name] = df._expr(expression)
            # to preserve the order, we first add the ones we want, then the rest
            add_columns(column_names)
            # then the rest
            rest = set(all_column_names) - set(column_names)
            add_columns(rest)
            # and remove them
            for name in rest:
                # if the column should not have been added, drop it. This checks if columns need
                # to be hidden instead, and expressions be rewritten.
                if name not in column_names:
                    df.drop(name, inplace=True)
                    assert name not in df.get_column_names(hidden=True)

        df.copy_metadata(self)
        return df

    def shallow_copy(self, virtual=True, variables=True):
        """Creates a (shallow) copy of the DataFrame.

        It will link to the same data, but will have its own state, e.g. virtual columns, variables, selection etc.

        """
        df = DataFrameLocal(self.name, self.path, self.column_names)
        df.columns.update(self.columns)
        df._length_unfiltered = self._length_unfiltered
        df._length_original = self._length_original
        df._index_end = self._index_end
        df._index_start = self._index_start
        df._active_fraction = self._active_fraction
        if virtual:
            df.virtual_columns.update(self.virtual_columns)
        if variables:
            df.variables.update(self.variables)
        # half shallow/deep copy
        # for key, value in self.selection_histories.items():
        # df.selection_histories[key] = list(value)
        # for key, value in self.selection_history_indices.items():
        # df.selection_history_indices[key] = value
        return df

    def is_local(self):
        """The local implementation of :func:`DataFrame.evaluate`, always returns True."""
        return True

    def length(self, selection=False):
        """Get the length of the DataFrames, for the selection of the whole DataFrame.

        If selection is False, it returns len(df).

        TODO: Implement this in DataFrameRemote, and move the method up in :func:`DataFrame.length`

        :param selection: When True, will return the number of selected rows
        :return:
        """
        if selection:
            return 0 if self.mask is None else np.sum(self.mask)
        else:
            return len(self)

    @_hidden
    def __call__(self, *expressions, **kwargs):
        """The local implementation of :func:`DataFrame.__call__`"""
        import vaex.legacy
        return vaex.legacy.SubspaceLocal(self, expressions, kwargs.get("executor") or self.executor, delay=kwargs.get("delay", False))

    def echo(self, arg): return arg

    @property
    def _dtype(self):
        dtypes = [self[k].dtype for k in self.get_column_names()]
        if not all([dtypes[0] == dtype for dtype in dtypes]):
            return ValueError("Not all dtypes are equal: %r" % dtypes)
        return dtypes[0]

    @property
    def shape(self):
        return (len(self), len(self.get_column_names()))

    def __array__(self, dtype=None, parallel=True):
        """Gives a full memory copy of the DataFrame into a 2d numpy array of shape (n_rows, n_columns).
        Note that the memory order is fortran, so all values of 1 column are contiguous in memory for performance reasons.

        Note this returns the same result as:

        >>> np.array(ds)

        If any of the columns contain masked arrays, the masks are ignored (i.e. the masked elements are returned as well).
        """
        if dtype is None:
            dtype = np.float64
        chunks = []
        column_names = self.get_column_names(strings=False)
        for name in column_names:
            if not np.can_cast(self.data_type(name), dtype):
                if self.data_type(name) != dtype:
                    raise ValueError("Cannot cast %r (of type %r) to %r" % (name, self.data_type(name), dtype))
        chunks = self.evaluate(column_names, parallel=parallel)
        if any(np.ma.isMaskedArray(chunk) for chunk in chunks):
            return np.ma.array(chunks, dtype=dtype).T
        else:
            return np.array(chunks, dtype=dtype).T

    @vaex.utils.deprecated('use DataFrame.join(other)')
    def _hstack(self, other, prefix=None):
        """Join the columns of the other DataFrame to this one, assuming the ordering is the same"""
        assert len(self) == len(other), "does not make sense to horizontally stack DataFrames with different lengths"
        for name in other.get_column_names():
            if prefix:
                new_name = prefix + name
            else:
                new_name = name
            self.add_column(new_name, other.columns[name])

    def concat(self, other):
        """Concatenates two DataFrames, adding the rows of the other DataFrame to the current, returned in a new DataFrame.

        No copy of the data is made.

        :param other: The other DataFrame that is concatenated with this DataFrame
        :return: New DataFrame with the rows concatenated
        :rtype: DataFrameConcatenated
        """
        dfs = []
        if isinstance(self, DataFrameConcatenated):
            dfs.extend(self.dfs)
        else:
            dfs.extend([self])
        if isinstance(other, DataFrameConcatenated):
            dfs.extend(other.dfs)
        else:
            dfs.extend([other])
        return DataFrameConcatenated(dfs)

    def _invalidate_caches(self):
        self._invalidate_selection_cache()
        self._cached_filtered_length = None

    def _invalidate_selection_cache(self):
        self._selection_mask_caches.clear()
        for key in self._selection_masks.keys():
            self._selection_masks[key] = vaex.superutils.Mask(self._length_unfiltered)

    def _filtered_range_to_unfiltered_indices(self, i1, i2):
        assert self.filtered
        count = self.count()  # force the cache to be filled
        assert i2 <= count
        cache = self._selection_mask_caches[FILTER_SELECTION_NAME]
        mask_blocks = iter(sorted(
            [(k1, k2, block) for (k1, k2), (selection, block) in cache.items()],
            key=lambda item: item[0]))
        done = False

        offset_unfiltered = 0  # points to the unfiltered arrays
        offset_filtered = 0    # points to the filtered array
        indices = []
        while not done:
            unfiltered_i1, unfiltered_i2, block = next(mask_blocks)
            count = block.sum()
            if (offset_filtered + count) < i1:  # i1 does not start in this block
                assert unfiltered_i2 == offset_unfiltered + len(block)
                offset_unfiltered = unfiltered_i2
                offset_filtered += count
            else:
                for block_index in range(len(block)):
                    if block[block_index]:  # if not filtered, we go to the next index
                        if i1 <= offset_filtered < i2:  # if this is in the range we want...
                            indices.append(offset_unfiltered)
                        offset_filtered += 1
                    offset_unfiltered += 1
            done = offset_filtered >= i2
        return np.array(indices, dtype=np.int64)

    def _evaluate(self, expression, i1, i2, out=None, selection=None, internal=False, filter_mask=None):
        scope = scopes._BlockScope(self, i1, i2, mask=filter_mask, **self.variables)
        if out is not None:
            scope.buffers[expression] = out
        value = scope.evaluate(expression)
        if isinstance(value, ColumnString) and not internal:
            value = value.to_numpy()
        return value

    def _unfiltered_chunk_slices(self, chunk_size):
        logical_length = len(self)
        if self.filtered:
            full_mask = self._selection_masks[FILTER_SELECTION_NAME]
            # TODO: python 3, use yield from
            for item in vaex.utils.subdivide_mask(full_mask, max_length=chunk_size, logical_length=logical_length):
                yield item
        else:
            for i1, i2 in vaex.utils.subdivide(logical_length, max_length=chunk_size):
                yield i1, i2, i1, i2

    def _evaluate_implementation(self, expression, i1=None, i2=None, out=None, selection=None, filtered=True, internal=False, parallel=True, chunk_size=None, raw=False):
        """The real implementation of :func:`DataFrame.evaluate` (not returning a generator).

        :param raw: Whether indices i1 and i2 refer to unfiltered (raw=True) or 'logical' offsets (raw=False)
        """
        # expression = _ensure_string_from_expression(expression)
        was_list, [expressions] = vaex.utils.listify(expression)
        expressions = vaex.utils._ensure_strings_from_expressions(expressions)
        expressions = [self._column_aliases.get(k, k) for k in expressions]

        selection = _ensure_strings_from_expressions(selection)
        max_stop = (len(self) if (self.filtered and filtered) else self.length_unfiltered())
        i1 = i1 or 0
        i2 = i2 or max_stop
        if parallel:
            df = self
            # first, reduce complexity for the parallel path
            if self.filtered and not filtered:
                df = df.drop_filter()
            if i1 != 0 or i2 != max_stop:
                if not raw and self.filtered and filtered:
                    count_check = len(self)  # fill caches and masks
                    mask = self._selection_masks[FILTER_SELECTION_NAME]
                    i1, i2 = mask.indices(i1, i2-1)
                    assert i1 != -1
                    i2 += 1
                # TODO: performance: can we collapse the two trims in one?
                df = df.trim()
                df.set_active_range(i1, i2)
                df = df.trim()
        expression = expressions[0]
        # here things are simpler or we don't go parallel
        mask = None

        if parallel:
            use_filter = df.filtered and filtered

            length = df.length_unfiltered()
            arrays = {}
            # maps to a dict of start_index -> apache arrow array (a chunk)
            chunks_map = {}
            dtypes = {}
            shapes = {}
            virtual = set()
            # TODO: For NEP branch: dtype -> dtype_evaluate

            for expression in set(expressions):
                dtypes[expression] = df.data_type(expression, internal=False)
                if expression not in df.columns:
                    virtual.add(expression)
                # since we will use pre_filter=True, we'll get chunks of the data at unknown offset
                # so we'll also have to stitch those back together
                if dtypes[expression] == str_type or use_filter or selection:
                    chunks_map[expression] = {}
                else:
                    # we know exactly where to place the chunks, so we pre allocate the arrays
                    shape = (length, ) + df._shape_of(expression, filtered=False)[1:]
                    shapes[expression] = shape
                    if df.is_masked(expression):
                        arrays[expression] = np.ma.empty(shapes.get(expression, length), dtype=dtypes[expression])
                    else:
                        arrays[expression] = np.zeros(shapes.get(expression, length), dtype=dtypes[expression])

            def assign(thread_index, i1, i2, *blocks):
                for i, expr in enumerate(expressions):
                    if expr in chunks_map:
                        # for non-primitive arrays we simply keep a reference to the chunk
                        chunks_map[expr][i1] = blocks[i]
                    else:
                        # for primitive arrays (and no filter/selection) we directly add it to the right place in contiguous numpy array
                        arrays[expr][i1:i2] = blocks[i]

            df.map_reduce(assign, lambda *_: None, expressions, ignore_filter=False, selection=selection, pre_filter=use_filter, info=True, to_numpy=False)
            def finalize_result(expression):
                if expression in chunks_map:
                    # put all chunks in order
                    chunks = [chunk for (i1, chunk) in sorted(chunks_map[expression].items(), key=lambda i1_and_chunk: i1_and_chunk[0])]
                    assert len(chunks) > 0
                    if len(chunks) == 1:
                        # TODO: For NEP Branch
                        # return pa.array(chunks[0])
                        if internal:
                            return chunks[0]
                        else:
                            values = to_numpy(chunks[0])
                            return values
                    else:
                        # TODO: For NEP Branch
                        # return pa.chunked_array(chunks)
                        if internal:
                            return chunks  # Returns a list of StringArrays
                        else:
                            # if isinstance(value, ColumnString) and not internal:
                            return np.concatenate([to_numpy(k) for k in chunks])
                else:
                    return arrays[expression]
            result = [finalize_result(k) for k in expressions]
            if not was_list:
                result = result[0]
            return result
        else:
            if not raw and self.filtered and filtered:
                count_check = len(self)  # fill caches and masks
                mask = self._selection_masks[FILTER_SELECTION_NAME]
                if _DEBUG:
                    if i1 == 0 and i2 == count_check:
                        # we cannot check it if we just evaluate a portion
                        assert not mask.is_dirty()
                        # assert mask.count() == count_check
                i1, i2 = mask.indices(i1, i2-1) # -1 since it is inclusive
                assert i1 != -1
                assert i2 != -1
                i2 = i2+1  # +1 to make it inclusive
            values = []
            for expression in expressions:
                # for both a selection or filtering we have a mask
                if selection not in [None, False] or (self.filtered and filtered):
                    mask = self.evaluate_selection_mask(selection, i1, i2)
                scope = scopes._BlockScope(self, i1, i2, mask=mask, **self.variables)
                # value = value[mask]
                if out is not None:
                    scope.buffers[expression] = out
                value = scope.evaluate(expression)
                if isinstance(value, ColumnString) and not internal:
                    value = value.to_numpy()
                values.append(value)
            if not was_list:
                return values[0]
            return values

    def _equals(self, other):
        values = self.compare(other)
        return values == ([], [], [], [])

    def compare(self, other, report_missing=True, report_difference=False, show=10, orderby=None, column_names=None):
        """Compare two DataFrames and report their difference, use with care for large DataFrames"""
        if column_names is None:
            column_names = self.get_column_names(virtual=False)
            for other_column_name in other.get_column_names(virtual=False):
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
            if column_name not in self.get_column_names(virtual=False):
                missing.append(column_name)
                if report_missing:
                    print("%s missing from this DataFrame" % column_name)
            elif column_name not in other.get_column_names(virtual=False):
                missing.append(column_name)
                if report_missing:
                    print("%s missing from other DataFrame" % column_name)
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
                type1 = self.data_type(column_name)
                if type1 != str_type:
                    type1 = type1.type
                type2 = other.data_type(column_name)
                if type2 != str_type:
                    type2 = type2.type
                if type1 != type2:
                    print("different dtypes: %s vs %s for %s" % (self.data_type(column_name), other.data_type(column_name), column_name))
                    type_mismatch.append(column_name)
                else:
                    # a = self.columns[column_name]
                    # b = other.columns[column_name]
                    # if self.filtered:
                    #   a = a[self.evaluate_selection_mask(None)]
                    # if other.filtered:
                    #   b = b[other.evaluate_selection_mask(None)]
                    a = self.evaluate(column_name)
                    b = other.evaluate(column_name)
                    if orderby:
                        a = a[index1]
                        b = b[index2]

                    def normalize(ar):
                        if ar.dtype == str_type:
                            return ar
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
                        if self.data_type(column_name) != str_type and self.data_type(column_name).kind == 'f':  # floats with nan won't equal itself, i.e. NaN != NaN
                            boolean_mask |= (np.isnan(a) & np.isnan(b))
                        return boolean_mask
                    boolean_mask = equal_mask(a, b)
                    all_equal = np.all(boolean_mask)
                    if not all_equal:
                        count = np.sum(~boolean_mask)
                        print("%s does not match for both DataFrames, %d rows are diffent out of %d" % (column_name, count, len(self)))
                        different_values.append(column_name)
                        if report_difference:
                            indices = np.arange(len(self))[~boolean_mask]
                            values1 = self.columns[column_name][:][~boolean_mask]
                            values2 = other.columns[column_name][:][~boolean_mask]
                            print("\tshowing difference for the first 10")
                            for i in range(min(len(values1), show)):
                                try:
                                    diff = values1[i] - values2[i]
                                except:
                                    diff = "does not exists"
                                print("%s[%d] == %s != %s other.%s[%d] (diff = %s)" % (column_name, indices[i], values1[i], values2[i], column_name, indices[i], diff))
        return different_values, missing, type_mismatch, meta_mismatch

    @docsubst
    def join(self, other, on=None, left_on=None, right_on=None, lprefix='', rprefix='', lsuffix='', rsuffix='', how='left', allow_duplication=False, inplace=False):
        """Return a DataFrame joined with other DataFrames, matched by columns/expression on/left_on/right_on

        If neither on/left_on/right_on is given, the join is done by simply adding the columns (i.e. on the implicit
        row index).

        Note: The filters will be ignored when joining, the full DataFrame will be joined (since filters may
        change). If either DataFrame is heavily filtered (contains just a small number of rows) consider running
        :func:`DataFrame.extract` first.

        Example:

        >>> a = np.array(['a', 'b', 'c'])
        >>> x = np.arange(1,4)
        >>> ds1 = vaex.from_arrays(a=a, x=x)
        >>> b = np.array(['a', 'b', 'd'])
        >>> y = x**2
        >>> ds2 = vaex.from_arrays(b=b, y=y)
        >>> ds1.join(ds2, left_on='a', right_on='b')

        :param other: Other DataFrame to join with (the right side)
        :param on: default key for the left table (self)
        :param left_on: key for the left table (self), overrides on
        :param right_on: default key for the right table (other), overrides on
        :param lprefix: prefix to add to the left column names in case of a name collision
        :param rprefix: similar for the right
        :param lsuffix: suffix to add to the left column names in case of a name collision
        :param rsuffix: similar for the right
        :param how: how to join, 'left' keeps all rows on the left, and adds columns (with possible missing values)
                'right' is similar with self and other swapped. 'inner' will only return rows which overlap.
        :param bool allow_duplication: Allow duplication of rows when the joined column contains non-unique values.
        :param inplace: {inplace}
        :return:
        """
        inner = False
        left = self
        right = other
        if how == 'left':
            pass
        elif how == 'right':
            left, right = right, left
            lprefix, rprefix = rprefix, lprefix
            lsuffix, rsuffix = rsuffix, lsuffix
            left_on, right_on = right_on, left_on
        elif how == 'inner':
            inner = True
        else:
            raise ValueError('join type not supported: {}, only left and right'.format(how))
        left = left if inplace else left.copy()

        on = _ensure_string_from_expression(on)
        left_on = _ensure_string_from_expression(left_on)
        right_on = _ensure_string_from_expression(right_on)
        left_on = left_on or on
        right_on = right_on or on
        left_on = self._column_aliases.get(left_on, left_on)
        right_on = self._column_aliases.get(right_on, right_on)
        for name in right:
            if left_on and (rprefix + name + rsuffix == lprefix + left_on + lsuffix):
                continue  # it's ok when we join on the same column name
            if name in left and rprefix + name + rsuffix == lprefix + name + lsuffix:
                raise ValueError('column name collision: {} exists in both column, and no proper suffix given'
                                 .format(name))

        right = right.extract()  # get rid of filters and active_range
        assert left.length_unfiltered() == left.length_original()
        N = left.length_unfiltered()
        N_other = len(right)
        if left_on is None and right_on is None:
            lookup = None
        else:
            df = left
            # we index the right side, this assumes right is smaller in size
            index = right._index(right_on)
            dtype = left.data_type(left_on)
            duplicates_right = index.has_duplicates

            if duplicates_right and not allow_duplication:
                raise ValueError('This join will lead to duplication of rows which is disabled, pass allow_duplication=True')

            # our max value for the lookup table is the row index number, so if we join a small
            # df with say 100 rows, we can do it with a int8
            lookup_dtype = vaex.utils.required_dtype_for_max(len(right))
            # we put in the max value to maximize triggering failures in the case of a bug (we don't want
            # to point to row 0 in case we do, we'd rather crash)
            lookup = np.full(left._length_original, np.iinfo(lookup_dtype).max, dtype=lookup_dtype)
            nthreads = self.executor.thread_pool.nthreads
            lookup_masked = [False] * nthreads  # does the lookup contain masked/-1 values?
            lookup_extra_chunks = []

            from vaex.column import _to_string_sequence
            def map(thread_index, i1, i2, ar):
                if dtype == str_type:
                    previous_ar = ar
                    ar = _to_string_sequence(ar)
                if np.ma.isMaskedArray(ar):
                    mask = np.ma.getmaskarray(ar)
                    found_masked = index.map_index_masked(ar.data, mask, lookup[i1:i2])
                    lookup_masked[thread_index] = lookup_masked[thread_index] or found_masked
                    if duplicates_right:
                        extra = index.map_index_duplicates(ar.data, mask, i1)
                        lookup_extra_chunks.append(extra)
                else:
                    found_masked = index.map_index(ar, lookup[i1:i2])
                    lookup_masked[thread_index] = lookup_masked[thread_index] or found_masked
                    if duplicates_right:
                        extra = index.map_index_duplicates(ar, i1)
                        lookup_extra_chunks.append(extra)
            def reduce(a, b):
                pass
            left.map_reduce(map, reduce, [left_on], delay=False, name='fill looking', info=True, to_numpy=False, ignore_filter=True)
            if len(lookup_extra_chunks):
                # if the right has duplicates, we increase the left of left, and the lookup array
                lookup_left = np.concatenate([k[0] for k in lookup_extra_chunks])
                lookup_right = np.concatenate([k[1] for k in lookup_extra_chunks])
                left = left.concat(left.take(lookup_left))
                lookup = np.concatenate([lookup, lookup_right])

            if inner:
                left_mask_matched = lookup != -1  # all the places where we found a match to the right
                lookup = lookup[left_mask_matched]  # filter the lookup table to the right
                left_indices_matched = np.where(left_mask_matched)[0]  # convert mask to indices for the left
                # indices can still refer to filtered rows, so do not drop the filter
                left = left.take(left_indices_matched, filtered=False, dropfilter=False)
        direct_indices_map = {}  # for performance, keeps a cache of two levels of indirection of indices

        def mangle_name(prefix, name, suffix):
            if name.startswith('__'):
                return '__' + prefix + name[2:] + suffix
            else:
                return prefix + name + suffix

        # first, do renaming, so all column names are unique
        right_names = right.get_names(hidden=True)
        left_names = left.get_names(hidden=True)
        for name in right_names:
            if name in left_names:
                # find a unique name across both dataframe, including the new name for the left
                all_names = list(set(right_names + left_names))
                all_names.append(mangle_name(lprefix, name, lsuffix))  # we dont want to steal the left's name
                all_names.remove(name)  # we could even claim the original name
                new_name = mangle_name(rprefix, name, rsuffix)
                # we will not add this column twice when it is the join column
                if new_name != left_on:
                    if new_name in all_names:  # it's still not unique
                        new_name = vaex.utils.find_valid_name(new_name, all_names)
                    right.rename(name, new_name)
                    right_names[right_names.index(name)] = new_name

                # and the same for the left
                all_names = list(set(right_names + left_names))
                all_names.remove(name)
                new_name = mangle_name(lprefix, name, lsuffix)
                if new_name in all_names:  # still not unique
                    new_name = vaex.utils.find_valid_name(new_name, all_names)
                left.rename(name, new_name)
                left_names[left_names.index(name)] = new_name

        # now we add columns from the right, to the left
        right_names = right.get_names(hidden=True)
        left_names = left.get_names(hidden=True)
        for name in right_names:
            column_name = right._column_aliases.get(name, name)
            if name == left_on and name in left_names:
                continue  # skip when it's the join column
            assert name not in left_names
            if name in right.variables:
                left.set_variable(name, right.variables[name])
            elif column_name in right.virtual_columns:
                left.add_virtual_column(name, right.virtual_columns[column_name])
            else:
                if lookup is not None:
                    column = ColumnIndexed.index(right, right.columns[column_name], name, lookup, direct_indices_map, any(lookup_masked))
                else:
                    column = right.columns[name]
                left.add_column(name, column)
        return left

    def export(self, path, column_names=None, byteorder="=", shuffle=False, selection=False, progress=None, virtual=True, sort=None, ascending=True):
        """Exports the DataFrame to a file written with arrow

        :param DataFrameLocal df: DataFrame to export
        :param str path: path for file
        :param lis[str] column_names: list of column names to export or None for all columns
        :param str byteorder: = for native, < for little endian and > for big endian (not supported for fits)
        :param bool shuffle: export rows in random order
        :param bool selection: export selection or not
        :param progress: progress callback that gets a progress fraction as argument and should return True to continue,
                or a default progress bar when progress=True
        :param: bool virtual: When True, export virtual columns
        :param str sort: expression used for sorting the output
        :param bool ascending: sort ascending (True) or descending
        :return:
        """
        if path.endswith('.arrow'):
            self.export_arrow(path, column_names, byteorder, shuffle, selection, progress=progress, virtual=virtual, sort=sort, ascending=ascending)
        elif path.endswith('.hdf5'):
            self.export_hdf5(path, column_names, byteorder, shuffle, selection, progress=progress, virtual=virtual, sort=sort, ascending=ascending)
        elif path.endswith('.fits'):
            self.export_fits(path, column_names, shuffle, selection, progress=progress, virtual=virtual, sort=sort, ascending=ascending)
        elif path.endswith('.parquet'):
            self.export_parquet(path, column_names, shuffle, selection, progress=progress, virtual=virtual, sort=sort, ascending=ascending)
        elif path.endswith('.csv'):
            self.export_csv(path, selection=selection, progress=progress, virtual=virtual)
        else:
            raise ValueError('''Unrecognized file extension. Please use .arrow, .hdf5, .parquet, .fits, or .csv to export to the particular file format.''')

    def export_arrow(self, path, column_names=None, byteorder="=", shuffle=False, selection=False, progress=None, virtual=True, sort=None, ascending=True):
        """Exports the DataFrame to a file written with arrow

        :param DataFrameLocal df: DataFrame to export
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
        import vaex_arrow.export
        vaex_arrow.export.export(self, path, column_names, byteorder, shuffle, selection, progress=progress, virtual=virtual, sort=sort, ascending=ascending)

    def export_parquet(self, path, column_names=None, byteorder="=", shuffle=False, selection=False, progress=None, virtual=True, sort=None, ascending=True):
        """Exports the DataFrame to a parquet file

        :param DataFrameLocal df: DataFrame to export
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
        import vaex_arrow.export
        vaex_arrow.export.export_parquet(self, path, column_names, byteorder, shuffle, selection, progress=progress, virtual=virtual, sort=sort, ascending=ascending)

    def export_hdf5(self, path, column_names=None, byteorder="=", shuffle=False, selection=False, progress=None, virtual=True, sort=None, ascending=True):
        """Exports the DataFrame to a vaex hdf5 file

        :param DataFrameLocal df: DataFrame to export
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

    def export_fits(self, path, column_names=None, shuffle=False, selection=False, progress=None, virtual=True, sort=None, ascending=True):
        """Exports the DataFrame to a fits file that is compatible with TOPCAT colfits format

        :param DataFrameLocal df: DataFrame to export
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

    @docsubst
    def export_csv(self, path, virtual=True, selection=False, progress=None, chunk_size=1_000_000, **kwargs):
        """ Exports the DataFrame to a CSV file.

        :param str path: Path for file
        :param bool virtual: If True, export virtual columns as well
        :param bool selection: {selection1}
        :param progress: {progress}
        :param int chunk_size: {chunk_size_export}
        :param **kwargs: Extra keyword arguments to be passed on pandas.DataFrame.to_csv()
        :return:
        """
        import pandas as pd

        expressions = self.get_column_names(virtual=virtual)
        progressbar = vaex.utils.progressbars(progress)
        dtypes = self[expressions].dtypes
        n_samples = len(self)

        for i1, i2, chunks in self.evaluate_iterator(expressions, chunk_size=chunk_size, selection=selection):
            progressbar( i1 / n_samples)
            chunk_dict = {col: values for col, values in zip(expressions, chunks)}
            chunk_pdf = pd.DataFrame(chunk_dict)

            if i1 == 0:  # Only the 1st chunk should have a header and the rest will be appended
                mode = 'w'
                header = True
            else:
                mode = 'a'
                header = False

            chunk_pdf.to_csv(path_or_buf=path, mode=mode, header=header, index=False, **kwargs)
        progressbar(1.0)
        return

    def _needs_copy(self, column_name):
        import vaex.file.other
        return not \
            ((column_name in self.column_names and not
              isinstance(self.columns[column_name], Column) and not
              isinstance(self.columns[column_name], vaex.file.other.DatasetTap.TapColumn) and
              self.columns[column_name].dtype.type == np.float64 and
              self.columns[column_name].strides[0] == 8 and
              column_name not in
              self.virtual_columns) or self.data_type(column_name) == str_type or self.data_type(column_name).kind == 'S')
        # and False:

    def selected_length(self, selection="default"):
        """The local implementation of :func:`DataFrame.selected_length`"""
        return int(self.count(selection=selection).item())
        # np.sum(self.mask) if self.has_selection() else None

    # def _set_mask(self, mask):
    #     self.mask = mask
    #     self._has_selection = mask is not None
    #     # self.signal_selection_changed.emit(self)

    def groupby(self, by=None, agg=None):
        """Return a :class:`GroupBy` or :class:`DataFrame` object when agg is not None

        Examples:

        >>> import vaex
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> x = np.random.randint(1, 5, 10)
        >>> y = x**2
        >>> df = vaex.from_arrays(x=x, y=y)
        >>> df.groupby(df.x, agg='count')
        #    x    y_count
        0    3          4
        1    4          2
        2    1          3
        3    2          1
        >>> df.groupby(df.x, agg=[vaex.agg.count('y'), vaex.agg.mean('y')])
        #    x    y_count    y_mean
        0    3          4         9
        1    4          2        16
        2    1          3         1
        3    2          1         4
        >>> df.groupby(df.x, agg={'z': [vaex.agg.count('y'), vaex.agg.mean('y')]})
        #    x    z_count    z_mean
        0    3          4         9
        1    4          2        16
        2    1          3         1
        3    2          1         4

        Example using datetime:

        >>> import vaex
        >>> import numpy as np
        >>> t = np.arange('2015-01-01', '2015-02-01', dtype=np.datetime64)
        >>> y = np.arange(len(t))
        >>> df = vaex.from_arrays(t=t, y=y)
        >>> df.groupby(vaex.BinnerTime.per_week(df.t)).agg({'y' : 'sum'})
        #  t                      y
        0  2015-01-01 00:00:00   21
        1  2015-01-08 00:00:00   70
        2  2015-01-15 00:00:00  119
        3  2015-01-22 00:00:00  168
        4  2015-01-29 00:00:00   87


        :param dict, list or agg agg: Aggregate operation in the form of a string, vaex.agg object, a dictionary
            where the keys indicate the target column names, and the values the operations, or the a list of aggregates.
            When not given, it will return the groupby object.
        :return: :class:`DataFrame` or :class:`GroupBy` object.
        """
        from .groupby import GroupBy
        groupby = GroupBy(self, by=by)
        if agg is None:
            return groupby
        else:
            return groupby.agg(agg)

    def binby(self, by=None, agg=None):
        """Return a :class:`BinBy` or :class:`DataArray` object when agg is not None

        The binby operation does not return a 'flat' DataFrame, instead it returns an N-d grid
        in the form of an xarray.


        :param dict, list or agg agg: Aggregate operation in the form of a string, vaex.agg object, a dictionary
            where the keys indicate the target column names, and the values the operations, or the a list of aggregates.
            When not given, it will return the binby object.
        :return: :class:`DataArray` or :class:`BinBy` object.
        """
        from .groupby import BinBy
        binby = BinBy(self, by=by)
        if agg is None:
            return binby
        else:
            return binby.agg(agg)

    def _selection(self, create_selection, name, executor=None, execute_fully=False):
        def create_wrapper(current):
            selection = create_selection(current)
            # only create a mask when we have a selection, so we do not waste memory
            if selection is not None and name not in self._selection_masks:
                self._selection_masks[name] = vaex.superutils.Mask(self._length_unfiltered)
            return selection
        return super()._selection(create_wrapper, name, executor, execute_fully)

class DataFrameConcatenated(DataFrameLocal):
    """Represents a set of DataFrames all concatenated. See :func:`DataFrameLocal.concat` for usage.
    """

    def __init__(self, dfs, name=None):
        super(DataFrameConcatenated, self).__init__(None, None, [])
        # to reduce complexity, we 'extract' the dataframes (i.e. remove filter)
        self.dfs = dfs = [df.extract() for df in dfs]
        self.name = name or "-".join(df.name for df in self.dfs)
        self.path = "-".join(df.path for df in self.dfs)
        first, tail = dfs[0], dfs[1:]
        for column_name in first.get_column_names(virtual=False, hidden=True, alias=False):
            if all([column_name in df.get_column_names(virtual=False, hidden=True, alias=False) for df in tail]):
                self.column_names.append(column_name)
        self.columns = {}
        for column_name in self.get_column_names(virtual=False, hidden=True, alias=False):
            self.columns[column_name] = ColumnConcatenatedLazy([df[column_name] for df in dfs])
            self._save_assign_expression(column_name)

        for name in list(first.virtual_columns.keys()):
            if all([first.virtual_columns[name] == df.virtual_columns.get(name, None) for df in tail]):
                self.add_virtual_column(name, first.virtual_columns[name])
            else:
                self.columns[name] = ColumnConcatenatedLazy([df[name] for df in dfs])
                self.column_names.append(name)
            self._save_assign_expression(name)

        for df in tail:
            if first._column_aliases != df._column_aliases:
                raise ValueError(f'Concatenating dataframes where different column aliases not supported: {first._column_aliases} != {df._column_aliases}')
        self._column_aliases = first._column_aliases.copy()

        for df in dfs[:1]:
            for name, value in list(df.variables.items()):
                if name not in self.variables:
                    self.set_variable(name, value, write=False)
        # self.write_virtual_meta()

        self._length_unfiltered = sum(len(ds) for ds in self.dfs)
        self._length_original = self._length_unfiltered
        self._index_end = self._length_unfiltered

    def is_masked(self, column):
        column = _ensure_string_from_expression(column)
        if column in self.columns:
            return self.columns[column].is_masked
        else:
            ar = self.evaluate(column, i1=0, i2=1, parallel=False)
            if isinstance(ar, np.ndarray) and np.ma.isMaskedArray(ar):
                return True
        return False


def _is_dtype_ok(dtype):
    return dtype.type in [np.bool_, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16,
                          np.uint32, np.uint64, np.float32, np.float64, np.datetime64] or\
        dtype.type == np.string_ or dtype.type == np.unicode_


def _is_array_type_ok(array):
    return _is_dtype_ok(array.dtype)


class DataFrameArrays(DataFrameLocal):
    """Represent an in-memory DataFrame of numpy arrays, see :func:`from_arrays` for usage."""

    def __init__(self, name="arrays"):
        super(DataFrameArrays, self).__init__(None, None, [])
        self.name = name
        self.path = "/has/no/path/" + name

    # def __len__(self):
    #   return len(self.columns.values()[0])

    def add_column(self, name, data, dtype=None):
        """Add a column to the DataFrame

        :param str name: name of column
        :param data: numpy array with the data
        """
        # assert _is_array_type_ok(data), "dtype not supported: %r, %r" % (data.dtype, data.dtype.type)
        # self._length = len(data)
        # if self._length_unfiltered is None:
        #     self._length_unfiltered = len(data)
        #     self._length_original = len(data)
        #     self._index_end = self._length_unfiltered
        super(DataFrameArrays, self).add_column(name, data, dtype=dtype)
        self._length_unfiltered = int(round(self._length_original * self._active_fraction))
        # self.set_active_fraction(self._active_fraction)

    @property
    def values(self):
        """Gives a full memory copy of the DataFrame into a 2d numpy array of shape (n_rows, n_columns).
        Note that the memory order is fortran, so all values of 1 column are contiguous in memory for performance reasons.

        Note this returns the same result as:

        >>> np.array(ds)

        If any of the columns contain masked arrays, the masks are ignored (i.e. the masked elements are returned as well).
        """
        return self.__array__()
