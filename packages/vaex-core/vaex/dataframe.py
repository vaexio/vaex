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
        main_executor = vaex.execution.Executor(vaex.multithreading.get_main_pool())
    return main_executor


# we import after function_mapping is defined
from .expression import Expression


_doc_snippets = {}
_doc_snippets["expression"] = "expression or list of expressions, e.g. 'x', or ['x, 'y']"
_doc_snippets["expression_single"] = "if previous argument is not a list, this argument should be given"
_doc_snippets["binby"] = "List of expressions for constructing a binned grid"
_doc_snippets["limits"] = """description for the min and max values for the expressions, e.g. 'minmax', '99.7%', [0, 10], or a list of, e.g. [[0, 10], [0, 20], 'minmax']"""
_doc_snippets["shape"] = """shape for the array where the statistic is calculated on, if only an integer is given, it is used for all dimensions, e.g. shape=128, shape=[128, 256]"""
_doc_snippets["percentile_limits"] = """description for the min and max values to use for the cumulative histogram, should currently only be 'minmax'"""
_doc_snippets["percentile_shape"] = """shape for the array where the cumulative histogram is calculated on, integer type"""
_doc_snippets["selection"] = """Name of selection to use (or True for the 'default'), or all the data (when selection is None or False), or a list of selections"""
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
_doc_snippets['note_copy'] = '.. note:: Note that no copy of the underlying data is made, only a view/reference is make.'
_doc_snippets['note_filter'] = '.. note:: Note that filtering will be ignored (since they may change), you may want to consider running :meth:`extract` first.'
_doc_snippets['inplace'] = 'Make modifications to self or return a new DataFrame'
_doc_snippets['return_shallow_copy'] = 'Returns a new DataFrame with a shallow copy/view of the underlying data'
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

        self.variables = collections.OrderedDict()
        self.variables["pi"] = np.pi
        self.variables["e"] = np.e
        self.variables["km_in_au"] = 149597870700 / 1000.
        self.variables["seconds_per_year"] = 31557600
        # leads to k = 4.74047 to go from au/year to km/s
        self.virtual_columns = collections.OrderedDict()
        self.functions = collections.OrderedDict()
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

        self.favorite_selections = collections.OrderedDict()

        self.mask = None  # a bitmask for the selection does not work for server side

        # maps from name to list of Selection objets
        self.selection_histories = collections.defaultdict(list)
        # after an undo, the last one in the history list is not the active one, -1 means no selection
        self.selection_history_indices = collections.defaultdict(lambda: -1)
        assert self.filtered is False
        self._auto_fraction = False

        self._sparse_matrices = {}  # record which sparse columns belong to which sparse matrix

        self._categories = collections.OrderedDict()
        self._selection_mask_caches = collections.defaultdict(dict)
        self._selection_masks = {}  # maps to vaex.superutils.Mask object
        self._renamed_columns = []

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
        dtype = self.dtype(expression)
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

    def execute(self):
        '''Execute all delayed jobs.'''
        self.executor.execute()
        self._task_aggs.clear()

    @property
    def filtered(self):
        return self.has_selection(FILTER_SELECTION_NAME)

    def map_reduce(self, map, reduce, arguments, progress=False, delay=False, info=False, ordered_reduce=False, to_numpy=True, name='map reduce (custom)'):
        # def map_wrapper(*blocks):
        task = tasks.TaskMapReduce(self, arguments, map, reduce, info=info, ordered_reduce=ordered_reduce, to_numpy=to_numpy)
        progressbar = vaex.utils.progressbars(progress)
        progressbar.add_task(task, name)
        self.executor.schedule(task)
        return self._delay(delay, task)

    def apply(self, f, arguments=None, dtype=None, delay=False, vectorize=False):
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

    def _set(self, expression, progress=False, delay=False):
        column = _ensure_string_from_expression(expression)
        columns = [column]
        from .hash import ordered_set_type_from_dtype
        from vaex.column import _to_string_sequence

        transient = self[str(expression)].transient or self.filtered or self.is_masked(expression)
        if self.dtype(expression) == str_type and not transient:
            # string is a special case, only ColumnString are not transient
            ar = self.columns[str(expression)]
            if not isinstance(ar, ColumnString):
                transient = True

        dtype = self.dtype(column)
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
        self.map_reduce(map, reduce, columns, delay=delay, name='set', info=True, to_numpy=False)
        sets = [k for k in sets if k is not None]
        set0 = sets[0]
        for other in sets[1:]:
            set0.merge(other)
        return set0

    def unique(self, expression, return_inverse=False, progress=False, delay=False):
        expression = _ensure_string_from_expression(expression)
        ordered_set = self._set(expression, progress=progress)
        transient = True
        if return_inverse:
            # inverse type can be smaller, depending on length of set
            inverse = np.zeros(self._length_unfiltered, dtype=np.int64)
            dtype = self.dtype(expression)
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
            self.map_reduce(map, reduce, [expression], delay=delay, name='unique_return_inverse', info=True, to_numpy=False)
        keys = ordered_set.keys()
        if ordered_set.has_nan:
            keys = [np.nan] + keys
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

    @delayed
    def _old_count_calculation(self, expression, binby, limits, shape, selection, edges, progressbar):
        if shape:
            limits, shapes = limits
        else:
            limits, shapes = limits, shape
        # print(limits, shapes)
        if expression in ["*", None]:
            task = tasks.TaskStatistic(self, binby, shapes, limits, op=tasks.OP_ADD1, selection=selection, edges=edges)
        else:
            task = tasks.TaskStatistic(self, binby, shapes, limits, weight=expression, op=tasks.OP_COUNT, selection=selection, edges=edges)
        self.executor.schedule(task)
        progressbar.add_task(task, "count for %s" % expression)
        @delayed
        def finish(counts):
            counts = np.array(counts)
            counts = counts[...,0]
            return counts
        return finish(task)

    @docsubst
    def _old_count(self, expression=None, binby=[], limits=None, shape=default_shape, selection=False, delay=False, edges=False, progress=None):
        logger.debug("count(%r, binby=%r, limits=%r)", expression, binby, limits)
        logger.debug("count(%r, binby=%r, limits=%r)", expression, binby, limits)
        expression = _ensure_string_from_expression(expression)
        binby = _ensure_strings_from_expressions(binby)
        waslist, [expressions,] = vaex.utils.listify(expression)
        @delayed
        def finish(*counts):
           return vaex.utils.unlistify(waslist, counts)
        progressbar = vaex.utils.progressbars(progress)
        limits = self.limits(binby, limits, delay=True, shape=shape)
        stats = [self._old_count_calculation(expression, binby=binby, limits=limits, shape=shape, selection=selection, edges=edges, progressbar=progressbar) for expression in expressions]
        var = finish(*stats)
        return self._delay(delay, var)

    @delayed
    def _count_calculation(self, expression, grid, selection, edges, progressbar):
        if expression in ["*", None]:
            agg = vaex.agg.count()
        else:
            agg = vaex.agg.count(expression)
        task = self._get_task_agg(grid)
        agg_subtask = task.add_aggregation_operation(agg, selection, edges=edges)
        progressbar.add_task(task, "count for %s" % expression)
        @delayed
        def finish(counts):
            counts = np.asarray(counts)
            return counts
        return finish(agg_subtask)

    def _compute_agg(self, name, expression, binby=[], limits=None, shape=default_shape, selection=False, delay=False, edges=False, progress=None, extra_expressions=None):
        logger.debug("aggregate %s(%r, binby=%r, limits=%r)", name, expression, binby, limits)
        expression = _ensure_strings_from_expressions(expression)
        if extra_expressions:
            extra_expressions = _ensure_strings_from_expressions(extra_expressions)
        expression_waslist, [expressions,] = vaex.utils.listify(expression)
        grid = self._create_grid(binby, limits, shape, delay=True)
        @delayed
        def compute(expression, grid, selection, edges, progressbar):
            if expression in ["*", None]:
                agg = vaex.agg.aggregates[name]()
            else:
                if extra_expressions:
                    agg = vaex.agg.aggregates[name](expression, *extra_expressions)
                else:
                    agg = vaex.agg.aggregates[name](expression)
            task = self._get_task_agg(grid)
            agg_subtask = agg.add_operations(task, selection=selection, edges=edges)
            progressbar.add_task(task, "%s for %s" % (name, expression))
            @delayed
            def finish(counts):
                counts = np.asarray(counts)
                return counts
            return finish(agg_subtask)
        @delayed
        def finish(*counts):
            return np.asarray(vaex.utils.unlistify(expression_waslist, counts))
        progressbar = vaex.utils.progressbars(progress)
        stats = [compute(expression, grid, selection=selection, edges=edges, progressbar=progressbar) for expression in expressions]
        var = finish(*stats)
        return self._delay(delay, var)

    @docsubst
    def count(self, expression=None, binby=[], limits=None, shape=default_shape, selection=False, delay=False, edges=False, progress=None):
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
        :return: {return_stat_scalar}
        """
        return self._compute_agg('count', expression, binby, limits, shape, selection, delay, edges, progress)

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
    def first(self, expression, order_expression, binby=[], limits=None, shape=default_shape, selection=False, delay=False, edges=False, progress=None):
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
        :return: Ndarray containing the first elements.
        :rtype: numpy.array
        """
        return self._compute_agg('first', expression, binby, limits, shape, selection, delay, edges, progress, extra_expressions=[order_expression])
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
    def mean(self, expression, binby=[], limits=None, shape=default_shape, selection=False, delay=False, progress=None, edges=False):
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
        :return: {return_stat_scalar}
        """
        return self._compute_agg('mean', expression, binby, limits, shape, selection, delay, edges, progress)
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
    def sum(self, expression, binby=[], limits=None, shape=default_shape, selection=False, delay=False, progress=None, edges=False):
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
        :return: {return_stat_scalar}
        """
        return self._compute_agg('sum', expression, binby, limits, shape, selection, delay, edges, progress)
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
    def std(self, expression, binby=[], limits=None, shape=default_shape, selection=False, delay=False, progress=None):
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
        :return: {return_stat_scalar}
        """
        @delayed
        def finish(var):
            return var**0.5
        return self._delay(delay, finish(self.var(expression, binby=binby, limits=limits, shape=shape, selection=selection, delay=True, progress=progress)))

    @docsubst
    @stat_1d
    def var(self, expression, binby=[], limits=None, shape=default_shape, selection=False, delay=False, progress=None):
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
        :return: {return_stat_scalar}
        """
        edges = False
        return self._compute_agg('var', expression, binby, limits, shape, selection, delay, edges, progress)
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
        """Calculate the covariance cov[x,y] between and x and y, possibly on a grid defined by binby.

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
        """Calculate the correlation coefficient cov[x,y]/(std[x]*std[y]) between and x and y, possibly on a grid defined by binby.

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

        Either x and y are expressions, e.g:

        >>> df.cov("x", "y")

        Or only the x argument is given with a list of expressions, e,g.:

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
        dtypes = [self.dtype(expr) for expr in expressions]
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
    def min(self, expression, binby=[], limits=None, shape=default_shape, selection=False, delay=False, progress=None, edges=False):
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
        :return: {return_stat_scalar}, the last dimension is of shape (2)
        """
        return self._compute_agg('min', expression, binby, limits, shape, selection, delay, edges, progress)
        @delayed
        def finish(result):
            return result[..., 0]
        return self._delay(delay, finish(self.minmax(expression, binby=binby, limits=limits, shape=shape, selection=selection, delay=delay, progress=progress)))

    @docsubst
    @stat_1d
    def max(self, expression, binby=[], limits=None, shape=default_shape, selection=False, delay=False, progress=None, edges=False):
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
        :return: {return_stat_scalar}, the last dimension is of shape (2)
        """
        return self._compute_agg('max', expression, binby, limits, shape, selection, delay, edges, progress)
        @delayed
        def finish(result):
            return result[..., 1]
        return self._delay(delay, finish(self.minmax(expression, binby=binby, limits=limits, shape=shape, selection=selection, delay=delay, progress=progress)))

    @docsubst
    @stat_1d
    def median_approx(self, expression, percentage=50., binby=[], limits=None, shape=default_shape, percentile_shape=256, percentile_limits="minmax", selection=False, delay=False):
        """Calculate the median , possibly on a grid defined by binby.

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
                values = np.array((totalcounts + 1) * percentage / 100.)  # make sure it's an ndarray
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
                results.append(x)

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
        if delay:
            return task
        else:
            self.execute()
            return task.get()

    @docsubst
    def limits_percentage(self, expression, percentage=99.73, square=False, delay=False):
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
        # percentiles = self.percentile(expression, [100-percentage/2, 100-(100-percentage/2.)], delay=True)
        # return self._delay(delay, percentiles)
        # print(percentage)
        import scipy
        logger.info("limits_percentage for %r, with percentage=%r", expression, percentage)
        waslist, [expressions, ] = vaex.utils.listify(expression)
        limits = []
        for expr in expressions:
            subspace = self(expr)
            limits_minmax = subspace.minmax()
            vmin, vmax = limits_minmax[0]
            size = 1024 * 16
            counts = subspace.histogram(size=size, limits=limits_minmax)
            cumcounts = np.concatenate([[0], np.cumsum(counts)])
            cumcounts /= cumcounts.max()
            # TODO: this is crude.. see the details!
            f = (1 - percentage / 100.) / 2
            x = np.linspace(vmin, vmax, size + 1)
            l = scipy.interp([f, 1 - f], cumcounts, x)
            limits.append(l)
        # return limits
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
            size = 1024 * 16
            counts = subspace.histogram(size=size, limits=limits_minmax)
            cumcounts = np.concatenate([[0], np.cumsum(counts)])
            cumcounts /= cumcounts.max()
            # TODO: this is crude.. see the details!
            f = percentage / 100.
            x = np.linspace(vmin, vmax, size + 1)
            l = scipy.interp([f], cumcounts, x)
            values.append(l[0])
        return vaex.utils.unlistify(waslist, values)

    @docsubst
    def limits(self, expression, value=None, square=False, selection=None, delay=False, shape=None):
        """Calculate the [min, max] range for expression, as described by value, which is '99.7%' by default.

        If value is a list of the form [minvalue, maxvalue], it is simply returned, this is for convenience when using mixed
        forms.

        Example:

        >>> df.limits("x")
        array([-28.86381927,  28.9261226 ])
        >>> df.limits(["x", "y"])
        (array([-28.86381927,  28.9261226 ]), array([-28.60476934,  28.96535249]))
        >>> df.limits(["x", "y"], "minmax")
        (array([-128.293991,  271.365997]), array([ -71.5523682,  146.465836 ]))
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
        # values =
        # values = _expand_limits(value, len(expressions))
        # logger.debug("limits %r", list(zip(expressions, values)))
        if value is None:
            value = "99.73%"
        # print("value is seq/limit?", _issequence(value), _is_limit(value), value)
        if _is_limit(value) or not _issequence(value):
            values = (value,) * len(expressions)
        else:
            values = value

        # print("expressions 1)", expressions)
        # print("values      1)", values)
        initial_expressions, initial_values = expressions, values
        expression_values = dict()
        expression_shapes = dict()
        for i, (expression, value) in enumerate(zip(expressions, values)):
            # print(">>>", expression, value)
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
            # print("expressions 2)", expressions)
            # print("values      2)", values)
            for j, (expression, value) in enumerate(zip(expressions, values)):
                if shape is not None:
                    if _issequence(shape):
                        shapes = shape
                    else:
                        shapes = (shape, ) * (len(expressions) if nested else len(initial_expressions))

                shape_index = j if nested else i

                if not _is_limit(value):  # if a
                    # value = tuple(value) # list is not hashable
                    expression_values[(expression, value)] = None
                if self.is_category(expression):
                    N = self._categories[_ensure_string_from_expression(expression)]['N']
                    expression_shapes[expression] = min(N, shapes[shape_index] if shape is not None else default_shape)
                else:
                    expression_shapes[expression] = shapes[shape_index] if shape is not None else default_shape

        # print("##### 1)", expression_values.keys())

        limits_list = []
        # for expression, value in zip(expressions, values):
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
                                limits = self.limits_percentage(expression, number, delay=False)
                            elif type in ["%s", "%square", "percentsquare"]:
                                limits = self.limits_percentage(expression, number, square=True, delay=True)
                elif value is None:
                    limits = self.limits_percentage(expression, square=square, delay=True)
                else:
                    limits = value
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

    def plot_widget(self, x, y, z=None, grid=None, shape=256, limits=None, what="count(*)", figsize=None,
                    f="identity", figure_key=None, fig=None, axes=None, xlabel=None, ylabel=None, title=None,
                    show=True, selection=[None, True], colormap="afmhot", grid_limits=None, normalize="normalize",
                    grid_before=None,
                    what_kwargs={}, type="default",
                    scales=None, tool_select=False, bq_cleanup=True,
                    backend="bqplot",
                    **kwargs):
        """Viz 1d, 2d or 3d in a Jupyter notebook

        .. note::
            This API is not fully settled and may change in the future

        Example:

        >>> df.plot_widget(df.x, df.y, backend='bqplot')
        >>> df.plot_widget(df.pickup_longitude, df.pickup_latitude, backend='ipyleaflet')

        :param backend: Widget backend to use: 'bqplot', 'ipyleaflet', 'ipyvolume', 'matplotlib'

        """
        import vaex.jupyter.plot
        backend = vaex.jupyter.plot.create_backend(backend)
        cls = vaex.jupyter.plot.get_type(type)
        x = _ensure_strings_from_expressions(x)
        y = _ensure_strings_from_expressions(y)
        z = _ensure_strings_from_expressions(z)
        for name in 'vx vy vz'.split():
            if name in kwargs:
                kwargs[name] = _ensure_strings_from_expressions(kwargs[name])
        plot2d = cls(backend=backend, dataset=self, x=x, y=y, z=z, grid=grid, shape=shape, limits=limits, what=what,
                     f=f, figure_key=figure_key, fig=fig,
                     selection=selection, grid_before=grid_before,
                     grid_limits=grid_limits, normalize=normalize, colormap=colormap, what_kwargs=what_kwargs, **kwargs)
        if show:
            plot2d.show()
        return plot2d

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

    # """Use bqplot to create an interactive plot, this method is subject to change, it is currently a tech demo"""
        # subspace = self(x, y)
        # return subspace.plot_bq(grid, size, limits, square, center, weight, figsize, aspect, f, fig, axes, xlabel, ylabel, title,
        #                       group_by, group_limits, group_colors, group_labels, group_count, cmap, scales, tool_select, bq_cleanup, **kwargs)

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

        Columns can be accesed by there names, which are attributes. The attribues are currently expressions, so you can
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
            dtype = self.dtype(column)
            dtype_internal = self.dtype(column, internal=True)
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

    def dtype(self, expression, internal=False):
        """Return the numpy dtype for the given expression, if not a column, the first row will be evaluated to get the dtype."""
        expression = _ensure_string_from_expression(expression)
        if expression in self._dtypes_override:
            return self._dtypes_override[expression]
        if expression in self.variables:
            return np.float64(1).dtype
        elif expression in self.columns.keys():
            column = self.columns[expression]
            data = column[0:1]
            dtype = data.dtype
        else:
            data = self.evaluate(expression, 0, 1, filtered=False)
            dtype = data.dtype
        if not internal:
            if dtype != str_type:
                if dtype.kind in 'US':
                    return str_type
                if dtype.kind == 'O':
                    # we lie about arrays containing strings
                    if isinstance(data[0], six.string_types):
                        return str_type
        return dtype

    @property
    def dtypes(self):
        """Gives a Pandas series object containing all numpy dtypes of all columns (except hidden)."""
        from pandas import Series
        return Series({column_name:self.dtype(column_name) for column_name in self.get_column_names()})

    def is_masked(self, column):
        '''Return if a column is a masked (numpy.ma) column.'''
        column = _ensure_string_from_expression(column)
        if column in self.columns:
            return np.ma.isMaskedArray(self.columns[column])
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
                     active_range=[self._index_start, self._index_end])
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
            self.virtual_columns = collections.OrderedDict()
            for name, value in state['virtual_columns'].items():
                self[name] = self._expr(value)
                # self._save_assign_expression(name)
            self.column_names = list(state['column_names'])
        else:
            # old behaviour
            self.virtual_columns = collections.OrderedDict()
            for name, value in state['virtual_columns'].items():
                self[name] = self._expr(value)
        self.variables = state['variables']
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
        """Load a state previously stored by :meth:`DataFrame.state_store`, see also :meth:`DataFrame.state_set`."""
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

    @_hidden
    def subspace(self, *expressions, **kwargs):
        """Return a :class:`Subspace` for this DataFrame with the given expressions:

        Example:

        >>> subspace_xy = some_df("x", "y")

        :rtype: Subspace
        :param list[str] expressions: list of expressions
        :param kwargs:
        :return:
        """
        return self(*expressions, **kwargs)

    @_hidden
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
        import vaex.legacy
        return vaex.legacy.Subspaces([self(*expressions, **kwargs) for expressions in expressions_list])

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

    @vaex.utils.deprecated('legacy system')
    @_hidden
    def __call__(self, *expressions, **kwargs):
        """Alias/shortcut for :func:`DataFrame.subspace`"""
        raise NotImplementedError

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

    def _evaluate_selection_mask(self, name="default", i1=None, i2=None, selection=None, cache=False):
        """Internal use, ignores the filter"""
        i1 = i1 or 0
        i2 = i2 or len(self)
        scope = scopes._BlockScopeSelection(self, i1, i2, selection, cache=cache)
        return scope.evaluate(name)

    def evaluate_selection_mask(self, name="default", i1=None, i2=None, selection=None, cache=False):
        i1 = i1 or 0
        i2 = i2 or self.length_unfiltered()
        if name in [None, False] and self.filtered:
            scope_global = scopes._BlockScopeSelection(self, i1, i2, None, cache=cache)
            mask_global = scope_global.evaluate(FILTER_SELECTION_NAME)
            return mask_global
        elif self.filtered and name != FILTER_SELECTION_NAME:
            scope = scopes._BlockScopeSelection(self, i1, i2, selection)
            scope_global = scopes._BlockScopeSelection(self, i1, i2, None, cache=cache)
            mask = scope.evaluate(name)
            mask_global = scope_global.evaluate(FILTER_SELECTION_NAME)
            return mask & mask_global
        else:
            scope = scopes._BlockScopeSelection(self, i1, i2, selection, cache=cache)
            return scope.evaluate(name)

        # if _is_string(selection):

    def evaluate(self, expression, i1=None, i2=None, out=None, selection=None):
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
        raise NotImplementedError

    @docsubst
    def to_items(self, column_names=None, selection=None, strings=True, virtual=False):
        """Return a list of [(column_name, ndarray), ...)] pairs where the ndarray corresponds to the evaluated data

        :param column_names: list of column names, to export, when None DataFrame.get_column_names(strings=strings, virtual=virtual) is used
        :param selection: {selection}
        :param strings: argument passed to DataFrame.get_column_names when column_names is None
        :param virtual: argument passed to DataFrame.get_column_names when column_names is None
        :return: list of (name, ndarray) pairs
        """
        items = []
        for name in column_names or self.get_column_names(strings=strings, virtual=virtual):
            items.append((name, self.evaluate(name, selection=selection)))
        return items

    @docsubst
    def to_dict(self, column_names=None, selection=None, strings=True, virtual=False):
        """Return a dict containing the ndarray corresponding to the evaluated data

        :param column_names: list of column names, to export, when None DataFrame.get_column_names(strings=strings, virtual=virtual) is used
        :param selection: {selection}
        :param strings: argument passed to DataFrame.get_column_names when column_names is None
        :param virtual: argument passed to DataFrame.get_column_names when column_names is None
        :return: dict
        """
        return dict(self.to_items(column_names=column_names, selection=selection, strings=strings, virtual=virtual))

    @docsubst
    def to_copy(self, column_names=None, selection=None, strings=True, virtual=False, selections=True):
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
        self.description = other.description

    @docsubst
    def to_pandas_df(self, column_names=None, selection=None, strings=True, virtual=False, index_name=None):
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
    def to_arrow_table(self, column_names=None, selection=None, strings=True, virtual=False):
        """Returns an arrow Table object containing the arrays corresponding to the evaluated data

        :param column_names: list of column names, to export, when None DataFrame.get_column_names(strings=strings, virtual=virtual) is used
        :param selection: {selection}
        :param strings: argument passed to DataFrame.get_column_names when column_names is None
        :param virtual: argument passed to DataFrame.get_column_names when column_names is None
        :return: pyarrow.Table object
        """
        from vaex_arrow.convert import arrow_table_from_vaex_df
        return arrow_table_from_vaex_df(self, column_names, selection, strings, virtual)

    @docsubst
    def to_astropy_table(self, column_names=None, selection=None, strings=True, virtual=False, index=None):
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
        for name, data in self.to_items(column_names=column_names, selection=selection, strings=strings, virtual=virtual):
            if self.dtype(name) == str_type:  # for astropy we convert it to unicode, it seems to ignore object type
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

    def validate_expression(self, expression):
        """Validate an expression (may throw Exceptions)"""
        # return self.evaluate(expression, 0, 2)
        vars = set(self.get_column_names()) | set(self.variables.keys())
        funcs = set(expression_namespace.keys())
        return vaex.expresso.validate_expression(expression, vars, funcs)

    def _block_scope(self, i1, i2):
        variables = {key: self.evaluate_variable(key) for key in self.variables.keys()}
        return scopes._BlockScope(self, i1, i2, **variables)

    def select(self, boolean_expression, mode="replace", name="default"):
        """Select rows based on the boolean_expression, if there was a previous selection, the mode is taken into account.

        if boolean_expression is None, remove the selection, has_selection() will returns false

        Note that per DataFrame, only one selection is possible.

        :param str boolean_expression: boolean expression, such as 'x < 0', '(x < 0) || (y > -10)' or None to remove the selection
        :param str mode: boolean operation to perform with the previous selection, "replace", "and", "or", "xor", "subtract"
        :return: None
        """
        raise NotImplementedError

    def add_column(self, name, f_or_array, dtype=None):
        """Add an in memory array as a column."""
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
            self.columns[name] = f_or_array
            if name not in self.column_names:
                self.column_names.append(name)
            ar = f_or_array
            if dtype is not None:
                self._dtypes_override[name] = dtype
            else:
                if isinstance(ar, np.ndarray) and ar.dtype.kind == 'O':
                    types = list({type(k) for k in ar if k == k and k is not None})
                    if len(types) == 1 and issubclass(types[0], six.string_types):
                        self._dtypes_override[name] = str_type
        else:
            raise ValueError("functions not yet implemented")
        self._save_assign_expression(name, Expression(self, name))

    def _sparse_matrix(self, column):
        column = _ensure_string_from_expression(column)
        return self._sparse_matrices.get(column)

    def add_columns(self, names, columns):
        from scipy.sparse import csc_matrix, csr_matrix
        if isinstance(columns, csr_matrix):
            if len(names) != columns.shape[1]:
                raise ValueError('number of columns ({}) does not match number of column names ({})'.format(columns.shape[1], len(names)))
            for i, name in enumerate(names):
                self.columns[name] = ColumnSparse(columns, i)
                self.column_names.append(name)
                self._sparse_matrices[name] = columns
                self._save_assign_expression(name, Expression(self, name))
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

    def rename_column(self, name, new_name, unique=False, store_in_state=True):
        """Renames a column, not this is only the in memory name, this will not be reflected on disk"""
        new_name = vaex.utils.find_valid_name(new_name, used=[] if not unique else list(self))
        data = self.columns.get(name)
        if data is not None:
            del self.columns[name]
            self.column_names[self.column_names.index(name)] = new_name
            self.columns[new_name] = data
        else:
            expression = self.virtual_columns[name]
            del self.virtual_columns[name]
            self.virtual_columns[new_name] = expression
        if store_in_state:
            self._renamed_columns.append((name, new_name))
        for d in [self.ucds, self.units, self.descriptions]:
            if name in d:
                d[new_name] = d[name]
                del d[name]
        return new_name

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
        expression = _ensure_string_from_expression(expression)
        if name in self.get_column_names():
            renamed = '__' +vaex.utils.find_valid_name(name, used=self.get_column_names())
            expression = self._rename(name, renamed, expression)[0].expression

        name = vaex.utils.find_valid_name(name, used=[] if not unique else self.get_column_names())
        self.virtual_columns[name] = expression
        self.column_names.append(name)
        self._save_assign_expression(name)
        self.signal_column_changed.emit(self, name, "add")
        # self.write_virtual_meta()

    def _rename(self, old, new, *expressions):
        #for name, expr in self.virtual_columns.items():
        if old in self.columns:
            self.columns[new] = self.columns.pop(old)
        if old in self.virtual_columns:
            self.virtual_columns[new] = self.virtual_columns.pop(old)
        self._renamed_columns.append((old, new))
        index = self.column_names.index(old)
        self.column_names[index] = new
        self.virtual_columns = {k:self[v]._rename(old, new).expression for k, v in self.virtual_columns.items()}
        for key, value in self.selection_histories.items():
            self.selection_histories[key] = list([k if k is None else k._rename(self, old, new) for k in value])
        return [self[_ensure_string_from_expression(e)]._rename(old, new) for e in expressions]


    def delete_virtual_column(self, name):
        """Deletes a virtual column from a DataFrame."""
        del self.virtual_columns[name]
        self.signal_column_changed.emit(self, name, "delete")
        # self.write_virtual_meta()

    def add_variable(self, name, expression, overwrite=True, unique=True):
        """Add a variable to to a DataFrame.

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
            parts += ["<tr>"]
            parts += ["<td>%s</td>" % name]
            virtual = name not in self.column_names
            if name in self.column_names:
                dtype = str(self.dtype(name)) if self.dtype(name) != str else 'str'
            else:
                dtype = "</i>virtual column</i>"
            parts += ["<td>%s</td>" % dtype]
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
            dtype = str(self.dtype(feature)) if self.dtype(feature) != str else 'str'
            if self.dtype(feature) == str_type or self.dtype(feature).kind in ['S', 'U', 'O']:
                count = self.count(feature, selection=selection, delay=True)
                self.execute()
                count = count.get()
                columns[feature] = ((dtype, count, N-count, '--', '--', '--', '--'))
            else:
                count = self.count(feature, selection=selection, delay=True)
                mean = self.mean(feature, selection=selection, delay=True)
                std = self.std(feature, selection=selection, delay=True)
                minmax = self.minmax(feature, selection=selection, delay=True)
                self.execute()
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

        column_names = self.get_column_names()
        values_list = []
        values_list.append(['#', []])
        # parts += ["<thead><tr>"]
        for name in column_names:
            values_list.append([name, []])
            # parts += ["<th>%s</th>" % name]
        # parts += ["</tr></thead>"]

        def table_part(k1, k2, parts):
            values = {}
            N = k2 - k1
            # slicing will invoke .extract which will make the evaluation
            # much quicker
            df = self[k1:k2]
            for i, name in enumerate(column_names):
                try:
                    values[name] = df.evaluate(name)
                except:
                    values[name] = ["error"] * (N)
                    logger.exception('error evaluating: %s at rows %i-%i' % (name, k1, k2))
                # values_list[i].append(value)
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
        parts = table_part(i1, i2, parts)
        if j1 is not None and j2 is not None:
            values_list[0][1].append('...')
            for i in range(len(column_names)):
                # parts += ["<td>...</td>"]
               values_list[i+1][1].append('...')

            # parts = table_part(j1, j2, parts)
            table_part(j1, j2, parts)
        # parts += "</table>"
        # html = "".join(parts)
        # return html
        values_list = dict(values_list)
        # print(values_list)
        import tabulate
        return tabulate.tabulate(values_list, headers="keys", tablefmt=format)

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

    def __repr__(self):
        return self._head_and_tail_table(format='plain')

    def __current_sequence_index(self):
        """TODO"""
        return 0

    def has_current_row(self):
        """Returns True/False is there currently is a picked row."""
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

    def column_count(self):
        """Returns the number of columns (including virtual columns)."""
        return len(self.column_names)

    def get_column_names(self, virtual=True, strings=True, hidden=False, regex=None):
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
        :rtype: list of str

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
        """
        def column_filter(name):
            '''Return True if column with specified name should be returned'''
            if regex and not re.match(regex, name):
                return False
            if not virtual and name in self.virtual_columns:
                return False
            if not strings and (self.dtype(name) == str_type or self.dtype(name).type == np.string_):
                return False
            if not hidden and name.startswith('__'):
                return False
            return True
        return [name for name in self.column_names if column_filter(name)]

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
        for name in df.get_column_names(hidden=True):
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
    def take(self, indices, unfiltered=False):
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
        :param unfiltered: (for internal use) The indices refer to the unfiltered data.
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
        if df.filtered and not unfiltered:
            df.count() # make sure the mask is filled
            # translate the indices to unfiltered indices
            max_index = indices.max()
            mask = df._selection_masks[FILTER_SELECTION_NAME]
            filtered_indices = mask.first(max_index+1)
            indices = filtered_indices[indices]
        for name, column in df.columns.items():
            if column is not None:
                # we optimize this somewhere, so we don't do multiple
                # levels of indirection
                if isinstance(column, ColumnIndexed):
                    # TODO: think about what happpens when the indices are masked.. ?
                    if id(column.indices) not in direct_indices_map:
                        direct_indices = column.indices[indices]
                        direct_indices_map[id(column.indices)] = direct_indices
                    else:
                        direct_indices = direct_indices_map[id(column.indices)]
                    df.columns[name] = ColumnIndexed(column.df, direct_indices, column.name)
                else:
                    df.columns[name] = ColumnIndexed(df_trimmed, indices, name)
        df._length_original = len(indices)
        df._length_unfiltered = df._length_original
        df._cached_filtered_length = None
        df._index_start = 0
        df._index_end = df._length_original
        df.set_selection(None, name=FILTER_SELECTION_NAME)
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
            return self.take(indices, unfiltered=True)
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
        values = self.evaluate(by)
        indices = np.argsort(values, kind=kind)
        if not ascending:
            indices = indices[::-1].copy()  # this may be used a lot, so copy for performance
        return self.take(indices)

    @docsubst
    def fillna(self, value, fill_nan=True, fill_masked=True, column_names=None, prefix='__original_', inplace=False):
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
            if column is not None:
                new_name = df.rename_column(name, prefix + name)
                expr = df[new_name]
                df[name] = df.func.fillna(expr, value, fill_nan=fill_nan, fill_masked=fill_masked)
            else:
                df[name] = df.func.fillna(df[name], value, fill_nan=fill_nan, fill_masked=fill_masked)
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
        self.signal_selection_changed.emit(self)
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
        self.signal_selection_changed.emit(self)
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
            self.signal_selection_changed.emit(self)  # TODO: unittest want to know, does this make sense?
        else:
            def create(current):
                return selections.SelectionExpression(boolean_expression, current, mode) if boolean_expression else None
            self._selection(create, name)

    def select_non_missing(self, drop_nan=True, drop_masked=True, column_names=None, mode="replace", name="default"):
        """Create a selection that selects rows having non missing values for all columns in column_names.

        The name reflect Panda's, no rows are really dropped, but a mask is kept to keep track of the selection

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
        expression = f(self._expr(column_names[0]))
        for column in column_names[1:]:
            expression = expression & f(self._expr(column))
        copy.select(~expression, name=FILTER_SELECTION_NAME, mode='and')
        return copy

    def select_nothing(self, name="default"):
        """Select nothing."""
        logger.debug("selecting nothing")
        self.select(None, name=name)
    # self.signal_selection_changed.emit(self)

    def select_rectangle(self, x, y, limits, mode="replace", name="default"):
        """Select a 2d rectangular box in the space given by x and y, bounds by limits.

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
        # TODO: maybe we also want free up selection masks
        if name not in self._selection_masks:
            self._selection_masks[name] = vaex.superutils.Mask(self._length_unfiltered)
        selection_history = self.selection_histories[name]
        previous_index = self.selection_history_indices[name]
        current = selection_history[previous_index] if selection_history else None
        selection = create_selection(current)
        executor = executor or self.executor
        selection_history.append(selection)
        self.selection_history_indices[name] += 1
        # clip any redo history
        del selection_history[self.selection_history_indices[name]:-1]
        self.signal_selection_changed.emit(self)
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
            if isinstance(value, Expression):
                value = value.expression
            if isinstance(value, np.ndarray):
                self.add_column(name, value)
            else:
                self.add_virtual_column(name, value)
        else:
            raise TypeError('__setitem__ only takes strings as arguments, not {}'.format(type(name)))

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
            if hasattr(self, item) and isinstance(getattr(self, item), Expression):
                return getattr(self, item)
            # if item in self.virtual_columns:
            #   return Expression(self, self.virtual_columns[item])
            return Expression(self, item)  # TODO we'd like to return the same expression if possible
        elif isinstance(item, Expression):
            expression = item.expression
            df = self.copy()
            df.select(expression, name=FILTER_SELECTION_NAME, mode='and')
            df._cached_filtered_length = None  # invalide cached length
            # WARNING: this is a special case where we create a new filter
            # the cache mask chunks still hold references to views on the old
            # mask, and this new mask will be filled when required
            df._selection_masks[FILTER_SELECTION_NAME] = vaex.superutils.Mask(df._length_unfiltered)
            return df
        elif isinstance(item, (tuple, list)):
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
            if self.filtered and start == 0:
                self.count()  # fill caches and masks
                mask = self._selection_masks[FILTER_SELECTION_NAME]
                indices = mask.first(stop-start)
                df = self.trim().take(indices, unfiltered=True)
            elif self.filtered and stop == len(self):
                self.count()  # fill caches and masks
                mask = self._selection_masks[FILTER_SELECTION_NAME]
                indices = mask.last(stop-start)
                df = self.trim().take(indices, unfiltered=True)
            else:
                df = self.extract()
                df.set_active_range(start, stop)
            return df.trim()

    def __delitem__(self, item):
        '''Removes a (virtual) column from the DataFrame.

        Note: this does not remove check if the column is used in a virtual expression or in the filter\
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
            self.column_names.remove(name)
        else:
            raise KeyError('no such column or virtual_columns named %r' % name)
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
        self.columns = collections.OrderedDict()
        self._task_aggs = {}
        self._binners = {}
        self._grids = {}

    def _readonly(self, inplace=False):
        # make arrays read only if possib;e
        df = self if inplace else self.copy()
        for key, ar in self.columns.items():
            if isinstance(ar, np.ndarray):
                df.columns[key] = ar = ar.view() # make new object so we don't modify others
                ar.flags['WRITEABLE'] = False
        return df

    def categorize(self, column, labels=None, check=True):
        """Mark column as categorical, with given labels, assuming zero indexing"""
        column = _ensure_string_from_expression(column)
        if check:
            vmin, vmax = self.minmax(column)
            if labels is None:
                N = int(vmax + 1)
                labels = list(map(str, range(N)))
            if (vmax - vmin) >= len(labels):
                raise ValueError('value of {} found, which is larger than number of labels {}'.format(vmax, len(labels)))
        self._categories[column] = dict(labels=labels, N=len(labels))

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

        original_column = df.rename_column(column, '__original_' + column, unique=True)
        labels = [str(k) for k in values]
        df.add_column(column, codes)
        df._categories[column] = dict(labels=labels, N=len(values), values=values)
        return df

    # for backward compatibility
    label_encode = _hidden(vaex.utils.deprecated('use is_category')(ordinal_encode))

    @property
    def data(self):
        """Gives direct access to the data as numpy arrays.

        Convenient when working with IPython in combination with small DataFrames, since this gives tab-completion.
        Only real columns (i.e. no virtual) columns can be accessed, for getting the data from virtual columns, use
        DataFrame.evalulate(...).

        Columns can be accesed by there names, which are attributes. The attribues are of type numpy.ndarray.

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
        df.variables.update(self.variables)
        df._categories.update(self._categories)
        column_names = column_names or self.get_column_names(hidden=True)
        all_column_names = self.get_column_names(hidden=True)

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

    def __array__(self, dtype=None):
        """Gives a full memory copy of the DataFrame into a 2d numpy array of shape (n_rows, n_columns).
        Note that the memory order is fortran, so all values of 1 column are contiguous in memory for performance reasons.

        Note this returns the same result as:

        >>> np.array(ds)

        If any of the columns contain masked arrays, the masks are ignored (i.e. the masked elements are returned as well).
        """
        if dtype is None:
            dtype = np.float64
        chunks = []
        for name in self.get_column_names(strings=False):
            if not np.can_cast(self.dtype(name), dtype):
                if self.dtype(name) != dtype:
                    raise ValueError("Cannot cast %r (of type %r) to %r" % (name, self.dtype(name), dtype))
            else:
                chunks.append(self.evaluate(name))
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
        """Concatenates two DataFrames, adding the rows of one the other DataFrame to the current, returned in a new DataFrame.

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

    def _evaluate(self, expression, i1, i2, out=None, selection=None, internal=False):
        scope = scopes._BlockScope(self, i1, i2, **self.variables)
        if out is not None:
            scope.buffers[expression] = out
        value = scope.evaluate(expression)
        if isinstance(value, ColumnString) and not internal:
            value = value.to_numpy()
        return value

    def evaluate(self, expression, i1=None, i2=None, out=None, selection=None, filtered=True, internal=False):
        """The local implementation of :func:`DataFrame.evaluate`"""
        expression = _ensure_string_from_expression(expression)
        selection = _ensure_strings_from_expressions(selection)
        i1 = i1 or 0
        i2 = i2 or (len(self) if (self.filtered and filtered) else self.length_unfiltered())
        mask = None

        if self.filtered and filtered:  # if we filter, i1:i2 has a different meaning
            if 1:
                count_check = self.count()  # fill caches and masks
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
            else:
                indices = self._filtered_range_to_unfiltered_indices(i1, i2)
                i1 = indices[0]
                i2 = indices[-1] + 1
        # for both a selection or filtering we have a mask
        if selection is not None or (self.filtered and filtered):
            mask = self.evaluate_selection_mask(selection, i1, i2)
        scope = scopes._BlockScope(self, i1, i2, mask=mask, **self.variables)
        # value = value[mask]
        if out is not None:
            scope.buffers[expression] = out
        value = scope.evaluate(expression)
        if isinstance(value, ColumnString) and not internal:
            value = value.to_numpy()
        return value

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
                type1 = self.dtype(column_name)
                if type1 != str_type:
                    type1 = type1.type
                type2 = other.dtype(column_name)
                if type2 != str_type:
                    type2 = type2.type
                if type1 != type2:
                    print("different dtypes: %s vs %s for %s" % (self.dtype(column_name), other.dtype(column_name), column_name))
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
                        if self.dtype(column_name) != str_type and self.dtype(column_name).kind == 'f':  # floats with nan won't equal itself, i.e. NaN != NaN
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
    def join(self, other, on=None, left_on=None, right_on=None, lsuffix='', rsuffix='', how='left', inplace=False):
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
        :param lsuffix: suffix to add to the left column names in case of a name collision
        :param rsuffix: similar for the right
        :param how: how to join, 'left' keeps all rows on the left, and adds columns (with possible missing values)
                'right' is similar with self and other swapped.
        :param inplace: {inplace}
        :return:
        """
        ds = self if inplace else self.copy()
        if how == 'left':
            left = ds
            right = other
        elif how == 'right':
            left = other
            right = ds
            lsuffix, rsuffix = rsuffix, lsuffix
            left_on, right_on = right_on, left_on
        else:
            raise ValueError('join type not supported: {}, only left and right'.format(how))

        for name in right:
            if name in left and name + rsuffix == name + lsuffix:
                raise ValueError('column name collision: {} exists in both column, and no proper suffix given'
                                 .format(name))

        right = right.extract()  # get rid of filters and active_range
        assert left.length_unfiltered() == left.length_original()
        N = left.length_unfiltered()
        N_other = len(right)
        left_on = left_on or on
        right_on = right_on or on
        if left_on is None and right_on is None:
            for name in right:
                right_name = name
                if name in left:
                    left.rename_column(name, name + lsuffix)
                    right_name = name + rsuffix
                if name in right.virtual_columns:
                    left.add_virtual_column(right_name, right.virtual_columns[name])
                else:
                    left.add_column(right_name, right.columns[name])
        else:
            left_values = left.evaluate(left_on, filtered=False)
            right_values = right.evaluate(right_on)
            # maps from the left_values to row #
            if np.ma.isMaskedArray(left_values):
                mask = ~left_values.mask
                left_values = left_values.data
                index_left = dict(zip(left_values[mask], np.arange(N)[mask]))
            else:
                index_left = dict(zip(left_values, np.arange(N)))
            # idem for right
            if np.ma.isMaskedArray(right_values):
                mask = ~right_values.mask
                right_values = right_values.data
                index_other = dict(zip(right_values[mask], np.arange(N_other)[mask]))
            else:
                index_other = dict(zip(right_values, np.arange(N_other)))

            # we do a left join, find all rows of the right DataFrame
            # that has an entry on the left
            # for each row in the right
            # find which row it needs to go to in the right
            # from_indices = np.zeros(N_other, dtype=np.int64)  # row # of right
            # to_indices = np.zeros(N_other, dtype=np.int64)    # goes to row # on the left
            # keep a boolean mask of which rows are found
            left_mask = np.ones(N, dtype=np.bool)
            # and which row they point to in the right
            left_row_to_right = np.zeros(N, dtype=np.int64) - 1
            for i in range(N_other):
                left_row = index_left.get(right_values[i])
                if left_row is not None:
                    left_mask[left_row] = False  # unmask, it exists
                    left_row_to_right[left_row] = i

            lookup = np.ma.array(left_row_to_right, mask=left_mask)
            for name in right:
                right_name = name
                if name in left:
                    left.rename_column(name, name + lsuffix)
                    right_name = name + rsuffix
                if name in right.virtual_columns:
                    left.add_virtual_column(right_name, right.virtual_columns[name])
                else:
                    left.add_column(right_name, ColumnIndexed(right, lookup, name))
        return left

    def export(self, path, column_names=None, byteorder="=", shuffle=False, selection=False, progress=None, virtual=False, sort=None, ascending=True):
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
        if path.endswith('.parquet'):
            self.export_parquet(path, column_names, shuffle, selection, progress=progress, virtual=virtual, sort=sort, ascending=ascending)

    def export_arrow(self, path, column_names=None, byteorder="=", shuffle=False, selection=False, progress=None, virtual=False, sort=None, ascending=True):
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

    def export_parquet(self, path, column_names=None, byteorder="=", shuffle=False, selection=False, progress=None, virtual=False, sort=None, ascending=True):
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

    def export_hdf5(self, path, column_names=None, byteorder="=", shuffle=False, selection=False, progress=None, virtual=False, sort=None, ascending=True):
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

    def export_fits(self, path, column_names=None, shuffle=False, selection=False, progress=None, virtual=False, sort=None, ascending=True):
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

    def _needs_copy(self, column_name):
        import vaex.file.other
        return not \
            ((column_name in self.column_names and not
              isinstance(self.columns[column_name], Column) and not
              isinstance(self.columns[column_name], vaex.file.other.DatasetTap.TapColumn) and
              self.columns[column_name].dtype.type == np.float64 and
              self.columns[column_name].strides[0] == 8 and
              column_name not in
              self.virtual_columns) or self.dtype(column_name) == str_type or self.dtype(column_name).kind == 'S')
        # and False:

    def selected_length(self, selection="default"):
        """The local implementation of :func:`DataFrame.selected_length`"""
        return int(self.count(selection=selection).item())
        # np.sum(self.mask) if self.has_selection() else None

    def _set_mask(self, mask):
        self.mask = mask
        self._has_selection = mask is not None
        self.signal_selection_changed.emit(self)

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

        The binby operations does not return a 'flat' DataFrame, instead it returns an N-d grid
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

    def _get_task_agg(self, grid):
        if grid not in self._task_aggs:
            self._task_aggs[grid] = task = vaex.tasks.TaskAggregate(self, grid)
            self.executor.schedule(task)
        return self._task_aggs[grid]

    @docsubst
    @stat_1d
    def _agg(self, aggregator, grid, selection=False, delay=False, progress=None):
        """

        :param selection: {selection}
        :param delay: {delay}
        :param progress: {progress}
        :return: {return_stat_scalar}
        """
        task_agg = self._get_task_agg(grid)
        sub_task = aggregator.add_operations(task_agg)
        return self._delay(delay, sub_task)

    def _binner(self, expression, limits=None, shape=None, delay=False):
        expression = str(expression)
        if limits is not None and not isinstance(limits, (tuple, str)):
            limits = tuple(limits)
        key = (expression, limits, shape)
        if key not in self._binners:
            if expression in self._categories:
                N = self._categories[expression]['N']
                binner = self._binner_ordinal(expression, N)
                self._binners[key] = vaex.promise.Promise.fulfilled(binner)
            else:
                self._binners[key] = vaex.promise.Promise()
                @delayed
                def create_binner(limits):
                    return self._binner_scalar(expression, limits, shape)
                self._binners[key] = create_binner(self.limits(expression, limits, delay=True))
        return self._delay(delay, self._binners[key])

    def _grid(self, binners):
        key = tuple(binners)
        if key in self._grids:
            return self._grids[key]
        else:
            self._grids[key] = grid = vaex.superagg.Grid(binners)
            return grid

    def _binner_scalar(self, expression, limits, shape):
        type = vaex.utils.find_type_from_dtype(vaex.superagg, "BinnerScalar_", self.dtype(expression))
        vmin, vmax = limits
        return type(expression, vmin, vmax, shape)

    def _binner_ordinal(self, expression, ordinal_count, min_value=0):
        type = vaex.utils.find_type_from_dtype(vaex.superagg, "BinnerOrdinal_", self.dtype(expression))
        return type(expression, ordinal_count, min_value)

    def _create_grid(self, binby, limits, shape, delay=False):
        if isinstance(binby, (list, tuple)):
            binbys = binby
        else:
            binbys = [binby]
        binbys = _ensure_strings_from_expressions(binbys)
        binners = []
        if len(binbys):
            limits = _expand_limits(limits, len(binbys))
        else:
            limits = []
        shapes = _expand_shape(shape, len(binbys))
        for binby, limits1, shape in zip(binbys, limits, shapes):
            binners.append(self._binner(binby, limits1, shape, delay=True))
        @delayed
        def finish(*binners):
            return self._grid(binners)
        return self._delay(delay, finish(*binners))


class DataFrameConcatenated(DataFrameLocal):
    """Represents a set of DataFrames all concatenated. See :func:`DataFrameLocal.concat` for usage.
    """

    def __init__(self, dfs, name=None):
        super(DataFrameConcatenated, self).__init__(None, None, [])
        self.dfs = dfs
        self.name = name or "-".join(df.name for df in self.dfs)
        self.path = "-".join(df.path for df in self.dfs)
        first, tail = dfs[0], dfs[1:]
        for df in dfs:
            assert df.filtered is False, "we don't support filtering for concatenated DataFrames"
        for column_name in first.get_column_names(virtual=False):
            if all([column_name in df.get_column_names(virtual=False) for df in tail]):
                self.column_names.append(column_name)
        self.columns = {}
        for column_name in self.get_column_names(virtual=False):
            self.columns[column_name] = ColumnConcatenatedLazy(dfs, column_name)
            self._save_assign_expression(column_name)

        for name in list(first.virtual_columns.keys()):
            if all([first.virtual_columns[name] == df.virtual_columns.get(name, None) for df in tail]):
                self.virtual_columns[name] = first.virtual_columns[name]
            else:
                self.columns[name] = ColumnConcatenatedLazy(dfs, name)
                self.column_names.append(name)
            self._save_assign_expression(name)


        for df in dfs[:1]:
            for name, value in list(df.variables.items()):
                if name not in self.variables:
                    self.set_variable(name, value, write=False)
        # self.write_virtual_meta()

        self._length_unfiltered = sum(len(ds) for ds in self.dfs)
        self._length_original = self._length_unfiltered
        self._index_end = self._length_unfiltered

    def is_masked(self, column):
        if column in self.columns:
            return self.columns[column].is_masked
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

