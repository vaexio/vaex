from functools import reduce
import operator
import os
import sys
import logging

import numpy as np
import pyarrow as pa

import dask.base
from vaex.expression import Expression
from vaex.utils import _normalize_selection_name
from .expression import _unary_ops, _binary_ops, reversable


from .stat import _Statistic
from vaex import encoding
from .datatype import DataType
from .docstrings import docsubst
import vaex.utils


list_ = list # dangerous, we override list in this module (the aggregator)

logger = logging.getLogger("vaex.agg")

if vaex.utils.has_c_extension:
    import vaex.superagg

_min = min  # we're gonna overwrite builtin min
aggregates = {}


def register(f, name=None):
    name = name or f.__name__
    aggregates[name] = f
    return f


@encoding.register('aggregation')
class aggregation_encoding:
    @staticmethod
    def encode(encoding, agg):
        return agg.encode(encoding)

    @staticmethod
    def decode(encoding, agg_spec):
        agg_spec = agg_spec.copy()
        type = agg_spec.pop('aggregation')
        f = aggregates[type]
        args = []
        if type == '_sum_moment':
            if 'parameters' in agg_spec:  # renameing between spec and implementation
                agg_spec['moment'] = agg_spec.pop('parameters')[0]
        if 'expressions' in agg_spec:
            args = agg_spec.pop('expressions')
        if type == "list":
            if "parameters" in agg_spec:  # renameing between spec and implementation
                agg_spec["dropnan"], agg_spec["dropmissing"] = agg_spec.pop("parameters")
        return f(*args, **agg_spec)


class AggregatorDescriptor:
    def __repr__(self):
        args = [*self.expressions]
        return 'vaex.agg.{}({!r})'.format(self.short_name, ", ".join(map(str, args)))

    def pretty_name(self, id, df):
        if id is None:
            id = "_".join(map(lambda k: df[k]._label, self.expressions))
        return '{0}_{1}'.format(id, self.short_name)

    def finish(self, value):
        return value


class AggregatorExpressionUnary(AggregatorDescriptor):
    def __init__(self, name, op, code, agg):
        self.agg = agg
        self.name = name
        self.op = op
        self.code = code
        self.expressions = self.agg.expressions
        self.selection = self.agg.selection

    def __repr__(self):
        return f'{self.code}{self.agg!r}'

    @property
    def edges(self):
        return self.agg.edges

    @edges.setter
    def edges(self, value):
        self.agg.edges = value

    def add_tasks(self, df, binners, progress):
        tasks, result = self.agg.add_tasks(df, binners, progress)
        @vaex.delayed
        def finish(value):
            return self.finish(value)
        return tasks, finish(result)

    def finish(self, value):
        return self.op(value)


class AggregatorExpressionBinary(AggregatorDescriptor):
    def __init__(self, name, op, code, agg1, agg2, reverse=False):
        self.agg1 = agg1
        self.agg2 = agg2
        self.reverse = reverse
        self.name = name
        self.op = op
        self.code = code
        self.expressions = self.agg1.expressions + self.agg2.expressions
        self.selection = self.agg1.selection
        self.short_name = f'{self.code}{self.agg2.short_name}'
        if self.agg1.selection != self.agg2.selection:
            raise ValueError(f'Selections of aggregator for binary op {self.op} should be the same')

    def __repr__(self):
        if self.reverse:
            return f'({self.agg2!r} {self.code} {self.agg1!r})'
        else:
            return f'({self.agg1!r} {self.code} {self.agg2!r})'

    @property
    def edges(self):
        assert self.agg1.edges == self.agg2.edges
        return self.agg1.edges

    @edges.setter
    def edges(self, value):
        self.agg1.edges = value
        self.agg2.edges = value

    def add_tasks(self, df, binners, progress):
        progressbar = vaex.utils.progressbars(progress, title=repr(self))
        tasks1, result1 = self.agg1.add_tasks(df, binners, progress=progressbar)
        tasks2, result2 = self.agg2.add_tasks(df, binners, progress=progressbar)
        @vaex.delayed
        def finish(value1, value2):
            return self.finish(value1, value2)
        return tasks1 + tasks2, finish(result1, result2)

    def finish(self, value1, value2):
        if self.reverse:
            return self.op(value2, value1)
        return self.op(value1, value2)


class AggregatorExpressionBinaryScalar(AggregatorDescriptor):
    def __init__(self, name, op, code, agg, scalar, reverse=False):
        self.agg = agg
        self.scalar = scalar
        self.name = name
        self.code = code
        self.op = op
        self.reverse = reverse
        self.expressions = self.agg.expressions
        self.selection = self.agg.selection

    def __repr__(self):
        if self.reverse:
            return f'({self.scalar!r} {self.code} {self.agg!r})'
        else:
            return f'({self.agg!r} {self.code} {self.scalar!r})'

    @property
    def edges(self):
        return self.agg.edges

    @edges.setter
    def edges(self, value):
        self.agg.edges = value

    def add_tasks(self, df, binners, progress):
        progressbar = vaex.utils.progressbars(progress, title=repr(self))
        tasks, result = self.agg.add_tasks(df, binners, progress=progressbar)
        @vaex.delayed
        def finish(value):
            return self.finish(value)
        return tasks, finish(result)

    def finish(self, value):
        if self.reverse:
            return self.op(self.scalar, value)
        return self.op(value, self.scalar)


for op in _binary_ops:
    def wrap(op=op):
        def f(a, b):
            if isinstance(a, AggregatorDescriptor):
                if isinstance(b, AggregatorDescriptor):
                    return AggregatorExpressionBinary(op['name'], op['op'], op['code'], a, b)
                else:
                    return AggregatorExpressionBinaryScalar(op['name'], op['op'], op['code'], a, b)
            else:
                raise RuntimeError('Cannot happen')
        setattr(AggregatorDescriptor, '__%s__' % op['name'], f)
        if op['name'] in reversable:
            def f(a, b):
                if isinstance(a, AggregatorDescriptor):
                    if isinstance(b, AggregatorDescriptor):
                        raise RuntimeError('Cannot happen')
                    else:
                        return AggregatorExpressionBinaryScalar(op['name'], op['op'], op['code'], a, b, reverse=True)
                else:
                    raise RuntimeError('Cannot happen')
            setattr(AggregatorDescriptor, '__r%s__' % op['name'], f)
    wrap(op)
for op in _unary_ops:
    def wrap(op=op):
        def f(a):
            return AggregatorExpressionUnary(op['name'], op['op'], op['code'], a)
        setattr(AggregatorDescriptor, '__%s__' % op['name'], f)
    wrap(op)


class AggregatorDescriptorBasic(AggregatorDescriptor):
    def __init__(self, name, expressions, short_name, multi_args=False, agg_args=[], selection=None, edges=False):
        self.name = name
        self.short_name = short_name
        self.agg_args = agg_args
        self.edges = edges
        self.selection = _normalize_selection_name(selection)
        assert isinstance(expressions, (list_, tuple))
        for e in expressions:
            assert not isinstance(e, (list_, tuple))
        self.expressions = [str(k) for k in expressions]
        if len(self.expressions) == 1 and self.expressions[0] == '*':
            self.expressions = []

    def __repr__(self):
        args = [*self.expressions, *self.agg_args]
        return 'vaex.agg.{}({!r})'.format(self.short_name, ", ".join(map(str, args)))

    def encode(self, encoding):
        spec = {'aggregation': self.short_name}
        if len(self.expressions) == 0:
            pass
        else:
            spec['expressions'] = [str(k) for k in self.expressions]
        if self.selection is not None:
            spec['selection'] = str(self.selection) if isinstance(self.selection, Expression) else self.selection
        if self.edges:
            spec['edges'] = True
        if self.agg_args and self.short_name not in ['first', 'last']:
            spec['parameters'] = self.agg_args
        return spec

    def _prepare_types(self, df):
        if len(self.expressions) == 0 and self.short_name == "count":
            self.dtype_in = DataType(np.dtype('int64'))
            self.dtype_out = DataType(np.dtype('int64'))
        else:
            self.dtypes_in = [df[str(e)].data_type().index_type for e in self.expressions]
            self.dtype_in = self.dtypes_in[0]
            self.dtype_out = self.dtype_in
            if self.short_name == "count":
                self.dtype_out = DataType(np.dtype('int64'))
            if self.short_name in ['sum', 'summoment']:
                self.dtype_out = self.dtype_in.upcast()

    def add_tasks(self, df, binners, progress):
        progressbar = vaex.utils.progressbars(progress)
        self._prepare_types(df)
        task = vaex.tasks.TaskAggregation(df, binners, self)
        task = df.executor.schedule(task)
        progressbar.add_task(task, repr(self))
        @vaex.delayed
        def finish(value):
            return self.finish(value)
        return [task], finish(task)

    def _create_operation(self, grid, nthreads):
        if self.name in ["AggFirst", "AggList"]:
            if len(self.dtypes_in) == 1:
                # rows use int64
                agg_op_type = vaex.utils.find_type_from_dtype(vaex.superagg, self.name + "_", self.dtypes_in[0], vaex.dtype(np.dtype('int64')))
            else:
                agg_op_type = vaex.utils.find_type_from_dtype(vaex.superagg, self.name + "_", self.dtypes_in[0], self.dtypes_in[1])
        else:
            agg_op_type = vaex.utils.find_type_from_dtype(vaex.superagg, self.name + "_", self.dtype_in)
        if self.dtype_out.is_primitive or self.dtype_out.is_temporal:
            bytes_per_cell = self.dtype_out.numpy.itemsize
        else:
            bytes_per_cell = self.dtype_out.value_type.numpy.itemsize
        cells = reduce(operator.mul, [len(binner) for binner in grid.binners], 1)
        grids = nthreads
        ncells = len(grid)
        # the more grid cells we have, the less more work have to do to
        # merge/reduce the grids
        if ncells >= 1e4:
            grids = _min(32, nthreads)
        if ncells >= 1e5:
            grids = _min(16, nthreads)
        if ncells >= 1e6:
            grids = _min(8, nthreads)
        if grids < 1:
            grids = 1
        if logger.isEnabledFor(logging.INFO):
            logger.info("Using %r grids for %r thread for aggerator %r (total grid cells %s)", grids, nthreads, self, f"{ncells:,}")
        if self.short_name in ["list"]:
            # cannot predict memory usage
            predicted_memory_usage = None
            grids = 1
        else:
            predicted_memory_usage = bytes_per_cell * cells * grids
            vaex.memory.local.agg.pre_alloc(predicted_memory_usage, f"aggregator data for {agg_op_type}")
        agg_op = agg_op_type(grid, grids, nthreads, *self.agg_args)
        used_memory = sys.getsizeof(agg_op)

        if predicted_memory_usage is not None:
            if predicted_memory_usage != used_memory:
                raise RuntimeError(f'Wrong prediction for {agg_op_type}, expected to take {predicted_memory_usage} bytes but actually used {used_memory}')
        else:
            vaex.memory.local.agg.pre_alloc(used_memory, f"aggregator data for {agg_op_type}")
        return agg_op

    def get_result(self, agg_operation):
        grid = agg_operation.get_result()
        if not self.edges:
            def binner2slice(binner):
                if 'BinnerScalar_' in str(binner):
                    return slice(2, -1)
                elif 'BinnerOrdinal_' in str(binner):
                    return slice(0, -2)
                else:
                    raise TypeError(f'Binner not supported with edges=False {binner}')
            slices = [binner2slice(binner) for binner in agg_operation.grid.binners]
            grid = grid[tuple(slices)]
        return grid


class AggregatorDescriptorNUnique(AggregatorDescriptorBasic):
    def __init__(self, name, expression, short_name, dropmissing, dropnan, selection=None, edges=False):
        super(AggregatorDescriptorNUnique, self).__init__(name, expression, short_name, selection=selection, edges=edges)
        self.dropmissing = dropmissing
        self.dropnan = dropnan

    def encode(self, encoding):
        spec = super().encode(encoding)
        if self.dropmissing:
            spec['dropmissing'] = self.dropmissing
        if self.dropnan:
            spec['dropnan'] = self.dropnan
        return spec

    def _prepare_types(self, df):
        super()._prepare_types(df)
        self.dtype_out = DataType(np.dtype('int64'))

    def _create_operation(self, grid, nthreads):
        grids = 1  # this is using a shared hashmap, which is thread safe
        agg_op_type = vaex.utils.find_type_from_dtype(vaex.superagg, self.name + "_", self.dtype_in)
        cells = reduce(operator.mul, [len(binner) for binner in grid.binners], 1)
        grid0 = vaex.superagg.Grid([])
        agg_op_test = agg_op_type(grid0, grids, nthreads, self.dropmissing, self.dropnan)
        predicted_memory_usage = sys.getsizeof(agg_op_test) * cells
        vaex.memory.local.agg.pre_alloc(predicted_memory_usage, f"aggregator data for {agg_op_type}")
        agg_op = agg_op_type(grid, grids, nthreads, self.dropmissing, self.dropnan)
        used_memory = sys.getsizeof(agg_op)
        if predicted_memory_usage != used_memory:
            raise RuntimeError(f'Wrong prediction for {agg_op_type}, expected to take {predicted_memory_usage} bytes but actually used {used_memory}')

        return agg_op


class AggregatorDescriptorMulti(AggregatorDescriptor):
    """Uses multiple operations/aggregation to calculate the final aggretation"""
    def __init__(self, name, expressions, short_name, selection=None, edges=False):
        self.name = name
        self.short_name = short_name
        self.expressions = expressions
        self.selection = selection
        self.edges = edges
        assert isinstance(expressions, (list_, tuple))
        for e in expressions:
            assert not isinstance(e, (list_, tuple))
        self.expressions = [str(k) for k in expressions]


class AggregatorDescriptorMean(AggregatorDescriptorMulti):
    def __init__(self, name, expressions, short_name="mean", selection=None, edges=False):
        super(AggregatorDescriptorMean, self).__init__(name, expressions, short_name, selection=selection, edges=edges)
        assert len(expressions) == 1

    def add_tasks(self, df, binners, progress):
        progressbar = vaex.utils.progressbars(progress, title=repr(self))
        expression = expression_sum = expression = df[str(self.expressions[0])]

        sum_agg = sum(expression_sum, selection=self.selection, edges=self.edges)
        count_agg = count(expression, selection=self.selection, edges=self.edges)

        task_sum = sum_agg.add_tasks(df, binners, progress=progressbar)[0][0]
        task_count = count_agg.add_tasks(df, binners, progress=progressbar)[0][0]
        self.dtype_in = sum_agg.dtype_in
        self.dtype_out = sum_agg.dtype_out

        @vaex.delayed
        def finish(sum, count):
            sum = np.array(sum)
            dtype = sum.dtype
            sum_kind = sum.dtype.kind
            if sum_kind == 'M':
                sum = sum.view('uint64')
                count = count.view('uint64')
            with np.errstate(divide='ignore', invalid='ignore'):
                mean = sum / count
            if dtype.kind != mean.dtype.kind and sum_kind == "M":
                # TODO: not sure why view does not work
                mean = mean.astype(dtype)
            return mean

        return [task_sum, task_count], finish(task_sum, task_count)


class AggregatorDescriptorVar(AggregatorDescriptorMulti):
    def __init__(self, name, expression, short_name="var", ddof=0, selection=None, edges=False):
        super(AggregatorDescriptorVar, self).__init__(name, expression, short_name, selection=selection, edges=edges)
        self.ddof = ddof

    def add_tasks(self, df, binners, progress):
        progressbar = vaex.utils.progressbars(progress, title=repr(self))
        expression_sum = expression = df[str(self.expressions[0])]
        expression = expression_sum = expression.astype('float64')
        sum_moment = _sum_moment(str(expression_sum), 2, selection=self.selection, edges=self.edges)
        sum_ = sum(str(expression_sum), selection=self.selection, edges=self.edges)
        count_ = count(str(expression), selection=self.selection, edges=self.edges)

        task_sum_moment = sum_moment.add_tasks(df, binners, progress=progressbar)[0][0]
        task_sum = sum_.add_tasks(df, binners, progress=progressbar)[0][0]
        task_count = count_.add_tasks(df, binners, progress=progressbar)[0][0]
        self.dtype_in = sum_.dtype_in
        self.dtype_out = sum_.dtype_out
        @vaex.delayed
        def finish(sum_moment, sum, count):
            sum = np.array(sum)
            dtype = sum.dtype
            if sum.dtype.kind == 'M':
                sum = sum.view('uint64')
                sum_moment = sum_moment.view('uint64')
                count = count.view('uint64')
            with np.errstate(divide='ignore', invalid='ignore'):
                mean = sum / count
                raw_moments2 = sum_moment/count
                variance = (raw_moments2 - mean**2) #* count/(count-self.ddof)
            if dtype.kind != mean.dtype.kind:
                # TODO: not sure why view does not work
                variance = variance.astype(dtype)
            return self.finish(variance)
        return [task_sum_moment, task_sum, task_count], finish(task_sum_moment, task_sum, task_count)


class AggregatorDescriptorSkew(AggregatorDescriptorMulti):
    def __init__(self, name, expression, short_name='skew', selection=None, edges=False):
        super(AggregatorDescriptorSkew, self).__init__(name, [expression], short_name, selection=selection, edges=edges)

    def add_tasks(self, df, binners, progress):
        progressbar = vaex.utils.progressbars(progress, title=repr(self))
        expression = expression_sum = expression = df[str(self.expressions[0])]
        expression = expression_sum = expression.astype('float64')

        sum_moment1 = _sum_moment(str(expression_sum), 1, selection=self.selection, edges=self.edges)
        sum_moment2 = _sum_moment(str(expression_sum), 2, selection=self.selection, edges=self.edges)
        sum_moment3 = _sum_moment(str(expression_sum), 3, selection=self.selection, edges=self.edges)
        count_ = count(str(expression), selection=self.selection, edges=self.edges)

        task_sum_moment1 = sum_moment1.add_tasks(df, binners, progress=progressbar)[0][0]
        task_sum_moment2 = sum_moment2.add_tasks(df, binners, progress=progressbar)[0][0]
        task_sum_moment3 = sum_moment3.add_tasks(df, binners, progress=progressbar)[0][0]
        task_count = count_.add_tasks(df, binners, progress=progressbar)[0][0]

        @vaex.delayed
        def finish(sum_moment1, sum_moment2, sum_moment3, count):
            with np.errstate(divide='ignore', invalid='ignore'):
                m1 = sum_moment1 / count
                m2 = sum_moment2 / count
                m3 = sum_moment3 / count
                skew = (m3 - 3*m1*m2 + 2*m1**3) / (m2 - m1**2)**(3/2)
            return self.finish(skew)
        return [task_sum_moment1, task_sum_moment2, task_sum_moment3, task_count], finish(task_sum_moment1, task_sum_moment2, task_sum_moment3, task_count)


class AggregatorDescriptorKurtosis(AggregatorDescriptorMulti):
    def __init__(self, name, expression, short_name='kurtosis', selection=None, edges=False):
        super(AggregatorDescriptorKurtosis, self).__init__(name, [expression], short_name, selection=selection, edges=edges)

    def add_tasks(self, df, binners, progress):
        progressbar = vaex.utils.progressbars(progress, title=repr(self))
        expression = expression_sum = expression = df[str(self.expressions[0])]
        expression = expression_sum = expression.astype('float64')

        sum_moment1 = _sum_moment(str(expression_sum), 1, selection=self.selection, edges=self.edges)
        sum_moment2 = _sum_moment(str(expression_sum), 2, selection=self.selection, edges=self.edges)
        sum_moment3 = _sum_moment(str(expression_sum), 3, selection=self.selection, edges=self.edges)
        sum_moment4 = _sum_moment(str(expression_sum), 4, selection=self.selection, edges=self.edges)
        count_ = count(str(expression), selection=self.selection, edges=self.edges)

        task_sum_moment1 = sum_moment1.add_tasks(df, binners, progress=progressbar)[0][0]
        task_sum_moment2 = sum_moment2.add_tasks(df, binners, progress=progressbar)[0][0]
        task_sum_moment3 = sum_moment3.add_tasks(df, binners, progress=progressbar)[0][0]
        task_sum_moment4 = sum_moment4.add_tasks(df, binners, progress=progressbar)[0][0]
        task_count = count_.add_tasks(df, binners, progress=progressbar)[0][0]

        @vaex.delayed
        def finish(sum_moment1, sum_moment2, sum_moment3, sum_moment4, count):
            with np.errstate(divide='ignore', invalid='ignore'):
                m1 = sum_moment1 / count
                m2 = sum_moment2 / count
                m3 = sum_moment3 / count
                m4 = sum_moment4 / count
                kurtosis = (m4 - 4*m1*m3 + 6*m1**2*m2 - 3*m1**4) / (m2 - m1**2)**2 -3.0
            return self.finish(kurtosis)
        return [task_sum_moment1, task_sum_moment2, task_sum_moment3, task_sum_moment4, task_count], finish(task_sum_moment1, task_sum_moment2, task_sum_moment3, task_sum_moment4, task_count)


class AggregatorDescriptorStd(AggregatorDescriptorVar):
    def finish(self, value):
        return value**0.5

@register
def count(expression='*', selection=None, edges=False):
    '''Creates a count aggregation'''
    return AggregatorDescriptorBasic('AggCount', [expression], 'count', selection=selection, edges=edges)

@register
def sum(expression, selection=None, edges=False):
    '''Creates a sum aggregation'''
    return AggregatorDescriptorBasic('AggSum', [expression], 'sum', selection=selection, edges=edges)

@register
def mean(expression, selection=None, edges=False):
    '''Creates a mean aggregation'''
    return AggregatorDescriptorMean('mean', [expression], 'mean', selection=selection, edges=edges)

@register
def min(expression, selection=None, edges=False):
    '''Creates a min aggregation'''
    return AggregatorDescriptorBasic('AggMin', [expression], 'min', selection=selection, edges=edges)

@register
def _sum_moment(expression, moment, selection=None, edges=False):
    '''Creates a sum of moment aggregator'''
    return AggregatorDescriptorBasic('AggSumMoment', [expression], '_sum_moment', agg_args=[moment], selection=selection, edges=edges)

@register
def max(expression, selection=None, edges=False):
    '''Creates a max aggregation'''
    return AggregatorDescriptorBasic('AggMax', [expression], 'max', selection=selection, edges=edges)

@register
def first(expression, order_expression=None, selection=None, edges=False):
    '''Creates a first aggregation.

    :param expression: {expression_one}.
    :param order_expression:  Order the values in the bins by this expression.
    :param selection: {selection1}
    :param edges: {edges}
    '''
    return AggregatorDescriptorBasic('AggFirst', [expression, order_expression] if order_expression is not None else [expression], 'first', multi_args=True, selection=selection, edges=edges, agg_args=[False])

@register
@docsubst
def last(expression, order_expression=None, selection=None, edges=False):
    '''Creates a first aggregation.

    :param expression: {expression_one}.
    :param order_expression:  Order the values in the bins by this expression.
    :param selection: {selection1}
    :param edges: {edges}
    '''
    return AggregatorDescriptorBasic('AggFirst', [expression, order_expression] if order_expression is not None else [expression], 'last', multi_args=True, selection=selection, edges=edges, agg_args=[True])

@register
def std(expression, ddof=0, selection=None, edges=False):
    '''Creates a standard deviation aggregation'''
    return AggregatorDescriptorStd('std', [expression], 'std', ddof=ddof, selection=selection, edges=edges)

@register
def var(expression, ddof=0, selection=None, edges=False):
    '''Creates a variance aggregation'''
    return AggregatorDescriptorVar('var', [expression], 'var', ddof=ddof, selection=selection, edges=edges)

@register
def skew(expression, selection=None, edges=False):
    '''Create a skew aggregation.'''
    return AggregatorDescriptorSkew('skew', expression, 'skew', selection=selection, edges=edges)

@register
def kurtosis(expression, selection=None, edges=False):
    '''Create a kurtosis aggregation.'''
    return AggregatorDescriptorKurtosis('kurtosis', expression, 'kurtosis', selection=selection, edges=edges)

@register
@docsubst
def nunique(expression, dropna=False, dropnan=False, dropmissing=False, selection=None, edges=False):
    """Aggregator that calculates the number of unique items per bin.

    :param expression: {expression_one}
    :param dropmissing: {dropmissing}
    :param dropnan: {dropnan}
    :param dropna: {dropna}
    :param selection: {selection1}
    """
    if dropna:
        dropnan = True
        dropmissing = True
    return AggregatorDescriptorNUnique('AggNUnique', [expression], 'nunique', dropmissing, dropnan, selection=selection, edges=edges)

@docsubst
def any(expression=None, selection=None):
    '''Aggregator that returns True when any of the values in the group are True, or when there is any data in the group that is valid (i.e. not missing values or np.nan).
    The aggregator returns False if there is no data in the group when the selection argument is used.

    :param expression: {expression_one}
    :param selection: {selection1}
    '''
    if expression is None and selection is None:
        return count(selection=selection) > -1  # trivial
    else:
        if expression is None:
            return count(selection=selection) > 0
        else:
            return sum(expression, selection=selection) > 0

@docsubst
def all(expression=None, selection=None):
    '''Aggregator that returns True when all of the values in the group are True,
    or when all of the data in the group is valid (i.e. not missing values or np.nan).
    The aggregator returns False if there is no data in the group when the selection argument is used.

    :param expression: {expression_one}
    :param selection: {selection1}
    '''
    if expression is None and selection is None:
        return count(selection=selection) > -1  # trivial
    else:
        if expression is None:
            # counting how often the selection is true == counting how many rows
            return sum(selection) == count(selection)
        else:
            if selection is None:
                # counting how often true == counting how much data there is
                return sum(expression) == count(expression)
            else:
                # since we cannot mix different selections:
                return sum(f'astype({expression}, "bool") & astype({selection}, "bool")') == count(expression)


@register
@docsubst
class list(AggregatorDescriptorBasic):
    '''Aggregator that returns a list of values belonging to the specified expression.

    :param expression: {expression_one}
    :param selection: {selection1}
    :param dropmissing: {dropmissing}
    :param dropnan: {dropnan}
    :param dropna: {dropna}
    :param edges: {edges}
    '''
    def __init__(self, expression, selection=None, dropna=False, dropnan=False, dropmissing=False, edges=False):
        if dropna:
            dropnan = True
            dropmissing = True
        super(list, self).__init__("AggList", [expression], "list", selection=selection, edges=edges, agg_args=[dropnan, dropmissing])

    def _prepare_types(self, df):
        super()._prepare_types(df)
        self.dtype_out = vaex.dtype(pa.large_list(self.dtype_out.arrow))


# TODO: we could maybe refactor this to support more struct aggregators
@register
class describe(AggregatorDescriptor):
    def __init__(self, expression):
        self.expression = expression
        self.expressions = [self.expression]
        self.short_name = "describe"
        self.edges = True

    def __repr__(self):
        return f"describe({self.expression!r})"

    def add_tasks(self, df, binners, progress):
        expression: Expression = df[str(self.expression)]
        col = expression._label
        if expression.data_type() != "string":
            aggs = {
                f"count": vaex.agg.count(expression, edges=self.edges),
                f"count_na": vaex.agg.count(edges=self.edges) - vaex.agg.count(expression, edges=self.edges),
                f"mean": vaex.agg.mean(expression, edges=self.edges),
                f"std": vaex.agg.std(expression, edges=self.edges),
                f"min": vaex.agg.min(expression, edges=self.edges),
                f"max": vaex.agg.max(expression, edges=self.edges),
            }
        else:
            aggs = {f"count": vaex.agg.count(expression, edges=self.edges), f"count_na": vaex.agg.count(edges=self.edges) - vaex.agg.count(expression, edges=self.edges)}

        progressbar = vaex.utils.progressbars(progress, title=repr(self))
        tasks = []
        results = []
        names = []
        for name, agg in aggs.items():
            tasks1, result = agg.add_tasks(df, binners, progress=progressbar)
            tasks.extend(tasks1)
            results.append(result)
            names.append(name)

        @vaex.delayed
        def finish(*values):
            return self.finish(values, names)

        return tasks, finish(*results)

    def finish(self, values, names):
        if len(values):
            if vaex.array_types.is_scalar(values[0]):
                return pa.StructArray.from_arrays(arrays=[[k] for k in values], names=names)
        return pa.StructArray.from_arrays(arrays=values, names=names)


# @register
# def covar(x, y):
#     '''Creates a standard deviation aggregation'''
#     return _Statistic('covar', x, y)

# @register
# def correlation(x, y):
#     '''Creates a standard deviation aggregation'''
#     return _Statistic('correlation', x, y)



@dask.base.normalize_token.register(AggregatorDescriptor)
def normalize(agg):
    return agg.__class__.__name__, repr(agg)
