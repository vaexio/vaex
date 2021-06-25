import os
import numpy as np

import dask.base
from vaex.expression import Expression

from .stat import _Statistic
from vaex import encoding
from .datatype import DataType


on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if not on_rtd:
    import vaex.superagg


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
        if type == 'first':
            args = agg_spec.pop('expression')
        return f(*args, **agg_spec)


class AggregatorDescriptor(object):
    def __repr__(self):
        return 'vaex.agg.{}({!r})'.format(self.short_name, str(self.expression))

    def pretty_name(self, id, df):
        if id is None:
            id = "_".join(map(lambda k: df[k]._label, self.expressions))
        return '{0}_{1}'.format(id, self.short_name)

    def finish(self, value):
        return value

class AggregatorDescriptorBasic(AggregatorDescriptor):
    def __init__(self, name, expression, short_name, multi_args=False, agg_args=[], selection=None, edges=False):
        self.name = name
        self.short_name = short_name
        self.expression = str(expression)
        self.agg_args = agg_args
        self.edges = edges
        self.selection = selection
        if not multi_args:
            if self.expression == '*':
                self.expressions = []
            else:
                self.expressions = [self.expression]
        else:
            self.expressions = expression

    def encode(self, encoding):
        spec = {'aggregation': self.short_name}
        if len(self.expressions) == 0:
            pass
        elif len(self.expressions) == 1:
            spec['expression'] = self.expression
        else:
            spec['expression'] = [str(k) for k in self.expressions]
        if self.selection is not None:
            spec['selection'] = str(self.selection) if isinstance(self.selection, Expression) else self.selection
        if self.edges:
            spec['edges'] = True
        if self.agg_args:
            spec['parameters'] = self.agg_args
        return spec

    def _prepare_types(self, df):
        if self.expression == '*':
            self.dtype_in = DataType(np.dtype('int64'))
            self.dtype_out = DataType(np.dtype('int64'))
        else:
            self.dtype_in = df[str(self.expressions[0])].data_type().index_type
            self.dtype_out = self.dtype_in
            if self.short_name == "count":
                self.dtype_out = DataType(np.dtype('int64'))
            if self.short_name in ['sum', 'summoment']:
                self.dtype_out = self.dtype_in.upcast()

    def add_tasks(self, df, binners):
        self._prepare_types(df)
        task = vaex.tasks.TaskAggregation(df, binners, self)
        task = df.executor.schedule(task)
        @vaex.delayed
        def finish(value):
            return self.finish(value)
        return [task], finish(task)

    def _create_operation(self, grid):
        agg_op_type = vaex.utils.find_type_from_dtype(vaex.superagg, self.name + "_", self.dtype_in)
        agg_op = agg_op_type(grid, *self.agg_args)
        return agg_op

    def get_result(self, agg_operation):
        grid = np.asarray(agg_operation)
        if not self.edges:
            grid = vaex.utils.extract_central_part(grid)
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

    def _create_operation(self, grid):
        agg_op_type = vaex.utils.find_type_from_dtype(vaex.superagg, self.name + "_", self.dtype_in)
        agg_op = agg_op_type(grid, self.dropmissing, self.dropnan)
        return agg_op


class AggregatorDescriptorMulti(AggregatorDescriptor):
    """Uses multiple operations/aggregation to calculate the final aggretation"""
    def __init__(self, name, expression, short_name, selection=None, edges=False):
        self.name = name
        self.short_name = short_name
        self.expression = str(expression)
        self.expressions = [self.expression]
        self.selection = selection
        self.edges = edges


class AggregatorDescriptorMean(AggregatorDescriptorMulti):
    def __init__(self, name, expression, short_name="mean", selection=None, edges=False):
        super(AggregatorDescriptorMean, self).__init__(name, expression, short_name, selection=selection, edges=edges)

    def add_tasks(self, df, binners):
        expression = expression_sum = expression = df[str(self.expression)]

        sum_agg = sum(expression_sum, selection=self.selection, edges=self.edges)
        count_agg = count(expression, selection=self.selection, edges=self.edges)

        task_sum = sum_agg.add_tasks(df, binners)[0][0]
        task_count = count_agg.add_tasks(df, binners)[0][0]
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

    def add_tasks(self, df, binners):
        expression_sum = expression = df[str(self.expression)]
        expression = expression_sum = expression.astype('float64')
        sum_moment = _sum_moment(str(expression_sum), 2, selection=self.selection, edges=self.edges)
        sum_ = sum(str(expression_sum), selection=self.selection, edges=self.edges)
        count_ = count(str(expression), selection=self.selection, edges=self.edges)

        task_sum_moment = sum_moment.add_tasks(df, binners)[0][0]
        task_sum = sum_.add_tasks(df, binners)[0][0]
        task_count = count_.add_tasks(df, binners)[0][0]
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


class AggregatorDescriptorStd(AggregatorDescriptorVar):
    def finish(self, value):
        return value**0.5

@register
def count(expression='*', selection=None, edges=False):
    '''Creates a count aggregation'''
    return AggregatorDescriptorBasic('AggCount', expression, 'count', selection=selection, edges=edges)

@register
def sum(expression, selection=None, edges=False):
    '''Creates a sum aggregation'''
    return AggregatorDescriptorBasic('AggSum', expression, 'sum', selection=selection, edges=edges)

@register
def mean(expression, selection=None, edges=False):
    '''Creates a mean aggregation'''
    return AggregatorDescriptorMean('mean', expression, 'mean', selection=selection, edges=edges)

@register
def min(expression, selection=None, edges=False):
    '''Creates a min aggregation'''
    return AggregatorDescriptorBasic('AggMin', expression, 'min', selection=selection, edges=edges)

@register
def _sum_moment(expression, moment, selection=None, edges=False):
    '''Creates a sum of moment aggregator'''
    return AggregatorDescriptorBasic('AggSumMoment', expression, '_sum_moment', agg_args=[moment], selection=selection, edges=edges)

@register
def max(expression, selection=None, edges=False):
    '''Creates a max aggregation'''
    return AggregatorDescriptorBasic('AggMax', expression, 'max', selection=selection, edges=edges)

@register
def first(expression, order_expression, selection=None, edges=False):
    '''Creates a max aggregation'''
    return AggregatorDescriptorBasic('AggFirst', [expression, order_expression], 'first', multi_args=True, selection=selection, edges=edges)

@register
def std(expression, ddof=0, selection=None, edges=False):
    '''Creates a standard deviation aggregation'''
    return AggregatorDescriptorStd('std', expression, 'std', ddof=ddof, selection=selection, edges=edges)

@register
def var(expression, ddof=0, selection=None, edges=False):
    '''Creates a variance aggregation'''
    return AggregatorDescriptorVar('var', expression, 'var', ddof=ddof, selection=selection, edges=edges)

@register
def nunique(expression, dropna=False, dropnan=False, dropmissing=False, selection=None, edges=False):
    """Aggregator that calculates the number of unique items per bin.

    :param expression: Expression for which to calculate the unique items
    :param dropmissing: do not count missing values
    :param dropnan: do not count nan values
    :param dropna: short for any of the above, (see :func:`Expression.isna`)
    """
    if dropna:
        dropnan = True
        dropmissing = True
    return AggregatorDescriptorNUnique('AggNUnique', expression, 'nunique', dropmissing, dropnan, selection=selection, edges=edges)

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
