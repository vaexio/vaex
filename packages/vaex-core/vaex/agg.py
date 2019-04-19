import numpy as np

from .stat import _Statistic
import vaex.superagg

aggregates = {}

def register(f, name=None):
    name = name or f.__name__
    aggregates[name] = f
    return f

class AggregatorDescriptor:
    pass

class AggregatorDescriptorBasic(AggregatorDescriptor):
    def __init__(self, name, expression, short_name):
        self.name = name
        self.short_name = short_name
        self.expression = expression
        if self.expression == '*':
            self.expressions = []
        else:
            self.expressions = [self.expression]

    def pretty_name(self, id=None):
        id = id or "_".join(map(str, self.expression))
        return '{0}_{1}'.format(id, self.short_name)

    def add_operations(self, agg_task, edges=True, **kwargs):
        return agg_task.add_aggregation_operation(self, edges=edges, **kwargs)

    def _create_operation(self, df, grid):
        if grid is None and binners is None:
            raise ValueError('Provide either binners or grid')
        if grid is None:
            binners = [binner.copy() for binner in binners]
            grid = vaex.superagg.Grid(binners)
        # expression = df[str(self.expression)]
        if self.expression == '*':
            self.dtype = np.dtype('int64')
        else:
            self.dtype = df[str(self.expression)].dtype
        agg_op_type = vaex.utils.find_type_from_dtype(vaex.superagg, self.name + "_", self.dtype)
        agg_op = agg_op_type(grid)
        return agg_op

class AggregatorDescriptorMean(AggregatorDescriptor):
    def __init__(self, name, expression, short_name):
        self.name = name
        self.short_name = short_name
        self.expression = expression
        if self.expression == '*':
            self.expressions = []
        else:
            self.expressions = [self.expression]
        self.sum = sum(expression)
        self.count = count(expression)

    def pretty_name(self, id=None):
        id = id or "_".join(map(str, self.expression))
        return '{0}_{1}'.format(id, self.short_name)

    def add_operations(self, agg_task, **kwargs):
        task_sum = self.sum.add_operations(agg_task, **kwargs)
        task_count = self.count.add_operations(agg_task, **kwargs)
        @vaex.delayed
        def finish(sum, count):
            with np.errstate(divide='ignore', invalid='ignore'):
                mean = sum / count
            return mean
        return finish(task_sum, task_count)



@register
def count(expression='*'):
    '''Creates a count aggregation'''
    return AggregatorDescriptorBasic('AggCount', expression, 'count')

@register
def sum(expression):
    '''Creates a sum aggregation'''
    return AggregatorDescriptorBasic('AggSum', expression, 'sum')

@register
def mean(expression):
    '''Creates a mean aggregation'''
    return AggregatorDescriptorMean('AggSum', expression, 'mean')

@register
def std(expression):
    '''Creates a standard deviation aggregation'''
    return _Statistic('std', expression)

@register
def covar(x, y):
    '''Creates a standard deviation aggregation'''
    return _Statistic('covar', x, y)

@register
def correlation(x, y):
    '''Creates a standard deviation aggregation'''
    return _Statistic('correlation', x, y)

