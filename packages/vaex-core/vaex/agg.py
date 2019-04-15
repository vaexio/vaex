from .stat import _Statistic

aggregates = {}

def register(f, name=None):
    name = name or f.__name__
    aggregates[name] = f
    return f

@register
def count(expression='*'):
    '''Creates a count aggregation'''
    return _Statistic('count', expression)

@register
def sum(expression):
    '''Creates a sum aggregation'''
    return _Statistic('sum', expression)

@register
def mean(expression):
    '''Creates a mean aggregation'''
    return _Statistic('mean', expression)

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

