from .expression import _unary_ops, _binary_ops, reversable
from future.utils import with_metaclass
from vaex.expression import expression_namespace
from vaex.delayed import delayed


class Meta(type):
    def __new__(upperattr_metaclass, future_class_name,
                future_class_parents, attrs):
        for op in _binary_ops:
            def wrap(op=op):
                def f(a, b):
                    return _StatisticsCalculation(op['name'], op['op'], [a, b], binary=True, code=op['code'])
                attrs['__%s__' % op['name']] = f
                if op['name'] in reversable:
                    def f(a, b):
                        return _StatisticsCalculation(op['name'], op['op'], [b, a], binary=True, code=op['code'])
                    attrs['__r%s__' % op['name']] = f
            wrap(op)
        for op in _unary_ops:
            def wrap(op=op):
                def f(a):
                    return _StatisticsCalculation(op['name'], op['op'], [a], unary=True, code=op['code'])
                attrs['__%s__' % op['name']] = f
            wrap(op)
        for name, func_real in expression_namespace.items():
            def wrap(name=name, func_real=func_real):
                def f(*args, **kwargs):
                    return _StatisticsCalculation(name, func_real, args)
                attrs['%s' % name] = f
            if name not in attrs:
                wrap(name)
        return type(future_class_name, future_class_parents, attrs)


class Expression(with_metaclass(Meta)):
    '''Describes an expression for a statistic'''

    def calculate(self, ds, binby=[], shape=256, limits=None, selection=None):
        '''Calculate the statistic for a :class:`Dataset`'''
        raise NotImplementedError()

    def __repr__(self):
        return '{}'.format(self)


class _StatisticsCalculation(Expression):
    def __init__(self, name, op, args, binary=False, unary=False, code='"<??>"'):
        self.name = name
        self.op = op
        self.args = args
        self.binary = binary
        self.unary = unary
        self.code = code

    def __str__(self):
        if self.binary:
            return "({0} {1} {2})".format(repr(self.args[0]), self.code, repr(self.args[1]))
        if self.unary:
            return "{0}{1}".format(self.code, repr(self.args[0]))
        return "{0}({1})".format(self.name, ", ".join(repr(k) for k in self.args))

    def calculate(self, ds, binby=[], shape=256, limits=None, selection=None, delay=False, progress=None):
        import vaex.arrow.numpy_dispatch
        @delayed
        def unwrap(x):
            return vaex.arrow.numpy_dispatch.unwrap(x)
        def to_value(v):
            if isinstance(v, Expression):
                return unwrap(v.calculate(ds, binby=binby, shape=shape, limits=limits, selection=selection, delay=delay, progress=progress))
            return unwrap(v)
        values = [to_value(v) for v in self.args]
        op = self.op
        op = delayed(op)
        value = unwrap(op(*values))
        return value if delay else value.get()


class _Statistic(Expression):
    def __init__(self, name, *expression):
        self.name = name
        self.expression = expression
        self.args = self.expression

    def pretty_name(self, id=None):
        id = id or "_".join(map(str, self.expression))
        return '{0}_{1}'.format(id, self.name)

    def __str__(self):
        return "{0}({1})".format(self.name, ", ".join(str(k) for k in self.args))

    def calculate(self, ds, binby=[], shape=256, limits=None, selection=None, delay=False, progress=None):
        method = getattr(ds, self.name)
        return method(*self.args, binby=binby, shape=shape, limits=limits, selection=selection, delay=delay, progress=progress)


def count(expression='*'):
    '''Creates a count statistic'''
    return _Statistic('count', expression)


def sum(expression):
    '''Creates a sum statistic'''
    return _Statistic('sum', expression)


def mean(expression):
    '''Creates a mean statistic'''
    return _Statistic('mean', expression)


def std(expression):
    '''Creates a standard deviation statistic'''
    return _Statistic('std', expression)


def covar(x, y):
    '''Creates a standard deviation statistic'''
    return _Statistic('covar', x, y)


def correlation(x, y):
    '''Creates a standard deviation statistic'''
    return _Statistic('correlation', x, y)
