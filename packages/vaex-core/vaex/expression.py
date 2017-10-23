import operator
import six
import functools
from future.utils import with_metaclass
from vaex.dataset import expression_namespace

_binary_ops = [
    dict(code="+", name='add',  op=operator.add),
    dict(code="in", name='contains',  op=operator.contains),
    dict(code="/", name='truediv',  op=operator.truediv),
    dict(code="//", name='floordiv',  op=operator.floordiv),
    dict(code="&", name='and',  op=operator.and_),
    dict(code="^", name='xor',  op=operator.xor),

    dict(code="|", name='or',  op=operator.or_),
    dict(code="**", name='pow',  op=operator.pow),
    dict(code="is", name='is',  op=operator.is_),
    dict(code="is not", name='is_not',  op=operator.is_not),

    dict(code="<<", name='lshift',  op=operator.lshift),
    dict(code="%", name='mod',  op=operator.mod),
    dict(code="*", name='mul',  op=operator.mul),

    dict(code=">>", name='rshift',  op=operator.rshift),
    dict(code="-", name='sub',  op=operator.sub),

    dict(code="<",  name='lt',   op=operator.lt),
    dict(code="<=", name='le',  op=operator.le),
    dict(code="==", name='eq',  op=operator.eq),
    dict(code="!=", name='ne',  op=operator.ne),
    dict(code=">=", name='ge',  op=operator.ge),
    dict(code=">",  name='gt',  op=operator.gt),
]
if hasattr(operator, 'div'):
    _binary_ops.append(dict(code="/", name='div',  op=operator.div))
if hasattr(operator, 'matmul'):
    _binary_ops.append(dict(code="@", name='matmul',  op=operator.matmul))

reversable = 'add sub mul matmul truediv floordiv mod divmod pow lshift rshift and xor or'.split()

_unary_ops = [
    dict(code="~", name='invert',  op=operator.invert),
    dict(code="-", name='neg',  op=operator.neg),
    dict(code="+", name='pos',  op=operator.pos),
]
class Meta(type):
    def __new__(upperattr_metaclass, future_class_name,
                future_class_parents, attrs):
        #attrs = {}
        for op in _binary_ops:
            def wrap(op=op):
                def f(a, b):
                    self = a
                    #print(op, a, b)
                    if isinstance(b, Expression):
                        assert b.ds == a.ds
                        b = b.expression
                    expression = '({0}) {1} ({2})'.format(a.expression, op['code'], b)
                    return Expression(self.ds, expression=expression)
                attrs['__%s__' % op['name']] = f
                if op['name'] in reversable:
                    def f(a, b):
                        self = a
                        #print(op, a, b)
                        if isinstance(b, Expression):
                            assert b.ds == a.ds
                            b = b.expression
                        expression = '({2}) {1} ({0})'.format(a.expression, op['code'], b)
                        return Expression(self.ds, expression=expression)
                    attrs['__r%s__' % op['name']] = f

            wrap(op)
        for op in _unary_ops:
            def wrap(op=op):
                def f(a):
                    self = a
                    expression = '{0}({1})'.format(op['code'], a.expression)
                    return Expression(self.ds, expression=expression)
                attrs['__%s__' % op['name']] = f
            wrap(op)
        for name, func_real in expression_namespace.items():
            def wrap(name=name):
                def f(*args, **kwargs):
                    self = args[0]
                    def to_expression(expression):
                        if isinstance(expression, Expression):
                            assert expression.ds == self.ds
                            expression = expression.expression
                        return expression
                    expressions = [to_expression(e) for e in args]
                    #print(name, expressions)
                    expression = '{0}({1})'.format(name, ", ".join(expressions))
                    return Expression(self.ds, expression=expression)
                try:
                    f = functools.wraps(func_real)(f)
                except AttributeError:
                    pass # numpy ufuncs don't have a __module__, which may choke wraps

                attrs['%s' % name] = f
            if name not in attrs:
                wrap(name)
        return type(future_class_name, future_class_parents, attrs)

class Expression(with_metaclass(Meta)):
    def __init__(self, ds, expression):
        self.ds = ds
        self.expression = expression
    def __str__(self):
        return self.expression

    # def __array__(self, dtype=None):
    #     '''For casting to a numpy array

    #     Example:
    #         >>> np.array(ds.x**2)

    #     '''
    #     return self.ds.evaluate(self)

    def tolist(self):
        '''Short for expr.evaluate().tolist()'''
        return self.evaluate().tolist()

    def __repr__(self):
        name = self.__class__.__module__ + "." +self.__class__.__name__
        try:
            N = len(self.ds)
            if N <= 10:
                values = ", ".join(str(k) for k in np.array(self))
            else:
                values_head = ", ".join(str(k) for k in self.evaluate(0, 5))
                values_tail = ", ".join(str(k) for k in self.evaluate(N-5, N))
                values = '{} ... (total {} values) ... {}'.format(  values_head, N, values_tail)
        except Exception as e:
            values = 'Error evaluating: %r' % e
        return "<%s(expressions=%r)> instance at 0x%x [%s] " % (name, self.expression, id(self), values)

    def count(self):
        return self.ds.count(self.expression)
    def sum(self):
        return self.ds.sum(self.expression)

    def evaluate(self, i1=None, i2=None, out=None, selection=None):
        return self.ds.evaluate(self, i1, i2, out=out, selection=selection)

    # def fillna(self, value, fill_nan=True, fill_masked=True):
    #     return self.ds.func.fillna(self, value, fill_nan=fill_nan, fill_masked=fill_masked)

    def clip(self, lower=None, upper=None):
        return self.ds.func.clip(self, lower, upper)

    def optimized(self):
        import pythran
        import imp
        import hashlib
        #self._import_all(module)
        names =  []
        funcs = set(vaex.dataset.expression_namespace.keys())
        vaex.expresso.validate_expression(self.expression, self.ds.get_column_names(virtual=True, strings=True), funcs, names)
        names = list(set(names))
        types = ", ".join(str(self.ds.dtype(name)) + "[]" for name in names)
        argstring = ", ".join(names)
        code = '''
    from numpy import *
    #pythran export f({2})
    def f({0}):
    return {1}'''.format(argstring, self.expression, types)
        print(code)
        m = hashlib.md5()
        m.update(code.encode('utf-8'))
        module_name = "pythranized_" + m.hexdigest()
        print(m.hexdigest())
        module_path = pythran.compile_pythrancode(module_name, code, extra_compile_args=["-DBOOST_SIMD", "-march=native"])
        module = imp.load_dynamic(module_name, module_path)
        function_name = "f_" +m.hexdigest()
        vaex.dataset.expression_namespace[function_name] = module.f

        return Expression(self.ds, "{0}({1})".format(function_name, argstring))

import types
import vaex.serialize
import base64
import cloudpickle as pickle
try:
    from StringIO import StringIO
except ImportError:
    from io import BytesIO as StringIO


@vaex.serialize.register
class FunctionSerializable(object):
    def __init__(self, f=None):
        self.f = f

    def pickle(self, function):
        return pickle.dumps(function)

    def unpickle(self, data):
        return pickle.loads(data)

    def state_get(self):
        data = self.pickle(self.f)
        if vaex.utils.PY2:
            pickled = base64.encodestring(data)
        else:
            pickled = base64.encodebytes(data).decode('ascii')
        return dict(pickled=pickled)

    def state_set(self, state):
        data = state['pickled']
        if vaex.utils.PY2:
            data = base64.decodestring(data)
        else:
            data = base64.decodebytes(data.encode('ascii'))
        self.f = self.unpickle(data)

    def __call__(self, *args, **kwargs):
        '''Forward the call to the real function'''
        return self.f(*args, **kwargs)

import numpy as np

class FunctionToScalar(FunctionSerializable):
    def __call__(self, *args, **kwargs):
        length = len(args[0])
        result = []
        for i in range(length):
            scalar_result = self.f(*[k[i] for k in args], **{key:value[i] for key, value in kwargs.items()})
            result.append(scalar_result)
        result = np.array(result)
        print(result, result.dtype)
        return result


class Function(object):
    
    def __init__(self, dataset, name, f):
        self.dataset = dataset
        self.name = name
        self.f = FunctionSerializable(f)

    def __call__(self, *args, **kwargs):
        arg_string = ", ".join([str(k) for k in args] + ['{}={}'.format(name, value) for name, value in kwargs.items()])
        expression = "{}({})".format(self.name, arg_string)
        return Expression(self.dataset, expression)

class FunctionBuiltin(object):

    def __init__(self, dataset, name, **kwargs):
        self.dataset = dataset
        self.name = name
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        kwargs = dict(kwargs, **self.kwargs)
        arg_string = ", ".join([str(k) for k in args] + ['{}={}'.format(name, value) for name, value in kwargs.items()])
        expression = "{}({})".format(self.name, arg_string)
        return Expression(self.dataset, expression)
