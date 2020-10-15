import numpy as np
import pyarrow as pa
import vaex
from ..expression import _binary_ops, _unary_ops, reversable

class NumpyDispatch:
    def __init__(self, arrow_array):
        self.arrow_array = arrow_array
        self._numpy_array = None

    @property
    def numpy_array(self):
        # convert lazily, since not all arrow arrays (e.g. lists) can be converted
        if self._numpy_array is None:
            self._numpy_array = vaex.array_types.to_numpy(self.arrow_array)
        return self._numpy_array

    def __eq__(self, rhs):
        if vaex.array_types.is_string(self.arrow_array):
            # this does not support scalar input
            # return pc.equal(self.arrow_array, rhs)
            return NumpyDispatch(pa.array(vaex.functions.str_equals(self.arrow_array, rhs)))
        else:
            if isinstance(rhs, NumpyDispatch):
                rhs = rhs.numpy_array
            return NumpyDispatch(pa.array(self.numpy_array == rhs))

for op in _binary_ops:
    def closure(op=op):
        def operator(a, b):
            if isinstance(a, NumpyDispatch):
                a = a.numpy_array
            if isinstance(b, NumpyDispatch):
                b = b.numpy_array
            return NumpyDispatch(pa.array(op['op'](a, b)))
        return operator
    method_name = '__%s__' % op['name']
    if op['name'] != "eq":
        setattr(NumpyDispatch, method_name, closure())
     # to support e.g. (1 + ...)    # to support e.g. (1 + ...)
    if op['name'] in reversable:
        def closure(op=op):
            def operator(b, a):
                return NumpyDispatch(pa.array(op['op'](a, b.numpy_array)))
            return operator
        method_name = '__r%s__' % op['name']
        setattr(NumpyDispatch, method_name, closure())


for op in _unary_ops:
    def closure(op=op):
        def operator(a):
            a = a.numpy_array
            return NumpyDispatch(pa.array(op['op'](a)))
        return operator
    method_name = '__%s__' % op['name']
    setattr(NumpyDispatch, method_name, closure())



def unwrap(value):
    if isinstance(value, NumpyDispatch):
        # import pdb
        # pdb.set_trace()
        return value.arrow_array
    # for performance reasons we don't visit lists and dicts
    return value


def autowrapper(f):
    '''Takes a function f, and will unwrap all its arguments and wrap the return value'''
    def wrapper(*args, **kwargs):
        args = map(unwrap, args)
        kwargs = {k: unwrap(v) for k, v, in kwargs.items()}
        result = f(*args, **kwargs)
        if isinstance(result, vaex.array_types.supported_arrow_array_types):
            result = NumpyDispatch(result)
        return result
    return wrapper
