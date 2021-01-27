import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import vaex
from ..expression import _binary_ops, _unary_ops, reversable
from .utils import combine_missing

class NumpyDispatch:
    def __init__(self, ar):
        self._array = ar
        if isinstance(ar, vaex.column.ColumnStringArrow):
            ar = pa.array(ar)
            self._array = ar
        if isinstance(ar, np.ndarray):
            self._numpy_array = ar
            self._arrow_array = None
        elif isinstance(ar, vaex.array_types.supported_arrow_array_types):
            self._numpy_array = None
            self._arrow_array = ar
        else:
            raise TypeError(f'Only support numpy and arrow, not {type(ar)}')

    def add_missing(self, ar):
        if isinstance(ar, np.ndarray):
            # if we are an arrow array, we upgrade ar to one
            if isinstance(self._array, vaex.array_types.supported_arrow_array_types):
                ar = vaex.array_types.to_arrow(ar)
                ar = combine_missing(ar, self._array)
            # else: both numpy, handled by numpy
        else:
            if isinstance(self._array, vaex.array_types.supported_arrow_array_types):
                ar = combine_missing(ar, self._array)
            # else: was numpy, handled by numpy
        return ar


    @property
    def numpy_array(self):
        if self._numpy_array is None:
            import vaex.arrow.convert
            arrow_array = self._arrow_array
            arrow_array = vaex.arrow.convert.ensure_not_chunked(arrow_array)
            buffers = arrow_array.buffers()
            # for math, we don't care about the nulls
            if buffers[0] is not None:
                buffers[0] = None
                arrow_array = pa.Array.from_buffers(arrow_array.type, len(arrow_array), buffers, offset=arrow_array.offset)
            self._numpy_array = vaex.array_types.to_numpy(arrow_array)
        return self._numpy_array

    @property
    def arrow_array(self):
        if self._arrow_array is None:
            # convert lazily, since not all arrow arrays (e.g. lists) can be converted
            if self._arrow_array is None:
                self._arrow_array = vaex.array_types.to_arrow(self._numpy_array)
        return self._arrow_array


for op in _binary_ops:
    def closure(op=op):
        def operator(a, b):
            a_data = a
            b_data = b
            if isinstance(a, NumpyDispatch):
                a_data = a.numpy_array
            if isinstance(b, NumpyDispatch):
                b_data = b.numpy_array
            if op['name'] == 'eq' and (vaex.array_types.is_string(a_data) or vaex.array_types.is_string(b_data)):
                result_data = vaex.functions.str_equals(a_data, b_data)
            else:
                result_data = op['op'](a_data, b_data)
            if isinstance(a, NumpyDispatch):
                result_data = a.add_missing(result_data)
            if isinstance(b, NumpyDispatch):
                result_data = b.add_missing(result_data)
            return NumpyDispatch(result_data)
        return operator
    method_name = '__%s__' % op['name']
    setattr(NumpyDispatch, method_name, closure())
     # to support e.g. (1 + ...)    # to support e.g. (1 + ...)
    if op['name'] in reversable:
        def closure(op=op):
            def operator(b, a):
                a_data = a
                b_data = b
                if isinstance(a, NumpyDispatch):
                    a_data = a.numpy_array
                if isinstance(b, NumpyDispatch):
                    b_data = b.numpy_array
                result_data = op['op'](a_data, b_data)
                if isinstance(a, NumpyDispatch):
                    result_data = a.add_missing(result_data)
                if isinstance(b, NumpyDispatch):
                    result_data = b.add_missing(result_data)
                return NumpyDispatch(result_data)
            return operator
        method_name = '__r%s__' % op['name']
        setattr(NumpyDispatch, method_name, closure())


for op in _unary_ops:
    def closure(op=op):
        def operator(a):
            a_data = a.numpy_array
            result_data = op['op'](a_data)
            if isinstance(a, NumpyDispatch):
                result_data = a.add_missing(result_data)
            return NumpyDispatch(result_data)
        return operator
    method_name = '__%s__' % op['name']
    setattr(NumpyDispatch, method_name, closure())


def wrap(value):
    if not isinstance(value, NumpyDispatch): # and not isinstance(value, np.ndarray):
        if isinstance(value, vaex.array_types.supported_array_types + (vaex.column.ColumnStringArrow,)):
            return NumpyDispatch(value)
    # for performance reasons we don't visit lists and dicts
    return value


def unwrap(value):
    if isinstance(value, NumpyDispatch):
        return value._array
    # for performance reasons we don't visit lists and dicts
    return value


def autowrapper(f):
    '''Takes a function f, and will unwrap all its arguments and wrap the return value'''
    def wrapper(*args, **kwargs):
        args_original = args
        args = list(map(unwrap, args))
        kwargs = {k: unwrap(v) for k, v, in kwargs.items()}
        result = f(*args, **kwargs)
        return wrap(result)
    return wrapper
