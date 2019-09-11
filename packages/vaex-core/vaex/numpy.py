import numpy as np

import vaex
from vaex import Expression


# implementing nep18: https://numpy.org/neps/nep-0018-array-function-protocol.html
_nep18_method_mapping = {}  # maps from numpy function to an Expression method
def nep18_method(numpy_function):
    def decorator(f):
        _nep18_method_mapping[numpy_function] = f
        return f
    return decorator


# # implementing nep13: https://numpy.org/neps/nep-0013-ufunc-overrides.html
_nep13_method_mapping = {}  # maps from numpy function to an Expression method
# def nep13_method(numpy_function):
#     def decorator(f):
#         _nep13_method_mapping[numpy_function] = f
#         return f
#     return decorator


def nep13_and_18_method(numpy_function):
    def decorator(f):
        _nep13_method_mapping[numpy_function] = f
        _nep18_method_mapping[numpy_function] = f
        return f
    return decorator


class DataFrameAccessorNumpy:
    def __init__(self, df, transposed=False, column_names=None):
        self._df = df
        self._transposed = transposed
        if column_names is None:
            column_names = df.get_column_names()
        self.column_names = column_names
        self._allow_array_casting = df._allow_array_casting

    @property
    def df(self):
        return self._df[self.column_names]

    def __repr__(self):
        names = ', '.join(self.column_names)
        return f'numpy wrapper for columns {names}: \n'  + repr(self.df)

    def __len__(self):
        return len(self._df)

    def __array__(self, dtype=None, parallel=True):
        if not self._allow_array_casting:
            raise RuntimeError('casting a dataframe numpy view to an array is explicitly disabled')
        try:
            _previous__allow_array_casting = self._df._allow_array_casting
            self._df._allow_array_casting = True
            ar = self.df.__array__(dtype=dtype, parallel=parallel)
        finally:
            self._df._allow_array_casting = _previous__allow_array_casting
        return ar.T if self._transposed else ar

    # def __iter__(self):
    #     """Iterator over the column names."""
    #     if self._transposed:
    #         for name in self.df.get_column_names():
    #             yield self[name]
    #     else:
    #         raise ValueError("Iterating over rows is not supported")

    def __getitem__(self, item):
        if isinstance(item, tuple):
            row, col = item
            if row != slice(None, None, None):
                raise ValueError("Please don't slice on row basis")
            if isinstance(col, int):
                return self._df[self.column_names[col]]
            elif isinstance(col, slice):
                column_names = self.column_names.__getitem__(col)
            elif isinstance(col, (list, tuple)):
                column_names = [self.column_names[k] for k in col]
            return DataFrameAccessorNumpy(self._df, self._transposed, column_names)
        elif isinstance(item, slice):
            return self[item, :]
        if isinstance(item, int):
            if self._transposed:
                return self[:, item]
            else:
                raise ValueError(f'Cannot get row {item}, only getting columns is supported')
        elif isinstance(item, str):
            return self._df[item]
        else:
            raise ValueError(f'Item not understood: {item}')

    def __setitem__(self, item, value):
        if isinstance(item, tuple):
            row, col = item
            if row != slice(None, None, None):
                raise ValueError("Please don't slice on row basis")
            if isinstance(col, int):
                column_names = self.column_names[col]
            elif isinstance(col, slice):
                column_names = self.column_names.__getitem__(col)
            elif isinstance(col, (list, tuple)):
                column_names = [self.column_names[k] for k in col]
            item = (row, column_names)
            if isinstance(value, DataFrameAccessorNumpy):
                value = value._df[value.column_names]
            self._df.__setitem__(item, value)
        elif isinstance(item, slice):
            self[item, :] = value
        elif isinstance(item, int):
            if not self._transposed:
                raise ValueError("Cannot assign to rows")
            item = (slice(None), item)
            self._df.__setitem__(item, value)
        elif isinstance(item, (tuple, list)):
            if not self._transposed:
                raise ValueError("Cannot assign to rows")
            item = (slice(None), item)
            self._df.__setitem__(item, value)
        else:
            raise ValueError(f'Item not understood: {item}')

    @property
    def numpy(self):
        return self

    @property
    def T(self):
        return type(self)(self._df, transposed=not self._transposed, column_names=self.column_names)

    @property
    def shape(self):
        if self._transposed:
            return (len(self.column_names), len(self))
        else:
            return (len(self), len(self.column_names))

    @property
    def ndim(self):
        return 2

    @property
    def dtype(self):
        dtypes = [self._df[k].dtype for k in self.column_names]
        assert all([dtypes[0] == dtype for dtype in dtypes])
        return dtypes[0]

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        method = _nep13_method_mapping.get(ufunc)
        if method is None:
            return NotImplemented
        return method(*inputs, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        method = _nep18_method_mapping.get(func)
        if method is None:
            return NotImplemented
        assert args[0] is self._df or args[0] is self
        result = method(*args, **kwargs)
        return result


    @nep18_method(np.empty_like)
    def _np_full_like(self, value, dtype=None, order='K', subok=True, shape=None):
        # we ignore order
        if shape is None:
            shape = self.shape
        assert subok
        # assert dtype in [None, self.dtype]
        if dtype is None:
            dtype = self.dtype
        rows, columns = shape
        # TODO: instead of vrange, we might want to have a vaex.zeros
        vcol = vaex.vrange(0, rows, dtype=dtype)
        df = vaex.from_dict({f'c{i}': vcol for i in range(columns)})
        if value:
            for name in df:
                df[name] = df[name] * 0 + value
        return df.numpy

    @nep18_method(np.empty_like)
    def _np_empty_like(self, dtype=None, order='K', subok=True, shape=None):
        return self._np_full_like(0, dtype, order, subok, shape)

    @nep18_method(np.zeros_like)
    def _np_zeros_like(self, dtype=None, order='K', subok=True, shape=None):
        return self._np_full_like(0, dtype, order, subok, shape)

    @nep18_method(np.ones_like)
    def _np_ones_like(self, dtype=None, order='K', subok=True, shape=None):
        return self._np_full_like(1, dtype, order, subok, shape)

    @nep18_method(np.mean)
    def _np_mean(self, axis=None):
        assert axis in [0, None]
        return self.mean(self.get_column_names())

    @nep18_method(np.unique)
    def _np_unique(self, axis=None):
        assert axis is None
        import pdb; pdb.set_trace()
        # assert axis in [0, None]
        # if axis is None:
        # return self.mean(self.get_column_names())

    @nep18_method(np.dot)
    def _dot(self, b):
        b = np.asarray(b)
        assert b.ndim == 2
        N = b.shape[1]
        df = self.copy()
        names = df.get_column_names()
        output_names = ['c'+str(i) for i in range(N)]
        columns = [df[names[j]] for j in range(b.shape[0])]
        for name in names:
            df._hide_column(name)
        for i in range(N):
            def dot_product(a, b):
                products = ['%s * %s' % (ai, bi) for ai, bi in zip(a, b)]
                return ' + '.join(products)
            df[output_names[i]] = dot_product(columns, b[:,i])
        return df.numpy

    @nep18_method(np.may_share_memory)
    def _np__may_share_memory(self, b):
        return True  # be conservative

    @nep18_method(np.linalg.svd)
    def _np_linalg_svd(self, full_matrices=True):
        import dask.array as da
        import dask
        X = self.to_dask_array()
        # TODO: we ignore full_matrices
        u, s, v = da.linalg.svd(X)#, full_matrices=full_matrices)
        u, s, v = dask.compute(u, s, v)
        return u, s, v

    @nep18_method(np.linalg.qr)
    def _np_linalg_qr(self):
        import dask.array as da
        import dask
        X = self.to_dask_array()
        result = da.linalg.qr(X)
        result = dask.compute(*result)
        return result


for op in vaex.expression._binary_ops:
    name = op.get('numpy_name', op['name'])
    if name in ['contains', 'is', 'is_not']:
        continue
    numpy_function = getattr(np, name)
    assert numpy_function, 'numpy does not have {}'.format(name)

    def closure(name=name, numpy_function=numpy_function):
        def binary_op_method(self, rhs, out=None, casting=None):
            assert casting in [None, 'no', 'same_kind']
            if isinstance(self, DataFrameAccessorNumpy):
                df = self.df
            else:
                df = self
                self = DataFrameAccessorNumpy(self)
            df = df.copy()
            if isinstance(rhs, np.ndarray):
                if self._transposed:
                    name = vaex.utils.find_valid_name('__aux', used=df.get_column_names(hidden=True))
                    df.add_column(name, rhs)
                    rhs = df[name]
                    for i, name in enumerate(self.df.get_column_names()):
                        df[name] = numpy_function(df[name], rhs)
                else:
                    for i, name in enumerate(self.df.get_column_names()):
                        df[name] = numpy_function(df[name], rhs[i])
            elif isinstance(rhs, (vaex.dataframe.DataFrame, DataFrameAccessorNumpy)):
                if isinstance(rhs, DataFrameAccessorNumpy):
                    rhs = rhs.df[rhs.column_names]
                # when the rhs is a dataframe, we join/stack the dataframes first
                dfj = df.join(rhs, rprefix="rhs_")
                names_left = df.get_column_names()
                names_right = dfj.get_column_names()[len(names_left):]
                df = dfj.copy()
                if len(names_left) == 1 and len(names_right) > 1:
                    # broadcast left
                    name_left = names_left[0]
                    for name_right in names_right:
                        df[name_right] = numpy_function(df[name_left], df[name_right])
                    df._hide_column(name_left)
                elif len(names_right) == 1 and len(names_left) > 1:
                    # broadcast right
                    name_right = names_right[0]
                    for name_left in names_left:
                        df[name_left] = numpy_function(df[name_left], df[name_right])
                    df._hide_column(name_right)
                elif len(names_right) == len(names_left):
                    for name_left, name_right in zip(names_left, names_right):
                        df[name_left] = numpy_function(df[name_left], df[name_right])
                        df._hide_column(name_right)
                else:
                    raise ValueError("cannot broadcast")
            else:
                for i, name in enumerate(self.df.get_column_names()):
                    df[name] = numpy_function(df[name], rhs)
            if out is not None:
                if isinstance(out, tuple):
                    assert len(out) == 1, "unexpected length of tuple of out argument"
                    out = out[0]
                assert isinstance(out, (vaex.dataframe.DataFrame, DataFrameAccessorNumpy)), 'only output to dataframe or numpy accessor supported'
                if isinstance(out, DataFrameAccessorNumpy):
                    names_left = out.column_names
                    out = out._df
                else:
                    names_left = out.get_column_names()
                out[:, names_left] = df
                df = out
                out[:, names_left] = df
                df = out
            if self._transposed:
                return df.T
            else:
                return df
        return binary_op_method

    # this implements e.g. numpy.multiply
    nep13_and_18_method(numpy_function)(closure())

    # while this implements the __mul__ method
    def closure2(numpy_function=numpy_function):
        def f(a, b):
            return numpy_function(a, b)
        return f

    dundername = '__{}__'.format(name)
    setattr(DataFrameAccessorNumpy, dundername, closure2())


for op in vaex.expression._unary_ops:
    name = op['name']
    numpy_name = op.get('numpy_name', name)
    numpy_function = getattr(np, numpy_name)
    assert numpy_function, 'numpy does not have {}'.format(name)
    def closure(name=name, numpy_function=numpy_function):
        def unary_op_method(self):
            if isinstance(self, DataFrameAccessorNumpy):
                df = self.df
            else:
                df = self
                self = DataFrameAccessorNumpy(self)
            df = df.copy()
            for i, name in enumerate(df.get_column_names()):
                df[name] = numpy_function(df[name])
            if self._transposed:
                return df.T
            else:
                return df
        return unary_op_method
    nep13_and_18_method(numpy_function)(closure())

    def closure2(numpy_function=numpy_function):
        def f(a):
            return numpy_function(a)
        return f

    dundername = '__{}__'.format(name)
    setattr(DataFrameAccessorNumpy, dundername, closure2())


for name, numpy_name in vaex.functions.numpy_function_mapping + [('isnan', 'isnan')]:
    numpy_function = getattr(np, numpy_name)
    assert numpy_function, 'numpy does not have {}'.format(numpy_name)
    def closure(name=name, numpy_name=numpy_name, numpy_function=numpy_function):
        def forward_call(self, *args, **kwargs):
            if isinstance(self, DataFrameAccessorNumpy):
                df = self.df
            else:
                df = self
                self = df.numpy
            df = df.copy()
            out = None
            if 'out' in kwargs:
                out = kwargs.pop('out')
            print(numpy_function, args, kwargs)
            if numpy_function == np.clip:
                # TODO: can we copy something from numpy here?
                args = [np.array(arg) if isinstance(arg, (tuple, list)) else arg for arg in args]
                broadcast_arg = [isinstance(arg, np.ndarray) for arg in args]
                # args_copy = args.copy()
                for i, name in enumerate(df.get_column_names()):
                    broadcasted_args = [arg[i] if broadcast else arg for arg, broadcast in zip(args, broadcast_arg)]
                    df[name] = numpy_function(df[name], *broadcasted_args, **kwargs)
                # import pdb; pdb.set_trace()
            else:
                for name in df.get_column_names():
                    df[name] = numpy_function(df[name], *args, **kwargs)
            if isinstance(out, tuple):
                assert len(out) == 1, "unexpected length of tuple of out argument"
                out = out[0]
            if out is not None:
                # import pdb; pdb.set_trace()
                assert isinstance(out, (vaex.dataframe.DataFrame, DataFrameAccessorNumpy)),\
                    'only output to dataframe or numpy accessor supported, not %r' % out
                if isinstance(out, DataFrameAccessorNumpy):
                    names_left = out.column_names
                    out = out._df
                else:
                    names_left = out.get_column_names()
                out[:, names_left] = df
                df = out
                out[:, names_left] = df
                df = out
            if self._transposed:
                return df.T
            else:
                return df
        return forward_call

    nep13_and_18_method(numpy_function)(closure())


aggregates_functions = [
    'nanmin',
    'nanmax',
    'nansum',
    'sum',
    'var',
    'nanvar'
]

for numpy_name in aggregates_functions:
    numpy_function = getattr(np, numpy_name)
    assert numpy_function, 'numpy does not have {}'.format(numpy_name)
    def closure(numpy_name=numpy_name, numpy_function=numpy_function):
        def forward_call(self, *args, **kwargs):
            if isinstance(self, DataFrameAccessorNumpy):
                df = self.df
            else:
                df = self
            results = []
            forward_kwargs = kwargs.copy()
            forward_kwargs['delay'] = True  # we do all aggregates in 1 pass
            if 'axis' in kwargs:
                if kwargs['axis'] == 0 and not self._transposed:
                    pass  # this is fine
                elif kwargs['axis'] == 1 and self._transposed:
                    forward_kwargs['axis'] = 0  # since we are transposed we need to change this axis
                else:
                    raise ValueError("not supported: numpy.%s with kwargs %r" % (numpy_name, kwargs))                
            for name in df.get_column_names():
                method = vaex.expression._nep18_method_mapping[numpy_function]
                results.append(method(*(df[name],) + args, **forward_kwargs))
            df.execute()
            results = [k.get() for k in results]
            # TODO: support axis argument
            results = np.array(results)
            if 'axis' in kwargs:
                return results
            return numpy_function(results)
        return forward_call

    nep13_and_18_method(numpy_function)(closure())
