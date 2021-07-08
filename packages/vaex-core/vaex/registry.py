"""This module contains the `register_function` decorator to add expression methods to vaex dataframe."""

import functools

import vaex.arrow
import vaex.expression
import vaex.multiprocessing

scopes = {
    'str': vaex.expression.StringOperations,
    'str_pandas': vaex.expression.StringOperationsPandas,
    'dt': vaex.expression.DateTime,
    'td': vaex.expression.TimeDelta,
    'struct': vaex.expression.StructOperations
}


def register_function(scope=None, as_property=False, name=None, on_expression=True, df_accessor=None,
                      multiprocessing=False):
    """Decorator to register a new function with vaex.

    If on_expression is True, the function will be available as a method on an
    Expression, where the first argument will be the expression itself.

    If `df_accessor` is given, it is added as a method to that dataframe accessor (see e.g. vaex/geo.py)

    Example:

    >>> import vaex
    >>> df = vaex.example()
    >>> @vaex.register_function()
    >>> def invert(x):
    >>>     return 1/x
    >>> df.x.invert()


    >>> import numpy as np
    >>> df = vaex.from_arrays(departure=np.arange('2015-01-01', '2015-12-05', dtype='datetime64'))
    >>> @vaex.register_function(as_property=True, scope='dt')
    >>> def dt_relative_day(x):
    >>>     return vaex.functions.dt_dayofyear(x)/365.
    >>> df.departure.dt.relative_day
    """
    import vaex.multiprocessing
    prefix = ''
    if scope:
        prefix = scope + "_"
        if scope not in scopes:
            raise KeyError("unknown scope")

    def wrapper(f, name=name):
        name = name or f.__name__
        # remove possible prefix
        if name.startswith(prefix):
            name = name[len(prefix):]
        full_name = prefix + name
        if df_accessor:
            def closure(name=name, full_name=full_name, function=f):
                def wrapper(self, *args, **kwargs):
                    lazy_func = getattr(self.df.func, full_name)
                    lazy_func = vaex.arrow.numpy_dispatch.autowrapper(lazy_func)
                    return vaex.multiprocessing.apply(lazy_func, args, kwargs, multiprocessing)

                return functools.wraps(function)(wrapper)

            if as_property:
                setattr(df_accessor, name, property(closure()))
            else:
                setattr(df_accessor, name, closure())
        else:
            if on_expression:
                if scope:
                    def closure(name=name, full_name=full_name, function=f):
                        def wrapper(self, *args, **kwargs):
                            lazy_func = getattr(self.expression.ds.func, full_name)
                            lazy_func = vaex.arrow.numpy_dispatch.autowrapper(lazy_func)
                            args = (self.expression,) + args
                            return vaex.multiprocessing.apply(lazy_func, args, kwargs, multiprocessing)

                        return functools.wraps(function)(wrapper)

                    if as_property:
                        setattr(scopes[scope], name, property(closure()))
                    else:
                        setattr(scopes[scope], name, closure())
                else:
                    def closure(name=name, full_name=full_name, function=f):
                        def wrapper(self, *args, **kwargs):
                            lazy_func = getattr(self.ds.func, full_name)
                            lazy_func = vaex.arrow.numpy_dispatch.autowrapper(lazy_func)
                            args = (self,) + args
                            return vaex.multiprocessing.apply(lazy_func, args, kwargs, multiprocessing=multiprocessing)

                        return functools.wraps(function)(wrapper)

                    setattr(vaex.expression.Expression, name, closure())
        vaex.expression.expression_namespace[prefix + name] = vaex.arrow.numpy_dispatch.autowrapper(f)
        return f  # we leave the original function as is

    return wrapper
