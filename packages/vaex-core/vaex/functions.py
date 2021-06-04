import vaex.serialize
import json
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from vaex import column
from vaex.column import _to_string_sequence, _to_string_column, _to_string_list_sequence, _is_stringy
from vaex.dataframe import docsubst
import vaex.arrow.numpy_dispatch
import vaex.arrow.utils
import re
import vaex.expression
import functools
import six


# @vaex.serialize.register_function
# class Function(FunctionSerializable):


def _arrow_string_might_grow(ar, factor):
    # will upcast utf8 string to large_utf8 string when needed
    # TODO: in the future we might be able to rely on Apache Arrow for this
    # TODO: placeholder, needs a test
    return ar


def _arrow_string_kernel_dispatch(name, ascii, *args):
    # helper function to call a pyarrow kernel
    variant = 'ascii' if ascii else 'utf8'
    kernel_name = f'{variant}_{name}'  # eg utf8_istitle / ascii_istitle
    return pc.call_function(kernel_name, args)


scopes = {
    'str': vaex.expression.StringOperations,
    'str_pandas': vaex.expression.StringOperationsPandas,
    'dt': vaex.expression.DateTime,
    'td': vaex.expression.TimeDelta
}

def register_function(scope=None, as_property=False, name=None, on_expression=True, df_accessor=None, multiprocessing=False):
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
                            args = (self.expression, ) + args
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


def auto_str_unwrap(f):
    '''Will take the first argument, and if a list, will unwrap/unraval, and apply f, and wrap again'''
    @functools.wraps(f)
    def decorated(x, *args, **kwargs):
        if not isinstance(x, vaex.column.supported_column_types):
            return f(x, *args, **kwargs)
        dtype = vaex.dtype_of(x)
        if dtype.is_list:
            x, wrapper = vaex.arrow.utils.list_unwrap(x)
            x = f(x, *args, **kwargs)
            return wrapper(x)
        else:
            return f(x, *args, **kwargs)
    return decorated

# name maps to numpy function
# <vaex name>:<numpy name>
numpy_function_mapping = [name.strip().split(":") if ":" in name else (name, name) for name in """
abs
arccos
arccosh
arcsin
arcsinh
arctan
arctan2
arctanh
clip
cos
cosh
deg2rad
digitize
exp
expm1
isfinite
isinf
log
log10
log1p
maximum
minimum
rad2deg
round
searchsorted
sin
sinc
sinh
sqrt
tan
tanh
""".strip().split()]
for name, numpy_name in numpy_function_mapping:
    if not hasattr(np, numpy_name):
        raise SystemError("numpy does not have: %s" % numpy_name)
    else:
        function = getattr(np, numpy_name)
        def f(function=function):
            def wrapper(*args, **kwargs):
                # convert to numpy
                args = [vaex.arrow.numpy_dispatch.wrap(k) for k in args]
                numpy_data = [k.numpy_array if isinstance(k, vaex.arrow.numpy_dispatch.NumpyDispatch) else k for k in args]
                result = function(*numpy_data, **kwargs)
                # and put back the masks
                for arg in args:
                    if isinstance(arg, vaex.arrow.numpy_dispatch.NumpyDispatch):
                        result = arg.add_missing(result)
                return vaex.arrow.numpy_dispatch.wrap(result)
            return wrapper
        function = f()
        function.__doc__ = "Lazy wrapper around :py:data:`numpy.%s`" % name

        register_function(name=name)(function)


@register_function()
def list_sum(ar,  fill_empty=0):
    fill_missing_item = 0  # a missing value gets replaced by 0, so it is effectively ignored
    fill_empty = 0  # empty list [], would result in 0
    # TODO: we might want to extract some of this to vaex.arrow.utils or even
    # upstream (it Arrow)
    dtype = vaex.dtype_of(ar)
    assert dtype.is_list
    offsets = vaex.array_types.to_numpy(ar.offsets)
    values = pc.fill_null(ar.values.slice(offsets[0]), fill_missing_item)
    values = vaex.array_types.to_numpy(values)

    zero_length = (offsets[1:] - offsets[:-1]) == 0
    if len(zero_length) == 1:  # special case, values is an empy array
        return pa.array([fill_empty], type=dtype.value_type.arrow)

    # we skip over empty lists and nulls, otherwise we'll have indices of -1
    skips = (~zero_length).argmax(axis=0)
    cumsum = values.cumsum()
    list_end_offset = offsets[skips+1:] - offsets[0] - 1
    sums = np.diff(cumsum[list_end_offset], prepend=0)
    if skips:
        sums = np.concatenate([np.zeros(skips, dtype=sums.dtype), sums])

    if fill_empty != 0:  # by default this is already the case
        sums[zero_length] = fill_empty
    sums = vaex.array_types.to_arrow(sums)
    from .arrow.utils import combine_missing
    sums = combine_missing(sums, ar)
    return sums


@register_function()
def fillmissing(ar, value):
    '''Returns an array where missing values are replaced by value.
    See :`ismissing` for the definition of missing values.
    '''
    dtype = vaex.dtype_of(ar)
    if dtype == str:
        return pc.fill_null(ar, value)
    ar = ar if not isinstance(ar, column.Column) else ar.to_numpy()
    mask = ismissing(ar)
    if np.any(mask):
        if np.ma.isMaskedArray(ar):
            ar = ar.data.copy()
        else:
            ar = ar.copy()
        ar[mask] = value
    return ar


@register_function()
def fillnan(ar, value):
    '''Returns an array where nan values are replaced by value.
    See :`isnan` for the definition of missing values.
    '''
    # TODO: optimize, we don't want to convert string to numpy
    # they will never contain nan
    if not _is_stringy(ar):
        ar = ar if not isinstance(ar, column.Column) else ar.to_numpy()
        ar = vaex.array_types.to_numpy(ar, strict=True)
        if ar.dtype.kind in 'fO':
            mask = isnan(ar)
            if np.any(mask):
                ar = ar.copy()
                ar[mask] = value
    return ar


@register_function()
def fillna(ar, value):
    '''Returns an array where NA values are replaced by value.
    See :`isna` for the definition of missing values.

    '''
    dtype = vaex.dtype_of(ar)
    if dtype == str:
        # str cannot contain anything other than missing values
        return fillmissing(ar, value)
    # TODO: should use arrow fill_null in the future
    if isinstance(ar, vaex.array_types.supported_arrow_array_types):
        ar = fillna(vaex.array_types.to_numpy(ar, strict=True), value)
        return vaex.array_types.to_arrow(ar)
    ar = ar if not isinstance(ar, column.Column) else ar.to_numpy()
    mask = isna(ar)
    if np.any(mask):
        if np.ma.isMaskedArray(ar):
            ar = ar.data.copy()
        else:
            ar = ar.copy()
        ar[mask] = value
    return ar

@register_function()
def ismissing(x):
    """Returns True where there are missing values (masked arrays), missing strings or None"""
    if np.ma.isMaskedArray(x):
        if x.dtype.kind in 'O':
            if x.mask is not None:
                return (x.data == None) | x.mask
            else:
                return (x.data == None)
        else:
            return x.mask == 1
    else:
        if isinstance(x, vaex.array_types.supported_arrow_array_types):
            return pa.compute.is_null(x)
        elif not isinstance(x, np.ndarray) or x.dtype.kind in 'US':
            x = _to_string_sequence(x)
            mask = x.mask()
            if mask is None:
                mask = np.zeros(x.length, dtype=np.bool)
            return mask
        elif isinstance(x, np.ndarray) and x.dtype.kind in 'O':
            return x == None
        else:
            return np.zeros(len(x), dtype=np.bool)


@register_function()
def notmissing(x):
    return ~ismissing(x)


@register_function()
def isnan(x):
    """Returns an array where there are NaN values"""
    if isinstance(x, vaex.array_types.supported_arrow_array_types):
        x = vaex.array_types.to_numpy(x)
    if isinstance(x, np.ndarray):
        if np.ma.isMaskedArray(x):
            # we don't want a masked arrays
            w = x.data != x.data
            w[x.mask] = False
            return w
        else:
            return x != x
    else:
        return np.zeros(len(x), dtype=np.bool)


@register_function()
def notnan(x):
    return ~isnan(x)


@register_function()
def isna(x):
    """Returns a boolean expression indicating if the values are Not Availiable (missing or NaN)."""
    return isnan(x) | ismissing(x)


@register_function()
def notna(x):
    """Opposite of isna"""
    return ~isna(x)


########## datetime operations ##########


def _pandas_dt_fix(x):
    # see https://github.com/pandas-dev/pandas/issues/23276
    import pandas as pd
    # not sure which version this is fixed in
    if not x.flags['WRITEABLE']:
        x = x.copy()
    return x

def _to_pandas_series(x):
    import pandas as pd
    # pandas seems to eager to infer dtype=object for v1.2
    return pd.Series(_pandas_dt_fix(x), dtype=x.dtype)

@register_function(scope='dt', as_property=True)
def dt_date(x):
    """Return the date part of the datetime value

    :returns: an expression containing the date portion of a datetime value

    Example:

    >>> import vaex
    >>> import numpy as np
    >>> date = np.array(['2009-10-12T03:31:00', '2016-02-11T10:17:34', '2015-11-12T11:34:22'], dtype=np.datetime64)
    >>> df = vaex.from_arrays(date=date)
    >>> df
      #  date
      0  2009-10-12 03:31:00
      1  2016-02-11 10:17:34
      2  2015-11-12 11:34:22

    >>> df.date.dt.date
    Expression = dt_date(date)
    Length: 3 dtype: datetime64[D] (expression)
    -------------------------------------------
    0  2009-10-12
    1  2016-02-11
    2  2015-11-12
    """
    import pandas as pd
    return _to_pandas_series(x).dt.date.values.astype(np.datetime64)

@register_function(scope='dt', as_property=True)
def dt_dayofweek(x):
    """Obtain the day of the week with Monday=0 and Sunday=6

    :returns: an expression containing the day of week.

    Example:

    >>> import vaex
    >>> import numpy as np
    >>> date = np.array(['2009-10-12T03:31:00', '2016-02-11T10:17:34', '2015-11-12T11:34:22'], dtype=np.datetime64)
    >>> df = vaex.from_arrays(date=date)
    >>> df
      #  date
      0  2009-10-12 03:31:00
      1  2016-02-11 10:17:34
      2  2015-11-12 11:34:22

    >>> df.date.dt.dayofweek
    Expression = dt_dayofweek(date)
    Length: 3 dtype: int64 (expression)
    -----------------------------------
    0  0
    1  3
    2  3
    """
    import pandas as pd
    return _to_pandas_series(x).dt.dayofweek.values

@register_function(scope='dt', as_property=True)
def dt_dayofyear(x):
    """The ordinal day of the year.

    :returns: an expression containing the ordinal day of the year.

    Example:

    >>> import vaex
    >>> import numpy as np
    >>> date = np.array(['2009-10-12T03:31:00', '2016-02-11T10:17:34', '2015-11-12T11:34:22'], dtype=np.datetime64)
    >>> df = vaex.from_arrays(date=date)
    >>> df
      #  date
      0  2009-10-12 03:31:00
      1  2016-02-11 10:17:34
      2  2015-11-12 11:34:22

    >>> df.date.dt.dayofyear
    Expression = dt_dayofyear(date)
    Length: 3 dtype: int64 (expression)
    -----------------------------------
    0  285
    1   42
    2  316
    """
    import pandas as pd
    return _to_pandas_series(x).dt.dayofyear.values

@register_function(scope='dt', as_property=True)
def dt_is_leap_year(x):
    """Check whether a year is a leap year.

    :returns: an expression which evaluates to True if a year is a leap year, and to False otherwise.

    Example:

    >>> import vaex
    >>> import numpy as np
    >>> date = np.array(['2009-10-12T03:31:00', '2016-02-11T10:17:34', '2015-11-12T11:34:22'], dtype=np.datetime64)
    >>> df = vaex.from_arrays(date=date)
    >>> df
      #  date
      0  2009-10-12 03:31:00
      1  2016-02-11 10:17:34
      2  2015-11-12 11:34:22

    >>> df.date.dt.is_leap_year
    Expression = dt_is_leap_year(date)
    Length: 3 dtype: bool (expression)
    ----------------------------------
    0  False
    1   True
    2  False
    """
    import pandas as pd
    return _to_pandas_series(x).dt.is_leap_year.values

@register_function(scope='dt', as_property=True)
def dt_year(x):
    """Extracts the year out of a datetime sample.

    :returns: an expression containing the year extracted from a datetime column.

    Example:

    >>> import vaex
    >>> import numpy as np
    >>> date = np.array(['2009-10-12T03:31:00', '2016-02-11T10:17:34', '2015-11-12T11:34:22'], dtype=np.datetime64)
    >>> df = vaex.from_arrays(date=date)
    >>> df
      #  date
      0  2009-10-12 03:31:00
      1  2016-02-11 10:17:34
      2  2015-11-12 11:34:22

    >>> df.date.dt.year
    Expression = dt_year(date)
    Length: 3 dtype: int64 (expression)
    -----------------------------------
    0  2009
    1  2016
    2  2015
    """
    import pandas as pd
    return _to_pandas_series(x).dt.year.values

@register_function(scope='dt', as_property=True)
def dt_month(x):
    """Extracts the month out of a datetime sample.

    :returns: an expression containing the month extracted from a datetime column.

    Example:

    >>> import vaex
    >>> import numpy as np
    >>> date = np.array(['2009-10-12T03:31:00', '2016-02-11T10:17:34', '2015-11-12T11:34:22'], dtype=np.datetime64)
    >>> df = vaex.from_arrays(date=date)
    >>> df
      #  date
      0  2009-10-12 03:31:00
      1  2016-02-11 10:17:34
      2  2015-11-12 11:34:22

    >>> df.date.dt.month
    Expression = dt_month(date)
    Length: 3 dtype: int64 (expression)
    -----------------------------------
    0  10
    1   2
    2  11
    """
    import pandas as pd
    return _to_pandas_series(x).dt.month.values

@register_function(scope='dt', as_property=True)
def dt_month_name(x):
    """Returns the month names of a datetime sample in English.

    :returns: an expression containing the month names extracted from a datetime column.

    Example:

    >>> import vaex
    >>> import numpy as np
    >>> date = np.array(['2009-10-12T03:31:00', '2016-02-11T10:17:34', '2015-11-12T11:34:22'], dtype=np.datetime64)
    >>> df = vaex.from_arrays(date=date)
    >>> df
      #  date
      0  2009-10-12 03:31:00
      1  2016-02-11 10:17:34
      2  2015-11-12 11:34:22

    >>> df.date.dt.month_name
    Expression = dt_month_name(date)
    Length: 3 dtype: str (expression)
    ---------------------------------
    0   October
    1  February
    2  November
    """
    import pandas as pd
    return pa.array(_to_pandas_series(x).dt.month_name())

@register_function(scope='dt', as_property=True)
def dt_quarter(x):
    """Extracts the quarter from a datetime sample.

    :returns: an expression containing the number of the quarter extracted from a datetime column.

    Example:

    >>> import vaex
    >>> import numpy as np
    >>> date = np.array(['2009-10-12T03:31:00', '2016-02-11T10:17:34', '2015-11-12T11:34:22'], dtype=np.datetime64)
    >>> df = vaex.from_arrays(date=date)
    >>> df
      #  date
      0  2009-10-12 03:31:00
      1  2016-02-11 10:17:34
      2  2015-11-12 11:34:22

    >>> df.date.dt.quarter
    Expression = dt_quarter(date)
    Length: 3 dtype: int64 (expression)
    -----------------------------------
    0  4
    1  1
    2  4
    """
    import pandas as pd
    return _to_pandas_series(x).dt.quarter.values

@register_function(scope='dt', as_property=True)
def dt_day(x):
    """Extracts the day from a datetime sample.

    :returns: an expression containing the day extracted from a datetime column.

    Example:

    >>> import vaex
    >>> import numpy as np
    >>> date = np.array(['2009-10-12T03:31:00', '2016-02-11T10:17:34', '2015-11-12T11:34:22'], dtype=np.datetime64)
    >>> df = vaex.from_arrays(date=date)
    >>> df
      #  date
      0  2009-10-12 03:31:00
      1  2016-02-11 10:17:34
      2  2015-11-12 11:34:22

    >>> df.date.dt.day
    Expression = dt_day(date)
    Length: 3 dtype: int64 (expression)
    -----------------------------------
    0  12
    1  11
    2  12
    """
    import pandas as pd
    return _to_pandas_series(x).dt.day.values

@register_function(scope='dt', as_property=True)
def dt_day_name(x):
    """Returns the day names of a datetime sample in English.

    :returns: an expression containing the day names extracted from a datetime column.

    Example:

    >>> import vaex
    >>> import numpy as np
    >>> date = np.array(['2009-10-12T03:31:00', '2016-02-11T10:17:34', '2015-11-12T11:34:22'], dtype=np.datetime64)
    >>> df = vaex.from_arrays(date=date)
    >>> df
      #  date
      0  2009-10-12 03:31:00
      1  2016-02-11 10:17:34
      2  2015-11-12 11:34:22

    >>> df.date.dt.day_name
    Expression = dt_day_name(date)
    Length: 3 dtype: str (expression)
    ---------------------------------
    0    Monday
    1  Thursday
    2  Thursday
    """
    import pandas as pd
    return pa.array(_to_pandas_series(x).dt.day_name())

@register_function(scope='dt', as_property=True)
def dt_weekofyear(x):
    """Returns the week ordinal of the year.

    :returns: an expression containing the week ordinal of the year, extracted from a datetime column.

    Example:

    >>> import vaex
    >>> import numpy as np
    >>> date = np.array(['2009-10-12T03:31:00', '2016-02-11T10:17:34', '2015-11-12T11:34:22'], dtype=np.datetime64)
    >>> df = vaex.from_arrays(date=date)
    >>> df
      #  date
      0  2009-10-12 03:31:00
      1  2016-02-11 10:17:34
      2  2015-11-12 11:34:22

    >>> df.date.dt.weekofyear
    Expression = dt_weekofyear(date)
    Length: 3 dtype: int64 (expression)
    -----------------------------------
    0  42
    1   6
    2  46
    """
    import pandas as pd
    return _to_pandas_series(x).dt.weekofyear.values

@register_function(scope='dt', as_property=True)
def dt_hour(x):
    """Extracts the hour out of a datetime samples.

    :returns: an expression containing the hour extracted from a datetime column.

    Example:

    >>> import vaex
    >>> import numpy as np
    >>> date = np.array(['2009-10-12T03:31:00', '2016-02-11T10:17:34', '2015-11-12T11:34:22'], dtype=np.datetime64)
    >>> df = vaex.from_arrays(date=date)
    >>> df
      #  date
      0  2009-10-12 03:31:00
      1  2016-02-11 10:17:34
      2  2015-11-12 11:34:22

    >>> df.date.dt.hour
    Expression = dt_hour(date)
    Length: 3 dtype: int64 (expression)
    -----------------------------------
    0   3
    1  10
    2  11
    """
    import pandas as pd
    return _to_pandas_series(x).dt.hour.values

@register_function(scope='dt', as_property=True)
def dt_minute(x):
    """Extracts the minute out of a datetime samples.

    :returns: an expression containing the minute extracted from a datetime column.

    Example:

    >>> import vaex
    >>> import numpy as np
    >>> date = np.array(['2009-10-12T03:31:00', '2016-02-11T10:17:34', '2015-11-12T11:34:22'], dtype=np.datetime64)
    >>> df = vaex.from_arrays(date=date)
    >>> df
      #  date
      0  2009-10-12 03:31:00
      1  2016-02-11 10:17:34
      2  2015-11-12 11:34:22

    >>> df.date.dt.minute
    Expression = dt_minute(date)
    Length: 3 dtype: int64 (expression)
    -----------------------------------
    0  31
    1  17
    2  34
    """
    import pandas as pd
    return _to_pandas_series(x).dt.minute.values

@register_function(scope='dt', as_property=True)
def dt_second(x):
    """Extracts the second out of a datetime samples.

    :returns: an expression containing the second extracted from a datetime column.

    Example:

    >>> import vaex
    >>> import numpy as np
    >>> date = np.array(['2009-10-12T03:31:00', '2016-02-11T10:17:34', '2015-11-12T11:34:22'], dtype=np.datetime64)
    >>> df = vaex.from_arrays(date=date)
    >>> df
      #  date
      0  2009-10-12 03:31:00
      1  2016-02-11 10:17:34
      2  2015-11-12 11:34:22

    >>> df.date.dt.second
    Expression = dt_second(date)
    Length: 3 dtype: int64 (expression)
    -----------------------------------
    0   0
    1  34
    2  22
    """
    import pandas as pd
    return _to_pandas_series(x).dt.second.values

@register_function(scope='dt')
def dt_strftime(x, date_format):
    """Returns a formatted string from a datetime sample.

    :returns: an expression containing a formatted string extracted from a datetime column.

    Example:

    >>> import vaex
    >>> import numpy as np
    >>> date = np.array(['2009-10-12T03:31:00', '2016-02-11T10:17:34', '2015-11-12T11:34:22'], dtype=np.datetime64)
    >>> df = vaex.from_arrays(date=date)
    >>> df
      #  date
      0  2009-10-12 03:31:00
      1  2016-02-11 10:17:34
      2  2015-11-12 11:34:22

    >>> df.date.dt.strftime("%Y-%m")
    Expression = dt_strftime(date, '%Y-%m')
    Length: 3 dtype: object (expression)
    ------------------------------------
    0  2009-10
    1  2016-02
    2  2015-11
    """
    import pandas as pd
    return pa.array(_to_pandas_series(x).dt.strftime(date_format))

@register_function(scope='dt')
def dt_floor(x, freq, *args):
    """Perform floor operation on an expression for a given frequency.

    :param freq: The frequency level to floor the index to. Must be a fixed frequency like 'S' (second), or 'H' (hour), but not 'ME' (month end).
    :returns: an expression containing the floored datetime column.

    Example:

    >>> import vaex
    >>> import numpy as np
    >>> date = np.array(['2009-10-12T03:31:00', '2016-02-11T10:17:34', '2015-11-12T11:34:22'], dtype=np.datetime64)
    >>> df = vaex.from_arrays(date=date)
    >>> df
      #  date
      0  2009-10-12 03:31:00
      1  2016-02-11 10:17:34
      2  2015-11-12 11:34:22

    >>> df.date.dt.floor("H")
    Expression = dt_floor(date, 'H')
    Length: 3 dtype: datetime64[ns] (expression)
    --------------------------------------------
    0  2009-10-12 03:00:00.000000000
    1  2016-02-11 10:00:00.000000000
    2  2015-11-12 11:00:00.000000000
    """
    import pandas as pd
    return _to_pandas_series(x).dt.floor(freq, *args).values

########## timedelta operations ##########

@register_function(scope='td', as_property=True)
def td_days(x):
    """Number of days in each timedelta sample.

    :returns: an expression containing the number of days in a timedelta sample.

    Example:

    >>> import vaex
    >>> import numpy as np
    >>> delta = np.array([17658720110,   11047049384039, 40712636304958, -18161254954], dtype='timedelta64[s]')
    >>> df = vaex.from_arrays(delta=delta)
    >>> df
      #  delta
      0  204 days +9:12:00
      1  1 days +6:41:10
      2  471 days +5:03:56
      3  -22 days +23:31:15

    >>> df.delta.td.days
    Expression = td_days(delta)
    Length: 4 dtype: int64 (expression)
    -----------------------------------
    0  204
    1    1
    2  471
    3  -22
    """
    import pandas as pd
    return _to_pandas_series(x).dt.days.values

@register_function(scope='td', as_property=True)
def td_microseconds(x):
    """Number of microseconds (>= 0 and less than 1 second) in each timedelta sample.

    :returns: an expression containing the number of microseconds in a timedelta sample.

    Example:

    >>> import vaex
    >>> import numpy as np
    >>> delta = np.array([17658720110,   11047049384039, 40712636304958, -18161254954], dtype='timedelta64[s]')
    >>> df = vaex.from_arrays(delta=delta)
    >>> df
      #  delta
      0  204 days +9:12:00
      1  1 days +6:41:10
      2  471 days +5:03:56
      3  -22 days +23:31:15

    >>> df.delta.td.microseconds
    Expression = td_microseconds(delta)
    Length: 4 dtype: int64 (expression)
    -----------------------------------
    0  290448
    1  978582
    2   19583
    3  709551
    """
    import pandas as pd
    return _to_pandas_series(x).dt.microseconds.values

@register_function(scope='td', as_property=True)
def td_nanoseconds(x):
    """Number of nanoseconds (>= 0 and less than 1 microsecond) in each timedelta sample.

    :returns: an expression containing the number of nanoseconds in a timedelta sample.

    Example:

    >>> import vaex
    >>> import numpy as np
    >>> delta = np.array([17658720110,   11047049384039, 40712636304958, -18161254954], dtype='timedelta64[s]')
    >>> df = vaex.from_arrays(delta=delta)
    >>> df
      #  delta
      0  204 days +9:12:00
      1  1 days +6:41:10
      2  471 days +5:03:56
      3  -22 days +23:31:15

    >>> df.delta.td.nanoseconds
    Expression = td_nanoseconds(delta)
    Length: 4 dtype: int64 (expression)
    -----------------------------------
    0  384
    1   16
    2  488
    3  616
    """
    import pandas as pd
    return _to_pandas_series(x).dt.nanoseconds.values

@register_function(scope='td', as_property=True)
def td_seconds(x):
    """Number of seconds (>= 0 and less than 1 day) in each timedelta sample.

    :returns: an expression containing the number of seconds in a timedelta sample.

    Example:

    >>> import vaex
    >>> import numpy as np
    >>> delta = np.array([17658720110,   11047049384039, 40712636304958, -18161254954], dtype='timedelta64[s]')
    >>> df = vaex.from_arrays(delta=delta)
    >>> df
      #  delta
      0  204 days +9:12:00
      1  1 days +6:41:10
      2  471 days +5:03:56
      3  -22 days +23:31:15

    >>> df.delta.td.seconds
    Expression = td_seconds(delta)
    Length: 4 dtype: int64 (expression)
    -----------------------------------
    0  30436
    1  39086
    2  28681
    3  23519
    """
    import pandas as pd
    return _to_pandas_series(x).dt.seconds.values

@register_function(scope='td', as_property=False)
def td_total_seconds(x):
    """Total duration of each timedelta sample expressed in seconds.

    :return: an expression containing the total number of seconds in a timedelta sample.

    Example:
    >>> import vaex
    >>> import numpy as np
    >>> delta = np.array([17658720110,   11047049384039, 40712636304958, -18161254954], dtype='timedelta64[s]')
    >>> df = vaex.from_arrays(delta=delta)
    >>> df
      #  delta
      0  204 days +9:12:00
      1  1 days +6:41:10
      2  471 days +5:03:56
      3  -22 days +23:31:15

    >>> df.delta.td.total_seconds()
    Expression = td_total_seconds(delta)
    Length: 4 dtype: float64 (expression)
    -------------------------------------
    0  -7.88024e+08
    1  -2.55032e+09
    2   6.72134e+08
    3   2.85489e+08
    """
    import pandas as pd
    return _to_pandas_series(x).dt.total_seconds().values


########## string operations ##########

@register_function(scope='str')
@auto_str_unwrap
def str_equals(x, y):
    """Tests if strings x and y are the same

    :returns: a boolean expression

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.

    >>> df.text.str.equals(df.text)
    Expression = str_equals(text, text)
    Length: 5 dtype: bool (expression)
    ----------------------------------
    0  True
    1  True
    2  True
    3  True
    4  True

    >>> df.text.str.equals('our')
    Expression = str_equals(text, 'our')
    Length: 5 dtype: bool (expression)
    ----------------------------------
    0  False
    1  False
    2  False
    3   True
    4  False
    """
    xmask = None
    ymask = None
    if not isinstance(x, six.string_types):
        x = _to_string_sequence(x)
    if not isinstance(y, six.string_types):
        y = _to_string_sequence(y)
    equals_mask = x.equals(y)
    return equals_mask


@register_function(scope='str')
@auto_str_unwrap
def str_notequals(x, y):
    """Tests if strings x and y are the not same

    :returns: a boolean expression

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.

    >>> df.text.str.notequals(df.text)
    Expression = str_notequals(text, text)
    Length: 5 dtype: bool (expression)
    ----------------------------------
    0  False
    1  False
    2  False
    3  False
    4  False

    >>> df.text.str.notequals('our')
    Expression = str_notequals(text, 'our')
    Length: 5 dtype: bool (expression)
    ----------------------------------
    0   True
    1   True
    2   True
    3  False
    4   True
    """
    return ~str_equals(x, y)


@register_function(scope='str')
@auto_str_unwrap
def str_capitalize(x):
    """Capitalize the first letter of a string sample.

    :returns: an expression containing the capitalized strings.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.

    >>> df.text.str.capitalize()
    Expression = str_capitalize(text)
    Length: 5 dtype: str (expression)
    ---------------------------------
    0    Something
    1  Very pretty
    2    Is coming
    3          Our
    4         Way.
    """
    sl = _to_string_sequence(x).capitalize()
    return column.ColumnStringArrow.from_string_sequence(sl)

@register_function(scope='str')
@auto_str_unwrap
def str_cat(x, other):
    """Concatenate two string columns on a row-by-row basis.

    :param expression other: The expression of the other column to be concatenated.
    :returns: an expression containing the concatenated columns.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.

    >>> df.text.str.cat(df.text)
    Expression = str_cat(text, text)
    Length: 5 dtype: str (expression)
    ---------------------------------
    0      SomethingSomething
    1  very prettyvery pretty
    2      is comingis coming
    3                  ourour
    4                way.way.
    """
    if isinstance(x, six.string_types):
        other = _to_string_sequence(other)
        sl = other.concat_reverse(x)
    else:
        x = _to_string_sequence(x)
        if not isinstance(other, six.string_types):
            other = _to_string_sequence(other)
        sl = x.concat(other)
    return column.ColumnStringArrow.from_string_sequence(sl)

@register_function(scope='str')
@auto_str_unwrap
def str_center(x, width, fillchar=' '):
    """ Fills the left and right side of the strings with additional characters, such that the sample has a total of `width`
    characters.

    :param int width: The total number of characters of the resulting string sample.
    :param str fillchar: The character used for filling.
    :returns: an expression containing the filled strings.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.

    >>> df.text.str.center(width=11, fillchar='!')
    Expression = str_center(text, width=11, fillchar='!')
    Length: 5 dtype: str (expression)
    ---------------------------------
    0  !Something!
    1  very pretty
    2  !is coming!
    3  !!!!our!!!!
    4  !!!!way.!!!
    """
    sl = _to_string_sequence(x).pad(width, fillchar, True, True)
    return column.ColumnStringArrow.from_string_sequence(sl)


@register_function(scope='str')
@auto_str_unwrap
def str_contains(x, pattern, regex=True):
    """Check if a string pattern or regex is contained within a sample of a string column.

    :param str pattern: A string or regex pattern
    :param bool regex: If True,
    :returns: an expression which is evaluated to True if the pattern is found in a given sample, and it is False otherwise.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.

    >>> df.text.str.contains('very')
    Expression = str_contains(text, 'very')
    Length: 5 dtype: bool (expression)
    ----------------------------------
    0  False
    1   True
    2  False
    3  False
    4  False
    """
    if regex:
        return _to_string_sequence(x).search(pattern, regex)
    else:
        if not isinstance(x, vaex.array_types.supported_arrow_array_types):
            x = pa.array(x)
        return pc.match_substring(x, pattern)

# TODO: default regex is False, which breaks with pandas
@register_function(scope='str')
@auto_str_unwrap
def str_count(x, pat, regex=False):
    """Count the occurences of a pattern in sample of a string column.

    :param str pat: A string or regex pattern
    :param bool regex: If True,
    :returns: an expression containing the number of times a pattern is found in each sample.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.

    >>> df.text.str.count(pat="et", regex=False)
    Expression = str_count(text, pat='et', regex=False)
    Length: 5 dtype: int64 (expression)
    -----------------------------------
    0  1
    1  1
    2  0
    3  0
    4  0
    """
    return _to_string_sequence(x).count(pat, regex)

# TODO: what to do with decode and encode

@register_function(scope='str')
@auto_str_unwrap
def str_endswith(x, pat):
    """Check if the end of each string sample matches the specified pattern.

    :param str pat: A string pattern or a regex
    :returns: an expression evaluated to True if the pattern is found at the end of a given sample, False otherwise.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.

    >>> df.text.str.endswith(pat="ing")
    Expression = str_endswith(text, pat='ing')
    Length: 5 dtype: bool (expression)
    ----------------------------------
    0   True
    1  False
    2   True
    3  False
    4  False
    """
    return _to_string_sequence(x).endswith(pat)

# TODO: what to do with extract/extractall

# def str_extract(x, pat, flags=0, expand=True):
#     """Wraps pandas"""
#     import pandas
#     x = x.to_numpy()
#     series = pandas.Series(x)
#     return series.str.extract(pat, flags, expand).values

# def str_extractall(x, pat, flags=0, expand=True):
#     """Wraps pandas"""
#     import pandas
#     x = x.to_numpy()
#     series = pandas.Series(x)
#     return series.str.str_extractall(pat, flags, expand).values

# TODO: extract/extractall

@register_function(scope='str')
@auto_str_unwrap
def str_find(x, sub, start=0, end=None):
    """Returns the lowest indices in each string in a column, where the provided substring is fully contained between within a
    sample. If the substring is not found, -1 is returned.

    :param str sub: A substring to be found in the samples
    :param int start:
    :param int end:
    :returns: an expression containing the lowest indices specifying the start of the substring.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.

    >>> df.text.str.find(sub="et")
    Expression = str_find(text, sub='et')
    Length: 5 dtype: int64 (expression)
    -----------------------------------
    0   3
    1   7
    2  -1
    3  -1
    4  -1
    """
    return _to_string_sequence(x).find(sub, start, 0 if end is None else end, end is None, True)

# TODO: findall (not sure what the use/behaviour is)

# TODO get/index/join

@register_function(scope='str')
@auto_str_unwrap
def str_get(x, i):
    """Extract a character from each sample at the specified position from a string column.
    Note that if the specified position is out of bound of the string sample, this method returns '', while pandas retunrs nan.

    :param int i: The index location, at which to extract the character.
    :returns: an expression containing the extracted characters.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.

    >>> df.text.str.get(5)
    Expression = str_get(text, 5)
    Length: 5 dtype: str (expression)
    ---------------------------------
    0    h
    1    p
    2    m
    3
    4
    """
    x = _to_string_sequence(x)
    if i == -1:
        sl = x.slice_string_end(-1)
    else:
        sl = x.slice_string(i, i+1)
    return column.ColumnStringArrow.from_string_sequence(sl)


@register_function(scope='str')
@auto_str_unwrap
def str_index(x, sub, start=0, end=None):
    """Returns the lowest indices in each string in a column, where the provided substring is fully contained between within a
    sample. If the substring is not found, -1 is returned. It is the same as `str.find`.

    :param str sub: A substring to be found in the samples
    :param int start:
    :param int end:
    :returns: an expression containing the lowest indices specifying the start of the substring.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.

    >>> df.text.str.index(sub="et")
    Expression = str_find(text, sub='et')
    Length: 5 dtype: int64 (expression)
    -----------------------------------
    0   3
    1   7
    2  -1
    3  -1
    4  -1
    """
    return str_find(x, sub, start, end)


@register_function(scope='str')
def str_join(x, sep):
    """Same as find (difference with pandas is that it does not raise a ValueError)"""
    dtype = vaex.dtype_of(x)
    if not dtype.is_list:
        raise TypeError(f'join expected a list, not {x}')

    x, wrapper = vaex.arrow.utils.list_unwrap(x, level=-2)
    values = x.values
    offsets = vaex.array_types.to_numpy(x.offsets)
    ss = _to_string_sequence(values)
    x_joined_ss = vaex.strings.join(sep, offsets, ss, x.offset)
    # TODO: we require a copy here, because the x_column and the string_seqence will be
    # garbage collected, this is not idea, but once https://github.com/apache/arrow/pull/8990 is
    # released, we can rely on that
    x_column = column.ColumnStringArrow.from_string_sequence(x_joined_ss)
    x_joined = pa.array(x_column)
    return wrapper(x_joined)


@register_function(scope='str')
@auto_str_unwrap
def str_len(x):
    """Returns the length of a string sample.

    :returns: an expression contains the length of each sample of a string column.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.

    >>> df.text.str.len()
    Expression = str_len(text)
    Length: 5 dtype: int64 (expression)
    -----------------------------------
    0   9
    1  11
    2   9
    3   3
    4   4
    """
    return _to_string_sequence(x).len()

@register_function(scope='str')
@auto_str_unwrap
def str_byte_length(x):
    """Returns the number of bytes in a string sample.

    :returns: an expression contains the number of bytes in each sample of a string column.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.

    >>> df.text.str.byte_length()
    Expression = str_byte_length(text)
    Length: 5 dtype: int64 (expression)
    -----------------------------------
    0   9
    1  11
    2   9
    3   3
    4   4
    """
    return _to_string_sequence(x).byte_length()

@register_function(scope='str')
@auto_str_unwrap
def str_ljust(x, width, fillchar=' '):
    """Fills the right side of string samples with a specified character such that the strings are right-hand justified.

    :param int width: The minimal width of the strings.
    :param str fillchar: The character used for filling.
    :returns: an expression containing the filled strings.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.

    >>> df.text.str.ljust(width=10, fillchar='!')
    Expression = str_ljust(text, width=10, fillchar='!')
    Length: 5 dtype: str (expression)
    ---------------------------------
    0   Something!
    1  very pretty
    2   is coming!
    3   our!!!!!!!
    4   way.!!!!!!
    """
    sl = _to_string_sequence(x).pad(width, fillchar, False, True)
    return column.ColumnStringArrow.from_string_sequence(sl)


@register_function(scope='str')
@auto_str_unwrap
def str_lower(x):
    """Converts string samples to lower case.

    :returns: an expression containing the converted strings.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.

    >>> df.text.str.lower()
    Expression = str_lower(text)
    Length: 5 dtype: str (expression)
    ---------------------------------
    0    something
    1  very pretty
    2    is coming
    3          our
    4         way.
    """
    if not isinstance(x, vaex.array_types.supported_arrow_array_types):
        x = pa.array(x)
    return pc.utf8_lower(x)


@register_function(scope='str')
@auto_str_unwrap
def str_lstrip(x, to_strip=None):
    """Remove leading characters from a string sample.

    :param str to_strip: The string to be removed
    :returns: an expression containing the modified string column.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.

    >>> df.text.str.lstrip(to_strip='very ')
    Expression = str_lstrip(text, to_strip='very ')
    Length: 5 dtype: str (expression)
    ---------------------------------
    0  Something
    1     pretty
    2  is coming
    3        our
    4       way.
    """
    # in c++ we give empty string the same meaning as None
    sl = _to_string_sequence(x).lstrip('' if to_strip is None else to_strip) if to_strip != '' else x
    return column.ColumnStringArrow.from_string_sequence(sl)


@register_function(scope='str')
@auto_str_unwrap
def str_match(x, pattern):
    """Check if a string sample matches a given regular expression.

    :param str pattern: a string or regex to match to a string sample.
    :returns: an expression which is evaluated to True if a match is found, False otherwise.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.

    >>> df.text.str.match(pattern='our')
    Expression = str_match(text, pattern='our')
    Length: 5 dtype: bool (expression)
    ----------------------------------
    0  False
    1  False
    2  False
    3   True
    4  False
    """
    return _to_string_sequence(x).match(pattern)

# TODO: normalize, partition

@register_function(scope='str')
@auto_str_unwrap
def str_pad(x, width, side='left', fillchar=' '):
    """Pad strings in a given column.

    :param int width: The total width of the string
    :param str side: If 'left' than pad on the left, if 'right' than pad on the right side the string.
    :param str fillchar: The character used for padding.
    :returns: an expression containing the padded strings.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.

    >>> df.text.str.pad(width=10, side='left', fillchar='!')
    Expression = str_pad(text, width=10, side='left', fillchar='!')
    Length: 5 dtype: str (expression)
    ---------------------------------
    0   !Something
    1  very pretty
    2   !is coming
    3   !!!!!!!our
    4   !!!!!!way.
    """
    sl = _to_string_sequence(x).pad(width, fillchar, side in ['left', 'both'], side in ['right', 'both'])
    return column.ColumnStringArrow.from_string_sequence(sl)


@register_function(scope='str')
@auto_str_unwrap
def str_repeat(x, repeats):
    """Duplicate each string in a column.

    :param int repeats: number of times each string sample is to be duplicated.
    :returns: an expression containing the duplicated strings

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.

    >>> df.text.str.repeat(3)
    Expression = str_repeat(text, 3)
    Length: 5 dtype: str (expression)
    ---------------------------------
    0        SomethingSomethingSomething
    1  very prettyvery prettyvery pretty
    2        is comingis comingis coming
    3                          ourourour
    4                       way.way.way.
    """
    sl = _to_string_sequence(x).repeat(repeats)
    return column.ColumnStringArrow.from_string_sequence(sl)


@register_function(scope='str')
@auto_str_unwrap
def str_replace(x, pat, repl, n=-1, flags=0, regex=False):
    """Replace occurences of a pattern/regex in a column with some other string.

    :param str pattern: string or a regex pattern
    :param str replace: a replacement string
    :param int n: number of replacements to be made from the start. If -1 make all replacements.
    :param int flags: ??
    :param bool regex: If True, ...?
    :returns: an expression containing the string replacements.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.

    >>> df.text.str.replace(pat='et', repl='__')
    Expression = str_replace(text, pat='et', repl='__')
    Length: 5 dtype: str (expression)
    ---------------------------------
    0    Som__hing
    1  very pr__ty
    2    is coming
    3          our
    4         way.
    """
    sl = _to_string_sequence(x).replace(pat, repl, n, flags, regex)
    return column.ColumnStringArrow.from_string_sequence(sl)


@register_function(scope='str')
@auto_str_unwrap
def str_rfind(x, sub, start=0, end=None):
    """Returns the highest indices in each string in a column, where the provided substring is fully contained between within a
    sample. If the substring is not found, -1 is returned.

    :param str sub: A substring to be found in the samples
    :param int start:
    :param int end:
    :returns: an expression containing the highest indices specifying the start of the substring.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.

    >>> df.text.str.rfind(sub="et")
    Expression = str_rfind(text, sub='et')
    Length: 5 dtype: int64 (expression)
    -----------------------------------
    0   3
    1   7
    2  -1
    3  -1
    4  -1
    """
    return _to_string_sequence(x).find(sub, start, 0 if end is None else end, end is None, False)

@register_function(scope='str')
@auto_str_unwrap
def str_rindex(x, sub, start=0, end=None):
    """Returns the highest indices in each string in a column, where the provided substring is fully contained between within a
    sample. If the substring is not found, -1 is returned. Same as `str.rfind`.

    :param str sub: A substring to be found in the samples
    :param int start:
    :param int end:
    :returns: an expression containing the highest indices specifying the start of the substring.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.

    >>> df.text.str.rindex(sub="et")
    Expression = str_rindex(text, sub='et')
    Length: 5 dtype: int64 (expression)
    -----------------------------------
    0   3
    1   7
    2  -1
    3  -1
    4  -1
    """
    return str_rfind(x, sub, start, end)

@register_function(scope='str')
@auto_str_unwrap
def str_rjust(x, width, fillchar=' '):
    """Fills the left side of string samples with a specified character such that the strings are left-hand justified.

    :param int width: The minimal width of the strings.
    :param str fillchar: The character used for filling.
    :returns: an expression containing the filled strings.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.

    >>> df.text.str.rjust(width=10, fillchar='!')
    Expression = str_rjust(text, width=10, fillchar='!')
    Length: 5 dtype: str (expression)
    ---------------------------------
    0   !Something
    1  very pretty
    2   !is coming
    3   !!!!!!!our
    4   !!!!!!way.
    """
    sl = _to_string_sequence(x).pad(width, fillchar, True, False)
    return column.ColumnStringArrow.from_string_sequence(sl)


# TODO: rpartition

@register_function(scope='str')
@auto_str_unwrap
def str_rstrip(x, to_strip=None):
    """Remove trailing characters from a string sample.

    :param str to_strip: The string to be removed
    :returns: an expression containing the modified string column.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.

    >>> df.text.str.rstrip(to_strip='ing')
    Expression = str_rstrip(text, to_strip='ing')
    Length: 5 dtype: str (expression)
    ---------------------------------
    0       Someth
    1  very pretty
    2       is com
    3          our
    4         way.
    """
    # in c++ we give empty string the same meaning as None
    sl = _to_string_sequence(x).rstrip('' if to_strip is None else to_strip) if to_strip != '' else x
    return column.ColumnStringArrow.from_string_sequence(sl)


@register_function(scope='str')
@auto_str_unwrap
def str_slice(x, start=0, stop=None):  # TODO: support n
    """Slice substrings from each string element in a column.

    :param int start: The start position for the slice operation.
    :param int end: The stop position for the slice operation.
    :returns: an expression containing the sliced substrings.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.

    >>> df.text.str.slice(start=2, stop=5)
    Expression = str_pandas_slice(text, start=2, stop=5)
    Length: 5 dtype: str (expression)
    ---------------------------------
    0  met
    1   ry
    2   co
    3    r
    4   y.
    """
    if stop is None:
        ss = _to_string_sequence(x).slice_string_end(start)
    else:
        ss = _to_string_sequence(x).slice_string(start, stop)
    return column.ColumnStringArrow.from_string_sequence(ss)

# TODO: slice_replace (not sure it this makes sense)

@register_function(scope='str')
@auto_str_unwrap
def str_rsplit(x, pattern=None, max_splits=-1):
    if not isinstance(x, vaex.array_types.supported_arrow_array_types):
        x = pa.array(x)
    if pattern is None:
        return pc.utf8_split_whitespace(x, reverse=True, max_splits=max_splits)
    else:
        return pc.split_pattern(x, reverse=True, max_splits=max_splits)


@register_function(scope='str')
@auto_str_unwrap
def str_split(x, pattern=None, max_splits=-1):
    if not isinstance(x, vaex.array_types.supported_arrow_array_types):
        x = pa.array(x)
    if pattern is None:
        return pc.utf8_split_whitespace(x, max_splits=max_splits)
    else:
        return pc.split_pattern(x, pattern=pattern, max_splits=max_splits)



@register_function(scope='str')
@auto_str_unwrap
def str_startswith(x, pat):
    """Check if a start of a string matches a pattern.

    :param str pat: A string pattern. Regular expressions are not supported.
    :returns: an expression which is evaluated to True if the pattern is found at the start of a string sample, False otherwise.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.

    >>> df.text.str.startswith(pat='is')
    Expression = str_startswith(text, pat='is')
    Length: 5 dtype: bool (expression)
    ----------------------------------
    0  False
    1  False
    2   True
    3  False
    4  False
    """
    return _to_string_sequence(x).startswith(pat)

@register_function(scope='str')
@auto_str_unwrap
def str_strip(x, to_strip=None):
    """Removes leading and trailing characters.

    Strips whitespaces (including new lines), or a set of specified
    characters from each string saple in a column, both from the left
    right sides.

    :param str to_strip: The characters to be removed. All combinations of the characters will be removed.
                         If None, it removes whitespaces.
    :param returns: an expression containing the modified string samples.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.

    >>> df.text.str.strip(to_strip='very')
    Expression = str_strip(text, to_strip='very')
    Length: 5 dtype: str (expression)
    ---------------------------------
    0  Something
    1      prett
    2  is coming
    3         ou
    4       way.
    """
    # in c++ we give empty string the same meaning as None
    sl = _to_string_sequence(x).strip('' if to_strip is None else to_strip) if to_strip != '' else x
    return column.ColumnStringArrow.from_string_sequence(sl)

# TODO: swapcase, translate

@register_function(scope='str')
@auto_str_unwrap
def str_title(x):
    """Converts all string samples to titlecase.

    :returns: an expression containing the converted strings.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.

    >>> df.text.str.title()
    Expression = str_title(text)
    Length: 5 dtype: str (expression)
    ---------------------------------
    0    Something
    1  Very Pretty
    2    Is Coming
    3          Our
    4         Way.
    """
    sl = _to_string_sequence(x).title()
    return column.ColumnStringArrow.from_string_sequence(sl)


@register_function(scope='str')
@docsubst
@auto_str_unwrap
def str_upper(x, ascii=False):
    """Converts all strings in a column to uppercase.

    :param bool ascii: {ascii}
    :returns: an expression containing the converted strings.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.


    >>> df.text.str.upper()
    Expression = str_upper(text)
    Length: 5 dtype: str (expression)
    ---------------------------------
    0    SOMETHING
    1  VERY PRETTY
    2    IS COMING
    3          OUR
    4         WAY.

    """
    if ascii:
        return pc.ascii_upper(x)
    else:
        if not isinstance(x, vaex.array_types.supported_arrow_array_types):
            x = pa.array(x)
        return pc.utf8_upper(_arrow_string_might_grow(x, 3/2))


# TODO: wrap, is*, get_dummies(maybe?)

@register_function(scope='str')
@auto_str_unwrap
def str_zfill(x, width):
    """Pad strings in a column by prepanding "0" characters.

    :param int width: The minimum length of the resulting string. Strings shorter less than `width` will be prepended with zeros.
    :returns: an expression containing the modified strings.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.

    >>> df.text.str.zfill(width=12)
    Expression = str_zfill(text, width=12)
    Length: 5 dtype: str (expression)
    ---------------------------------
    0  000Something
    1  0very pretty
    2  000is coming
    3  000000000our
    4  00000000way.
    """
    sl = _to_string_sequence(x).pad(width, '0', True, False)
    return column.ColumnStringArrow.from_string_sequence(sl)


@register_function(scope='str')
@docsubst
@auto_str_unwrap
def str_isalnum(x, ascii=False):
    """Check if all characters in a string sample are alphanumeric.

    :param bool ascii: {ascii}
    :returns: an expression evaluated to True if a sample contains only alphanumeric characters, otherwise False.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.

    >>> df.text.str.isalnum()
    Expression = str_isalnum(text)
    Length: 5 dtype: bool (expression)
    ----------------------------------
    0   True
    1  False
    2  False
    3   True
    4  False
    """
    kernel_function = pc.ascii_is_alnum if ascii else pc.utf8_is_alnum
    if not isinstance(x, vaex.array_types.supported_arrow_array_types):
        x = pa.array(x)
    return kernel_function(x)


@register_function(scope='str')
@auto_str_unwrap
def str_isalpha(x):
    """Check if all characters in a string sample are alphabetic.

    :returns: an expression evaluated to True if a sample contains only alphabetic characters, otherwise False.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.

    >>> df.text.str.isalpha()
    Expression = str_isalpha(text)
    Length: 5 dtype: bool (expression)
    ----------------------------------
    0   True
    1  False
    2  False
    3   True
    4  False
    """
    return _to_string_sequence(x).isalpha()

@register_function(scope='str')
@auto_str_unwrap
def str_isdigit(x):
    """Check if all characters in a string sample are digits.

    :returns: an expression evaluated to True if a sample contains only digits, otherwise False.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', '6']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  6

    >>> df.text.str.isdigit()
    Expression = str_isdigit(text)
    Length: 5 dtype: bool (expression)
    ----------------------------------
    0  False
    1  False
    2  False
    3  False
    4   True
    """
    return _to_string_sequence(x).isdigit()

@register_function(scope='str')
@auto_str_unwrap
def str_isspace(x):
    """Check if all characters in a string sample are whitespaces.

    :returns: an expression evaluated to True if a sample contains only whitespaces, otherwise False.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', '      ', ' ']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3
      4

    >>> df.text.str.isspace()
    Expression = str_isspace(text)
    Length: 5 dtype: bool (expression)
    ----------------------------------
    0  False
    1  False
    2  False
    3   True
    4   True
    """
    return _to_string_sequence(x).isspace()

@register_function(scope='str')
@auto_str_unwrap
def str_islower(x):
    """Check if all characters in a string sample are lowercase characters.

    :returns: an expression evaluated to True if a sample contains only lowercase characters, otherwise False.

    Example:

    >>> import vaex
    >>> text = ['Something', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  Something
      1  very pretty
      2  is coming
      3  our
      4  way.

    >>> df.text.str.islower()
    Expression = str_islower(text)
    Length: 5 dtype: bool (expression)
    ----------------------------------
    0  False
    1   True
    2   True
    3   True
    4   True
    """
    return _to_string_sequence(x).islower()

@register_function(scope='str')
@auto_str_unwrap
def str_isupper(x):
    """Check if all characters in a string sample are lowercase characters.

    :returns: an expression evaluated to True if a sample contains only lowercase characters, otherwise False.

    Example:

    >>> import vaex
    >>> text = ['SOMETHING', 'very pretty', 'is coming', 'our', 'way.']
    >>> df = vaex.from_arrays(text=text)
    >>> df
      #  text
      0  SOMETHING
      1  very pretty
      2  is coming
      3  our
      4  way.

    >>> df.text.str.isupper()
    Expression = str_isupper(text)
    Length: 5 dtype: bool (expression)
    ----------------------------------
    0   True
    1  False
    2  False
    3  False
    4  False
    """
    return _to_string_sequence(x).isupper()

@register_function(scope='str')
@auto_str_unwrap
def str_istitle(x, ascii=False):
    '''TODO'''
    return _arrow_string_kernel_dispatch('is_title', ascii, x)


# @register_function(scope='str')
# def str_isnumeric(x):
#     sl = _to_string_sequence(x).isnumeric()

# @register_function(scope='str')
# def str_isdecimal(x):
#     sl = _to_string_sequence(x).isnumeric()

@register_function()
def to_string(x):
    # don't change the dtype, otherwise for each block the dtype may be different (string length)
    sl = vaex.strings.to_string(x)
    return column.ColumnStringArrow.from_string_sequence(sl)


@register_function()
def format(x, format):
    """Uses http://www.cplusplus.com/reference/string/to_string/ for formatting"""
    # don't change the dtype, otherwise for each block the dtype may be different (string length)
    if not isinstance(x, np.ndarray) or x.dtype.kind in 'US':
        # sometimes the dtype can be object, but seen as an string array
        x = _to_string_sequence(x)
    sl = vaex.strings.format(x, format)
    return column.ColumnStringArrow.from_string_sequence(sl)


for name in dir(scopes['str']):
    if name.startswith('__'):
        continue
    force_string = ['get']
    def pandas_wrapper(name=name):
        def wrapper(*args, **kwargs):
            import pandas
            def fix_arg(x):
                if isinstance(x, column.Column):
                    x = x.to_numpy()
                return x
            args = list(map(fix_arg, args))
            x = args[0]
            args = args[1:]
            series = pandas.Series(x)
            method = getattr(series.str, name)
            value = method(*args, **kwargs)
            if name in force_string:
                value = _to_string_column(value.values, force=True)
                return value
            else:
                return value.values
        return wrapper
    wrapper = pandas_wrapper()
    wrapper.__doc__ = "Wrapper around pandas.Series.%s" % name
    register_function(scope='str_pandas')(wrapper, name=name)
    # expression_namespace['str_pandas_' + name] = wrapper
    # _scopes['str_pandas'][name] = wrapper

# expression_namespace['str_strip'] = str_strip

@register_function()
def _ordinal_values(x, ordered_set):
    from vaex.column import _to_string_sequence

    if not isinstance(x, vaex.array_types.supported_array_types) or isinstance(ordered_set, vaex.superutils.ordered_set_string):
        # sometimes the dtype can be object, but seen as an string array
        x = _to_string_sequence(x)
    return ordered_set.map_ordinal(x)

@register_function()
def _choose(ar, choices, default=None):
    from vaex.column import _to_string_sequence
    # if not isinstance(choices, np.ndarray) or choices.dtype.kind in 'US':
    #     choices = _to_string_sequence(choices)
    if default is not None:
        mask = ar == -1
        ar[mask] == 0  # we fill it in with some values, doesn't matter, since it will be replaced
    ar = choices[ar]
    if default is not None:
        ar[mask] = default
    return ar

@register_function()
def _choose_masked(ar, choices):
    """Similar to _choose, but -1 maps to NA"""
    from vaex.column import _to_string_sequence
    mask = ar == -1
    ar[mask] == 0  # we fill it in with some values, doesn't matter, since it is masked
    ar = choices[ar]
    return np.ma.array(ar, mask=mask)


@register_function()
def _map(ar, value_to_index, choices, default_value=None, use_missing=False, axis=None):
    flatten = axis is None
    if flatten:
        import vaex.arrow.utils
        ar, wrapper = vaex.arrow.utils.list_unwrap(ar)

    if not isinstance(ar, vaex.array_types.supported_array_types) or isinstance(value_to_index, vaex.superutils.ordered_set_string):
        # sometimes the dtype can be object, but seen as an string array
        ar = _to_string_sequence(ar)
        # -1 points to missing
        indices = value_to_index.map_ordinal(ar) + 1
    else:
        ar = vaex.array_types.to_numpy(ar)
        indices = value_to_index.map_ordinal(ar) + 1
        if np.ma.isMaskedArray(ar):
            mask = np.ma.getmaskarray(ar)
            indices[mask] = 1  # missing values are at offset 1 (see expression.map)
    values = choices.take(indices)
    if flatten:
        values = wrapper(values)
    return values


@register_function(name='float')
def _float(x):
    return x.astype(np.float64)

@register_function(name='astype', on_expression=False)
def _astype(x, dtype):
    if isinstance(x, vaex.column.ColumnString):
        x = x.to_arrow()
    if isinstance(x, vaex.array_types.supported_arrow_array_types):
        if pa.types.is_timestamp(x.type):
            # arrow does not support timestamp to int/float, so we use numpy
            y = x.to_numpy().astype(dtype)
            return y

        if dtype.startswith('datetime64'):  # parse dtype
            if len(dtype) > len('datetime64'):
                units = dtype[len('datetime64')+1:-1]
            else:
                units = 'ns'
            dtype = pa.timestamp(units)

        y = x.cast(dtype, safe=False)
        return y
    else:  # numpy case
        if dtype in ['str', 'string', 'large_string']:
            y = x.astype('str')
            return vaex.column._to_string_column(y)
        else:
            return x.astype(dtype)


@register_function(name='isin', on_expression=False)
def _isin(x, values):
    if vaex.column._is_stringy(x):
        x = vaex.column._to_string_column(x)
        return x.string_sequence.isin(values)
    else:
        # TODO: this happens when a column is of dtype=object
        # but only strings, the values gets converted to a superstring array
        # but numpy doesn't know what to do with that
        if hasattr(values, 'to_numpy'):
            values = values.to_numpy()
        mask = isnan(values)
        if np.any(mask):
            if np.ma.isMaskedArray(x):
                return np.ma.isin(x, values) | isnan(x)
            else:
                return np.isin(x, values) | isnan(x)
        else:
            if np.ma.isMaskedArray(x):
                return np.ma.isin(x, values)
            else:
                return np.isin(x, values)


@register_function(name='isin_set', on_expression=False)
def _isin_set(x, set):
    if vaex.column._is_stringy(x) or isinstance(set, vaex.superutils.ordered_set_string):
        x = vaex.column._to_string_column(x)
        x = _to_string_sequence(x)
        # return x.string_sequence.isin(values)
        return set.isin(x)
    else:
        if np.ma.isMaskedArray(x):
            isin = set.isin(x.data)
            isin[x.mask] = False
            return isin
        else:
            return set.isin(x)


@register_function()
def as_arrow(x):
    '''Lazily convert to Apache Arrow array type'''
    from .array_types import to_arrow
    # since we do this lazily, we can convert to native without wasting memory
    return to_arrow(x, convert_to_native=True)


@register_function()
def as_numpy(x, strict=False):
    '''Lazily convert to NumPy ndarray type'''
    from .array_types import to_numpy
    return to_numpy(x, strict=strict)


def add_geo_json(ds, json_or_file, column_name, longitude_expression, latitude_expresion, label=None, persist=True, overwrite=False, inplace=False, mapping=None):
    ds = ds if inplace else ds.copy()
    if not isinstance(json_or_file, (list, tuple)):
        with open(json_or_file) as f:
            geo_json = json.load(f)
    else:
        geo_json = json_or_file
    def default_label(properties):
        return " - ".join(properties.values())
    label = label or default_label
    features = geo_json['features']
    list_of_polygons = []
    labels = []
    if mapping:
        mapping_dict = {}
    for i, feature in enumerate(features[:]):
        geo = feature['geometry']
        properties = feature['properties']
        if mapping:
            mapping_dict[properties[mapping]] = i
    #     print(properties)
        # label = f"{properties['borough']} - {properties['zone']}'"
        labels.append(label(properties))#roperties[label_key])
        list_of_polygons.append([np.array(polygon_set[0]).T for polygon_set in geo['coordinates']])
        M = np.sum([polygon_set.shape[1] for polygon_set in list_of_polygons[-1]])
        # print(M)
        # N += M
    ds[column_name] = ds.func.inside_which_polygons(longitude_expression, latitude_expresion, list_of_polygons)
    if persist:
        ds.persist(column_name, overwrite=overwrite)
    ds.categorize(column_name, labels=labels, check=False)
    if mapping:
        return ds, mapping_dict
    else:
        return ds

import vaex.arrow.numpy_dispatch

@register_function()
def where(condition, x, y, dtype=None):
    # special where support for strings
    # TODO: this should be replaced by an arrow compute function in the future
    if type(x) == str:
        if type(y) == str:
            condition = vaex.array_types.to_arrow(condition).cast(pa.int8())
            choices = pa.array([y, x])
            values = choices.take(condition)
            return values
        else:
            condition = vaex.array_types.to_arrow(condition)
            indices = np.arange(len(y), dtype=np.int64)
            # point to the last value
            indices[vaex.array_types.to_numpy(condition)] = len(y)
            indices = vaex.array_types.to_arrow(indices)

            indices = vaex.arrow.numpy_dispatch.combine_missing(indices, condition)
            y = vaex.array_types.to_arrow(y)
            choices = vaex.array_types.concat([y, pa.array([x], type=y.type)])
            values = choices.take(indices)
            return values
    elif type(y) == str:
        condition = vaex.array_types.to_arrow(condition)
        indices = np.arange(len(x), dtype=np.int64)
        # point to the last value
        indices[~vaex.array_types.to_numpy(condition)] = len(x)
        indices = vaex.array_types.to_arrow(indices)

        indices = vaex.arrow.numpy_dispatch.combine_missing(indices, condition)
        x = vaex.array_types.to_arrow(x)
        choices = vaex.array_types.concat([x, pa.array([y], type=x.type)])
        values = choices.take(indices)
        return values
    elif vaex.column.is_column_like(x) and vaex.dtype_of(x).is_string:
        assert vaex.dtype_of(y).is_string
        condition = vaex.array_types.to_arrow(condition)
        assert len(x) == len(y) == len(condition)
        N = len(x)
        indices = np.arange(N, dtype=np.int64)
        mask = vaex.array_types.to_numpy(condition)
        indices[~mask] += len(x)
        indices = vaex.array_types.to_arrow(indices)

        indices = vaex.arrow.numpy_dispatch.combine_missing(indices, condition)
        x = vaex.array_types.to_arrow(x)
        y = vaex.array_types.to_arrow(y)
        x, y = vaex.arrow.convert.same_type(x, y)
        choices = vaex.array_types.concat([x, y])
        values = choices.take(indices)
        return values

    # cast x and y
    if dtype is not None:
        if np.can_cast(x, dtype):
            dtype_scalar = np.dtype(dtype)
            x = dtype_scalar.type(x)
        if np.can_cast(y, dtype):
            dtype_scalar = np.dtype(dtype)
            y = dtype_scalar.type(y)

    # default callback is on numpy
    # where() respects the dtypes of x and y; ex: if x and y are 'uint8', the resulting array will also be 'uint8'
    ar = np.where(condition, x, y)

    return ar


@register_function()
def index_values(ar):
    dtype = vaex.dtype_of(ar)
    if not dtype.is_encoded:
        raise TypeError(f'Can only get index values from a (dictionary) encoded array, not for {ar}')
    if isinstance(ar, pa.ChunkedArray):
        return pa.chunked_array([k.indices for k in ar.chunks], type=ar.type.index_type)
    else:
        return ar.indices
