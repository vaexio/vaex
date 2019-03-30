import vaex.serialize
import json
import numpy as np
from vaex import column
from vaex.column import _to_string_sequence
import re
import vaex.expression
import functools

# @vaex.serialize.register_function
# class Function(FunctionSerializable):

scopes = {
    'str': vaex.expression.StringOperations,
    'str_pandas': vaex.expression.StringOperationsPandas,
    'dt': vaex.expression.DateTime
}

def register_function(scope=None, as_property=False, name=None):
    """Decorator to register a new function with vaex.

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
        if scope:
            def closure(name=name, full_name=full_name, function=f):
                def wrapper(self, *args, **kwargs):
                    lazy_func = getattr(self.expression.ds.func, full_name)
                    args = (self.expression, ) + args
                    return lazy_func(*args, **kwargs)
                return functools.wraps(function)(wrapper)
            if as_property:
                setattr(scopes[scope], name, property(closure()))
            else:
                setattr(scopes[scope], name, closure())
        else:
            def closure(name=name, full_name=full_name, function=f):
                def wrapper(self, *args, **kwargs):
                    lazy_func = getattr(self.ds.func, full_name)
                    args = (self, ) + args
                    return lazy_func(*args, **kwargs)
                return functools.wraps(function)(wrapper)
            setattr(vaex.expression.Expression, name, closure())
        vaex.expression.expression_namespace[prefix + name] = f
        return f  # we leave the original function as is
    return wrapper

# name maps to numpy function
# <vaex name>:<numpy name>
numpy_function_mapping = [name.strip().split(":") if ":" in name else (name, name) for name in """
sinc
sin
cos
tan
arcsin
arccos
arctan
arctan2
sinh
cosh
tanh
arcsinh
arccosh
arctanh
log
log10
log1p
exp
expm1
sqrt
abs
where
rad2deg
deg2rad
minimum
maximum
clip
searchsorted
""".strip().split()]
for name, numpy_name in numpy_function_mapping:
    if not hasattr(np, numpy_name):
        raise SystemError("numpy does not have: %s" % numpy_name)
    else:
        function = getattr(np, numpy_name)
        def f(function=function):
            def wrapper(*args, **kwargs):
                return function(*args, **kwargs)
            return wrapper
        try:
            function = functools.wraps(function)(f())
        except AttributeError:
            function = f()  # python 2 case
        register_function(name=name)(function)


@register_function()
def fillna(ar, value, fill_nan=True, fill_masked=True):
    '''Returns an array where missing values are replaced by value.

    If the dtype is object, nan values and 'nan' string values
    are replaced by value when fill_nan==True.
    '''
    ar = ar if not isinstance(ar, column.Column) else ar.to_numpy()
    if ar.dtype.kind in 'O' and fill_nan:
        strings = ar.astype(str)
        mask = strings == 'nan'
        ar = ar.copy()
        ar[mask] = value
    elif ar.dtype.kind in 'f' and fill_nan:
        mask = np.isnan(ar)
        if np.any(mask):
            ar = ar.copy()
            ar[mask] = value
    if fill_masked and np.ma.isMaskedArray(ar):
        mask = ar.mask
        if np.any(mask):
            ar = ar.data.copy()
            ar[mask] = value
    return ar


########## datetime operations ##########

@register_function(scope='dt', as_property=True)
def dt_dayofweek(x):
    import pandas as pd
    return pd.Series(x).dt.dayofweek.values

@register_function(scope='dt', as_property=True)
def dt_dayofyear(x):
    import pandas as pd
    return pd.Series(x).dt.dayofyear.values

@register_function(scope='dt', as_property=True)
def dt_is_leap_year(x):
    import pandas as pd
    return pd.Series(x).dt.is_leap_year.values

@register_function(scope='dt', as_property=True)
def dt_year(x):
    import pandas as pd
    return pd.Series(x).dt.year.values

@register_function(scope='dt', as_property=True)
def dt_month(x):
    import pandas as pd
    return pd.Series(x).dt.month.values

@register_function(scope='dt', as_property=True)
def dt_month_name(x):
    import pandas as pd
    return pd.Series(x).dt.month_name().values.astype(str)

@register_function(scope='dt', as_property=True)
def dt_day(x):
    import pandas as pd
    return pd.Series(x).dt.day.values

@register_function(scope='dt', as_property=True)
def dt_day_name(x):
    import pandas as pd
    return pd.Series(x).dt.day_name().values.astype(str)

@register_function(scope='dt', as_property=True)
def dt_weekofyear(x):
    import pandas as pd
    return pd.Series(x).dt.weekofyear.values

@register_function(scope='dt', as_property=True)
def dt_hour(x):
    import pandas as pd
    return pd.Series(x).dt.hour.values

@register_function(scope='dt', as_property=True)
def dt_minute(x):
    import pandas as pd
    return pd.Series(x).dt.minute.values

@register_function(scope='dt', as_property=True)
def dt_second(x):
    import pandas as pd
    return pd.Series(x).dt.second.values


########## string operations ##########

@register_function(scope='str')
def str_capitalize(x):
    sl = _to_string_sequence(x).capitalize()
    return column.ColumnStringArrow(sl.bytes, sl.indices, sl.length, sl.offset, string_sequence=sl)

@register_function(scope='str')
def str_cat(x, other):
    sl1 = _to_string_sequence(x)
    sl2 = _to_string_sequence(other)
    sl = sl1.concat(sl2)
    return column.ColumnStringArrow.from_string_sequence(sl)

# TODO: center

@register_function(scope='str')
def str_contains(x, pattern, regex=True):
    return _to_string_sequence(x).search(pattern, regex)

# TODO: default regex is False, which breaks with pandas
@register_function(scope='str')
def str_count(x, pat, regex=False):
    return _to_string_sequence(x).count(pat, regex)

# TODO: what to do with decode and encode

@register_function(scope='str')
def str_endswith(x, pat):
    return _to_string_sequence(x).endswith(pat)

# TODO: extract/extractall
# TODO: find/findall/get/index/join

@register_function(scope='str')
def str_len(x):
    return _to_string_sequence(x).len()

@register_function(scope='str')
def str_byte_length(x):
    return _to_string_sequence(x).byte_length()

# TODO: ljust

@register_function(scope='str')
def str_lower(x):
    sl = _to_string_sequence(x).lower()
    return column.ColumnStringArrow(sl.bytes, sl.indices, sl.length, sl.offset, string_sequence=sl)


@register_function(scope='str')
def str_lstrip(x, to_strip=None):
    # in c++ we give empty string the same meaning as None
    return _to_string_sequence(x).lstrip('' if to_strip is None else to_strip) if to_strip != '' else x

# TODO: match, normalize, pad partition, repeat, replace, rfind, rindex, rjust, rpartition

@register_function(scope='str')
def str_rstrip(x, to_strip=None):
    # in c++ we give empty string the same meaning as None
    return _to_string_sequence(x).rstrip('' if to_strip is None else to_strip) if to_strip != '' else x

# TODO: slice, slice_replace,

@register_function(scope='str')
def str_split(x, pattern):  # TODO: support n
    sll = _to_string_sequence(x).split(pattern)
    return sll

# TODO: rsplit

@register_function(scope='str')
def str_startswith(x, pat):
    return _to_string_sequence(x).startswith(pat)

@register_function(scope='str')
def str_strip(x, to_strip=None):
    # in c++ we give empty string the same meaning as None
    return _to_string_sequence(x).strip('' if to_strip is None else to_strip) if to_strip != '' else x

# TODO: swapcase, title, translate

@register_function(scope='str')
def str_upper(x):
    sl = _to_string_sequence(x).upper()
    return column.ColumnStringArrow(sl.bytes, sl.indices, sl.length, sl.offset, string_sequence=sl)


# TODO: wrap, zfill, is*, get_dummies(maybe?)

# expression_namespace["str_contains"] = str_contains
# expression_namespace["str_capitalize"] = str_capitalize
# expression_namespace["str_contains2"] = str_contains2

@register_function(scope='str')
def str_strip(x, chars=None):
    """Removes leading and trailing characters."""
    # don't change the dtype, otherwise for each block the dtype may be different (string length)
    return np.char.strip(x, chars).astype(x.dtype)

@register_function()
def to_string(x):
    # don't change the dtype, otherwise for each block the dtype may be different (string length)
    sl = vaex.strings.to_string(x)
    return column.ColumnStringArrow(sl.bytes, sl.indices, sl.length, sl.offset, string_sequence=sl)

@register_function()
def format(x, format):
    """Uses http://www.cplusplus.com/reference/string/to_string/ for formatting"""
    # don't change the dtype, otherwise for each block the dtype may be different (string length)
    sl = vaex.strings.format(x, format)
    return column.ColumnStringArrow(sl.bytes, sl.indices, sl.length, sl.offset, string_sequence=sl)


for name in dir(scopes['str']):
    if name.startswith('__'):
        continue
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
            return method(*args, **kwargs)
        return wrapper
    wrapper = pandas_wrapper()
    wrapper.__doc__ = "Wrapper around pandas.Series.%s" % name
    register_function(scope='str_pandas')(wrapper, name=name)
    # expression_namespace['str_pandas_' + name] = wrapper
    # _scopes['str_pandas'][name] = wrapper

# expression_namespace['str_strip'] = str_strip

@register_function(name='float')
def _float(x):
    return x.astype(np.float64)

@register_function(name='astype')
def _astype(x, dtype):
    return x.astype(dtype)


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
