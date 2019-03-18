import vaex.serialize
import json
import numpy as np
from vaex import column
from vaex.column import _to_string_sequence
import re

# @vaex.serialize.register
# class Function(FunctionSerializable):

# name maps to numpy function
# <vaex name>:<numpy name>
function_mapping = [name.strip().split(":") if ":" in name else (name, name) for name in """
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
nan
searchsorted
""".strip().split()]
expression_namespace = {}
for name, numpy_name in function_mapping:
    if not hasattr(np, numpy_name):
        raise SystemError("numpy does not have: %s" % numpy_name)
    else:
        expression_namespace[name] = getattr(np, numpy_name)


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


expression_namespace['fillna'] = fillna


def dt_dayofweek(x):
    import pandas as pd
    # x = x.astype("<M8[ns]")
    return pd.Series(x).dt.dayofweek.values

def dt_dayofyear(x):
    import pandas as pd
    # x = x.astype("<M8[ns]")
    return pd.Series(x).dt.dayofyear.values

def dt_year(x):
    import pandas as pd
    # x = x.astype("<M8[ns]")
    return pd.Series(x).dt.year.values

def dt_weekofyear(x):
    import pandas as pd
    # x = x.astype("<M8[ns]")
    return pd.Series(x).dt.weekofyear.values

def dt_hour(x):
    import pandas as pd
    # x = x.astype("<M8[ns]")
    return pd.Series(x).dt.hour.values


expression_namespace["dt_dayofweek"] = dt_dayofweek
expression_namespace["dt_dayofyear"] = dt_dayofyear
expression_namespace["dt_year"] = dt_year
expression_namespace["dt_weekofyear"] = dt_weekofyear
expression_namespace["dt_hour"] = dt_hour

_scopes = {}


def register(scope=None):
    prefix = ''
    if scope:
        prefix = scope + "_"
        if scope not in _scopes:
            _scopes[scope] = {}
    def wrapper(f):
        name = f.__name__
        # remove possible prefix
        if name.startswith(prefix):
            name = name[len(prefix):]
        if scope:
            _scopes[scope][name] = f
        expression_namespace[prefix + name] = f
    return wrapper


@register(scope='str')
def str_capitalize(x):
    sl = _to_string_sequence(x).capitalize()
    return column.ColumnStringArrow(sl.bytes, sl.indices, sl.length, sl.offset, string_sequence=sl)

@register(scope='str')
def str_cat(x, other):
    sl1 = _to_string_sequence(x)
    sl2 = _to_string_sequence(other)
    sl = sl1.concat(sl2)
    return column.ColumnStringArrow.from_string_sequence(sl)

# TODO: center

@register(scope='str')
def str_contains(x, pattern, regex=True):
    return _to_string_sequence(x).search(pattern, regex)

# TODO: default regex is False, which breaks with pandas
@register(scope='str')
def str_count(x, pat, regex=False):
    return _to_string_sequence(x).count(pat, regex)

# TODO: what to do with decode and encode

@register(scope='str')
def str_endswith(x, pat):
    return _to_string_sequence(x).endswith(pat)

# TODO: extract/extractall
# TODO: find/findall/get/index/join

@register(scope='str')
def str_len(x):
    return _to_string_sequence(x).len()

@register(scope='str')
def str_byte_length(x):
    return _to_string_sequence(x).byte_length()

# TODO: ljust

@register(scope='str')
def str_lower(x):
    sl = _to_string_sequence(x).lower()
    return column.ColumnStringArrow(sl.bytes, sl.indices, sl.length, sl.offset, string_sequence=sl)


@register(scope='str')
def str_lstrip(x, to_strip=None):
    # in c++ we give empty string the same meaning as None
    return _to_string_sequence(x).lstrip('' if to_strip is None else to_strip) if to_strip != '' else x

# TODO: match, normalize, pad partition, repeat, replace, rfind, rindex, rjust, rpartition

@register(scope='str')
def str_rstrip(x, to_strip=None):
    # in c++ we give empty string the same meaning as None
    return _to_string_sequence(x).rstrip('' if to_strip is None else to_strip) if to_strip != '' else x

# TODO: slice, slice_replace, 

@register(scope='str')
def str_split(x, pattern):  # TODO: support n
    sll = _to_string_sequence(x).split(pattern)
    return sll

# TODO: rsplit

@register(scope='str')
def str_startswith(x, pat):
    return _to_string_sequence(x).startswith(pat)

@register(scope='str')
def str_strip(x, to_strip=None):
    # in c++ we give empty string the same meaning as None
    return _to_string_sequence(x).strip('' if to_strip is None else to_strip) if to_strip != '' else x

# TODO: swapcase, title, translate

@register(scope='str')
def str_upper(x):
    sl = _to_string_sequence(x).upper()
    return column.ColumnStringArrow(sl.bytes, sl.indices, sl.length, sl.offset, string_sequence=sl)


# TODO: wrap, zfill, is*, get_dummies(maybe?)

# expression_namespace["str_contains"] = str_contains
# expression_namespace["str_capitalize"] = str_capitalize
# expression_namespace["str_contains2"] = str_contains2

@register(scope='str')
def str_strip(x, chars=None):
    # don't change the dtype, otherwise for each block the dtype may be different (string length)
    return np.char.strip(x, chars).astype(x.dtype)

@register()
def to_string(x):
    # don't change the dtype, otherwise for each block the dtype may be different (string length)
    sl = vaex.strings.to_string(x)
    return column.ColumnStringArrow(sl.bytes, sl.indices, sl.length, sl.offset, string_sequence=sl)

@register()
def format(x, format):
    """Uses http://www.cplusplus.com/reference/string/to_string/ for formatting"""
    # don't change the dtype, otherwise for each block the dtype may be different (string length)
    sl = vaex.strings.format(x, format)
    return column.ColumnStringArrow(sl.bytes, sl.indices, sl.length, sl.offset, string_sequence=sl)


_scopes['str_pandas'] = {}

for name, function in _scopes['str'].items():
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
    expression_namespace['str_pandas_' + name] = wrapper
    _scopes['str_pandas'][name] = wrapper

# expression_namespace['str_strip'] = str_strip

def _float(x):
    return x.astype(np.float64)

def _astype(x, dtype):
    return x.astype(dtype)


expression_namespace["float"] = _float
expression_namespace["astype"] = _astype



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
