import vaex.serialize
import json
import numpy as np
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


def str_strip(x, chars=None):
    # don't change the dtype, otherwise for each block the dtype may be different (string length)
    return np.char.strip(x, chars).astype(x.dtype)

expression_namespace['str_strip'] = str_strip

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
