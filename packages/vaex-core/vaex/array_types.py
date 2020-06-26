"""Conversion between different types of arrays"""
import numpy as np
import pyarrow as pa
import vaex.utils


supported_arrow_array_types = (pa.Array, pa.ChunkedArray)
supported_array_types = (np.ndarray, ) + supported_arrow_array_types

string_types = [pa.string(), pa.large_string()]



def is_string_type(data_type):
    return not isinstance(data_type, np.dtype) and data_type in string_types


def same_type(type1, type2):
    try:
        return type1 == type2
    except TypeError:
        # numpy dtypes don't like to be compared
        return False


def to_numpy(x, strict=False):
    import vaex.arrow.convert
    import vaex.column
    if isinstance(x, vaex.column.ColumnStringArrow):
        if strict:
            return x.to_numpy()
        else:
            return x.to_arrow()
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, supported_arrow_array_types):
        x = vaex.arrow.convert.column_from_arrow_array(x)
        return to_numpy(x, strict=strict)
    elif hasattr(x, "to_numpy"):
        x = x.to_numpy()
    return np.asanyarray(x)


def to_arrow(x, convert_to_native=False):
    if isinstance(x, supported_arrow_array_types):
        return x
    if convert_to_native and isinstance(x, np.ndarray):
        x = vaex.utils.to_native_array(x)
    return pa.array(x)


def to_xarray(x):
    import xarray
    return xarray.DataArray(to_numpy(x))


def convert(x, type, default_type="numpy"):
    import vaex.column
    if type == "numpy":
        if isinstance(x, (list, tuple)):
            return np.concatenate([convert(k, type) for k in x])
        else:
            return to_numpy(x, strict=True)
    elif type == "arrow":
        if isinstance(x, (list, tuple)):
            return pa.chunked_array([convert(k, type) for k in x])
        else:
            return to_arrow(x)
    elif type == "xarray":
        return to_xarray(x)
    elif type == "list":
        return convert(x, 'numpy').tolist()
    elif type is None:
        if isinstance(x, (list, tuple)):
            chunks = [convert(k, type) for k in x]
            if isinstance(chunks[0], (pa.Array, pa.ChunkedArray, vaex.column.ColumnStringArrow)):
                return convert(chunks, "arrow")
            elif isinstance(chunks[0], np.ndarray):
                return convert(chunks, "numpy")
            else:
                raise ValueError("Unknown type: %r" % chunks[0])
        else:
            # return convert(x, Nonedefault_type)
            return x
    else:
        raise ValueError("Unknown type: %r" % type)


def numpy_dtype(x, strict=False):
    assert not strict
    from . import column
    if isinstance(x, column.ColumnString):
        return x.dtype
    elif isinstance(x, np.ndarray):
        return x.dtype
    elif isinstance(x, supported_arrow_array_types):
        arrow_type = x.type
        if isinstance(arrow_type, pa.DictionaryType):
            # we're interested in the type of the dictionary or the indices?
            if isinstance(x, pa.ChunkedArray):
                # take the first dictionaryu
                x = x.chunks[0]
            return numpy_dtype(x.dictionary)
        if arrow_type in string_types:
            return arrow_type
        dtype = arrow_type.to_pandas_dtype()
        dtype = np.dtype(dtype)  # turn into instance
        return dtype
    else:
        raise TypeError("Cannot determine numpy dtype from: %r" % x)


def arrow_type(x):
    if isinstance(x, supported_arrow_array_types):
        return x.type
    else:
        return to_arrow(x[0:1]).type


def to_arrow_type(data_type):
    if isinstance(data_type, np.dtype):
        return arrow_type_from_numpy_dtype(data_type)
    else:
        return data_type


def to_numpy_type(data_type):
    if isinstance(data_type, np.dtype):
        return data_type
    else:
        return numpy_dtype_from_arrow_type(data_type)


def arrow_type_from_numpy_dtype(dtype):
    data = np.empty(1, dtype=dtype)
    return arrow_type(data)


def numpy_dtype_from_arrow_type(arrow_type):
    data = pa.array([], type=arrow_type)
    return numpy_dtype(data)
