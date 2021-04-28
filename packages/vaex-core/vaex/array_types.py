"""Conversion between different types of arrays"""
import numpy as np
import pyarrow as pa
import vaex.utils

supported_arrow_array_types = (pa.Array, pa.ChunkedArray)
supported_array_types = (np.ndarray, ) + supported_arrow_array_types

string_types = [pa.string(), pa.large_string()]


def full(n, value, dtype):
    from .datatype import DataType
    dtype = DataType(dtype)
    values = np.full(n, value, dtype=dtype.numpy)
    if dtype.is_arrow:
        return pa.array(values)
    else:
        return values

def is_arrow_array(ar):
    return isinstance(ar, supported_arrow_array_types)


def is_numpy_array(ar):
    return isinstance(ar, np.ndarray)


def filter(ar, boolean_mask):
    if isinstance(ar, supported_arrow_array_types):
        return ar.filter(pa.array(boolean_mask))
    else:
        return ar[boolean_mask]


def slice(ar, offset, length=None):
    if isinstance(ar, supported_arrow_array_types):
        return ar.slice(offset, length)
    else:
        if length is not None:
            return ar[offset:offset + length]
        else:
            return ar[offset:]


def concat(arrays):
    if len(arrays) == 1:
        return arrays[0]
    if any([isinstance(k, vaex.array_types.supported_arrow_array_types) for k in arrays]):
        arrays = [to_arrow(k) for k in arrays]
        flat_chunks = []
        type = arrays[0].type
        for chunk in arrays:
            if len(chunk) == 0:
                continue
            if isinstance(chunk, pa.ChunkedArray):
                flat_chunks.extend(chunk.chunks)
            else:
                flat_chunks.append(chunk)
        return pa.chunked_array(flat_chunks, type=type)

        return pa.chunked_array(arrays)
    else:
        ar = np.ma.concatenate(arrays)
        # avoid useless masks
        if ar.mask is False:
            ar = ar.data
        if ar.mask is np.False_:
            ar = ar.data
        return ar


def is_string_type(data_type):
    return not isinstance(data_type, np.dtype) and data_type in string_types


def is_string(ar):
    return isinstance(ar, supported_arrow_array_types) and is_string_type(ar.type)


def filter(ar, boolean_mask):
    if isinstance(ar, supported_arrow_array_types):
        return ar.filter(pa.array(boolean_mask))
    else:
        return ar[boolean_mask]


def same_type(type1, type2):
    try:
        return type1 == type2
    except TypeError:
        # numpy dtypes don't like to be compared
        return False


def tolist(ar):
    if isinstance(ar, list):
        return ar
    if isinstance(ar, supported_arrow_array_types):
        return ar.to_pylist()
    else:
        return ar.tolist()


def data_type(ar):
    if isinstance(ar, supported_arrow_array_types):
        return ar.type
    else:
        return ar.dtype


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
        if not strict and is_string(x):
            return x
        x = vaex.arrow.convert.column_from_arrow_array(x)
        return to_numpy(x, strict=strict)
    elif hasattr(x, "to_numpy"):
        x = x.to_numpy()
    return np.asanyarray(x)


def to_arrow(x, convert_to_native=True):
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
            return concat([convert(k, type) for k in x])
        else:
            return to_numpy(x, strict=True)
    if type == "numpy-arrow":  # used internally, numpy if possible, otherwise arrow
        if isinstance(x, (list, tuple)):
            return concat([convert(k, type) for k in x])
        else:
            return to_numpy(x, strict=False)

    elif type == "arrow":
        if isinstance(x, (list, tuple)):
            chunks = [convert(k, type) for k in x]
            return concat(chunks)
        else:
            return to_arrow(x)
    elif type == "xarray":
        return to_xarray(x)
    elif type in ['list', 'python']:
        try:
            return pa.array(x).tolist()
        except:
            return np.array(x).tolist()
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


def numpy_dtype(x, strict=True):
    from . import column
    if isinstance(x, column.ColumnString):
        return x.dtype
    elif isinstance(x, np.ndarray):
        return x.dtype
    elif isinstance(x, supported_arrow_array_types):
        arrow_type = x.type
        from .datatype import DataType
        # dtype = DataType(arrow_type)
        dtype = arrow_type.to_pandas_dtype()
        dtype = np.dtype(dtype)  # turn into instance
        if strict:
            return dtype
        else:
            if dtype.kind in 'iufbMm':
                return dtype
            else:
                return arrow_type

        # I don't there is a reason anymore to return this type, the to_pandas_dtype should
        # handle that
        # if isinstance(arrow_type, pa.DictionaryType):
        #     # we're interested in the type of the dictionary or the indices?
        #     if isinstance(x, pa.ChunkedArray):
        #         # take the first dictionary
        #         x = x.chunks[0]
        #     return numpy_dtype(x.dictionary)
        # if arrow_type in string_types:
        #     return arrow_type
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


def to_numpy_type(data_type, strict=True):
    """

    Examples:
    >>> to_numpy_type(np.dtype('f8'))
    dtype('float64')
    >>> to_numpy_type(pa.float64())
    dtype('float64')
    >>> to_numpy_type(pa.string())
    dtype('O')
    >>> to_numpy_type(pa.string(), strict=False)
    DataType(string)
    """
    if isinstance(data_type, np.dtype):
        return data_type
    else:
        return numpy_dtype_from_arrow_type(data_type, strict=strict)


def arrow_type_from_numpy_dtype(dtype):
    data = np.empty(1, dtype=dtype)
    return arrow_type(data)


def numpy_dtype_from_arrow_type(arrow_type, strict=True):
    data = pa.array([], type=arrow_type)
    return numpy_dtype(data, strict=strict)


def type_promote(t1, t2):
    # when two ndarrays, we keep it like it
    if isinstance(t1, np.dtype) and isinstance(t2, np.dtype):
        return np.promote_types(t1, t2)
    # otherwise we go to arrow
    t1 = to_arrow_type(t1)
    t2 = to_arrow_type(t2)

    if pa.types.is_null(t1):
        return t2
    if pa.types.is_null(t2):
        return t1

    if t1 == t2:
        return t1


    # TODO: so far we only use this in in code that converts to arrow
    # if we want to support numpy, we have to check it types were numpy types
    is_numerics = [pa.types.is_floating, pa.types.is_integer]
    if any(test(t1) for test in is_numerics) and any(test(t2) for test in is_numerics):
        # leverage numpy for type promotion
        dtype1 = numpy_dtype_from_arrow_type(t1)
        dtype2 = numpy_dtype_from_arrow_type(t2)
        dtype = np.promote_types(dtype1, dtype2)
        return arrow_type_from_numpy_dtype(dtype)
    elif is_string_type(t1):
        return t1
    elif is_string_type(t2):
        return t2
    else:
        raise TypeError(f'Cannot promote {t1} and {t2} to a common type')


def upcast(type):
    if isinstance(type, np.dtype):
        if type.kind == "b":
            return np.dtype('int64')
        if type.kind == "i":
            return np.dtype('int64')
        if type.kind == "u":
            return np.dtype('uint64')
        if type.kind == "f":
            return np.dtype('float64')
    else:
        dtype = numpy_dtype_from_arrow_type(type)
        dtype = upcast(dtype)
        type = arrow_type_from_numpy_dtype(dtype)

    return type


def arrow_reduce_large(arrow_array):
    if arrow_array.type == pa.large_string():
        import vaex.arrow.convert
        return vaex.arrow.convert.large_string_to_string(arrow_array)
    return arrow_array
