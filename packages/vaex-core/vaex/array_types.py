"""Conversion between different types of arrays"""
import numpy as np
import pyarrow as pa
import vaex.column


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "to_numpy"):
        x = x.to_numpy()
    if isinstance(x, (pa.Array, pa.ChunkedArray)):
        import vaex.arrow.convert
        x = vaex.arrow.convert.column_from_arrow_array(x)
        return to_numpy(x)
    return np.asanyarray(x)


def to_arrow(x):
    if isinstance(x, (pa.Array, pa.ChunkedArray)):
        return x
    return pa.array(x)


def convert(x, type, default_type="numpy"):
    if type == "numpy":
        if isinstance(x, (list, tuple)):
            return np.concatenate([convert(k, type) for k in x])
        else:
            return to_numpy(x)
    elif type == "arrow":
        if isinstance(x, (list, tuple)):
            return pa.chunked_array([convert(k, type) for k in x])
        else:
            return to_arrow(x)
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


def numpy_dtype(x):
    from . import column
    if isinstance(x, column.ColumnString):
        return str
    elif isinstance(x, np.ndarray):
        return x.dtype
    elif isinstance(x, (pa.Array, pa.ChunkedArray)):
        arrow_type = x.type
        if isinstance(arrow_type, pa.DictionaryType):
            # we're interested in the type of the dictionary or the indices?
            if isinstance(x, pa.ChunkedArray):
                # take the first dictionaryu
                x = x.chunks[0]
            return numpy_dtype(x.dictionary)
        if arrow_type in (pa.string(), pa.large_string()):
            return str
        return arrow_type.to_pandas_dtype()().dtype
    else:
        raise TypeError("Cannot determine numpy dtype from: %r" % x)


def arrow_type(x):
    from . import column
    if isinstance(x, (pa.Array, pa.ChunkedArray)):
        return x.type
    else:
        return pa.array(x[0:1]).type
