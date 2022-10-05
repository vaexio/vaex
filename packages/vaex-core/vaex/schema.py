import operator
import numpy as np
import pyarrow as pa
from functools import reduce


import vaex

class resolver_flexible:
    @classmethod
    def resolve(cls, types, shapes=None):
        if any(k is None for k in types):  # because np.dtype('f8) == None #wtfpython
            # if it contains None (e.g. we need to generate missing values)
            # we move to arrow, since it is more efficient with all missing values
            types = [vaex.array_types.to_arrow_type(k) for k in types if k is not None]
        else:
            types = [k for k in types if k is not None]  # takes out None
        data_type = reduce(vaex.array_types.type_promote, types)
        if shapes is None:
            shapes = [None]
        else:
            shapes = [k for k in shapes if k is not None]  # take out None
        if shapes and any(shape != shapes[0] for shape in shapes):
            raise ValueError(f'Unequal shapes are not supported yet, please open an issue on https://github.com/vaexio/vaex/issues')
        return data_type, shapes[0]

    @classmethod
    def align(cls, N, ar, type, shape):
        # fast path for numpy
        if isinstance(ar, np.ndarray) and isinstance(type, np.dtype) and ar.dtype == type:
            return ar
        # needs a cast (or byteflip)
        if isinstance(ar, np.ndarray) and isinstance(type, np.dtype):
            return ar.astype(type)
        if ar is None:
            type = vaex.array_types.to_arrow_type(type)
            ar = pa.nulls(N, type=type)
        else:
            ar = vaex.array_types.to_arrow(ar)
            # convert null types to typed null types
            if pa.types.is_null(ar.type):
                ar = pa.nulls(len(ar), type=type)
        if ar.type != type:
            ar = ar.cast(type)
        return ar
