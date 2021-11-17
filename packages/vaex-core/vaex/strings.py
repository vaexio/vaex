import os

import numpy as np
import pyarrow as pa

import vaex


if vaex.utils.has_c_extension:
    from .superstrings import *

def array(ar):
    if isinstance(ar, (tuple, list)):
        ar = np.asarray(ar)
    if isinstance(ar, np.ndarray):
        return StringArray(ar.astype('O'))
    elif isinstance(ar, (StringList32, StringList64)):
        return ar
    else:
        raise ValueError('Cannot convert %r to a string array' % ar)

def to_string_sequence(strings):
    if not vaex.array_types.is_arrow_array(strings):
        strings = pa.array(strings)
    strings = vaex.column.ColumnStringArrow.from_arrow(strings)
    return strings.string_sequence


def to_arrow(ar):
    if isinstance(ar, (StringList32, StringList64)):
        col = vaex.column.ColumnStringArrow.from_string_sequence(ar)
        ar = pa.array(col)
    elif isinstance(ar, vaex.array_types.supported_arrow_array_types):
        pass
    else:
        raise TypeError(f'{ar} cannot be converted to array string array')
    return ar
