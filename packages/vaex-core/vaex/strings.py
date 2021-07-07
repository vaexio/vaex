import os

import numpy as np
import pyarrow as pa

import vaex

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if not on_rtd:
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
    strings = vaex.column.ColumnStringArrow.from_arrow(pa.array(strings))
    return strings.string_sequence
