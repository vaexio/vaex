from .superstrings import *
import numpy as np


def array(ar):
    if isinstance(ar, (tuple, list)):
        ar = np.asarray(ar)
    if isinstance(ar, np.ndarray):
        return StringArray(ar.astype('O'))
    elif isinstance(ar, (StringList32, StringList64)):
        return ar
    else:
        raise ValueError('Cannot convert %r to a string array' % ar)