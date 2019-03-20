import numpy as np
import numbers

MAX_LENGTH = 40

def _format_value(value):
    if isinstance(value, (str, bytes)):
        value = repr(value)
    elif isinstance(value, np.ma.core.MaskedConstant):
        value = str(value)
    elif not isinstance(value, numbers.Number):
        value = str(value)
    if isinstance(value, (str, bytes)):
        value = repr(value[:MAX_LENGTH-3])[:-1] + '...'
    return value
