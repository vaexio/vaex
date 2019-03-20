import numpy as np

MAX_LENGTH_STRING = 40

def _format_value(value):
    if isinstance(value, (str, bytes)):
        if len(value) > MAX_LENGTH_STRING:
            value = repr(value[:MAX_LENGTH_STRING-3])[:-1] + '...'
    if isinstance(value, np.ma.core.MaskedConstant):
        value = str(value)
    return value
