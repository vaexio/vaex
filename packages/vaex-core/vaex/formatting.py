import numpy as np
import numbers
import six


MAX_LENGTH = 40


def _format_value(value):
    if isinstance(value, six.string_types):
        value = str(value)
    elif isinstance(value, bytes):
        value = repr(value)
    elif isinstance(value, np.ma.core.MaskedConstant):
        value = str(value)
    if isinstance(value, (np.timedelta64, np.datetime64)):
        value = str(value)
    elif not isinstance(value, numbers.Number):
        value = str(value)
    if isinstance(value, float):
        value = repr(value)
    if isinstance(value, (str, bytes)):
        if len(value) > MAX_LENGTH:
            value = repr(value[:MAX_LENGTH-3])[:-1] + '...'
    return value
