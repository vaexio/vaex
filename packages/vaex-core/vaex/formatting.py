import numpy as np
import numbers
import six
import datetime


MAX_LENGTH = 50


def _format_value(value):
    if isinstance(value, six.string_types):
        value = str(value)
    elif isinstance(value, bytes):
        value = repr(value)
    elif isinstance(value, np.ma.core.MaskedConstant):
        value = str(value)
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            value = 'NaT'
        else:
            value = ' '.join(str(value).split('T'))
    if isinstance(value, np.timedelta64):
        if np.isnat(value):
            value = 'NaT'
        else:
            tmp = datetime.timedelta(seconds=value / np.timedelta64(1, 's'))
            ms = tmp.microseconds
            s = np.mod(tmp.seconds, 60)
            m = np.mod(tmp.seconds//60, 60)
            h = tmp.seconds // 3600
            d = tmp.days
            if ms:
                value = str('%i days %+02i:%02i:%02i.%i' % (d,h,m,s,ms))
            else:
                value = str('%i days %+02i:%02i:%02i' % (d,h,m,s))
    elif not isinstance(value, numbers.Number):
        value = str(value)
    if isinstance(value, float):
        value = repr(value)
    if isinstance(value, (str, bytes)):
        if len(value) > MAX_LENGTH:
            value = repr(value[:MAX_LENGTH-3])[:-1] + '...'
    return value
