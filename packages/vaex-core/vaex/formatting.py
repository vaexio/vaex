from base64 import b64encode

import numpy as np
import numbers
import six
import datetime
import pyarrow as pa
from vaex import datatype, struct

MAX_LENGTH = 50
IMAGE_WIDTH = 100
IMAGE_HEIGHT = 100



def _trim_string(value):
    if len(value) > MAX_LENGTH:
        value = repr(value[:MAX_LENGTH - 3])[:-1] + '...'
    return value


def _format_value(value, value_format='plain'):
    if value_format == "html" and hasattr(value, '_repr_png_'):
        data = value._repr_png_()
        base64_data = b64encode(data)
        data_encoded = base64_data.decode('ascii')
        url_data = f"data:image/png;base64,{data_encoded}"
        plain = f'<img src="{url_data}" width="{IMAGE_WIDTH}" height="{IMAGE_HEIGHT}"></img>'
        return plain

    # print("value = ", value, type(value), isinstance(value, numbers.Number))
    elif isinstance(value, pa.lib.Scalar):
        if datatype.DataType(value.type).is_struct:
            value = struct.format_struct_item_vaex_style(value)
        else:
            value = value.as_py()

        if value is None:
            return '--'
        else:
            return _trim_string(str(value))
    if isinstance(value, str):
        return _trim_string(str(value))
    elif isinstance(value, bytes):
        value = _trim_string(repr(value))
    elif isinstance(value, np.ma.core.MaskedConstant):
        return str(value)
    elif isinstance(value, np.datetime64):
        if np.isnat(value):
            value = 'NaT'
        else:
            value = ' '.join(str(value).split('T'))
        return value
    elif isinstance(value, np.timedelta64):
        if np.isnat(value):
            value = 'NaT'
        else:
            tmp = datetime.timedelta(seconds=value / np.timedelta64(1, 's'))
            ms = tmp.microseconds
            s = np.mod(tmp.seconds, 60)
            m = np.mod(tmp.seconds // 60, 60)
            h = tmp.seconds // 3600
            d = tmp.days
            if ms:
                value = str('%i days %+02i:%02i:%02i.%i' % (d, h, m, s, ms))
            else:
                value = str('%i days %+02i:%02i:%02i' % (d, h, m, s))
        return value
    elif isinstance(value, numbers.Number):
        value = str(value)
    else:
        value = repr(value)
        value = _trim_string(value)
    return value
