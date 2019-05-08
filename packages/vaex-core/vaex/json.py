from __future__ import absolute_import
import json
import numpy as np
import datetime
import re

serializers = []


def register(cls):
    serializers.append(cls)
    return cls


@register
class DateTime64Serializer:
    @staticmethod
    def can_encode(obj):
        return isinstance(obj, (np.datetime64, np.timedelta64))

    @staticmethod
    def encode(obj):
        value = int(obj.astype(int))
        typename = obj.dtype.name
        type, unit = re.match('(\w*)\[(\w*)\]', obj.dtype.name).groups()
        return {
            'type': type,
            'data': {
                'value': value,
                'unit': unit
            }
        }

    @staticmethod
    def can_decode(data):
        return data.get('type') in ['datetime64', 'timedelta64']

    @staticmethod
    def decode(data):
        dtype = np.dtype("%s[%s]" % (data['type'], data['data']['unit']))
        value = np.int64(data['data']['value']).astype(dtype)
        return value


def encode(obj):
    for serializer in serializers:
        if serializer.can_encode(obj):
            return serializer.encode(obj)


class VaexJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        encoded = encode(obj)
        if obj is not None and encoded is not None:
            return encoded
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bytes_):
            return obj.decode('UTF-8')
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8');
        else:
            return super(VaexJsonEncoder, self).default(obj)

class VaexJsonDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)
    
    def object_hook(self, dct):
        for serializer in serializers:
            if serializer.can_decode(dct):
                return serializer.decode(dct)
        return dct
