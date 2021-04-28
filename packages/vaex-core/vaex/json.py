from __future__ import absolute_import
import json
import numpy as np
import datetime
import re
import vaex.hash
import pyarrow as pa


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


@register
class NumpySerializer:
    @staticmethod
    def can_encode(obj):
        return isinstance(obj, np.ndarray)

    @staticmethod
    def encode(obj):
        if np.ma.isMaskedArray(obj):
            values = obj.data.tolist()
            mask = obj.mask.tolist()
        else:
            values = obj.tolist()
            mask = None
        dtype = str(obj.dtype)
        return {
            'type': 'ndarray',
            'data': {
                'values': values,
                'mask': mask,
                'dtype': dtype
            }
        }

    @staticmethod
    def can_decode(data):
        return data.get('type') == 'ndarray'

    @staticmethod
    def decode(data):
        dtype = np.dtype(data['data']['dtype'])
        if 'mask' in data['data'] and data['data']['mask'] is not None:
            value = np.ma.array(data['data']['values'], mask=data['data']['mask'], dtype=dtype)
        else:
            value = np.array(data['data']['values'], dtype)
        return value




@register
class ArrowSerializer:
    @staticmethod
    def can_encode(obj):
        return isinstance(obj, pa.Array)

    @staticmethod
    def encode(obj):
        encoding = vaex.encoding.Encoding()
        data = encoding.encode('arrow-array', obj)
        wiredata = vaex.encoding.inline.serialize(data, encoding)
        return {'type': 'arrow-array', 'data': wiredata}

    @staticmethod
    def can_decode(data):
        return data.get('type') == 'arrow-array'

    @staticmethod
    def decode(data):
        wiredata = data['data']
        encoding = vaex.encoding.Encoding()
        data = vaex.encoding.inline.deserialize(wiredata, encoding)
        ar = encoding.decode('arrow-array', data)
        return ar


@register
class OrdererSetSerializer:
    @staticmethod
    def can_encode(obj):
        return isinstance(obj, vaex.hash.ordered_set)

    @staticmethod
    def encode(obj):
        values = list(obj.extract().items())
        clsname = obj.__class__.__name__
        return {
            'type': clsname,
            'data': {
                'values': values,
                'count': obj.count,
                'nan_count': obj.nan_count,
                'missing_count': obj.null_count
            }
        }

    @staticmethod
    def can_decode(data):
        return data.get('type', '').startswith('ordered_set')

    @staticmethod
    def decode(data):
        clsname = data['type']
        cls = getattr(vaex.hash, clsname)
        value = cls(dict(data['data']['values']), data['data']['count'], data['data']['nan_count'], data['data']['missing_count'])
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
        if isinstance(obj, np.integer):
            return obj.item()
        elif isinstance(obj, np.floating):
            return obj.item()
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
