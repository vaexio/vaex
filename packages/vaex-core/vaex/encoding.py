"""Private module that determines how data is encoded and serialized, to be able to send it over a wire, or save to disk"""

import base64
import io
import json
import numbers
import uuid
import struct

import numpy as np

import vaex

registry = {}


def register(name):
    def wrapper(cls):
        assert name not in registry
        registry[name] = cls
        return cls
    return wrapper


@register("json")  # this will pass though data as is
class vaex_json_encoding:
    @classmethod
    def encode(cls, encoding, result):
        return result

    @classmethod
    def decode(cls, encoding, result_encoded):
        return result_encoded


@register("vaex-task-result")
class vaex_task_result_encoding:
    @classmethod
    def encode(cls, encoding, result):
        return encoding.encode('vaex-evaluate-result', result)

    @classmethod
    def decode(cls, encoding, result_encoded):
        return encoding.decode('vaex-evaluate-result', result_encoded)


@register("vaex-rmi-result")
class vaex_rmi_result_encoding:
    @classmethod
    def encode(cls, encoding, result):
        return encoding.encode('json', result)

    @classmethod
    def decode(cls, encoding, result_encoded):
        return encoding.decode('json', result_encoded)


@register("vaex-evaluate-result")
class vaex_evaluate_results_encoding:
    @classmethod
    def encode(cls, encoding, result):
        if isinstance(result, (list, tuple)):
            return [cls.encode(encoding, k) for k in result]
        else:
            if isinstance(result, np.ndarray):
                return {'type': 'ndarray', 'data': encoding.encode('ndarray', result)}
            elif isinstance(result, numbers.Number):
                try:
                    result = result.item()  # for numpy scalars
                except:  # noqa
                    pass
                return {'type': 'json', 'data': result}
            else:
                raise ValueError('Cannot encode: %r' % result)

    @classmethod
    def decode(cls, encoding, result_encoded):
        if isinstance(result_encoded, (list, tuple)):
            return [cls.decode(encoding, k) for k in result_encoded]
        else:
            return encoding.decode(result_encoded['type'], result_encoded['data'])


@register("ndarray")
class ndarray_encoding:
    @classmethod
    def encode(cls, encoding, array):
        # if array.dtype.kind == 'O':
        #     raise ValueError('Numpy arrays with objects cannot be serialized: %r' % array)
        mask = None
        dtype = array.dtype
        if np.ma.isMaskedArray(array):
            values = array.data
            mask = array.mask
        else:
            values = array
        if values.dtype.kind in 'mM':
            values = values.view(np.uint64)
        if values.dtype.kind == 'O':
            data = {
                    'values': values.tolist(),  # rely on json encoding
                    'shape': array.shape,
                    'dtype': encoding.encode('dtype', dtype)
            }
        else:
            data = {
                    'values': encoding.add_blob(values),
                    'shape': array.shape,
                    'dtype': encoding.encode('dtype', dtype)
            }
        if mask is not None:
            data['mask'] = encoding.add_blob(mask)
        return data

    @classmethod
    def decode(cls, encoding, result_encoded):
        if isinstance(result_encoded, (list, tuple)):
            return [cls.decode(encoding, k) for k in result_encoded]
        else:
            dtype = encoding.decode('dtype', result_encoded['dtype'])
            shape = result_encoded['shape']
            if dtype.kind == 'O':
                data = result_encoded['values']
                array = np.array(data, dtype=dtype)
            else:
                data = encoding.get_blob(result_encoded['values'])
                array = np.frombuffer(data, dtype=dtype).reshape(shape)
            if 'mask' in result_encoded:
                mask_data = encoding.get_blob(result_encoded['mask'])
                mask_array = np.frombuffer(mask_data, dtype=np.bool_).reshape(shape)
                array = np.ma.array(array, mask=mask_array)
            return array


@register("dtype")
class dtype_encoding:
    @staticmethod
    def encode(encoding, dtype):
        if dtype == str:
            return "str"
        else:
            if type(dtype) == type:
                dtype = dtype().dtype
        return str(dtype)

    @staticmethod
    def decode(encoding, type_spec):
        if type_spec == "str":
            return str
        else:
            return np.dtype(type_spec)


@register("binner")
class binner_encoding:
    @staticmethod
    def encode(encoding, binner):
        name = type(binner).__name__
        if name.startswith('BinnerOrdinal_'):
            datatype = name[len('BinnerOrdinal_'):]
            if datatype.endswith("_non_native"):
                datatype = datatype[:-len('64_non_native')]
                datatype = encoding.encode('dtype', np.dtype(datatype).newbyteorder())
            return {'type': 'ordinal', 'expression': binner.expression, 'datatype': datatype, 'count': binner.ordinal_count, 'minimum': binner.min_value}
        elif name.startswith('BinnerScalar_'):
            datatype = name[len('BinnerScalar_'):]
            if datatype.endswith("_non_native"):
                datatype = datatype[:-len('64_non_native')]
                datatype = encoding.encode('dtype', np.dtype(datatype).newbyteorder())
            return {'type': 'scalar', 'expression': binner.expression, 'datatype': datatype, 'count': binner.bins, 'minimum': binner.vmin, 'maximum': binner.vmax}
        else:
            raise ValueError('Cannot serialize: %r' % binner)

    @staticmethod
    def decode(encoding, binner_spec):
        type = binner_spec['type']
        dtype = encoding.decode('dtype', binner_spec['datatype'])
        if type == 'ordinal':
            cls = vaex.utils.find_type_from_dtype(vaex.superagg, "BinnerOrdinal_", dtype)
            return cls(binner_spec['expression'], binner_spec['count'], binner_spec['minimum'])
        elif type == 'scalar':
            cls = vaex.utils.find_type_from_dtype(vaex.superagg, "BinnerScalar_", dtype)
            return cls(binner_spec['expression'], binner_spec['minimum'], binner_spec['maximum'], binner_spec['count'])
        else:
            raise ValueError('Cannot deserialize: %r' % binner_spec)


@register("grid")
class grid_encoding:
    @staticmethod
    def encode(encoding, grid):
        return encoding.encode_list('binner', grid.binners)

    @staticmethod
    def decode(encoding, grid_spec):
        return vaex.superagg.Grid(encoding.decode_list('binner', grid_spec))


class Encoding:
    def __init__(self, next=None):
        self.registry = {**registry}
        self.blobs = {}

    def encode(self, typename, value):
        encoded = self.registry[typename].encode(self, value)
        return encoded

    def encode_list(self, typename, values):
        encoded = [self.registry[typename].encode(self, k) for k in values]
        return encoded

    def encode_dict(self, typename, values):
        encoded = {key: self.registry[typename].encode(self, value) for key, value in values.items()}
        return encoded

    def decode(self, typename, value, **kwargs):
        decoded = self.registry[typename].decode(self, value, **kwargs)
        return decoded

    def decode_list(self, typename, values, **kwargs):
        decoded = [self.registry[typename].decode(self, k, **kwargs) for k in values]
        return decoded

    def decode_dict(self, typename, values, **kwargs):
        decoded = {key: self.registry[typename].decode(self, value, **kwargs) for key, value in values.items()}
        return decoded

    def add_blob(self, buffer):
        blob_id = str(uuid.uuid4())
        self.blobs[blob_id] = memoryview(buffer).tobytes()
        return f'blob:{blob_id}'

    def get_blob(self, blob_ref):
        assert blob_ref.startswith('blob:')
        blob_id = blob_ref[5:]
        return self.blobs[blob_id]


class inline:
    @staticmethod
    def serialize(data, encoding):
        import base64
        blobs = {key: base64.b64encode(value).decode('ascii') for key, value in encoding.blobs.items()}
        return json.dumps({'data': data, 'blobs': blobs})

    @staticmethod
    def deserialize(data, encoding):
        data = json.loads(data)
        encoding.blobs = {key: base64.b64decode(value.encode('ascii')) for key, value in data['blobs'].items()}
        return data['data']


def _pack_blobs(*blobs):
    count = len(blobs)
    lenghts = [len(blob) for blob in blobs]
    stream = io.BytesIO()
    # header: <number of blobs>,<offset 0>, ... <offset N-1> with 8 byte unsigned ints
    header_length = 8 * (2 + count)
    offsets = (np.cumsum([0] + lenghts) + header_length).tolist()
    stream.write(struct.pack(f'{count+2}q', count, *offsets))
    for blob in blobs:
        stream.write(blob)
    bytes = stream.getvalue()
    assert offsets[-1] == len(bytes)
    return bytes


def _unpack_blobs(bytes):
    stream = io.BytesIO(bytes)

    count, = struct.unpack('q', stream.read(8))
    offsets = struct.unpack(f'{count+1}q', stream.read(8 * (count + 1)))
    assert offsets[-1] == len(bytes)
    blobs = []
    for i1, i2 in zip(offsets[:-1], offsets[1:]):
        blobs.append(bytes[i1:i2])
    return blobs


class binary:
    @staticmethod
    def serialize(data, encoding):
        blob_refs = list(encoding.blobs.keys())
        blobs = [encoding.blobs[k] for k in blob_refs]
        json_blob = json.dumps({'data': data, 'blob_refs': blob_refs})
        return _pack_blobs(json_blob.encode('utf8'), *blobs)

    @staticmethod
    def deserialize(data, encoding):
        json_data, *blobs = _unpack_blobs(data)
        json_data = json_data.decode('utf8')
        json_data = json.loads(json_data)
        data = json_data['data']
        encoding.blobs = {key: blob for key, blob in zip(json_data['blob_refs'], blobs)}
        return data


serialize = binary.serialize
deserialize = binary.deserialize
