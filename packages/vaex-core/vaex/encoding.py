"""Private module that determines how data is encoded and serialized, to be able to send it over a wire, or save to disk"""

import base64
import io
import json
import numbers
import pickle
import uuid
import struct
import collections.abc

import numpy as np
import pyarrow as pa
import vaex
from .datatype import DataType


registry = {}


def register(name):
    def wrapper(cls):
        assert name not in registry, f'{name} already in registry: {registry[name]}'
        registry[name] = cls
        return cls
    return wrapper




def make_class_registery(groupname):
    _encoding_types = {}
    def register_helper(cls):
        name = cls.snake_name #name or getattr(cls, 'snake_name') or cls.__name__
        _encoding_types[name] = cls
        return cls

    @register(groupname)
    class encoding:
        @staticmethod
        def encode(encoding, obj):
            spec = obj.encode(encoding)
            spec[f'{groupname}-type'] = obj.snake_name
            return spec

        @staticmethod
        def decode(encoding, spec, **kwargs):
            spec = spec.copy()
            type = spec.pop(f'{groupname}-type')
            cls = _encoding_types[type]
            return cls.decode(encoding, spec, **kwargs)
    return register_helper

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
           return encoding.encode('array', result)

    @classmethod
    def decode(cls, encoding, result_encoded):
        if isinstance(result_encoded, (list, tuple)):
            return [cls.decode(encoding, k) for k in result_encoded]
        else:
            return encoding.decode('array', result_encoded)


@register("array")
class array_encoding:
    @classmethod
    def encode(cls, encoding, result):
        if isinstance(result, np.ndarray):
            return {'type': 'ndarray', 'data': encoding.encode('ndarray', result)}
        elif isinstance(result, vaex.array_types.supported_arrow_array_types):
            return {'type': 'arrow-array', 'data': encoding.encode('arrow-array', result)}
        if isinstance(result, vaex.column.Column):
            return {'type': 'column', 'data': encoding.encode('column', result)}
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
        return encoding.decode(result_encoded['type'], result_encoded['data'])


@register("arrow-array")
class arrow_array_encoding:
    @classmethod
    def encode(cls, encoding, array):
        schema = pa.schema({'x': array.type})
        with pa.BufferOutputStream() as sink:
            with pa.ipc.new_stream(sink, schema) as writer:
                writer.write_table(pa.table({'x': array}))
        blob = sink.getvalue()
        return {'arrow-ipc-blob': encoding.add_blob(blob)}

    @classmethod
    def decode(cls, encoding, result_encoded):
        if 'arrow-serialized-blob' in result_encoded:  # backward compatibility
            blob = encoding.get_blob(result_encoded['arrow-serialized-blob'])
            return pa.deserialize(blob)
        else:
            blob = encoding.get_blob(result_encoded['arrow-ipc-blob'])
            with pa.BufferReader(blob) as source:
                with pa.ipc.open_stream(source) as reader:
                    table = reader.read_all()
                    assert table.num_columns == 1
                    ar = table.column(0)
                    if len(ar.chunks) == 1:
                        ar = ar.chunks[0]
            return ar

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
                    'dtype': encoding.encode('dtype', DataType(dtype))
            }
        else:
            data = {
                    'values': encoding.add_blob(values),
                    'shape': array.shape,
                    'dtype': encoding.encode('dtype', DataType(dtype))
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
                array = np.array(data, dtype=dtype.numpy)
            else:
                data = encoding.get_blob(result_encoded['values'])
                array = np.frombuffer(data, dtype=dtype.numpy).reshape(shape)
            if 'mask' in result_encoded:
                mask_data = encoding.get_blob(result_encoded['mask'])
                mask_array = np.frombuffer(mask_data, dtype=np.bool_).reshape(shape)
                array = np.ma.array(array, mask=mask_array)
            return array


@register("numpy-scalar")
class numpy_scalar_encoding:
    @classmethod
    def encode(cls, encoding, scalar):
        if scalar.dtype.kind in 'mM':
            value = int(scalar.astype(int))
        else:
            value = scalar.item()
        return {'value': value, 'dtype': encoding.encode('dtype', DataType(scalar.dtype))}

    @classmethod
    def decode(cls, encoding, scalar_spec):
        dtype = encoding.decode('dtype', scalar_spec['dtype'])
        value = scalar_spec['value']
        return np.array([value], dtype=dtype.numpy)[0]

@register("dtype")
class dtype_encoding:
    @staticmethod
    def encode(encoding, dtype):
        dtype = DataType(dtype)
        if dtype.is_arrow and dtype.is_timedelta:
            return {'type': 'duration', 'unit': dtype.arrow.unit}
        if dtype.is_arrow and dtype.is_datetime:
            return {'type': 'timestamp', 'unit': dtype.arrow.unit}
        if dtype.is_list:
            return {'type': 'list', 'value_type': encoding.encode('dtype', dtype.value_type)}
        elif dtype.is_encoded:
            return {'type': 'dict', 'value_type': encoding.encode('dtype', dtype.value_type), 'index_type': encoding.encode('dtype', dtype.index_type), 'ordered': dtype.arrow.ordered}
        dtype = DataType(dtype)
        return str(dtype)

    @staticmethod
    def decode(encoding, type_spec):
        if isinstance(type_spec, dict):
            if type_spec['type'] == 'duration':
                return DataType(pa.duration(type_spec['unit']))
            elif type_spec['type'] == 'timestamp':
                return DataType(pa.timestamp(type_spec['unit']))
            elif type_spec['type'] == 'list':
                sub = encoding.decode('dtype', type_spec['value_type']).arrow
                return DataType(pa.list_(sub))
            elif type_spec['type'] == 'dict':
                value_type = encoding.decode('dtype', type_spec["value_type"]).arrow
                index_type = encoding.decode('dtype', type_spec["index_type"]).arrow
                bool_ordered = type_spec["ordered"]
                return DataType(pa.dictionary(index_type, value_type, bool_ordered))
            else:
                raise ValueError(f'Do not understand type {type_spec}')
        if type_spec == 'string':
            return DataType(pa.string())
        if type_spec == 'large_string':
            return DataType(pa.large_string())
        # TODO: find a proper way to support all arrow types
        if type_spec == 'timestamp[ms]':
            return DataType(pa.timestamp('ms'))
        else:
            return DataType(np.dtype(type_spec))


@register("dataframe-state")
class dataframe_state_encoding:
    @staticmethod
    def encode(encoding, state):
        return state

    @staticmethod
    def decode(encoding, state_spec):
        return state_spec


@register("selection")
class selection_encoding:
    @staticmethod
    def encode(encoding, selection):
        return selection.to_dict() if selection is not None else None

    @staticmethod
    def decode(encoding, selection_spec):
        if selection_spec is None:
            return None
        selection = vaex.selections.selection_from_dict(selection_spec)
        return selection


@register("function")
class function_encoding:
    @staticmethod
    def encode(encoding, function):
        return vaex.serialize.to_dict(function.f)

    @staticmethod
    def decode(encoding, function_spec, trusted=False):
        if function_spec is None:
            return None
        function = vaex.serialize.from_dict(function_spec, trusted=trusted)
        return function



@register("variable")
class selection_encoding:
    @staticmethod
    def encode(encoding, obj):
        if isinstance(obj, np.ndarray):
            return {'type': 'ndarray', 'data': encoding.encode('ndarray', obj)}
        elif isinstance(obj, vaex.array_types.supported_arrow_array_types):
            return {'type': 'arrow-array', 'data': encoding.encode('arrow-array', obj)}
        elif isinstance(obj, vaex.hash.HashMapUnique):
            return {'type': 'hash-map-unique', 'data': encoding.encode('hash-map-unique', obj)}
        elif isinstance(obj, np.generic):
            return {'type': 'numpy-scalar', 'data': encoding.encode('numpy-scalar', obj)}
        elif isinstance(obj, np.integer):
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
            return obj

    @staticmethod
    def decode(encoding, obj_spec):
        if isinstance(obj_spec, dict):
            return encoding.decode(obj_spec['type'], obj_spec['data'])
        else:
            return obj_spec


class Encoding:
    def __init__(self, next=None):
        self.registry = {**registry}
        self.blobs = {}
        # for sharing objects
        self._object_specs = {}
        self._objects = {}

    def set_object(self, id, obj):
        assert id not in self._objects
        self._objects[id] = obj

    def get_object(self, id):
        return self._objects[id]

    def has_object(self, id):
        return id in self._objects

    def set_object_spec(self, id, obj):
        assert id not in self._object_specs, f"Overwriting id {id}"
        self._object_specs[id] = obj

    def get_object_spec(self, id):
        return self._object_specs[id]

    def has_object_spec(self, id):
        return id in self._object_specs

    def encode(self, typename, value):
        encoded = self.registry[typename].encode(self, value)
        return encoded

    def encode_collection(self, typename, values):
        if isinstance(values, (list, tuple)):
            return self.encode_list(typename, values)
        elif isinstance(values, dict):
            return self.encode_dict(typename, values)
        else:
            return self.encode(typename, values)

    def encode_list(self, typename, values):
        encoded = [self.registry[typename].encode(self, k) for k in values]
        return encoded

    def encode_list2(self, typename, values):
        encoded = [self.encode_list(typename, k) for k in values]
        return encoded

    def encode_dict(self, typename, values):
        encoded = {key: self.registry[typename].encode(self, value) for key, value in values.items()}
        return encoded

    def decode(self, typename, value, **kwargs):
        decoded = self.registry[typename].decode(self, value, **kwargs)
        return decoded

    def decode_collection(self, typename, values, **kwargs):
        if isinstance(values, (list, tuple)):
            return self.decode_list(typename, values, **kwargs)
        elif isinstance(values, dict):
            return self.decode_dict(typename, values, **kwargs)
        else:
            return self.decode(typename, values)

    def decode_list(self, typename, values, **kwargs):
        decoded = [self.registry[typename].decode(self, k, **kwargs) for k in values]
        return decoded

    def decode_list2(self, typename, values, **kwargs):
        decoded = [self.decode_list(typename, k, **kwargs) for k in values]
        return decoded

    def decode_dict(self, typename, values, **kwargs):
        decoded = {key: self.registry[typename].decode(self, value, **kwargs) for key, value in values.items()}
        return decoded

    def add_blob(self, buffer):
        bytes = memoryview(buffer).tobytes()
        hasher = vaex.utils.create_hasher(bytes, large_data=True)
        blob_id = hasher.hexdigest()
        self.blobs[blob_id] = bytes
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
        json_blob = json.dumps({'data': data, 'blob_refs': blob_refs, 'objects': encoding._object_specs})
        return _pack_blobs(json_blob.encode('utf8'), *blobs)

    @staticmethod
    def deserialize(data, encoding):
        json_data, *blobs = _unpack_blobs(data)
        json_data = json_data.decode('utf8')
        json_data = json.loads(json_data)
        data = json_data['data']
        encoding.blobs = {key: blob for key, blob in zip(json_data['blob_refs'], blobs)}
        if 'objects' in json_data:  # for backwards compatibility, otherwise we might not be able to parse old msg'es
            encoding._object_specs = json_data['objects']
        return data


def fingerprint(typename, object):
    '''Use the encoding framework to calculate a fingerprint'''
    encoding = vaex.encoding.Encoding()
    jsonable = encoding.encode(typename, object)
    blob_keys = list(encoding.blobs)  # blob keys are hashes, so they are unique and enough for a fingerprint
    return vaex.cache.fingerprint(jsonable, blob_keys)

serialize = binary.serialize
deserialize = binary.deserialize
