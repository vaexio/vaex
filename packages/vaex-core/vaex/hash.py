import copyreg
import sys
import os

import pyarrow as pa
import numpy as np

import vaex
import vaex.array_types
from vaex.column import _to_string_sequence
import vaex.utils


if vaex.utils.has_c_extension:
    from .superutils import *
    from . import superutils
    import dask.base

    ordered_set = tuple([cls for name, cls in vars(superutils).items() if name.startswith('ordered_set')])

    def pickle(x):
        keys = x.key_array()
        return type(x), (keys, x.null_value, x.nan_count, x.null_count, x.fingerprint)
    for cls in ordered_set:
        copyreg.pickle(cls, pickle)


    def create_set_string(keys, *other):
        keys = vaex.column.ColumnStringArrow.from_arrow(keys)
        return ordered_set_string(keys.string_sequence, *other)

    def pickle_set_string(x):
        keys = x.key_array()
        keys = pa.array(keys.to_numpy(), type=pa.large_utf8())
        return create_set_string, (keys, x.null_value, x.nan_count, x.null_count, x.fingerprint)
    copyreg.pickle(ordered_set_string, pickle_set_string)

    for ordered_set_cls in ordered_set:
        @dask.base.normalize_token.register(ordered_set_cls)
        def normalize(obj):
            if not obj.fingerprint:
                raise RuntimeError('No fingerprint present in set')
            return obj.fingerprint


def counter_type_from_dtype(dtype, transient=True):
    return vaex.utils.find_type_from_dtype(vaex.hash, 'counter_', dtype, transient=transient, support_non_native=False)


def ordered_set_type_from_dtype(dtype, transient=True):
    name = 'ordered_set_'
    return vaex.utils.find_type_from_dtype(vaex.hash, name, dtype, transient=transient, support_non_native=False)


def index_type_from_dtype(dtype, transient=True, prime_growth=False):
    name = 'index_hash_'
    if prime_growth:
        name += "_prime_growth"
    return vaex.utils.find_type_from_dtype(vaex.hash, name, dtype, transient=transient, support_non_native=False)


@vaex.encoding.register("hash-map-unique")
class HashMapUnique:
    '''HashMap that maps keys to unique integers'''
    def __init__(self, dtype, nmaps=1, limit=None, _internal=None):
        self.dtype = dtype
        self.dtype_item = self.dtype
        limit = -1 if limit is None else limit
        if _internal is None:
            cls = ordered_set_type_from_dtype(dtype)
            self._internal = cls(nmaps, limit)
        else:
            self._internal = _internal

    @staticmethod
    def encode(encoding, obj):
        keys = obj._internal.key_array()
        if isinstance(keys, (vaex.strings.StringList32, vaex.strings.StringList64)):
            keys = vaex.strings.to_arrow(keys)
        keys = encoding.encode('array', keys)
        clsname = obj._internal.__class__.__name__
        return {
            'class': clsname,
            'dtype': encoding.encode('dtype', vaex.dtype(obj.dtype)),
            'data': {
                'keys': keys,
                'null_value': obj.null_value,
                'nan_count': obj.nan_count,
                'missing_count': obj.null_count,
                'fingerprint': obj.fingerprint,
            }
        }

    @staticmethod
    def decode(encoding, obj_spec):
        clsname = obj_spec['class']
        cls = getattr(vaex.hash, clsname)
        keys = encoding.decode('array', obj_spec['data']['keys'])
        dtype = vaex.dtype_of(keys)
        if dtype.is_string:
            keys = vaex.strings.to_string_sequence(keys)
        _hash_map_internal = cls(keys, obj_spec['data']['null_value'], obj_spec['data']['nan_count'], obj_spec['data']['missing_count'], obj_spec['data']['fingerprint'])
        dtype = encoding.decode('dtype', obj_spec['dtype'])
        return vaex.hash.HashMapUnique(dtype, _internal=_hash_map_internal)

    @classmethod
    def from_keys(cls, keys, null_value, null_count, dtype=None, fingerprint=''):
        dtype = vaex.dtype_of(keys) if dtype is None else dtype
        if vaex.dtype_of(keys) == float:
            nancount = np.isnan(keys).sum()
        else:
            nancount = 0
        set_type = vaex.hash.ordered_set_type_from_dtype(dtype)
        if dtype.is_string:
            values = vaex.column.ColumnStringArrow.from_arrow(keys)
            string_sequence = values.string_sequence
            hash_map_unique_internal = set_type(string_sequence, null_value, nancount, null_count, fingerprint)
        else:
            hash_map_unique_internal = set_type(keys, null_value, nancount, null_count, fingerprint)
        return HashMapUnique(dtype, _internal=hash_map_unique_internal)

    def add(self, ar, return_inverse=False):
        if self.dtype_item.is_string:
            ar = _to_string_sequence(ar)
        else:
            ar = vaex.array_types.to_numpy(ar)
            if ar.strides != (ar.itemsize,):
                ar = ar.copy()

        chunk_size = 1024*1024
        if np.ma.isMaskedArray(ar):
            mask = np.ma.getmaskarray(ar)
            if return_inverse:
                return self._internal.update(ar, mask, -1, chunk_size=chunk_size, bucket_size=chunk_size*4, return_values=return_inverse)
            else:
                self._internal.update(ar, mask,  -1, chunk_size=chunk_size, bucket_size=chunk_size*4)
        else:
            if return_inverse:
                return self._internal.update(ar, -1, chunk_size=chunk_size, bucket_size=chunk_size*4, return_values=return_inverse)
            else:
                self._internal.update(ar, -1, chunk_size=chunk_size, bucket_size=chunk_size*4)
    
    def __len__(self):
        return len(self._internal)

    def merge(self, others):
        self._internal.merge(others)

    def keys(self, mask=True):
        ar = self._internal.key_array()
        if self.dtype_item.is_datetime or self.dtype_item.is_timedelta:
            ar = ar.view(self.dtype_item.numpy)
        if isinstance(ar, vaex.superstrings.StringList64):
            # TODO: find out why this more efficient path does not work
            # col = vaex.column.ColumnStringArrow.from_string_sequence(self.bin_values)
            # self.bin_values = pa.array(col)
            ar = pa.array(ar.to_numpy())
        if mask and self.has_null and (self.dtype_item.is_primitive or self.dtype_item.is_datetime):
            mask = np.zeros(shape=ar.shape, dtype="?")
            mask[self.null_value] = 1
            ar = np.ma.array(ar, mask=mask)
        return ar

    def map(self, keys):
        '''Map key values to unique integers'''
        from vaex.column import _to_string_sequence

        if not isinstance(keys, vaex.array_types.supported_array_types) or self.dtype == str:
            # sometimes the dtype can be object, but seen as an string array
            keys = _to_string_sequence(keys)
        else:
            keys = vaex.array_types.to_numpy(keys)
        indices = self._internal.map_ordinal(keys)
        if np.ma.isMaskedArray(keys):
            indices[keys.mask] = self.null_value
        return indices

    @property
    def has_null(self):
        return self._internal.has_null

    @property
    def has_nan(self):
        return self._internal.has_nan

    @property
    def null_value(self):
        return self._internal.null_value

    @property
    def nan_value(self):
        return self._internal.nan_value

    @property
    def fingerprint(self):
        return self._internal.fingerprint

    @property
    def nan_count(self):
        return self._internal.nan_count

    @property
    def null_count(self):
        return self._internal.null_count

    def sorted(self, keys=None, indices=None, return_keys=False):
        keys = self.keys() if keys is None else keys
        keys = vaex.array_types.to_arrow(keys)
        indices = pa.compute.sort_indices(keys) if indices is None else indices
        # arrow sorts with null last
        null_value = -1 if not self.has_null else len(keys)-1
        keys = pa.compute.take(keys, indices)
        fingerprint = self._internal.fingerprint + "-sorted"
        if self.dtype_item.is_string:
            string_sequence = vaex.column.ColumnStringArrow.from_arrow(keys).string_sequence
            set = type(self._internal)(string_sequence, null_value, self._internal.nan_count, self._internal.null_count, fingerprint)
        else:
            set = type(self._internal)(keys, null_value, self._internal.nan_count, self._internal.null_count, fingerprint)
        hash_map_sorted = HashMapUnique(self.dtype, _internal=set)
        if return_keys:
            return hash_map_sorted, keys
        else:
            return hash_map_sorted

    def limit(self, limit):
        fingerprint = self.fingerprint + f"-limit-{limit}"
        keys = self.keys(mask=False)
        keys = keys[:limit]
        null_value = self.null_value
        null_count = 1
        nan_count = 0
        if null_value >= limit:
            null_value = -1
            null_count = 0
        if self.dtype_item == float:
            nan_count = np.isnan(keys).sum()
        if self.dtype_item.is_string:
            bin_values = vaex.column.ColumnStringArrow.from_arrow(keys)
            string_sequence = bin_values.string_sequence
            hash_map_unique = type(self._internal)(string_sequence, null_value, nan_count, null_count, fingerprint)
        else:
            hash_map_unique = type(self._internal)(keys, null_value, nan_count, null_count, fingerprint)
        hash_map_unique.fingerprint = fingerprint
        return HashMapUnique(self.dtype, _internal=hash_map_unique)

    def __sizeof__(self) -> int:
        return sys.getsizeof(self._internal)


@dask.base.normalize_token.register(HashMapUnique)
def normalize(obj):
    if not obj.fingerprint:
        raise RuntimeError('No fingerprint present in HashMapUnique')
    return obj.fingerprint
