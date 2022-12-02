from abc import  abstractmethod, abstractproperty
import difflib
import os
from pathlib import Path
import collections.abc
import logging
import uuid
from urllib.parse import urlparse
from typing import Set, List
import threading

import numpy as np
from frozendict import frozendict
import pyarrow as pa

import vaex
import vaex.execution
import vaex.settings
import vaex.utils
from vaex.array_types import data_type
from .column import Column, ColumnIndexed, supported_column_types
from . import array_types
from vaex import encoding
try:
    from sys import version_info
    if version_info[:2] >= (3, 10):
        from importlib.metadata import entry_points
    else:
        from importlib_metadata import entry_points
except ImportError:
    import pkg_resources
    entry_points = pkg_resources.iter_entry_points

logger = logging.getLogger('vaex.dataset')

opener_classes = []
HASH_VERSION = "1"
HASH_VERSION_KEY = "version"
chunk_size_default = vaex.settings.main.chunk.size or 1024**2

_dataset_types = {}
lock = threading.Lock()


def register(cls, name=None):
    name = name or getattr(cls, 'snake_name') or cls.__name__
    _dataset_types[name] = cls
    return cls

@encoding.register('dataset')
class dataset_encoding:
    @staticmethod
    def encode(encoding, dataset):
        return dataset.encode(encoding)

    @staticmethod
    def decode(encoding, dataset_spec):
        dataset_spec = dataset_spec.copy()
        type = dataset_spec.pop('dataset_type')
        cls = _dataset_types[type]
        return cls.decode(encoding, dataset_spec)


def open(path, fs_options={}, fs=None, *args, **kwargs):
    failures = []
    with lock:  # since we cache, make this thread save
        if not opener_classes:
            for entry in entry_points(group='vaex.dataset.opener'):
                logger.debug('trying opener: ' + entry.name)
                try:
                    opener = entry.load()
                    opener_classes.append(opener)
                except Exception as e:
                    logger.exception('issue loading ' + entry.name)
                    failures.append((e, entry))

    # first the quick path
    for opener in opener_classes:
        if opener.quick_test(path, fs_options=fs_options, fs=fs):
            if opener.can_open(path, fs_options=fs_options, fs=fs, *args, **kwargs):
                return opener.open(path, fs_options=fs_options, fs=fs, *args, **kwargs)

    # otherwise try all openers
    for opener in opener_classes:
        try:
            if opener.can_open(path, fs_options=fs_options, fs=fs, *args, **kwargs):
                return opener.open(path, fs_options=fs_options, fs=fs, *args, **kwargs)
        except Exception as e:
            failures.append((e, opener))

    failures = "\n".join([f'\n-----{who}-----\n:' + vaex.utils.format_exception_trace(e) for e, who in failures])
    if failures:
        raise IOError(f'Cannot open {path}, failures: {failures}.')
    else:
        raise IOError(f'Cannot open {path} nobody knows how to read it.')


def _to_bytes(ar):
    try:
        return ar.view(np.uint8)
    except ValueError:
        return ar.copy().view(np.uint8)

def hash_combine(*hashes):
    hasher = vaex.utils.create_hasher(large_data=False)
    for hash in hashes:
        hasher.update(hash.encode())
    return hasher.hexdigest()


def hash_slice(hash, start, end):
    hasher = vaex.utils.create_hasher(hash.encode(), large_data=False)
    slice = np.array([start, end], dtype=np.int64)
    hasher.update(_to_bytes(slice))
    return hasher.hexdigest()


def hash_array_data(ar):
    # this function should stay consistent with all future versions
    # since this is the expensive part of the hashing
    if isinstance(ar, np.ndarray):
        ar = ar.ravel()
        if ar.dtype == np.object_:
            return {"type": "numpy", "data": str(uuid.uuid4()), "mask": None}
        if np.ma.isMaskedArray(ar):
            data_byte_ar = _to_bytes(ar.data)
            hasher = vaex.utils.create_hasher(data_byte_ar, large_data=True)
            hash_data = {"type": "numpy", "data": hasher.hexdigest(), "mask": None}
            if ar.mask is not True and ar.mask is not False and ar.mask is not np.True_ and ar.mask is not np.False_:
                mask_byte_ar = _to_bytes(ar.mask)
                hasher = vaex.utils.create_hasher(mask_byte_ar, large_data=True)
                hash_data["mask"] = hasher.hexdigest()
            return hash_data
        else:
            try:
                byte_ar = _to_bytes(ar)
            except ValueError:
                byte_ar = ar.copy().view(np.uint8)
            hasher = vaex.utils.create_hasher(byte_ar, large_data=True)
            hash_data = {"type": "numpy", "data": hasher.hexdigest(), "mask": None}
    elif isinstance(ar, (pa.Array, pa.ChunkedArray)):
        hasher = vaex.utils.create_hasher(large_data=True)
        buffer_hashes = []
        hash_data = {"type": "arrow", "buffers": buffer_hashes}
        if isinstance(ar, pa.ChunkedArray):
            chunks = ar.chunks
        else:
            chunks = [ar]
        for chunk in chunks:
            for buffer in chunk.buffers():
                if buffer is not None:
                    hasher.update(memoryview(buffer))
                    buffer_hashes.append(hasher.hexdigest())
                else:
                    buffer_hashes.append(None)
    elif isinstance(ar, vaex.column.Column):
        hash_data = {"type": "column", "fingerprint": ar.fingerprint()}
    else:
        raise TypeError
    return hash_data


def hash_array(ar, hash_info=None, return_info=False):
    # this function can change over time, as it builds on top of the expensive part
    # (hash_array_data), so we can cheaply calculate new hashes if we pass on hash_info
    if hash_info is None:
        hash_info = hash_array_data(ar)
    if hash_info.get(HASH_VERSION_KEY) == HASH_VERSION:  # TODO: semver check?
        return hash_info['hash'], hash_info
    if isinstance(ar, np.ndarray):
        if ar.dtype == np.object_:
            return hash_info['data']  # uuid, so always unique
        if np.ma.isMaskedArray(ar):
            if not (hash_info['type'] == 'numpy' and hash_info['data'] and hash_info['mask']):
                hash_info = hash_array_data(ar)
        else:
            if not (hash_info['type'] == 'numpy' and hash_info['data']):
                hash_info = hash_array_data(ar)
        keys = [HASH_VERSION, hash_info['type'], hash_info['data']]
        if hash_info['mask']:
            keys.append(hash_info['mask'])
    elif isinstance(ar, vaex.array_types.supported_arrow_array_types):
        if not (hash_info['type'] == 'arrow' and hash_info['buffers']):
            hash_info = hash_array_data(ar)
        keys = [HASH_VERSION]
        keys.extend(["NO_BUFFER" if not b else b for b in hash_info['buffers']])
    elif isinstance(ar, vaex.column.Column):
        if not (hash_info['type'] == 'column'):
            hash_info = hash_array_data(ar)
        keys = [HASH_VERSION]
        keys.append(hash_info['fingerprint'])
    hasher = vaex.utils.create_hasher(large_data=False)  # small amounts of data
    for key in keys:
        hasher.update(key.encode('ascii'))
    hash = hasher.hexdigest()
    if return_info:
        hash_info['hash'] = hash
        hash_info[HASH_VERSION_KEY] = HASH_VERSION
        return hash, hash_info
    else:
        return hash


def to_supported_array(ar):
    if not isinstance(ar, supported_column_types):
        ar = np.asanyarray(ar)

    if isinstance(ar, np.ndarray) and ar.dtype.kind == 'U':
        ar = vaex.column.ColumnArrowLazyCast(ar, pa.string())
    elif isinstance(ar, np.ndarray) and ar.dtype.kind == 'O':
        ar_data = ar
        if np.ma.isMaskedArray(ar):
            ar_data = ar.data

        try:
            # "k != k" is a way to detect NaN's and NaT's
            types = list({type(k) for k in ar_data if k is not None and k == k})
        except ValueError:
            # If there is an array value in the column, Numpy throws a ValueError
            # "The truth value of an array with more than one element is ambiguous".
            # We don't handle this by default as it is a bit slower.
            def is_missing(k):
                if k is None:
                    return True
                try:
                    # a way to detect NaN's and NaT
                    return not (k == k)
                except ValueError:
                    # if a value is an array, this will fail, and it is a non-missing
                    return False
            types = list({type(k) for k in ar_data if k is not is_missing(k)})

        if len(types) == 1 and issubclass(types[0], str):
            # TODO: how do we know it should not be large_string?
            # self._dtypes_override[valid_name] = pa.string()
            ar = vaex.column.ColumnArrowLazyCast(ar, pa.string())
        if len(types) == 0:  # can only be if all nan right?
            ar = ar.astype(np.float64)
    return ar


def _concat_chunk_list(list_of_chunks):
    dict_of_list_of_arrays = collections.defaultdict(list)
    for chunks in list_of_chunks:
        for name, array in chunks.items():
            if isinstance(array, pa.ChunkedArray):
                dict_of_list_of_arrays[name].extend(array.chunks)
            else:
                dict_of_list_of_arrays[name].append(array)
    chunks = {name: vaex.array_types.concat(arrays) for name, arrays in dict_of_list_of_arrays.items()}
    return chunks


def _slice_of_chunks(chunks_ready_list, chunk_size):
    current_row_count = 0
    chunks_current_list = []
    while current_row_count < chunk_size and chunks_ready_list:
        chunks_current = chunks_ready_list.pop(0)
        chunk = list(chunks_current.values())[0]
        # chunks too large, split, and put back a part
        if current_row_count + len(chunk) > chunk_size:
            strict = True
            if strict:
                needed_length = chunk_size - current_row_count
                current_row_count += needed_length
                assert current_row_count == chunk_size


                chunks_head = {name: vaex.array_types.slice(chunk, 0, needed_length) for name, chunk in chunks_current.items()}
                chunks_current_list.append(chunks_head)
                chunks_extra = {name: vaex.array_types.slice(chunk, needed_length) for name, chunk in chunks_current.items()}
                chunks_ready_list.insert(0, chunks_extra)  # put back the extra in front
            else:
                current_row_count += len(chunk)
                chunks_current_list.append(chunks_current)
        else:
            current_row_count += len(chunk)
            chunks_current_list.append(chunks_current)
    return chunks_current_list, current_row_count


def chunk_rechunk(chunk_iter, chunk_size):
    chunks_ready_list = []
    i1 = i2 = 0
    for _, _, chunks in chunk_iter:
        chunks_ready_list.append(chunks)
        total_row_count = sum([len(list(k.values())[0]) for k in chunks_ready_list])
        if total_row_count > chunk_size:
            chunks_current_list, current_row_count = vaex.dataset._slice_of_chunks(chunks_ready_list, chunk_size)
            i2 += current_row_count
            chunks = vaex.dataset._concat_chunk_list(chunks_current_list)
            yield i1, i2, chunks
            i1 = i2

    while chunks_ready_list:
        chunks_current_list, current_row_count = vaex.dataset._slice_of_chunks(chunks_ready_list, chunk_size)
        i2 += current_row_count
        chunks = vaex.dataset._concat_chunk_list(chunks_current_list)
        yield i1, i2, chunks
        i1 = i2


def _rechunk(chunk_iter, chunk_size):
    def wrapper():
        i1 = i2 = 0
        for chunks in chunk_iter:
            i2 += len(list(chunks.values())[0])
            yield i1, i2, chunks
            i1 = i2
    yield from chunk_rechunk(wrapper(), chunk_size)


def empty_chunk_iterator(start, end, chunk_size):
    length = end - start
    i1 = 0
    i2 = min(length, i1 + chunk_size)
    while i1 < length:
        yield i1, i2, {}
        i1 = i2
        i2 = min(length, i1 + chunk_size)


class Dataset(collections.abc.Mapping):
    def __init__(self):
        super().__init__()
        self._columns = frozendict()
        self._row_count = None
        self._id = str(uuid.uuid4())
        self._cached_fingerprint = None

    def _check_existence(self, column_name):
        matches = difflib.get_close_matches(column_name, list(self._columns))
        msg = "Column or variable %r does not exist." % column_name
        if matches:
            msg += ' Did you mean: ' + " or ".join(map(repr, matches))
        else:
            msg += ' Available columns or variables: ' + ", ".join(map(repr, list(self._columns)))
        raise KeyError(msg)

    def __repr__(self):
        import yaml
        data = self.__repr_data__()
        return yaml.dump(data, sort_keys=False, indent=4)

    def __repr_data__(self):
        state = self.__getstate__()
        def normalize(v):
            if isinstance(v, Dataset):
                return v.__repr_data__()
            if isinstance(v, frozendict):
                return dict(v)
            if isinstance(v, vaex.dataframe.DataFrame):
                return {'type': 'dataframe', 'repr': repr(v)}
            if isinstance(v, np.ndarray):
                return v.tolist()
            return v
        return {'type': self.snake_name, **{k: normalize(v) for k, v in state.items() if not k.startswith('_')}}

    @property
    def id(self):
        '''id that uniquely identifies a dataset at runtime'''
        return self.fingerprint

    @property
    def fingerprint(self):
        '''id that uniquely identifies a dataset cross runtime, might be more expensive and require hasing'''
        if self._cached_fingerprint is None:
            self._cached_fingerprint = self._fingerprint
        return self._cached_fingerprint

    @abstractproperty
    def _fingerprint(self):
        pass

    def encode(self, encoding):
        if not encoding.has_object_spec(self.id):
            spec = self._encode(encoding)
            encoding.set_object_spec(self.id, spec)
        return {'dataset_type': self.snake_name, 'object-id': self.id}

    @classmethod
    def decode(cls, encoding, spec):
        id = spec['object-id']
        if not encoding.has_object(id):
            spec = encoding.get_object_spec(id)
            ds = cls._decode(encoding, spec)
            encoding.set_object(id, ds)
        return encoding.get_object(id)

    @abstractmethod
    def _create_columns(self):
        pass

    @property
    def name(self):
        # TODO: in the future, we might want to use self.fingerprint or self.id
        return "no-name"

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_columns']
        del state['_cached_fingerprint']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state.copy())
        self._cached_fingerprint = None
        self._create_columns()

    def schema(self, array_type=None):
        return {name: vaex.array_types.data_type(col) for name, col in self.items()}

    def shapes(self):
        return {name: self.shape(name) for name, col in self.items()}

    def _set_row_count(self):
        if not self._columns:
            return
        values = list(self._columns.values())
        self._row_count = len(values[0])
        for name, value in list(self._columns.items())[1:]:
            if len(value) != self._row_count:
                raise ValueError(f'First columns has length {self._row_count}, while column {name} has length {len(value)}')

    @property
    def row_count(self):
        return self._row_count

    def project(self, *names):
        all = set(self)
        drop = all - set(names)
        # we want a deterministic order for fingerprints
        drop = list(drop)
        drop.sort()
        return self.dropped(*list(drop))

    def concat(self, *others, resolver='flexible'):
        datasets = []
        if isinstance(self, DatasetConcatenated):
            datasets.extend(self.datasets)
        else:
            datasets.extend([self])
        for other in others:
            if isinstance(other, DatasetConcatenated):
                datasets.extend(other.datasets)
            else:
                datasets.extend([other])
        return DatasetConcatenated(datasets, resolver=resolver)

    def take(self, indices, masked=False):
        return DatasetTake(self, indices, masked=masked)

    def renamed(self, renaming):
        return DatasetRenamed(self, renaming)

    def merged(self, rhs):
        return DatasetMerged(self, rhs)

    def dropped(self, *names):
        return DatasetDropped(self, names)

    def __getitem__(self, item):
        if isinstance(item, slice):
            assert item.step in [1, None]
            return self.slice(item.start or 0, item.stop or self.row_count)
        return self._columns[item]

    def __len__(self):
        return len(self._columns)

    def __iter__(self):
        return iter(self._columns)

    def get_data(self, i1, i2, names):
        raise NotImplementedError

    def __eq__(self, rhs):
        if not isinstance(rhs, Dataset):
            return NotImplemented
        # simple case, if fingerprints are equal, the data is equal
        
        if self.fingerprint == rhs.fingerprint:
            return True
        # but no the other way around
        keys = set(self)
        keys_hashed = set(self._ids)
        missing = keys ^ keys_hashed
        if missing:
            return self.fingerprint == rhs.fingerprint
        keys = set(rhs)
        keys_hashed = set(rhs._ids)
        missing = keys ^ keys_hashed
        if missing:
            return self.fingerprint == rhs.fingerprint
        return self._ids == rhs._ids

    def __hash__(self):
        keys = set(self)
        keys_hashed = set(self._ids)
        missing = keys ^ keys_hashed
        if missing:
            # if we don't have hashes for all columns, we just use the fingerprint
            return hash(self.fingerprint)
        return hash(tuple(self._ids.items()))

    def _default_lazy_chunk_iterator(self, array_map, columns, chunk_size, reverse=False):
        chunk_size = chunk_size or 1024**2
        chunk_count = (self.row_count + chunk_size - 1) // chunk_size
        chunks = range(chunk_count)
        if reverse:
            chunks = reversed(chunks)
        for i in chunks:
            i1 = i * chunk_size
            i2 = min((i + 1) * chunk_size, self.row_count)
            def reader(i1=i1, i2=i2):
                chunks = {k: array_map[k][i1:i2] for k in columns}
                length = i2 - i1
                for name, chunk in chunks.items():
                    assert len(chunk) == length, f'Oops, got a chunk ({name}) of length {len(chunk)} while it is expected to be of length {length} (at {i1}-{i2}'
                return chunks
            yield i1, i2, reader

    def _default_chunk_iterator(self, array_map, columns, chunk_size, reverse=False):
        for i1, i2, reader in self._default_lazy_chunk_iterator(array_map, columns, chunk_size, reverse):
            yield i1, i2, reader()

    @abstractmethod
    def chunk_iterator(self, columns, chunk_size=None, reverse=False):
        pass

    @abstractmethod
    def is_masked(self, column):
        pass

    @abstractmethod
    def shape(self, column):
        pass

    @abstractmethod
    def close(self):
        '''Close file handles or other resources, the DataFrame will not be in a usable state afterwards.'''
        pass

    @abstractmethod
    def slice(self, start, end):
        pass

    @abstractmethod
    def hashed(self):
        pass

    @abstractmethod
    def leafs(self) -> List["Dataset"]:
        pass


class DatasetDecorator(Dataset):
    def __init__(self, original):
        super().__init__()
        self.original = original

    def leafs(self) -> List[Dataset]:
        return self.original.leafs()

    def close(self):
        self.original.close()

    def is_masked(self, column):
        return self.original.is_masked(column)

    def shape(self, column):
        return self.original.shape(column)


class ColumnProxy(vaex.column.Column):
    '''To give the Dataset._columns object useful containers for debugging'''
    ds: Dataset

    def __init__(self, ds, name, type):
        self.ds = ds
        self.name = name
        self.dtype = type

    def _fingerprint(self):
        fp = vaex.cache.fingerprint(self.ds.fingerprint, self.name)
        return f'column-proxy-{fp}'

    def __len__(self):
        return self.ds.row_count

    def to_numpy(self):
        values = self[:]
        return np.array(values)

    def __getitem__(self, item):
        if isinstance(item, slice):
            array_chunks = []
            ds = self.ds.__getitem__(item)
            for chunk_start, chunk_end, chunks in ds.chunk_iterator([self.name]):
                ar = chunks[self.name]
                if isinstance(ar, pa.ChunkedArray):
                    array_chunks.extend(ar.chunks)
                else:
                    array_chunks.append(ar)
            if len(array_chunks) == 1:
                return array_chunks[0]
            if len(array_chunks) == 0:
                return vaex.dtype(self.dtype).create_array([])
            return vaex.array_types.concat(array_chunks)
        else:
            raise NotImplementedError


@register
class DatasetRenamed(DatasetDecorator):
    snake_name = 'rename'
    def __init__(self, original, renaming):
        super().__init__(original)
        self.renaming = renaming
        self.reverse = {v: k for k, v in renaming.items()}
        self._create_columns()
        self._ids = frozendict({renaming.get(name, name): ar for name, ar in original._ids.items()})
        self._set_row_count()

    def renamed(self, renaming):
        # # {'a': 'x', 'b': 'y'} and {'x': 'a', 'b': 'z', 'c', 'q'} -> {'b': 'z', 'c': 'q'}
        resulting = {}
        renaming = renaming.copy()  # we'll modify in place
        for old, new in self.renaming.items():
            if new in renaming:
                if old == renaming[new]:
                    pass # e.g.  x->a->x
                else:
                    resulting[old] = renaming[new]
                del renaming[new]  # we already covered this
            else:
                # e.g. x->a->a
                resulting[old] = new
        # e.g. x->x->a
        resulting.update(renaming)
        return DatasetRenamed(self.original, resulting)

    @property
    def _fingerprint(self):
        id = vaex.cache.fingerprint(self.original.fingerprint, self.renaming)
        return f'dataset-{self.snake_name}-{self.original.fingerprint}'

    def _create_columns(self):
        self._columns = frozendict({self.renaming.get(name, name): ar for name, ar in self.original.items()})

    def _encode(self, encoding):
        dataset_spec = encoding.encode('dataset', self.original)
        return {'renaming': dict(self.renaming), 'dataset': dataset_spec}

    @classmethod
    def _decode(cls, encoding, spec):
        dataset = encoding.decode('dataset', spec['dataset'])
        return cls(dataset, spec['renaming'])

    def chunk_iterator(self, columns, chunk_size=None, reverse=False):
        for name in columns:
            if name in self.renaming:
                rename = self.renaming[name]
                raise KeyError(f'Oops, you tried to get column {name}, but you renamed it to {rename}')
        columns = [self.reverse.get(name, name) for name in columns]
        for i1, i2, chunks in self.original.chunk_iterator(columns, chunk_size, reverse=reverse):
            yield i1, i2, {self.renaming.get(name, name): ar for name, ar in chunks.items()}

    def is_masked(self, column):
        return self.original.is_masked(self.reverse.get(column, column))

    def shape(self, column):
        return self.original.shape(self.reverse.get(column, column))

    def slice(self, start, end):
        if start == 0 and end == self.row_count:
            return self
        return type(self)(self.original.slice(start, end), self.renaming)

    def hashed(self):
        if set(self._ids) == set(self):
            return self
        return type(self)(self.original.hashed(), self.renaming)


@register
class DatasetConcatenated(Dataset):
    snake_name = "concat"
    def __init__(self, datasets, resolver):
        super().__init__()
        self.datasets = datasets
        self.resolver = resolver
        if self.resolver == 'strict':
            for dataset in datasets[1:]:
                if set(dataset) != set(datasets[0]):
                    l = set(dataset)
                    r = set(datasets[0])
                    diff = l ^ r
                    raise NameError(f'Concatenating datasets with different names: {l} and {r} (difference: {diff})')
            self._schema = datasets[0].schema()
            self._shapes = datasets[0].shapes()
            for dataset in datasets[1:]:
                if dataset.shapes() != self._shapes:
                    raise ValueError(f'Cannot concatenate with different shapes: {self._shapes} != {dataset.shapes()}')
            for dataset in datasets[1:]:
                schema = dataset.schema()
                if dataset.schema() != self._schema:
                    raise ValueError(f'Cannot concatenate with different schemas: {self._shapes} != {dataset.shapes()}')
        elif self.resolver == 'flexible':
            schemas = [ds.schema() for ds in datasets]
            shapes = [ds.shapes() for ds in datasets]
            # try to keep the order of the original dataset
            schema_list_map = {}
            for schema in schemas:
                for name, type in schema.items():
                    if name not in schema_list_map:
                        schema_list_map[name] = []
            for name, type_list in schema_list_map.items():
                for schema in schemas:
                    # None means it is means the column is missing
                    type_list.append(schema.get(name))
            from .schema import resolver_flexible

            # shapes
            shape_list_map = {}
            for shape in shapes:
                for name, type in shape.items():
                    if name not in shape_list_map:
                        shape_list_map[name] = []
            for name, shape_list in shape_list_map.items():
                for shapes_ in shapes:
                    # None means it is means the column is missing
                    shape_list.append(shapes_.get(name))
            self._schema = {}
            self._shapes = {}

            for name in shape_list_map:
                self._schema[name], self._shapes[name] = resolver_flexible.resolve(schema_list_map[name], shape_list_map[name])
        else:
            raise ValueError(f'Invalid resolver {resolver}, choose between "strict" or "flexible"')

        self._create_columns()
        self._set_row_count()

    @property
    def _fingerprint(self):
        ids = [ds.fingerprint for ds in self.datasets]
        id = vaex.cache.fingerprint(*ids)
        return f'dataset-{self.snake_name}-{id}'

    def _create_columns(self):
        columns = {}
        hashes = {}
        for name in self._schema:
            columns[name] = ColumnProxy(self, name, self._schema[name])
            if all(name in ds._ids for ds in self.datasets):
                hashes[name] = hash_combine(*[ds._ids[name] for ds in self.datasets])
        self._columns = frozendict(columns)
        self._ids = frozendict(hashes)

    def _encode(self, encoding, skip=set()):
        datasets = encoding.encode_list('dataset', self.datasets)
        spec = {'dataset_type': self.snake_name, 'datasets': datasets, 'resolver': self.resolver}
        return spec

    @classmethod
    def _decode(cls, encoding, spec):
        datasets = encoding.decode_list('dataset', spec['datasets'])
        ds = cls(datasets, spec['resolver'])
        return ds

    def is_masked(self, column):
        for dataset in self.datasets:
            if column not in dataset:
                # if the column is not in the dataset, we assume it is not masked
                # since we ues arrow in that case
                return False
        return any(k.is_masked(column) for k in self.datasets)

    def shape(self, column):
        return self._shapes[column]

    def _set_row_count(self):
        self._row_count = sum(ds.row_count for ds in self.datasets)

    def schema(self, array_type=None):
        return self._schema.copy()

    def _chunk_iterator_non_strict(self, columns, chunk_size=None, reverse=False, start=0, end=None):
        end = self.row_count if end is None else end
        offset = 0
        for dataset in self.datasets:
            present = [k for k in columns if k in dataset]
            # skip over whole datasets
            if start >= offset + dataset.row_count:
                offset += dataset.row_count
                continue
            # we are past the end
            if end <= offset:
                break
            for i1, i2, chunks in dataset.chunk_iterator(present, chunk_size=chunk_size, reverse=reverse):
                # chunks = {name: vaex.array_types.to_arrow(ar) for name, ar in chunks.items()}
                length = i2 - i1
                chunk_start = offset
                chunk_end = offset + length
                if start >= chunk_end:  # we didn't find the beginning yet
                    offset += length
                    continue
                if end <= chunk_start:  # we are past the end
                    # assert False
                    break

                if start > chunk_start:
                    # this means we have to cut off a piece of the beginning
                    if end < chunk_end:
                        # AND the end
                        length = end - chunk_start  # without the start cut off
                        length -= start - chunk_start  # correcting for the start cut off
                        assert length > 0
                        chunks = {name: vaex.array_types.slice(ar, start - chunk_start, length) for name, ar in chunks.items()}
                        for name, ar in chunks.items():
                            assert len(ar) == length, f'Oops, array was expected to be of length {length} but was {len(ar)}'
                    else:
                        length -= start - chunk_start  # correcting for the start cut off
                        assert length > 0
                        chunks = {name: vaex.array_types.slice(ar, start - chunk_start) for name, ar in chunks.items()}
                        for name, ar in chunks.items():
                            assert len(ar) == length, f'Oops, array was expected to be of length {length} but was {len(ar)}'
                else:
                    if end < chunk_end:
                        # we only need to cut off a piece of the end
                        length = end - chunk_start
                        assert length > 0
                        chunks = {name: vaex.array_types.slice(ar, 0, length) for name, ar in chunks.items()}
                        for name, ar in chunks.items():
                            assert len(ar) == length, f'Oops, array was expected to be of length {length} but was {len(ar)}'

                from .schema import resolver_flexible
                allchunks = {name: resolver_flexible.align(length, chunks.get(name), self._schema[name], self._shapes[name]) for name in columns}
                yield {k: allchunks[k] for k in columns}
                offset += (i2 - i1)

    def chunk_iterator(self, columns, chunk_size=None, reverse=False, start=0, end=None):
        chunk_size = chunk_size or 1024*1024
        i1 = 0
        i1 = i2 = 0
        if not columns:
            end = self.row_count if end is None else end
            yield from empty_chunk_iterator(start, end, chunk_size)
        else:
            chunk_iterator = self._chunk_iterator_non_strict(columns, chunk_size, reverse=reverse, start=start, end=self.row_count if end is None else end)
            yield from _rechunk(chunk_iterator, chunk_size)

    def close(self):
        for ds in self.datasets:
            ds.close()

    def slice(self, start, end):
        if start == 0 and end == self.row_count:
            return self
        # TODO: we can be smarter here, and trim off some datasets
        return DatasetSliced(self, start=start, end=end)

    def hashed(self):
        if set(self._ids) == set(self):
            return self
        return type(self)([dataset.hashed() for dataset in self.datasets], resolver=self.resolver)

    def leafs(self) -> List[Dataset]:
        return [self]

    # def leafs(self) -> List[Dataset]:
    #     leafs = list()
    #     for ds in self.datasets:
    #         leafs.extend(ds.leafs())
    #     return leafs

@register
class DatasetTake(DatasetDecorator):
    snake_name = "take"
    def __init__(self, original, indices, masked):
        super().__init__(original)
        self.indices = indices
        self.masked = masked
        self._lazy_hash_index = None
        self._create_columns()
        self._set_row_count()

    @property
    def _fingerprint(self):
        id = vaex.cache.fingerprint(self.original.fingerprint, self._hash_index, self.masked)
        return f'dataset-{self.snake_name}-{id}'

    @property
    def _hash_index(self):
        if self._lazy_hash_index is None:
            self._lazy_hash_index = hash_array(self.indices)
        return self._lazy_hash_index

    def _create_columns(self):
        # if the columns in ds already have a ColumnIndex
        # we could do, direct_indices = df.column['bla'].indices[indices]
        # which should be shared among multiple ColumnIndex'es, so we store
        # them in this dict
        direct_indices_map = {}
        columns = {}
        hashes = {}
        for name, column in self.original.items():
            columns[name] = ColumnIndexed.index(column, self.indices, direct_indices_map, masked=self.masked)
            if name in self.original._ids:
                hashes[name] = hash_combine(self._hash_index, self.original._ids[name])
        self._columns = frozendict(columns)
        self._ids = frozendict(hashes)

    def _encode(self, encoding, skip=set()):
        dataset_spec = encoding.encode('dataset', self.original)
        spec = {'dataset_type': self.snake_name, 'dataset': dataset_spec}
        spec['indices'] = encoding.encode('array', self.indices)
        spec['masked'] = self.masked
        return spec

    @classmethod
    def _decode(cls, encoding, spec):
        dataset = encoding.decode('dataset', spec['dataset'])
        indices = encoding.decode('array', spec['indices'])
        ds = cls(dataset, indices, spec['masked'])
        return ds

    def chunk_iterator(self, columns, chunk_size=None, reverse=False):
        # TODO: we may be able to do this slightly more efficient by first
        # materializing the columns
        yield from self._default_chunk_iterator(self._columns, columns, chunk_size, reverse=reverse)

    def slice(self, start, end):
        if start == 0 and end == self.row_count:
            return self
        return DatasetSlicedArrays(self, start=start, end=end)

    def hashed(self):
        if set(self._ids) == set(self):
            return self
        return type(self)(self.original.hashed(), self.indices, self.masked)


@register
class DatasetFiltered(DatasetDecorator):
    snake_name = 'filter'
    def __init__(self, original, filter, expected_length=None, state=None, selection=None):
        super().__init__(original)
        self._filter = filter
        self._lazy_hash_filter = None
        self._create_columns()
        self._row_count = np.sum(self._filter).item()
        self.state = state
        self.selection = selection
        if expected_length is not None:
            if expected_length != self._row_count:
                raise ValueError(f'Expected filter to have {expected_length} true values, but counted {self._row_count}')

    @property
    def _fingerprint(self):
        id = vaex.cache.fingerprint(self.original.id, self._hash_index, self.state, self.selection)
        return f'dataset-{self.snake_name}-{id}'

    @property
    def _hash_index(self):
        if self._lazy_hash_filter is None:
            self._lazy_hash_filter = hash_array(self._filter)
        return self._lazy_hash_filter

    def _create_columns(self):
        columns = {name: vaex.dataset.ColumnProxy(self, name, data_type(col)) for name, col in self.original._columns.items()}
        hashes = {}
        for name, column in self.original.items():
            if name in self.original._ids:
                hashes[name] = hash_combine(self._hash_index, self.original._ids[name])
        self._columns = frozendict(columns)
        self._ids = frozendict(hashes)

    def _encode(self, encoding, skip=set()):
        dataset_spec = encoding.encode('dataset', self.original)
        spec = {'dataset': dataset_spec}
        if self.state is not None and self.selection is not None:
            spec['state'] = encoding.encode('dataframe-state', self.state)
            spec['selection'] = encoding.encode('selection', self.selection)
        spec['filter_array'] = encoding.encode('array', self._filter)
        return spec

    @classmethod
    def _decode(cls, encoding, spec):
        dataset = encoding.decode('dataset', spec['dataset'])
        if 'filter_array' in spec:
            filter = encoding.decode('array', spec['filter_array'])
            ds = cls(dataset, filter)
        else:
            state = encoding.decode('dataframe-state', spec['state'])
            selection = encoding.decode('selection', spec['selection'])
            df = vaex.from_dataset(dataset)
            df.state_set(state)
            df.set_selection(vaex.dataframe.FILTER_SELECTION_NAME, selection)
            df._push_down_filter()
            filter = df.dataset.filter
            ds = cls(dataset, filter, state=state, selection=selection)
        return ds

    def chunk_iterator(self, columns, chunk_size=None, reverse=False):
        chunk_size = chunk_size or 1024**2
        if not columns:
            end = self.row_count
            length = end
            i1 = i2 = 0
            i2 = min(length, i1 + chunk_size)
            while i1 < length:
                yield i1, i2, {}
                i1 = i2
                i2 = min(length, i1 + chunk_size)
            return
        def filtered_chunks():
            for i1, i2, chunks in self.original.chunk_iterator(columns, chunk_size=chunk_size, reverse=reverse):
                chunks_filtered = {name: vaex.array_types.filter(ar, self._filter[i1:i2]) for name, ar in chunks.items()}
                yield chunks_filtered
        yield from _rechunk(filtered_chunks(), chunk_size)

    def hashed(self):
        if set(self._ids) == set(self):
            return self
        return type(self)(self.original.hashed(), self._filter)

    def slice(self, start, end):
        if start == 0 and end == self.row_count:
            return self
        expected_length = end - start
        mask = vaex.superutils.Mask(memoryview(self._filter))
        start, end = mask.indices(start, end-1)
        end += 1
        filter = self._filter[start:end]
        assert filter.sum() == expected_length
        return type(self)(self.original.slice(start, end), filter)

@register
class DatasetSliced(DatasetDecorator):
    snake_name = "slice"
    def __init__(self, original, start, end):
        super().__init__(original)
        self.start = start
        self.end = end
        self._row_count = end - start
        self._create_columns()
        # self._ids = {}
        self._ids = frozendict({name: hash_slice(hash, start, end) for name, hash in original._ids.items()})

    @property
    def _fingerprint(self):
        id = vaex.cache.fingerprint(self.original.fingerprint, self.start, self.end)
        return f'dataset-{self.snake_name}-{id}'

    def leafs(self) -> List[Dataset]:
        # we don't want to propagate slicing
        return [self]

    def _encode(self, encoding, skip=set()):
        dataset_spec = encoding.encode('dataset', self.original)
        return {'dataset': dataset_spec, 'start': self.start, 'end': self.end}

    @classmethod
    def _decode(cls, encoding, spec):
        dataset = encoding.decode('dataset', spec['dataset'])
        return cls(dataset, spec['start'], spec['end'])

    def _create_columns(self):
        self._columns = {name: vaex.dataset.ColumnProxy(self, name, data_type(col)) for name, col in self.original._columns.items()}

    def chunk_iterator(self, columns, chunk_size=None, reverse=False):
        yield from self.original.chunk_iterator(columns, chunk_size=chunk_size, reverse=reverse, start=self.start, end=self.end)

    def hashed(self):
        if set(self._ids) == set(self):
            return self
        return type(self)(self.original.hashed(), self.start, self.end)

    def slice(self, start, end):
        length = end - start
        start += self.start
        end = start + length
        if end > self.original.row_count:
            raise IndexError(f'Slice end ({end}) if larger than number of rows: {self.original.row_count}')
        return type(self)(self.original, start, end)


@register
class DatasetSlicedArrays(DatasetDecorator):
    snake_name = 'slice_arrays'
    def __init__(self, original, start, end):
        super().__init__(original)
        # maybe we want to avoid slicing twice, and collapse it to 1?
        self.start = start
        self.end = end
        # TODO: this is the old dataframe.trim method, we somehow need to test/capture that
        # if isinstance(column, array_types.supported_array_types):  # real array
        #     df.columns[name] = column[self._index_start:self._index_end]
        # else:
        #     df.columns[name] = column.trim(self._index_start, self._index_end)
        self._create_columns()
        self._ids = frozendict({name: hash_slice(hash, start, end) for name, hash in original._ids.items()})
        self._set_row_count()

    @property
    def _fingerprint(self):
        id = vaex.cache.fingerprint(self.original.fingerprint, self.start, self.end)
        return f'dataset-{self.snake_name}-{id}'

    def leafs(self) -> List[Dataset]:
        # we don't want to propagate slicing
        return [self]

    def _create_columns(self):
        columns = {}
        for name, column in self.original.items():
            if isinstance(column, array_types.supported_array_types):  # real array
                column = column[self.start:self.end]
            else:
                column = column.trim(self.start, self.end)
            columns[name] = column
        self._columns = frozendict(columns)

    def _encode(self, encoding, skip=set()):
        dataset_spec = encoding.encode('dataset', self.original)
        return {'dataset': dataset_spec, 'start': self.start, 'end': self.end}

    @classmethod
    def _decode(cls, encoding, spec):
        dataset = encoding.decode('dataset', spec['dataset'])
        return cls(dataset, spec['start'], spec['end'])

    def chunk_iterator(self, columns, chunk_size=None, reverse=False):
        yield from self._default_chunk_iterator(self._columns, columns, chunk_size, reverse=reverse)

    def hashed(self):
        if set(self._ids) == set(self):
            return self
        return type(self)(self.original.hashed(), self.start, self.end)

    def slice(self, start, end):
        if start == 0 and end == self.row_count:
            return self
        length = end - start
        start += self.start
        end = start + length
        if end > self.original.row_count:
            raise IndexError(f'Slice end ({end}) if larger than number of rows: {self.original.row_count}')
        return type(self)(self.original, start, end)


@register
class DatasetDropped(DatasetDecorator):
    snake_name = "drop"
    def __init__(self, original, names):
        super().__init__(original)
        self._dropped_names = tuple(names)
        self._create_columns()
        self._ids = frozendict({name: ar for name, ar in original._ids.items() if name not in names})
        self._set_row_count()

    def dropped(self, *names):
        return DatasetDropped(self.original, self._dropped_names + names)

    @property
    def _fingerprint(self):
        id = vaex.cache.fingerprint(self.original.fingerprint, self._dropped_names)
        return f'dataset-{self.snake_name}-{id}'

    def _create_columns(self):
        self._columns = frozendict({name: ar for name, ar in self.original.items() if name not in self._dropped_names})

    def _encode(self, encoding):
        dataset_spec = encoding.encode('dataset', self.original)
        return {'dataset': dataset_spec, 'names': list(self._dropped_names)}

    @classmethod
    def _decode(cls, encoding, spec):
        dataset = encoding.decode('dataset', spec['dataset'])
        ds = cls(dataset, spec['names'])
        return ds

    def chunk_iterator(self, columns, chunk_size=None, reverse=False):
        for column in columns:
            if column in self._dropped_names:
                raise KeyError(f'Oops, you tried to get column {column} while it is actually dropped')
        yield from self.original.chunk_iterator(columns, chunk_size=chunk_size, reverse=reverse)

    def hashed(self):
        if set(self._ids) == set(self):
            return self
        return type(self)(self.original.hashed(), self._dropped_names)

    def close(self):
        self.original.close()

    def slice(self, start, end):
        if start == 0 and end == self.row_count:
            return self
        return type(self)(self.original.slice(start, end), self._dropped_names)


@register
class DatasetMerged(Dataset):
    snake_name = "merge"
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right
        if self.left.row_count != self.right.row_count:
            raise ValueError(f'Merging datasets with unequal row counts ({self.left.row_count} != {self.right.row_count})')
        self._row_count = self.left.row_count
        overlap = set(left) & set(right)
        if overlap:
            raise NameError(f'Duplicate names: {overlap}')
        self._create_columns()
        self._ids = frozendict({**left._ids, **right._ids})
        self._set_row_count()

    @property
    def _fingerprint(self):
        id = vaex.cache.fingerprint(self.left.fingerprint, self.right.fingerprint)
        return f'dataset-{self.snake_name}-{id}'

    def leafs(self) -> List[Dataset]:
        return self.left.leafs() + self.right.leafs()

    def _create_columns(self):
        # TODO: for DatasetArray, we might want to just do this?
        # self._columns = frozendict({**left._columns, **right._columns})
        self._columns = {**{name: ColumnProxy(self.left, name, data_type(col)) for name, col in self.left._columns.items()},
                         **{name: ColumnProxy(self.right, name, data_type(col)) for name, col in self.right._columns.items()}}

    def _encode(self, encoding, skip=set()):
        dataset_spec_left = encoding.encode('dataset', self.left)
        dataset_spec_right = encoding.encode('dataset', self.right)
        spec = {'left': dataset_spec_left, 'right': dataset_spec_right}
        return spec

    @classmethod
    def _decode(cls, encoding, spec):
        left = encoding.decode('dataset', spec['left'])
        right = encoding.decode('dataset', spec['right'])
        ds = cls(left, right)
        return ds

    def chunk_iterator(self, columns, chunk_size=None, reverse=False):
        columns_left = [k for k in columns if k in self.left]
        columns_right = [k for k in columns if k in self.right]
        if not columns_left:
            yield from self.right.chunk_iterator(columns, chunk_size, reverse=reverse)
        elif not columns_right:
            yield from self.left.chunk_iterator(columns, chunk_size, reverse=reverse)
        else:
            for (i1, i2, ichunks), (j1, j2, jchunks) in zip(
                self.left.chunk_iterator(columns_left, chunk_size, reverse=reverse),
                self.right.chunk_iterator(columns_right, chunk_size, reverse=reverse)):
                # TODO: if one of the datasets does not respect the chunk_size (e.g. parquet)
                # this might fail
                assert i1 == j1
                assert i2 == j2
                yield i1, i2, {**ichunks, **jchunks}

    def is_masked(self, column):
        if column in self.left:
            return self.left.is_masked(column)
        else:
            return self.right.is_masked(column)

    def shape(self, column):
        if column in self.left:
            return self.left.shape(column)
        else:
            return self.right.shape(column)

    def hashed(self):
        if set(self._ids) == set(self):
            return self
        return type(self)(self.left.hashed(), self.right.hashed())

    def close(self):
        self.left.close()
        self.right.close()

    def slice(self, start, end):
        if start == 0 and end == self.row_count:
            return self
        return type(self)(self.left.slice(start, end), self.right.slice(start, end))


@register
class DatasetArrays(Dataset):
    snake_name = "arrays"
    def __init__(self, mapping=None, hashed=True, **kwargs):
        super().__init__()
        if mapping is None:
            mapping = {}
        columns = {**mapping, **kwargs}
        columns = {key: to_supported_array(ar) for key, ar in columns.items()}
        # TODO: we finally want to get rid of datasets with no columns
        self._columns = frozendict(columns)
        if hashed:
            self._ids = frozendict({key: hash_array(array) for key, array in self._columns.items()})
        else:
            self._ids = frozendict()
        self._set_row_count()

    @property
    def id(self):
        try:
            # requires hashing and is expensive
            return self.fingerprint
        except ValueError:
            return f'dataset-{self.snake_name}-uuid4-{self._id}'

    @property
    def _fingerprint(self):
        keys = set(self)
        keys_hashed = set(self._ids)
        missing = keys ^ keys_hashed
        if missing:
            # if we don't have hashes for all columns, we do it like id
            return f'dataset-{self.snake_name}-uuid4-{self._id}'
        # self.__hash__()  # invoke just to check we don't have missing hashes
        # but Python's hash functions are not deterministic (cross processs)
        fp = vaex.cache.fingerprint(tuple(self._ids.items()))
        return f'dataset-{self.snake_name}-hashed-{fp}'

    def leafs(self) -> List[Dataset]:
        return [self]

    def _encode(self, encoding):
        arrays = encoding.encode_dict('array', self._columns)
        spec = {'dataset_type': self.snake_name, 'arrays': arrays}
        if self._ids:
            fingerprints = dict(self._ids)
            spec['fingerprints'] = fingerprints
        return spec

    @classmethod
    def _decode(cls, encoding, spec):
        arrays = encoding.decode_dict('array', spec['arrays'])
        ds = cls(arrays)
        if 'fingerprints' in spec:
            ds._ids = frozendict(spec['fingerprints'])
        return ds

    def __getstate__(self):
        state = self.__dict__.copy()
        # here, we actually DO want to keep the columns
        # del state['_columns']
        return state

    def __setstate__(self, state):
        super().__setstate__(state)

    def _create_columns(self):
        pass

    def chunk_iterator(self, columns, chunk_size=None, reverse=False):
        yield from self._default_chunk_iterator(self._columns, columns, chunk_size, reverse=reverse)

    def is_masked(self, column):
        ar = self._columns[column]
        if not isinstance(ar, np.ndarray):
            ar = ar[0:1]  # take a small piece
        if isinstance(ar, np.ndarray):
            return np.ma.isMaskedArray(ar)
        else:
            return False  # an arrow array always has null value options

    def shape(self, column):
        ar = self._columns[column]
        if not isinstance(ar, np.ndarray):
            ar = ar[0:1]  # take a small piece
        if isinstance(ar, vaex.array_types.supported_arrow_array_types):
            return tuple()
        else:
            return ar.shape[1:]

    def merged(self, rhs):
        # TODO: if we don't allow emtpy datasets, we can remove this method
        if len(self) == 0:
            return rhs
        if len(rhs) == 0:
            return self
        # TODO: this is where we want to check if both are array like
        # and have faster version of merged
        return DatasetMerged(self, rhs)

    def slice(self, start, end):
        if start == 0 and end == self.row_count:
            return self
        return DatasetSlicedArrays(self, start=start, end=end)

    def hashed(self):
        if set(self._ids) == set(self):
            return self
        new = type(self)(self._columns)
        new._ids = frozendict({key: hash_array(array) for key, array in new._columns.items()})
        return new

    def close(self):
        pass  # nothing to do, maybe drop a refcount?

    # TODO: we might want to really get rid of these, since we want to avoid copying them over the network?
    # def dropped(self, names):

class DatasetFile(Dataset):
    """Datasets that map to a file can keep their ids/hashes in the file itself,
    or keep them in a meta file.
    """
    def __init__(self, path, write=False, fs_options={}, fs=None):
        super().__init__()
        self.path = path
        self.fs_options = fs_options
        self.fs = fs
        self.write = write
        self._columns = {}
        self._ids = {}
        self._frozen = False
        self._hash_calculations = 0  # track it for testing purposes
        self._hash_info = {}
        self._hash_cache_needs_write = False
        self._read_hashes()

    @property
    def name(self):
        base, ext, fs_options = vaex.file.split_ext(self.path)
        base = os.path.basename(base)
        return base

    @property
    def _fingerprint(self):
        if set(self._ids) == set(self):
            fingerprint = vaex.cache.fingerprint(dict(self._ids))
            return f'dataset-{self.snake_name}-hashed-{fingerprint}'
        else:
            # TODO: if the dataset is hashed, return a fingerprint based on that
            fingerprint = vaex.file.fingerprint(self.path, fs_options=self.fs_options, fs=self.fs)
            return f'dataset-{self.snake_name}-{fingerprint}'

    def leafs(self) -> List[Dataset]:
        return [self]

    def _create_columns(self):
        pass

    @classmethod
    def quick_test(cls, path, fs_options={}, fs=None, *args, **kwargs):
        return False

    @classmethod
    def open(cls, path, *args, **kwargs):
        return cls(path, *args, **kwargs)

    def chunk_iterator(self, columns, chunk_size=None, reverse=False):
        yield from self._default_chunk_iterator(self._columns, columns, chunk_size, reverse=reverse)

    def is_masked(self, column):
        ar = self._columns[column]
        if not isinstance(ar, np.ndarray):
            ar = ar[0:1]  # take a small piece
        if isinstance(ar, np.ndarray):
            return np.ma.isMaskedArray(ar)
        else:
            return False  # an arrow array always has null value options

    def shape(self, column):
        ar = self._columns[column]
        if not isinstance(ar, np.ndarray):
            ar = ar[0:1]  # take a small piece
        if isinstance(ar, vaex.array_types.supported_arrow_array_types):
            return tuple()
        else:
            return ar.shape[1:]

    def slice(self, start, end):
        if start == 0 and end == self.row_count:
            return self
        return DatasetSlicedArrays(self, start=start, end=end)

    def _read_hashes(self):
        path_hashes = Path(str(self.path )+ '.d') / 'hashes.yaml'
        try:
            exists = path_hashes.exists()
        except OSError:  # happens for windows py<38
            exists = False
        if exists:
            with path_hashes.open() as f:
                hashes = vaex.utils.yaml_load(f)
                if hashes is None:
                    raise ValueError(f'{path_hashes} was probably truncated due to another process writing.')
                self._hash_info = hashes.get('columns', {})

    def _freeze(self):
        self._ids = frozendict(self._ids)
        self._columns = frozendict(self._columns)
        self._set_row_count()
        self._frozen = True
        if self._hash_cache_needs_write:
            self._write_hash_info()

    def _encode(self, encoding):
        spec = {'write': self.write,
                'path': str(self.path),
                'fs_options': self.fs_options,
                'fs': self.fs}
        return spec

    @classmethod
    def _decode(cls, encoding, spec):
        return cls(**spec)

    def __getstate__(self):
        # we don't have the columns in the state, since we should be able
        # to get them from disk again
        state = {
            'write': self.write,
            'path': self.path,
            'fs_options': self.fs_options,
            'fs': self.fs,
            '_ids': dict(self._ids)  # serialize the hases as non-frozen dict
        }
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        # 'ctor' like initialization
        self._frozen = False
        self._hash_calculations = 0
        self._columns = {}
        self._hash_info = {}
        self._hash_cache_needs_write = False
        self._read_hashes()

    def add_column(self, name, data):
        self._columns[name] = data
        if self.write:
            return  # the columns don't include the final data
            # the hashes will be done in .freeze()
        hash_info = self._hash_info.get(name)
        if hash_info:
            hash_info_previous = hash_info.copy()
            hash, hash_info = hash_array(data, hash_info, return_info=True)
            if hash_info_previous != hash_info:
                self._hash_cache_needs_write = True
            self._ids[name] = hash
            self._hash_info[name] = hash_info  # always update the information

    @property
    def _local_hash_path(self):
        # TODO: support s3 and gcs
        # TODO: fallback directory when a user cannot write
        if Path(self.path).exists():
            directory = Path(self.path + '.d')
            directory.mkdir(exist_ok=True)
        else:
            o = urlparse(self.path)
            directory = Path(vaex.utils.get_private_dir('dataset', o.scheme, o.netloc, o.path[1:]))
        return directory / 'hashes.yaml'

    def hashed(self):
        if set(self._ids) == set(self):
            return self
        cls = type(self)
        # use pickle protocol to clone
        new = cls.__new__(cls)
        new.__setstate__(self.__getstate__())
        hashes = {}
        disk_cached_hashes = {}
        for name, column in new.items():
            hash_info = self._hash_info.get(name)
            if hash_info is None:
                logging.warning(f'Calculating hash for column {name} of length {len(column)} (1 time operation, will be cached on disk)')
                hash_info = hash_array_data(column)
            hash, hash_info = hash_array(column, hash_info, return_info=True)
            new._hash_calculations += 1
            hashes[name] = hash
            disk_cached_hashes[name] = hash_info
        new._ids = frozendict(hashes)
        new._hash_info = frozendict(disk_cached_hashes)
        path_hashes = new._local_hash_path
        # TODO: without this check, if multiple processes are writing (e.g. tests/execution_test.py::test_task_sum with ray)
        # this leads to a race condition, where we write the file, and while truncated, _read_hases() fails (because the file exists)
        # if new._hash_info != new._ids:
        new._write_hash_info()
        return new

    def _write_hash_info(self):
        if self._hash_info:  # TODO: file lock
            path_hashes = self._local_hash_path
            with path_hashes.open('w') as f:
                vaex.utils.yaml_dump(f, {'columns': dict(self._hash_info)})


class DatasetCached(DatasetDecorator):
    snake_name = "cached"
    shared_cache = {}

    def __init__(self, original, names, cache=None, to_numpy=False):
        super(DatasetCached, self).__init__(original)
        self.original = original
        self.names = names
        self._shared = cache is None or cache is self.shared_cache
        self.cache = cache if cache is not None else self.shared_cache
        self.to_numpy = to_numpy
        self._create_columns()
        self._row_count = self.original.row_count

    @property
    def _fingerprint(self):
        return self.original.fingerprint

    def _create_columns(self):
        columns = {}
        schema = self.original.schema()
        for name, column in self.original.items():
            columns[name] = ColumnProxy(self, name, schema[name])
        self._columns = frozendict(columns)
        self._ids = frozendict(self.original._ids)

    def _encode(self, encoding, skip=set()):
        raise NotImplementedError("cannot serialize cache")

    @classmethod
    def _decode(cls, encoding, spec):
        raise NotImplementedError("cannot serialize cache")

    def chunk_iterator(self, columns, chunk_size=None, reverse=False):
        chunk_size = chunk_size or chunk_size_default
        columns_all = set(columns)
        columns_cachable = columns_all & set(self.names)
        # avoids asking the cache twice, by using .get() and then testing for None
        columns_cached = {name: self.cache.get(self._cache_key(name)) for name in columns_cachable}
        columns_cached = {name: array for name, array in columns_cached.items() if array is not None}
        columns_to_cache = columns_cachable - set(columns_cached)
        column_required = columns_all - set(columns_cached)
        cache_chunks = {name: [] for name in columns_to_cache}

        def cached_iterator():
            chunks_list = [chunks for name, chunks in columns_cached.items()]
            # chunks_list is of form [[ar1x, ar2x, a3x], [ar1y, ar2y, a3y]]
            # and now we want to yield
            #  * i1, i2 {'x': ar1x, 'y': ar1y}
            #  * i1, i2 {'x': ar2x, 'y': ar2y}
            #  * i1, i2 {'x': ar3x, 'y': ar3y}
            names = [name for name, chunks in columns_cached.items()]
            i1 = 0
            i2 = 0
            for chunks in zip(*chunks_list):
                i2 += len(chunks[0])
                for chunk in chunks:
                    assert len(chunk) == len(chunks[0])
                yield i1, i2, dict(zip(names, chunks))
                i1 = i2

        if columns_cached:
            cached_iter = chunk_rechunk(cached_iterator(), chunk_size)
        else:
            cached_iter = empty_chunk_iterator(0, self.row_count, chunk_size)
        if column_required:
            original_iter = self.original.chunk_iterator(column_required, chunk_size, reverse=reverse)
        else:
            original_iter = empty_chunk_iterator(0, self.row_count, chunk_size)
        original_iter = list(original_iter)
        cached_iter = list(cached_iter)

        for (o1, o2, ochunks), (c1, c2, cchunks) in zip(original_iter, cached_iter):
            assert o1 == c1
            assert o2 == c2
            yield o1, o2, {**ochunks, **cchunks}
            for name in columns_to_cache:
                if self.to_numpy:
                    ochunks = {k: vaex.array_types.to_numpy(v) for k, v in ochunks.items()}
                cache_chunks[name].append(ochunks[name])
        # we write it too the cache in 1 go
        for name in columns_to_cache:
            self.cache[self._cache_key(name)] = cache_chunks[name]

    def slice(self, start, end):
        if start == 0 and end == self.row_count:
            return self
        return type(self)(self.original.slice(start, end), self.names, cache=self.cache)

    def hashed(self):
        if set(self._ids) == set(self):
            return self
        return type(self)(self.original.hashed(), self.names, cache=self.cache)

    def _cache_key(self, name):
        return f"{self.fingerprint}-{name}"
