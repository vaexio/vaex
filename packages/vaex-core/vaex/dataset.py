from abc import  abstractmethod
import os
from pathlib import Path
import collections.abc
import logging
import uuid
from urllib.parse import urlparse

import numpy as np
import blake3
from frozendict import frozendict
import pyarrow as pa

import vaex
from .column import Column, ColumnIndexed, ColumnConcatenatedLazy, supported_column_types
from . import array_types

logger = logging.getLogger('vaex.dataset')


HASH_VERSION = "1"


def _to_bytes(ar):
    try:
        return ar.view(np.uint8)
    except ValueError:
        return ar.copy().view(np.uint8)

def hash_combine(*hashes):
    blake = blake3.blake3(multithreading=False)
    for hash in hashes:
        blake.update(hash.encode())
    return blake.hexdigest()


def hash_slice(hash, start, end):
    blake = blake3.blake3(hash.encode(), multithreading=False)
    slice = np.array([start, end], dtype=np.int64)
    blake.update(_to_bytes(slice))
    return blake.hexdigest()


def hash_array_data(ar):
    # this function should stay consistent with all future versions
    # since this is the expensive part of the hashing
    if isinstance(ar, np.ndarray):
        ar = ar.ravel()
        if ar.dtype == np.object_:
            return {"type": "numpy", "data": str(uuid.uuid4()), "mask": None}
        if np.ma.isMaskedArray(ar):
            data_byte_ar = _to_bytes(ar.data)
            blake = blake3.blake3(data_byte_ar, multithreading=True)
            hash_data = {"type": "numpy", "data": blake.hexdigest(), "mask": None}
            if ar.mask is not True and ar.mask is not False and ar.mask is not np.True_ and ar.mask is not np.False_:
                mask_byte_ar = _to_bytes(ar.mask)
                blake = blake3.blake3(mask_byte_ar, multithreading=True)
                hash_data["mask"] = blake.hexdigest()
            return hash_data
        else:
            try:
                byte_ar = _to_bytes(ar)
            except ValueError:
                byte_ar = ar.copy().view(np.uint8)
            blake = blake3.blake3(byte_ar, multithreading=True)
            hash_data = {"type": "numpy", "data": blake.hexdigest(), "mask": None}
    else:
        try:
            ar = pa.array(ar)
        except:  # dtype=o for lazy columns doesn't work..
            return str(uuid.uuid4())
        blake = blake3.blake3(multithreading=True)
        buffer_hashes = []
        hash_data = {"type": "arrow", "buffers": buffer_hashes}
        for buffer in ar.buffers():
            if buffer is not None:
                # TODO: we need to make a copy here, a memoryview would be better
                # or possible patch the blake module to accept a memoryview https://github.com/oconnor663/blake3-py/issues/9
                # or feed in the buffer in batches
                # blake.update(buffer)
                blake.update(memoryview((buffer)).tobytes())
                buffer_hashes.append(blake.hexdigest())
            else:
                buffer_hashes.append(None)
    return hash_data


def hash_array(ar, hash_info=None, return_info=False):
    # this function can change over time, as it builds on top of the expensive part
    # (hash_array_data), so we can cheaply calculate new hashes if we pass on hash_info
    if hash_info is None:
        hash_info = hash_array_data(ar)
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
    blake = blake3.blake3(multithreading=False)  # small amounts of data
    for key in keys:
        blake.update(key.encode('ascii'))
    hash = blake.hexdigest()
    if return_info:
        return hash, hash_info
    else:
        return hash


def to_supported_array(ar):
    if not isinstance(ar, supported_column_types):
        ar = np.asanyarray(ar)
    if isinstance(ar, np.ndarray) and ar.dtype.kind == 'O':
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


class Dataset(collections.abc.Mapping):
    def __init__(self):
        super().__init__()
        self._columns = frozendict()
        self._row_count = None

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
        print(drop)
        return self.dropped(*list(drop))

    def concat(self, *others):
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
        return DatasetConcatenated(datasets)

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
            return DatasetSliced(self, item.start or 0, item.stop or self.row_count)
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
        keys = set(self)
        keys_hashed = set(self._ids)
        missing = keys ^ keys_hashed
        if missing:
            raise ValueError(f'Comparing datasets where the left hand side is missing hashes for columns: {missing} (tip: use dataset.hashed())')
        keys = set(rhs)
        keys_hashed = set(rhs._ids)
        missing = keys ^ keys_hashed
        if missing:
            raise ValueError(f'Comparing datasets where the right hand side is missing hashes for columns: {missing} (tip: use dataset.hashed())')
        return self._ids == rhs._ids

    def __hash__(self):
        keys = set(self)
        keys_hashed = set(self._ids)
        missing = keys ^ keys_hashed
        if missing:
            raise ValueError(f'Trying to hash a dataset with unhashed columns: {missing} (tip: use dataset.hashed())')
        return hash(self._ids)

    @abstractmethod
    def close(self):
        '''Close file handles or other resources, the DataFrame will not be in a usable state afterwards.'''
        pass

class DatasetRenamed(Dataset):
    def __init__(self, original, renaming):
        super().__init__()
        self.original = original
        self._columns = frozendict({renaming.get(name, name): ar for name, ar in original.items()})
        self._ids = frozendict({renaming.get(name, name): ar for name, ar in original._ids.items()})
        self._set_row_count()

    def close(self):
        self.original.close()


class DatasetConcatenated(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        for dataset in datasets[1:]:
            if set(dataset) != set(datasets[0]):
                l = set(dataset)
                r = set(datasets[0])
                diff = l ^ r
                raise NameError(f'Concatenating datasets with different names: {l} and {r} (difference: {diff})')
        # we need to work with a dataframe :( because the column expects that
        # maybe we should split the column into a lazy evaluate (virtual column -> column)
        # and a concatenated one (that only works with columns)
        dfs = [vaex.dataframe.DataFrameLocal(ds) for ds in datasets]
        columns = {}
        hashes = {}
        for name in datasets[0]:
            columns[name] = ColumnConcatenatedLazy([df[name] for df in dfs])
            if all(name in ds._ids for ds in datasets):
                hashes[name] = hash_combine(*[ds._ids[name] for ds in datasets])
        self._columns = frozendict(columns)
        self._ids = frozendict(hashes)
        self._set_row_count()

    def close(self):
        for ds in self.datasets:
            ds.close()


class DatasetTake(Dataset):
    def __init__(self, original, indices, masked):
        super().__init__()
        self.original = original
        self.indices = indices
        self.masked = masked
        columns = dict(original)
        # if the columns in ds already have a ColumnIndex
        # we could do, direct_indices = df.column['bla'].indices[indices]
        # which should be shared among multiple ColumnIndex'es, so we store
        # them in this dict
        direct_indices_map = {}
        columns = {}
        hashes = {}
        hash_index = hash_array(indices)
        for name, column in original.items():
            columns[name] = ColumnIndexed.index(column, indices, direct_indices_map, masked=masked)
            if name in original._ids:
                hashes[name] = hash_combine(hash_index, original._ids[name])
        self._columns = frozendict(columns)
        self._ids = frozendict(hashes)
        self._set_row_count()

    def hashed(self):
        if set(self._ids) == set(self):
            return self
        return type(self)(self.original.hashed(), self.indices, self.masked)

    def close(self):
        self.original.close()


class DatasetSliced(Dataset):
    def __init__(self, original, start, end):
        super().__init__()
        # maybe we want to avoid slicing twice, and collapse it to 1?
        self.original = original
        self.start = start
        self.end = end
        # TODO: this is the old dataframe.trim method, we somehow need to test/capture that
        # if isinstance(column, array_types.supported_array_types):  # real array
        #     df.columns[name] = column[self._index_start:self._index_end]
        # else:
        #     df.columns[name] = column.trim(self._index_start, self._index_end)
        columns = {}
        for name, column in original.items():
            if isinstance(column, array_types.supported_array_types):  # real array
                column = column[start:end]
            else:
                column = column.trim(start, end)
            columns[name] = column

        self._columns = frozendict(columns)
        self._ids = frozendict({name: hash_slice(hash, start, end) for name, hash in original._ids.items()})
        self._set_row_count()

    def hashed(self):
        if set(self._ids) == set(self):
            return self
        return type(self)(self.original.hashed(), self.start, self.end)

    def close(self):
        self.original.close()


class DatasetDropped(Dataset):
    def __init__(self, original, names):
        super().__init__()
        self.original = original
        self.names = tuple(names)
        self._columns = frozendict({name: ar for name, ar in original.items() if name not in names})
        self._ids = frozendict({name: ar for name, ar in original._ids.items() if name not in names})
        self._set_row_count()

    def hashed(self):
        if set(self._ids) == set(self):
            return self
        return type(self)(self.original.hashed(), self.names)

    def close(self):
        self.original.close()


class DatasetMerged(Dataset):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right
        overlap = set(left) & set(right)
        if overlap:
            raise NameError(f'Duplicate names: {overlap}')
        self._columns = frozendict({**left._columns, **right._columns})
        self._ids = frozendict({**left._ids, **right._ids})
        self._set_row_count()

    def hashed(self):
        if set(self._ids) == set(self):
            return self
        return type(self)(self.left.hashed(), self.right.hashed())

    def close(self):
        self.left.close()
        self.right.close()


class DatasetArrays(Dataset):
    def __init__(self, mapping=None, **kwargs):
        super().__init__()
        if mapping is None:
            mapping = {}
        columns = {**mapping, **kwargs}
        columns = {key: to_supported_array(ar) for key, ar in columns.items()}
        self._columns = frozendict(columns)
        self._ids = frozendict()
        self._set_row_count()

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
    def __init__(self, path, write=False):
        super().__init__()
        self.path = path
        self.write = write
        self._columns = {}
        self._ids = {}
        self._frozen = False
        self._hash_calculations = 0  # track it for testing purposes
        self._hash_info = {}
        self._read_hashes()

    def _read_hashes(self):
        path_hashes = Path(self.path + '.d') / 'hashes.yaml'
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

    def __getstate__(self):
        # we don't have the columns in the state, since we should be able
        # to get them from disk again
        return {
            'write': self.write,
            'path': self.path,
            '_ids': dict(self._ids)  # serialize the hases as non-frozen dict
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        # 'ctor' like initialization
        self._frozen = False
        self._hash_calculations = 0
        self._columns = {}
        self._hash_info = {}
        self._read_hashes()

    def add_column(self, name, data):
        self._columns[name] = data
        if self.write:
            return  # the columns don't include the final data
            # the hashes will be done in .freeze()
        hash_info = self._hash_info.get(name)
        if hash_info:
            hash, hash_info = hash_array(data, hash_info, return_info=True)
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
        if 1:  # TODO: file lock
            with path_hashes.open('w') as f:
                vaex.utils.yaml_dump(f, {'columns': dict(new._hash_info)})
        return new
