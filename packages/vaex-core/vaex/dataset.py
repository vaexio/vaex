from pathlib import Path
from .dataframe import *
from .dataframe import (
    # .dataframe imports from .utils
    _ensure_strings_from_expressions,
    _ensure_string_from_expression,
    _ensure_list,
    _is_limit,
    _isnumber,
    _issequence,
    _is_string,
    _parse_reduction,
    _parse_n,
    _normalize_selection_name,
    _normalize,
    _parse_f,
    _expand,
    _expand_shape,
    _expand_limits,
    _split_and_combine_mask,
    # dataframe definitions
    ColumnConcatenatedLazy as _ColumnConcatenatedLazy,
    _doc_snippets,
    _functions_statistics_1d,
    _hidden,
    _is_array_type_ok,
    _is_dtype_ok,
    _requires
)

# alias kept for backward compatibility
Dataset = DataFrame
DatasetLocal = DataFrameLocal
# DatasetArrays = DataFrameArrays
DatasetConcatenated = DataFrameConcatenated

import os
import collections.abc
import numpy as np
import uuid
import logging

logger = logging.getLogger('vaex.dataset')

import blake3
from frozendict import frozendict

from .column import Column, supported_column_types
import pyarrow as pa

def _to_bytes(ar):
    try:
        return ar.view(np.uint8)
    except ValueError:
        return ar.copy().view(np.uint8)


def hash_array(ar):
    if isinstance(ar, np.ndarray):
        if ar.dtype == np.object_:
            return str(uuid.uuid4())
        if np.ma.isMaskedArray(ar):
            data_byte_ar = _to_bytes(ar.data)
            blake = blake3.blake3(data_byte_ar, multithreading=True)
            if ar.mask is not True and ar.mask is not False and ar.mask is not np.True_ and ar.mask is not np.False_:
                mask_byte_ar = _to_bytes(ar.mask)
                blake.update(mask_byte_ar)
            return blake.hexdigest()
        else:
            try:
                byte_ar = _to_bytes(ar)
            except ValueError:
                byte_ar = ar.copy().view(np.uint8)
            return blake3.blake3(byte_ar, multithreading=True).hexdigest()
    else:
        # TODO: dtype is not included in hash
        try:
            ar = pa.array(ar)
        except:  # dtype=o for lazy columns doesn't work..
            return str(uuid.uuid4())
        blake = blake3.blake3(multithreading=True)
        for buffer in ar.buffers():
            if buffer is not None:
                # TODO: we need to make a copy here, a memoryview would be better
                # or possible patch the blake module to accept a memoryview https://github.com/oconnor663/blake3-py/issues/9
                # or feed in the buffer in batches
                # blake.update(buffer)
                blake.update(memoryview((buffer)).tobytes())
        return blake.hexdigest()


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

        if len(types) == 1 and issubclass(types[0], six.string_types):
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

    def renamed(self, renaming):
        return DatasetRenamed(self, renaming)

    def merged(self, rhs):
        return DatasetMerged(self, rhs)

    def dropped(self, *names):
        return DatasetDropped(self, names)

    def __getitem__(self, name):
        return self._columns[name]
    
    def __len__(self):
        return len(self._columns)

    def __iter__(self):
        return iter(self._columns)

    def get_data(self, i1, i2, names):
        raise NotImplementedError

    def __eq__(self, rhs):
        if not isinstance(rhs, Dataset):
            return NotImplemented
        return self._ids == rhs._ids

    def __hash__(self):
        return hash(self._ids)

class DatasetRenamed(Dataset):
    def __init__(self, original, renaming):
        self.original = original
        self._columns = frozendict({renaming.get(name, name): ar for name, ar in original.items()})
        self._ids = frozendict({renaming.get(name, name): ar for name, ar in original._ids.items()})


class DatasetDropped(Dataset):
    def __init__(self, original, names):
        self.original = original
        self._columns = frozendict({name: ar for name, ar in original.items() if name not in names})
        self._ids = frozendict({name: ar for name, ar in original._ids.items() if name not in names})


class DatasetMerged(Dataset):
    def __init__(self, left, right):
        overlap = set(left) & set(right)
        if overlap:
            raise NameError(f'Duplicate names: {overlap}')
        self._columns = frozendict({**left._columns, **right._columns})
        self._ids = frozendict({**left._ids, **right._ids})


class DatasetArrays(Dataset):
    def __init__(self, mapping=None, **kwargs):
        if mapping is None:
            mapping = {}
        columns = {**mapping, **kwargs}
        columns = {key: to_supported_array(ar) for key, ar in columns.items()}
        self._columns = frozendict(columns)
        self._ids = frozendict({key: hash_array(array) for key, array in self._columns.items()})

    # TODO: we might want to really get rid of these, since we want to avoid copying them over the network?
    # def dropped(self, names):

class DatasetFile(Dataset):
    """Datasets that map to a file can keep their ids/hashes in the file itself,
    or keep them in a meta file.
    """
    def __init__(self, path, write=False):
        self.path = path
        self.write = write
        self._columns = {}
        self._ids = {}
        self._frozen = False
        self._hash_calculations = 0  # track it for testing purposes
        self._disk_cached_hashes = {}
        self._read_hashes()

    def _read_hashes(self):
        path_hashes = Path(self.path + '.d') / 'hashes.yaml'
        if path_hashes.exists():
            with path_hashes.open() as f:
                hashes = vaex.utils.yaml_load(f)
                if hashes is None:
                    raise ValueError(f'{path_hashes} was probably truncated due to another process writing.')
                self._disk_cached_hashes = hashes.get('columns', {})

    def _freeze(self):
        if self.write:
            for name, column in self.items():
                logging.error(f'Calculating hash for column {name} of length {len(column)} (1 time operation, will be cached on disk)')
                hash = hash_array(column)
                self._hash_calculations += 1
                self._ids[name] = hash
        self._ids = frozendict(self._ids)
        self._columns = frozendict(self._columns)
        directory = Path(self.path + '.d')
        directory.mkdir(exist_ok=True)
        path_hashes = directory / 'hashes.yaml'
        # TODO: without this check, if multiple processes are writing (e.g. tests/execution_test.py::test_task_sum with ray)
        # this leads to a race condition, where we write the file, and while truncated, _read_hases() fails (because the file exists)
        if self._disk_cached_hashes != self._ids:
            with path_hashes.open('w') as f:
                vaex.utils.yaml_dump(f, {'columns': dict(self._ids)})
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
        self._disk_cached_hashes = {}
        self._read_hashes()

    def add_column(self, name, data):
        self._columns[name] = data
        if self.write:
            return  # the columns don't include the final data
            # the hashes will be done in .freeze()
        hash = self._disk_cached_hashes.get(name)
        if hash is None:
            hash = hash_array(data)
            logging.error(f'Calculating hash for column {name} of length {len(data)} (1 time operation, will be cached on disk)')
            self._hash_calculations += 1
        self._ids[name] = hash

        # self._columns, self._ids, self.attrs = read_hdf5(path)


    def _get_private_dir(self, create=False):
        """Each DataFrame has a directory where files are stored for metadata etc.

        Example

        >>> ds._get_private_dir()
        '/Users/users/breddels/.vaex/dataset/_Users_users_breddels_vaex-testing_data_helmi-dezeeuw-2000-10p.hdf5'

        :param bool create: is True, it will create the directory if it does not exist
        """
        name = os.path.abspath(self.path).replace(os.path.sep, "_")[:250]  # should not be too long for most os'es
        name = name.replace(":", "_")  # for windows drive names
        dir = os.path.join(vaex.utils.get_private_dir(), "dataset", name)
        os.makedirs(dir, exist_ok=create)
        return dir
