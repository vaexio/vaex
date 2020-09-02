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
from frozendict import frozendict
import blake3
import numpy as np
import uuid

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
            mask_byte_ar = _to_bytes(ar.mask)
            blake = blake3.blake3(data_byte_ar, multithreading=True)
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
    def __init__(self, path):
        self.path = path
        self._columns = {}
        self._ids = {}

    def __getstate__(self):
        # we don't have the columns in the state, since we should be able
        # to get them from disk again
        return {
            '_ids': self._ids
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._columns = {}

    def add_column(self, name, data):
        self._columns[name] = data
        # we wanna cache this
        # if unpickled, the ids are already there
        if name not in self._ids:
            self._ids[name] = hash_array(data)

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
