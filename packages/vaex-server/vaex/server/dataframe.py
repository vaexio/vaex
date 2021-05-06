from __future__ import absolute_import
__author__ = 'breddels'
import numpy as np
import logging
from vaex.dataframe import DataFrame
import vaex
from frozendict import frozendict


logger = logging.getLogger("vaex.server.dataframe")
allowed_method_names = []


# remote method invokation
def _rmi(f=None):
    def decorator(method):
        method_name = method.__name__
        allowed_method_names.append(method_name)

        def wrapper(df, *args, **kwargs):
            return df.executor._rmi(df, method_name, args, kwargs)
        return wrapper
    if f is None:
        return decorator
    else:
        return decorator(f)

class ColumnProxyRemote(vaex.column.Column):
    '''To give the Dataset._columns object useful containers for debugging'''
    ds: vaex.dataset.Dataset

    def __init__(self, ds, name, type):
        self.ds = ds
        self.name = name
        self.dtype = type

    def _fingerprint(self):
        fp = vaex.cache.fingerprint(self.ds.fingerprint, self.name)
        return f'column-remote-{fp}'

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
            return vaex.array_types.concat(array_chunks)
        else:
            raise NotImplementedError

@vaex.dataset.register
class DatasetRemote(vaex.dataset.Dataset):
    snake_name = "remote"
    def __init__(self, name, column_names, dtypes, row_count):
        super().__init__()
        self._name = name
        self.column_names = column_names
        self.dtypes = dtypes
        self._row_count = row_count
        self._ids = frozendict({name: vaex.cache.fingerprint(self.name, name, dtype, row_count) for name, dtype in zip(column_names, dtypes)})
        self._columns = frozendict({name: ColumnProxyRemote(self, name, dtype) for name, dtype in zip(column_names, dtypes)})

    @property
    def id(self):
        id = self._name
        return f'dataset-{self.snake_name}-{id}'

    @property
    def _fingerprint(self):
        return self.id  # TODO: this should be an id from the server

    def _create_columns(self):
        pass

    def chunk_iterator(self):
        raise RuntimeError('Remote')

    def close(self):
        pass

    def hashed(self):
        return self

    def is_masked(self):
        pass

    def leafs(self):
        return [self]

    def shape(self):
        pass

    def slice(self):
        pass


# TODO: we should not inherit from local
class DataFrameRemote(DataFrame):
    def __init__(self, name, column_names, dtypes, length_original):
        super(DataFrameRemote, self).__init__(name)
        self.dataset = DatasetRemote(name, column_names, dtypes, length_original)
        self.column_names = column_names
        self._dtypes = dtypes
        for column_name in self.get_column_names(virtual=True, strings=True):
            self._save_assign_expression(column_name)
        self._length_original = length_original
        self._length_unfiltered = length_original
        self._index_end = length_original
        self._dtype_cache = {}
        self.fraction = 1

    def hashed(self, inplace=False) -> DataFrame:
        # we're always hashed
        return self if inplace else self.copy()

    def __getstate__(self):
        return {
            'length_original': self._length_original,
            'column_names': self.column_names,
            'dtypes': self._dtypes,
            'url': self.executor.client.url,
            'state': self.state_get(),
        }

    def __setstate__(self, state):
        self._init()
        self.column_names = state['column_names']
        self._dtypes = state['dtypes']
        self._length_original = state['length_original']
        self._length_unfiltered = self._length_original
        self._index_start = 0
        self._index_end = self._length_original
        self.fraction = 1
        client = vaex.server.connect(state['url'])
        self.executor = vaex.server.executor.Executor(client)
        self.state_set(state['state'], use_active_range=True, trusted=True)

    def is_local(self):
        return False

    def copy(self, column_names=None, virtual=True):
        dtypes = {name: self.data_type(name) for name in self.get_column_names(strings=True, virtual=False)}
        df = DataFrameRemote(self.name, self.column_names, dtypes=dtypes, length_original=self._length_original)
        df.executor = self.executor
        state = self.state_get()
        if not virtual:
            state['virtual_columns'] = {}
        df.state_set(state, use_active_range=True)
        return df

    def trim(self, inplace=False):
        df = self if inplace else self.copy()
        # can we get away with not trimming?
        return df

    @_rmi
    def _evaluate_implementation(self, *args, **kwargs):
        pass

    @_rmi
    def _repr_mimebundle_(self, *args, **kwargs):
        pass

    @_rmi
    def _head_and_tail_table(self, *args, **kwargs):
        pass

    def _shape_of(self, expression, filtered=True):
        # sample = self.evaluate(expression, 0, 1, filtered=False, internal=True, parallel=False)
        # TODO: support this properly
        rows = len(self) if filtered else self.length_unfiltered()
        return (rows,)
        # return (rows,) + sample.shape[1:]

    def data_type(self, expression, internal=False, array_type=None, axis=0):
        if str(expression) in self._dtypes:
            return self._dtypes[str(expression)]
        else:
            if str(expression) not in self._dtype_cache:
                self._dtype_cache[str(expression)] = super().data_type(expression, array_type=array_type, axis=axis)
            # TODO: invalidate cache
            return self._dtype_cache[str(expression)]

    # TODO: would be nice to get some info on the remote dataframe
    # def __repr__(self):
    #     name = self.__class__.__module__ + "." + self.__class__.__name__
    #     return "<%s(server=%r, name=%r, column_names=%r, __len__=%r)> instance at 0x%x" % (name, self.server, self.name, self.column_names, len(self), id(self))
