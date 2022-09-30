__author__ = 'maartenbreddels'
from collections import defaultdict
import logging
from typing import List
from frozendict import frozendict

import numpy as np
import pyarrow as pa
import pyarrow.dataset
import pyarrow.parquet as pq

import vaex.dataset
import vaex.file
from vaex.dataset import DatasetSlicedArrays
from ..itertools import buffer
from vaex.multithreading import get_main_io_pool


logger = logging.getLogger("vaex.multithreading")


class DatasetArrowBase(vaex.dataset.Dataset):
    def __init__(self, max_rows_read=1024**2*10):
        super().__init__()
        self.max_rows_read = max_rows_read
        self._create_columns()

    def _create_columns(self):
        self._create_dataset()
        # we have to get the metadata again, this does not pickle
        row_count = 0
        for fragment in self._arrow_ds.get_fragments():
            if hasattr(fragment, "ensure_complete_metadata"):
                fragment.ensure_complete_metadata()
            if hasattr(fragment, "count_rows"):
                row_count += fragment.count_rows()
            else:
                for rg in fragment.row_groups:
                    row_count += rg.num_rows
        self._row_count = row_count
        self._columns = {name: vaex.dataset.ColumnProxy(self, name, type) for name, type in
                          zip(self._arrow_ds.schema.names, self._arrow_ds.schema.types)}
        for name, dictionary in self._partitions.items():
            # TODO: make int32 dependant on the data?
            self._columns[name] = vaex.dataset.ColumnProxy(self, name, pa.dictionary(pa.int32(), dictionary.type))

    def leafs(self) -> List[vaex.dataset.Dataset]:
        return [self]

    def slice(self, start, end):
        # TODO: we can be smarter here, and trim off some fragments
        if start == 0 and end == self.row_count:
            return self
        return vaex.dataset.DatasetSliced(self, start=start, end=end)

    def is_masked(self, column):
        return False

    def shape(self, column):
        return tuple()

    def __getitem__(self, item):
        if isinstance(item, slice):
            assert item.step in [1, None]
            return vaex.dataset.DatasetSliced(self, item.start or 0, item.stop or self.row_count)
        return self._columns[item]

    def close(self):
        # no need to close it, it seem
        pass

    def _chunk_producer(self, columns, chunk_size=None, reverse=False, start=0, end=None):
        import pyarrow.parquet
        pool = get_main_io_pool()
        offset = 0
        columns = tuple(columns)
        columns_physical = []
        columns_partition = []
        for column in columns:
            if column in self._partitions:
                columns_partition.append(column)
            else:
                columns_physical.append(column)
        columns_physical = tuple(columns_physical)
        columns_partition = tuple(columns_partition)

        for fragment_large in self._arrow_ds.get_fragments():
            if hasattr(fragment_large, "count_rows"):
                fragment_large_rows = fragment_large.count_rows()
                fragments = [fragment_large]
            else:
                fragment_large_rows = sum([rg.num_rows for rg in fragment_large.row_groups])
                fragments = [fragment_large]
                # when do we want to split up? File size? max chunk size?
                if fragment_large_rows > self.max_rows_read:
                    fragments = fragment_large.split_by_row_group()
            for fragment in fragments:
                if hasattr(fragment_large, "count_rows"):
                    rows = fragment_large.count_rows()
                else:
                    rows = sum([rg.num_rows for rg in fragment.row_groups])
                chunk_start = offset
                chunk_end = offset + rows

                length = chunk_end - chunk_start  # default length

                if start >= chunk_end:  # we didn't find the beginning yet
                    offset += length
                    continue
                if end <= chunk_start:  # we are past the end
                    # assert False
                    break
                def reader(fragment=fragment):
                    table = fragment.to_table(columns=list(columns_physical), use_threads=False)
                    chunks_physical = dict(zip(table.column_names, table.columns))
                    chunks_partition = {}
                    partition_keys = self._partition_keys[fragment.path]
                    for name in columns_partition:
                        partition_index = partition_keys[name]
                        partition_index_ar = pa.array(np.full(len(table), partition_index, dtype=np.int32))
                        chunks_partition[name] = pa.DictionaryArray.from_arrays(partition_index_ar, self._partitions[name])
                    chunks = {name: chunks_physical.get(name, chunks_partition.get(name)) for name in columns}
                    return chunks
                assert length > 0
                if start > chunk_start:
                    # this means we have to cut off a piece of the beginning
                    if end < chunk_end:
                        # AND the end
                        length = end - chunk_start  # without the start cut off
                        length -= start - chunk_start  # correcting for the start cut off
                        assert length > 0
                        def slicer(chunk_start=chunk_start, reader=reader, length=length):
                            chunks = reader()
                            chunks = {name: ar.slice(start - chunk_start, length) for name, ar in chunks.items()}
                            for name, ar in chunks.items():
                                assert len(ar) == length, f'Oops, array was expected to be of length {length} but was {len(ar)}'
                            return chunks
                        reader = slicer
                    else:
                        length -= start - chunk_start  # correcting for the start cut off
                        assert length > 0
                        def slicer(chunk_start=chunk_start, reader=reader, length=length):
                            chunks = reader()
                            chunks = {name: ar.slice(start - chunk_start) for name, ar in chunks.items()}
                            for name, ar in chunks.items():
                                assert len(ar) == length, f'Oops, array was expected to be of length {length} but was {len(ar)}'
                            return chunks
                        reader = slicer
                else:
                    if end < chunk_end:
                        # we only need to cut off a piece of the end
                        length = end - chunk_start
                        assert length > 0
                        def slicer(chunk_start=chunk_start, reader=reader, length=length):
                            chunks = reader()
                            chunks = {name: ar.slice(0, length) for name, ar in chunks.items()}
                            for name, ar in chunks.items():
                                assert len(ar) == length, f'Oops, array was expected to be of length {length} but was {len(ar)}'
                            return chunks
                        reader = slicer
                offset += rows
                yield pool.submit(reader)

    def chunk_iterator(self, columns, chunk_size=None, reverse=False, start=0, end=None):
        chunk_size = chunk_size or 1024*1024
        i1 = 0
        chunks_ready_list = []
        i1 = i2 = 0
        # TODO: merge this with DatsetConcatenated.chunk_iterator
        if not columns:
            end = self.row_count if end is None else end
            length = end - start
            i2 = min(length, i1 + chunk_size)
            while i1 < length:
                yield i1, i2, {}
                i1 = i2
                i2 = min(length, i1 + chunk_size)
            return

        workers = get_main_io_pool()._max_workers
        for chunks_future in buffer(self._chunk_producer(columns, chunk_size, start=start, end=end or self._row_count), workers+3):
            chunks = chunks_future.result()
            chunks_ready_list.append(chunks)
            total_row_count = sum([len(list(k.values())[0]) for k in chunks_ready_list])
            if total_row_count > chunk_size:
                chunks_current_list, current_row_count = vaex.dataset._slice_of_chunks(chunks_ready_list, chunk_size)
                i2 += current_row_count
                yield i1, i2, vaex.dataset._concat_chunk_list(chunks_current_list)
                i1 = i2

        while chunks_ready_list:
            chunks_current_list, current_row_count = vaex.dataset._slice_of_chunks(chunks_ready_list, chunk_size)
            i2 += current_row_count
            yield i1, i2, vaex.dataset._concat_chunk_list(chunks_current_list)
            i1 = i2


@vaex.dataset.register
class DatasetParquet(DatasetArrowBase):
    snake_name = "arrow-parquet"
    def __init__(self, path, fs_options, fs=None, max_rows_read=1024**2*10, partitioning=None, kwargs=None):
        self.path = path
        self.fs_options = fs_options
        self.fs = fs
        self.partitioning = partitioning
        self.kwargs = kwargs or {}
        super().__init__(max_rows_read=max_rows_read)

    @property
    def _fingerprint(self):
        if isinstance(self.path, (list, tuple)):
            paths = self.path
        else:
            paths = [self.path]
        fingerprints = [vaex.file.fingerprint(path, fs_options=self.fs_options, fs=self.fs) for path in paths]
        fp = vaex.cache.fingerprint(*fingerprints)
        return f'dataset-{self.snake_name}-{fp}'

    def hashed(self):
        return self

    def _encode(self, encoding):
        if self.fs:
            raise ValueError('Serializing filesystem not supported yet')
        spec = {'path': self.path, 'fs_options': self.fs_options, 'partitioning': self.partitioning,
                'max_rows_read': self.max_rows_read}
        if self.kwargs:
            spec['extra_options'] = self.kwargs
        return spec

    @classmethod
    def _decode(cls, encoding, spec):
        kwargs = spec.pop('extra_options', None)
        return cls(**spec, kwargs=kwargs)

    def _create_columns(self):
        super()._create_columns()
        self._ids = frozendict({name: vaex.cache.fingerprint(self._fingerprint, name) for name in self._columns})

    def _create_dataset(self):
        file_system, source = vaex.file.parse(self.path, self.fs_options, fs=self.fs, for_arrow=True)
        self._arrow_ds = pyarrow.dataset.dataset(source, filesystem=file_system, partitioning=self.partitioning)

        self._partitions = defaultdict(list) # path -> list (which will be an arrow array later on)
        self._partition_keys = defaultdict(dict)  # path -> key -> int/index

        for fragment in self._arrow_ds.get_fragments():
            keys = pa.dataset._get_partition_keys(fragment.partition_expression)
            for name, value in keys.items():
                if value not in self._partitions[name]:
                    self._partitions[name].append(value)
                self._partition_keys[fragment.path][name] = self._partitions[name].index(value)
        self._partitions = {name: pa.array(values) for name, values in self._partitions.items()}

    def __getstate__(self):
        state = super().__getstate__()
        del state['_arrow_ds']
        del state['_partitions']
        del state['_partition_keys']
        return state



class DatasetArrow(DatasetArrowBase):
    snake_name = "arrow-dataset"
    def __init__(self, ds, max_rows_read=1024**2*10):
        self._arrow_ds = ds
        super().__init__(max_rows_read=max_rows_read)

    @property
    def _fingerprint(self):
        return self._id

    def hashed(self):
        raise NotImplementedError

    def _create_columns(self):
        super()._create_columns()
        # self._ids = frozendict({name: vaex.cache.fingerprint(self._fingerprint, name) for name in self._columns})
        self._ids = frozendict()

    def _create_dataset(self):
        self._partitions = defaultdict(list) # path -> list (which will be an arrow array later on)
        self._partition_keys = defaultdict(dict)  # path -> key -> int/index

        for fragment in self._arrow_ds.get_fragments():
            keys = pa.dataset._get_partition_keys(fragment.partition_expression)
            for name, value in keys.items():
                if value not in self._partitions[name]:
                    self._partitions[name].append(value)
                self._partition_keys[fragment.path][name] = self._partitions[name].index(value)
        self._partitions = {name: pa.array(values) for name, values in self._partitions.items()}



class DatasetArrowFileBase(vaex.dataset.Dataset):
    def __init__(self, path, fs_options, fs=None):
        super().__init__()
        self.fs_options = fs_options
        self.fs = fs
        self.path = path
        self._create_columns()
        self._set_row_count()
        self._ids = frozendict({name: vaex.cache.fingerprint(self._fingerprint, name) for name in self._columns})

    @property
    def _fingerprint(self):
        fingerprint = vaex.file.fingerprint(self.path, fs_options=self.fs_options, fs=self.fs)
        return f'dataset-{self.snake_name}-{fingerprint}'

    def leafs(self) -> List[vaex.dataset.Dataset]:
        return [self]

    def _encode(self, encoding):
        if self.fs:
            raise ValueError('Serializing filesystem not supported yet')
        spec = {'path': self.path, 'fs_options': self.fs_options}
        return spec

    @classmethod
    def _decode(cls, encoding, spec):
        return cls(**spec)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_source']
        del state['_columns']
        return state

    def chunk_iterator(self, columns, chunk_size=None, reverse=False):
        yield from self._default_chunk_iterator(self._columns, columns, chunk_size, reverse=reverse)

    def close(self):
        self._source.close()

    def hashed(self):
        return self

    def shape(self, column):
        return tuple()

    def is_masked(self, column):
        return False

    def slice(self, start, end):
        if start == 0 and end == self.row_count:
            return self
        return DatasetSlicedArrays(self, start=start, end=end)

@vaex.dataset.register
class DatasetArrowIPCFile(DatasetArrowFileBase):
    snake_name = "arrow-ipc-file"
    def _create_columns(self):
        self._source = vaex.file.open(path=self.path, mode='rb', fs_options=self.fs_options, fs=self.fs, mmap=True, for_arrow=True)
        reader = pa.ipc.open_file(self._source)
        batches = [reader.get_batch(i) for i in range(reader.num_record_batches)]
        table = pa.Table.from_batches(batches, schema=reader.schema)
        self._columns = dict(zip(table.schema.names, table.columns))


@vaex.dataset.register
class DatasetArrowIPCStream(DatasetArrowFileBase):
    snake_name = "arrow-ipc-stream"
    def _create_columns(self):
        self._source = vaex.file.open(path=self.path, mode='rb', fs_options=self.fs_options, fs=self.fs, mmap=True, for_arrow=True)
        reader = pa.ipc.open_stream(self._source)
        table = pa.Table.from_batches(reader, schema=reader.schema)
        self._columns = dict(zip(table.schema.names, table.columns))


def from_table(table):
    columns = dict(zip(table.schema.names, table.columns))
    dataset = vaex.dataset.DatasetArrays(columns)
    return dataset


def open(path, fs_options, fs):
    with vaex.file.open(path=path, mode='rb', fs_options=fs_options, fs=fs, mmap=True, for_arrow=True) as f:
        file_signature = bytes(f.read(6))
        is_arrow_file = file_signature == b'ARROW1'
    if is_arrow_file:
        return DatasetArrowIPCFile(path, fs_options=fs_options, fs=fs)
    else:
        return DatasetArrowIPCStream(path, fs_options=fs_options, fs=fs)


def open_parquet(path, fs_options={}, fs=None, partitioning='hive', **kwargs):
    return DatasetParquet(path, fs_options=fs_options, fs=fs, partitioning=partitioning, kwargs=kwargs)

