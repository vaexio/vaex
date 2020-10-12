__author__ = 'maartenbreddels'
import collections
import concurrent.futures
import logging
import multiprocessing
import os

import pyarrow as pa
import pyarrow.dataset

import vaex.dataset
import vaex.file.other
from ..itertools import buffer


logger = logging.getLogger("vaex.arrow.dataset")

thread_count_default_io = os.environ.get('VAEX_NUM_THREADS_IO', multiprocessing.cpu_count() * 2 + 1)
thread_count_default_io = int(thread_count_default_io)
main_io_pool = None

logger = logging.getLogger("vaex.multithreading")


def get_main_io_pool():
    global main_io_pool
    if main_io_pool is None:
        main_io_pool = concurrent.futures.ThreadPoolExecutor(max_workers=thread_count_default_io)
    return main_io_pool


class DatasetSliced(vaex.dataset.Dataset):
    def __init__(self, original, start, end):
        super().__init__()
        self.original = original
        self.start = start
        self.end = end
        self._row_count = end - start
        self._columns = {name: vaex.dataset.ColumnProxy(self, col.name, col.dtype) for name, col in self.original._columns.items()}
        self._ids = {}

    def chunk_iterator(self, columns, chunk_size=None, reverse=False):
        yield from self.original.chunk_iterator(columns, chunk_size=chunk_size, reverse=reverse, start=self.start, end=self.end)

    def hashed(self):
        raise NotImplementedError

    def close(self):
        self.original.close()

    def slice(self, start, end):
        length = end - start
        start += self.start
        end = start + length
        if end > self.original.row_count:
            raise IndexError(f'Slice end ({end}) if larger than number of rows: {self.original.row_count}')
        return type(self)(self.original, start, end)


class DatasetArrow(vaex.dataset.Dataset):
    def __init__(self, ds):
        super().__init__()
        self._arrow_ds = ds
        row_count = 0
        for fragment in self._arrow_ds.get_fragments():
            if hasattr(fragment, "ensure_complete_metadata"):
                fragment.ensure_complete_metadata()
            for rg in fragment.row_groups:
                row_count += rg.num_rows
        self._row_count = row_count
        self._columns = {name: vaex.dataset.ColumnProxy(self, name, type) for name, type in
                          zip(self._arrow_ds.schema.names, self._arrow_ds.schema.types)}
        self._ids = {}

    def hashed(self):
        raise NotImplementedError

    def slice(self, start, end):
        # TODO: we can be smarter here, and trim off some fragments
        if start == 0 and end == self.row_count:
            return self
        return DatasetSliced(self, start=start, end=end)

    def __getitem__(self, item):
        if isinstance(item, slice):
            assert item.step in [1, None]
            return DatasetSliced(self, item.start or 0, item.stop or self.row_count)
        return self._columns[item]

    def close(self):
        # no need to close it, it seem
        pass

    def _chunk_producer(self, columns, chunk_size=None, reverse=False, start=0, end=None):
        pool = get_main_io_pool()
        offset = 0
        for fragment_large in self._arrow_ds.get_fragments():
            fragment_large_rows = sum([rg.num_rows for rg in fragment_large.row_groups])
            # when do we want to split up? File size? max chunk size?
            # if fragment_large_rows > chunk_size:
            #     fragments = fragment_large.split_by_row_group()
            # else:
            #     # or not
            #     fragments = [fragment_large]
            import pyarrow.parquet
            fragments = [fragment_large]
            for fragment in fragments:
                rows = sum([rg.num_rows for rg in fragment.row_groups])
                chunk_start = offset
                chunk_end = offset + rows

                length = chunk_end - chunk_start  # default length

                if start >= chunk_end:  # we didn't find the beginning yet
                    offset += length
                    continue
                if end < chunk_start:  # we are past the end
                    # assert False
                    break
                def reader(fragment=fragment):
                    table = fragment.to_table(columns=columns, use_threads=False)
                    chunks = dict(zip(table.column_names, table.columns))
                    return chunks

                if start > chunk_start:
                    # this means we have to cut off a piece of the beginning
                    if end < chunk_end:
                        # AND the end
                        length = end - chunk_start  # without the start cut off
                        length -= start - chunk_start  # correcting for the start cut off
                        def slicer(chunk_start=chunk_start, reader=reader, length=length):
                            chunks = reader()
                            chunks = {name: ar.slice(start - chunk_start, length) for name, ar in chunks.items()}
                            for name, ar in chunks.items():
                                assert len(ar) == length, f'Oops, array was expected to be of length {length} but was {len(ar)}'
                            return chunks
                        reader = slicer
                    else:
                        length -= start - chunk_start  # correcting for the start cut off
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
                        def slicer(chunk_start=chunk_start, reader=reader, length=length):
                            chunks = reader()
                            chunks = {name: ar.slice(0, length) for name, ar in chunks.items()}
                            for name, ar in chunks.items():
                                assert len(ar) == length, f'Oops, array was expected to be of length {length} but was {len(ar)}'
                            return chunks
                        reader = slicer
                offset += length
                yield pool.submit(reader)

    def chunk_iterator(self, columns, chunk_size=None, reverse=False, start=0, end=None):
        chunk_size = chunk_size or 1024*1024
        i1 = 0
        lenghts = set()
        cuts = 0
        chunks_ready_list = []
        i1 = i2 = 0
        def get(chunks_ready_list):
            nonlocal cuts
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
                        if needed_length not in lenghts:
                            lenghts.add(needed_length)
                        cuts += 1

                        chunks_head = {name: chunk.slice(0, needed_length) for name, chunk in chunks_current.items()}
                        chunks_current_list.append(chunks_head)
                        chunks_extra = {name: chunk.slice(needed_length) for name, chunk in chunks_current.items()}
                        chunks_ready_list.insert(0, chunks_extra)  # put back the extra in front
                    else:
                        current_row_count += len(chunk)
                        chunks_current_list.append(chunks_current)
                else:
                    current_row_count += len(chunk)
                    chunks_current_list.append(chunks_current)
            return chunks_current_list, current_row_count

        for chunks_future in buffer(self._chunk_producer(columns, chunk_size, start=start, end=end or self._row_count), thread_count_default_io+3):
            chunks = chunks_future.result()
            chunks_ready_list.append(chunks)
            total_row_count = sum([len(list(k.values())[0]) for k in chunks_ready_list])
            if total_row_count > chunk_size:
                chunks_current_list, current_row_count = get(chunks_ready_list)
                i2 += current_row_count
                yield i1, i2, _concat(chunks_current_list)
                i1 = i2

        while chunks_ready_list:
            chunks_current_list, current_row_count = get(chunks_ready_list)
            i2 += current_row_count
            yield i1, i2, _concat(chunks_current_list)
            i1 = i2


def _concat(list_of_chunks):
    dict_of_list_of_arrays = collections.defaultdict(list)
    for chunks in list_of_chunks:
        for name, array in chunks.items():
            dict_of_list_of_arrays[name].extend(array.chunks)
    chunks = {name: pa.chunked_array(arrays) for name, arrays in dict_of_list_of_arrays.items()}
    return chunks


def from_table(table, as_numpy=False):
    columns = dict(zip(table.schema.names, table.columns))
    # TODO: this should be an DatasetArrow and/or DatasetParquet
    dataset = vaex.dataset.DatasetArrays(columns)
    df = vaex.dataframe.DataFrameLocal(dataset)
    return df.as_numpy() if as_numpy else df


def open(filename, as_numpy=False):
    source = pa.memory_map(filename)
    try:
        # first we try if it opens as stream
        reader = pa.ipc.open_stream(source)
    except pa.lib.ArrowInvalid:
        # if not, we open as file
        reader = pa.ipc.open_file(source)
        # for some reason this reader is not iterable
        batches = [reader.get_batch(i) for i in range(reader.num_record_batches)]
    else:
        # if a stream, we're good
        batches = reader  # this reader is iterable
    table = pa.Table.from_batches(batches)
    return from_table(table, as_numpy=as_numpy)


def open_parquet(filename, as_numpy=False):
    arrow_ds = pyarrow.dataset.dataset(filename)
    ds = DatasetArrow(arrow_ds)
    return vaex.from_dataset(ds)

# vaex.file.other.dataset_type_map["arrow"] = DatasetArrow
# vaex.file.other.dataset_type_map["parquet"] = DatasetParquet

