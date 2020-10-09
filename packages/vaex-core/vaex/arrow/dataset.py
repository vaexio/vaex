__author__ = 'maartenbreddels'
import collections
import logging

import pyarrow as pa
import pyarrow.dataset

import vaex.dataset
import vaex.file.other
from .convert import column_from_arrow_array
logger = logging.getLogger("vaex.arrow")


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
        chunk_size = chunk_size or 1024*1024
        dict_or_list_of_arrays = collections.defaultdict(list)
        i1 = i2 = 0
        for chunk_start, chunk_end, reader in self.original.chunk_iterator(columns, chunk_size=chunk_size):
            if self.start >= chunk_end:  # we didn't find the beginning yet
                continue
            if self.end < chunk_start:  # we are past the end
                break
            slice_reader = reader  # default case, if we don't have to slice
            length = chunk_end - chunk_start  # default length
            if self.start > chunk_start:
                # this means we have to cut off a piece of the beginning
                if self.end < chunk_end:
                    # AND the end
                    length = self.end - chunk_start  # without the start cut off
                    length -= self.start - chunk_start  # correcting for the start cut off
                    def slice_reader(chunk_start=chunk_start, reader=reader, length=length):
                        chunks = reader()
                        chunks = {name: ar.slice(self.start - chunk_start, length) for name, ar in chunks.items()}
                        for name, ar in chunks.items():
                            assert len(ar) == length, f'Oops, array was expected to be of length {length} but was {len(ar)}'
                        return chunks
                else:
                    length -= self.start - chunk_start  # correcting for the start cut off
                    def slice_reader(chunk_start=chunk_start, reader=reader, length=length):
                        chunks = reader()
                        chunks = {name: ar.slice(self.start - chunk_start) for name, ar in chunks.items()}
                        for name, ar in chunks.items():
                            assert len(ar) == length, f'Oops, array was expected to be of length {length} but was {len(ar)}'
                        return chunks
            else:
                if self.end < chunk_end:
                    # we only need to cut off a piece of the end
                    length = self.end - chunk_start
                    def slice_reader(chunk_start=chunk_start, reader=reader, length=length):
                        chunks = reader()
                        chunks = {name: ar.slice(0, length) for name, ar in chunks.items()}
                        for name, ar in chunks.items():
                            assert len(ar) == length, f'Oops, array was expected to be of length {length} but was {len(ar)}'
                        return chunks
                # else, the defaults apply
            i2 = i1 + length
            yield i1, i2, slice_reader
            i1 = i2

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

    def chunk_iterator(self, columns, chunk_size=None, reverse=False):
        chunk_size = chunk_size or 1024*1024
        i1 = 0
        # Instead of looping over all fragments, they might be too big, so...
        for fragment_large in self._arrow_ds.get_fragments():
            fragment_large_rows = sum([rg.num_rows for rg in fragment_large.row_groups])
            # then we split them up
            if fragment_large_rows > chunk_size:
                fragments = fragment_large.split_by_row_group()
            else:
                # or not
                fragments = [fragment_large]
            # now we collect fragments, until we hit the chunk_size
            rows_planned = 0
            fragments_planned = []
            for fragment in fragments:
                fragment_rows = sum([rg.num_rows for rg in fragment.row_groups])
                # if adding the current fragment would make the chunk too large
                # and we already have some fragments
                if rows_planned + fragment_rows > chunk_size and rows_planned > 0:
                    i2 = i1 + rows_planned
                    yield i1, i2, self._make_reader(fragments_planned, chunk_size, columns, rows_planned)
                    i1 = i2
                    rows_planned = 0
                    fragments_planned = []
                fragments_planned.append(fragment)
                rows_planned += fragment_rows
            if rows_planned:
                i2 = i1 + rows_planned
                yield i1, i2, self._make_reader(fragments_planned, chunk_size, columns, rows_planned)
                i1 = i2
                rows_planned = 0
                fragments_planned = []

    def _make_reader(self, fragments, chunk_size, columns, rows_planned):
        def reader():
            record_batches = []
            for fragment in fragments:
                for scan_task in fragment.scan(batch_size=chunk_size, use_threads=False, columns=columns):
                    for record_batch in scan_task.execute():
                        record_batches.append((record_batch))
            dict_or_list_of_arrays = collections.defaultdict(list)
            for rb in record_batches:
                for name, array in zip(rb.schema.names, rb.columns):
                    dict_or_list_of_arrays[name].append(array)
            chunks = {name: pa.chunked_array(arrays) for name, arrays in dict_or_list_of_arrays.items()}
            for name, chunk in chunks.items():
                assert len(chunk) == rows_planned, f'Oops, got a chunk ({name}) of length {len(chunk)} while it is expected to be of length {rows_planned}'
            return chunks
        return reader


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

