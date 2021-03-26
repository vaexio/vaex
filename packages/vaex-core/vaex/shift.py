from typing import Dict

import pyarrow as pa
import numpy as np
from frozendict import frozendict

import vaex.dataset



def chunk_copy(chunk_iter, mapping):
    for i1, i2, chunks in chunk_iter:
        chunks = dict(chunks)
        chunks.update({mapping[name]: chunks[name] for name in mapping if name in chunks})
        yield i1, i2, chunks


def chunk_project(chunk_iter, keys):
    for i1, i2, chunks in chunk_iter:
        yield i1, i2, {name: chunks[name] for name in keys}


def chunk_trim(chunk_iter, count):
    seen = 0
    for i1, i2, chunks in chunk_iter:
        if i2 > count:  # last chunk
            max_length = count - i1
            if max_length == 0:
                return
            chunks = {name: vaex.array_types.slice(chunks[name], 0, max_length) for name in chunks}
            i2 = i1 + max_length
            yield i1, i2, chunks
            return
        else:
            yield i1, i2, chunks


def chunk_eat(chunk_iter, count):
    trimmed = 0
    for i1, i2, chunks in chunk_iter:
        chunk_length = len(list(chunks.values())[0])
        if trimmed < count:
            max_trim = min(chunk_length, count)
            chunks = {name: vaex.array_types.slice(chunks[name], max_trim) for name in chunks}
            trimmed += max_trim
            i1 += max_trim
        if i1 == i2:
            continue
        if trimmed < count:
            continue
        assert trimmed == count
        yield i1 - count, i2 - count, chunks


def sliding_matrix(ar, ar_next, steps):
    ar = vaex.array_types.to_numpy(ar)
    dtype = vaex.dtype_of(ar)
    assert ar.ndim == 1
    N = len(ar)
    M = np.empty(ar.shape + (steps,), dtype=dtype.numpy)
    M = np.ma.array(M, shrink=False, mask=np.full(M.shape, True))

    for i in range(N):
        part = ar[i:i+steps]
        M[i,:len(part)] = part
        M.mask[i,:len(part)] = part.mask if np.ma.isMaskedArray(part) else False

    if ar_next is not None:
        ar_next = vaex.array_types.to_numpy(ar_next)
        for i in range(steps-1):
            length = i + 1
            part = ar_next[:length]
            offset = N-steps+i+1
            M[offset,-length:] = part
            M.mask[offset,-length:] = part.mask if np.ma.isMaskedArray(part) else False

    return M

def chunk_sliding_matrix(chunk_iter, shift, keys, chunk_size):
    assert shift < chunk_size
    chunks_ready_list_prepend = []
    i1, i2, chunks = next(chunk_iter)
    for i1_next, i2_next, chunks_next in chunk_iter:
        chunks_extra = {name: sliding_matrix(chunks[name], chunks_next[name], shift) for name in keys}
        yield i1, i2, {**chunks, **chunks_extra}
        chunks = chunks_next
        i1 = i1_next
        i2 = i2_next
    chunks_extra = {name: sliding_matrix(chunks[name], None, shift) for name in keys}
    yield i1, i2, {**chunks, **chunks_extra}


def chunk_prepend(chunk_iter, prepend_chunks, chunk_size):
    chunks_ready_list_prepend = []
    chunks_ready_list_prepend.append(prepend_chunks)
    for i1, i2, chunks in chunk_iter:
        chunks = dict(chunks)
        for chunk in chunks.values():
            assert len(chunk) <= chunk_size, f'Expected chunk_size<={chunk_size}, but got {len(chunk)}'
        chunks_ready_list_prepend.append({name: ar for name, ar in chunks.items() if name in prepend_chunks})
        chunks_passthrough = {name: ar for name, ar in chunks.items() if name not in prepend_chunks}
        chunks_current_list_prepend, current_row_count_prepend = vaex.dataset._slice_of_chunks(chunks_ready_list_prepend, chunk_size)
        chunks_prepend = vaex.dataset._concat_chunk_list(chunks_current_list_prepend)
        chunks = {**chunks_prepend, **chunks_passthrough}
        length = i2 - i1
        def trim(ar):
            if len(ar) > length:
                return vaex.array_types.slice(ar, 0, length)
            else:
                return ar
        # and trim off the trailing part
        chunks = {name: trim(chunks[name]) for name in chunks}  # sort and trim
        yield i1, i2, chunks


def chunk_append(chunk_iter, append_chunks, chunk_size):
    trimmed = 0
    n = len(list(append_chunks.values())[0])

    chunks_ready_list_append = []
    chunks_ready_list_passthrough = []
    our_i1, our_i2 = 0, 0
    for i1, i2, chunks in chunk_iter:
        chunks = dict(chunks)

        # keep a list of chunks for which we want to append to
        chunks_ready_list_append.insert(-1, {name: ar for name, ar in chunks.items() if name in append_chunks})
        # and passthrough
        chunks_ready_list_passthrough.append({name: ar for name, ar in chunks.items() if name not in append_chunks})

        if trimmed < n:
            # eat n rows
            _, trimmed_now = vaex.dataset._slice_of_chunks(chunks_ready_list_append, n)
            if trimmed_now == 0:
                # continue getting chunks until we have enough
                # continue  # TODO: how to avoid inf loop
                continue
            else:
                # otherwise we are done
                trimmed = n
            continue

        chunks_current_list_append, current_row_count_append = vaex.dataset._slice_of_chunks(chunks_ready_list_append, chunk_size)
        chunks_append = vaex.dataset._concat_chunk_list(chunks_current_list_append)
        

        has_passthrough = any(name not in append_chunks for name in chunks)
        if has_passthrough:
            chunks_current_list_passthrough, current_row_count_passthrough = vaex.dataset._slice_of_chunks(chunks_ready_list_passthrough, current_row_count_append)
            chunks_passthrough = vaex.dataset._concat_chunk_list(chunks_current_list_passthrough)
            chunks = {**chunks_append, **chunks_passthrough}
            assert current_row_count_passthrough == current_row_count_append
        else:
            chunks = chunks_append
        our_i2 = our_i1 + current_row_count_append
        chunks = {name: chunks[name] for name in chunks}  # sort
        yield our_i1, our_i2, chunks
        our_i1 = our_i2

    # add the trailing part, and repeat the end of the loop
    chunks_ready_list_append.append(append_chunks)
    while chunks_ready_list_append:
        chunks_current_list_append, current_row_count_append = vaex.dataset._slice_of_chunks(chunks_ready_list_append, chunk_size)
        chunks_append = vaex.dataset._concat_chunk_list(chunks_current_list_append)
        

        has_passthrough = any(name not in append_chunks for name in chunks)
        if has_passthrough:
            chunks_current_list_passthrough, current_row_count_passthrough = vaex.dataset._slice_of_chunks(chunks_ready_list_passthrough, chunk_size)
            chunks_passthrough = vaex.dataset._concat_chunk_list(chunks_current_list_passthrough)
            chunks = {**chunks_append, **chunks_passthrough}
            assert current_row_count_passthrough == current_row_count_append
        else:
            chunks = chunks_append
        our_i2 = our_i1 + current_row_count_append
        chunks = {name: chunks[name] for name in chunks}  # sort
        yield our_i1, our_i2, chunks
        our_i1 = our_i2

class DatasetShifted(vaex.dataset.Dataset):
    column_mapping: Dict[str, str]  # original name: shifted name
    def __init__(self, dataset, column_mapping : Dict[str, str], start, end, fill_value=None):
        self.start = start
        self.end = end
        assert self.start <= self.end
        self.fill_value = fill_value
        self.original = dataset
        self.column_mapping = dict(column_mapping)
        self.column_mapping_reverse = {v: k for k, v in column_mapping.items()}
        self._shifted_column_names = list(column_mapping.values())
        for column in self.column_mapping:
            assert column in dataset, f'Expected {column} to be in the current dataset'
        self._columns = {name: vaex.dataset.ColumnProxy(self, name, vaex.array_types.data_type(col)) for name, col in self.original._columns.items()}
        self._ids = frozendict({name: ar for name, ar in self.original._ids.items()})
        schema = self.original.schema()
        for original_name, shifted_name in column_mapping.items():
            self._columns[shifted_name] = vaex.dataset.ColumnProxy(self, shifted_name, schema[original_name])
            if original_name in self.original._ids:
                self._ids[shifted_name] = self.original._ids[original_name]
        self._row_count = self.original.row_count

    def _create_columns(self):
        pass

    def chunk_iterator(self, columns, chunk_size=None, reverse=False, start=0, end=None):
        yield from self._chunk_iterator(columns, chunk_size, start=start, end=end)

    def _chunk_iterator(self, columns, chunk_size, reverse=False, start=0, end=None):
        end = self.row_count if end is None else end
        chunk_size = chunk_size or 1024**2  # TODO: should we have a constant somewhere
        # if self.n > chunk_size:
            # raise ValueError(f'Iterating with a chunk_size ({chunk_size}) smaller then window_size ({self.window_size})')
        assert not reverse
        # find all columns we need from the original dataset
        columns_original = set(self.column_mapping_reverse.get(name, name) for name in columns)
        # and the once we need to return shifted
        columns_shifted = set(columns) & set(self._shifted_column_names)
        iter = self.original.chunk_iterator(list(columns_original), chunk_size, reverse=reverse)

        # schema for shifted columns only
        schema = {name: vaex.array_types.to_arrow_type(dtype) for name, dtype in self.schema().items()}

        if columns_shifted:
            # (shallow) copy e.g. x to x_shifted
            iter = chunk_copy(iter, self.column_mapping)

            if self.start > 0:
                # for positive shift, we add the fill values at the front
                iter = chunk_prepend(iter, {name: self._filler(min(self.row_count, self.start), dtype=schema[name]) for name in columns_shifted}, chunk_size)
            elif self.start < 0:
                # negative at the end
                iter = chunk_append(iter, {name: self._filler(min(self.row_count, -self.end), dtype=schema[name]) for name in columns_shifted}, chunk_size)
            # for start 0 we don't need to do anything
            if self.end > self.start:
                shift = self.end - self.start
                iter = chunk_sliding_matrix(iter, shift, columns_shifted, chunk_size)

            # because of the shifting, we may has asked our child dataset for too many columns
            # so we remove them
            iter = chunk_project(iter, columns)
        if start != 0 or end != self.row_count:
            if start != 0:                
                iter = chunk_eat(iter, start)
            if end != self.row_count:
                iter = chunk_trim(iter, end - start)
            iter = vaex.dataset.chunk_rechunk(iter, chunk_size)
        yield from iter

    def _filler(self, n, dtype):
        if self.fill_value is None:
            type = vaex.array_types.to_arrow_type(dtype)
            return pa.nulls(n, type=type)
        else:
            return vaex.array_types.full(n, self.fill_value, dtype=dtype)

    def is_masked(self, column):
        column = self.column_mapping_reverse.get(column, column)
        return self.original.is_masked(column) or self.start != self.end

    def shape(self, column):
        column = self.column_mapping_reverse.get(column, column)
        return self.original.shape(column)

    def close(self):
        self.original.close()

    def slice(self, start, end):
        if start == 0 and end == self.row_count:
            return self
        # TODO: we can be smarter here, and trim off some datasets
        return vaex.dataset.DatasetSliced(self, start=start, end=end)

    def hashed(self):
        if set(self._ids) == set(self):
            return self
        return type(self)(self.original.hashed(), column_mapping=self.column_mapping, start=self.start, end=self.end, fill_value=self.fill_value)
