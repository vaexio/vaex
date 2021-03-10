from typing import Dict

import pyarrow as pa
from frozendict import frozendict

import vaex.dataset


class DatasetShifted(vaex.dataset.Dataset):
    column_mapping: Dict[str, str]  # original name: shifted name
    def __init__(self, dataset, column_mapping : Dict[str, str], n, fill_value=None):
        self.n = n
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

    def chunk_iterator(self, columns, chunk_size=None, reverse=False):
        yield from self._chunk_iterator(columns, chunk_size)

    def _chunk_iterator(self, columns, chunk_size, reverse=False):
        chunk_size = chunk_size or 1024**2  # TODO: should we have a constant somewhere
        if self.n > chunk_size:
            raise ValueError(f'Iterating with a chunk_size ({chunk_size}) smaller then window_size ({self.window_size})')
        assert not reverse
        # find all columns we need from the original dataset
        columns_original = set(self.column_mapping_reverse.get(name, name) for name in columns)
        # and the once we need to return shifted
        columns_shifted = set(columns) & set(self._shifted_column_names)
        iter = self.original.chunk_iterator(list(columns_original), chunk_size, reverse=reverse)
        chunks_ready_list_passthrough = []
        chunks_ready_list_shifted = []

        # schema for shifted columns only
        schema = {name: vaex.array_types.to_arrow_type(dtype) for name, dtype in self.schema().items()}
        # requested_shifted_columns = any(col in columns for col in self._shifted_column_names)

        def project(chunks):
            # to return only the chunks asked for
            return {name: chunks[name] for name in columns}

        if not columns_shifted or self.n == 0:
            yield from iter  # no shift needed
        if self.n > 0:
            # for positive shift, we add the fill values at the front
            chunks_ready_list_shifted.append({name: self._filler(self.n, dtype=schema[name]) for name in columns_shifted})
            for i1, i2, chunks in iter:
                chunks = dict(chunks)
                # copy chunks that we need to shift
                chunks.update({self.column_mapping[name]: ar for name, ar in chunks.items()})
                chunks_ready_list_shifted.append({name: ar for name, ar in chunks.items() if name in columns_shifted})
                chunks_passthrough = {name: ar for name, ar in chunks.items() if name not in self._shifted_column_names}
                chunks_current_list_shifted, current_row_count_shifted = vaex.dataset._slice_of_chunks(chunks_ready_list_shifted, chunk_size)
                chunks_shifted = vaex.dataset._concat_chunk_list(chunks_current_list_shifted)
                chunks = {**chunks_shifted, **chunks_passthrough}
                length = i2 - i1
                def trim(ar):
                    if len(ar) > length:
                        return vaex.array_types.slice(ar, 0, length)
                    else:
                        return ar
                # and trim off the trailing part
                chunks = {name: trim(chunks[name]) for name in columns}  # sort and trim
                # print(chunks)
                yield i1, i2, project(chunks)
        else:
            # for negative shift, it is a bit more complicated
            # we track how much we trimmed off from the start
            trimmed = 0
            has_passthrough = any(col not in self._shifted_column_names for col in columns)

            for i1, i2, chunks in iter:
                chunks = dict(chunks)
                # copy chunks that we need to shift
                chunks.update({self.column_mapping[name]: ar for name, ar in chunks.items()})
                chunks_ready_list_shifted.append({name: ar for name, ar in chunks.items() if name in columns_shifted})
                chunks_ready_list_passthrough.append({name: ar for name, ar in chunks.items() if name not in columns_shifted})
                chunks_current_list_shifted, current_row_count_shifted = vaex.dataset._slice_of_chunks(chunks_ready_list_shifted, chunk_size)
                chunks_shifted = vaex.dataset._concat_chunk_list(chunks_current_list_shifted)
                if trimmed < -self.n:
                    trim_max = min(current_row_count_shifted, -self.n)
                    def trim(ar):
                        return vaex.array_types.slice(ar, trim_max)
                    chunks_shifted = {name: trim(ar) for name, ar in chunks_shifted.items()}
                    trimmed += trim_max
                chunk_length_expected = i2 - i1
                chunk_length_shifted = len(list(chunks_shifted.values())[0])
                # and when we are at the end, we pad the chunks with the fill values
                if chunk_length_shifted < chunk_length_expected:
                    padding_length = chunk_length_expected - chunk_length_shifted
                    def pad(ar):
                        dtype = vaex.array_types.data_type(ar)
                        return vaex.array_types.concat([ar, self._filler(padding_length, dtype=dtype)])
                    chunks_shifted = {name: pad(ar) for name, ar in chunks_shifted.items()}
                if has_passthrough:
                    chunks_current_list_passthrough, current_row_count_shifted = vaex.dataset._slice_of_chunks(chunks_ready_list_passthrough, chunk_size)
                    chunks_passthrough = vaex.dataset._concat_chunk_list(chunks_current_list_passthrough)
                    chunks = {**chunks_shifted, **chunks_passthrough}
                else:
                    chunks = chunks_shifted
                length = i2 - i1
                chunks = {name: chunks[name] for name in columns}  # sort
                yield i1, i2, project(chunks)

    def _filler(self, n, dtype):
        if self.fill_value is None:
            type = vaex.array_types.to_arrow_type(dtype)
            return pa.nulls(n, type=type)
        else:
            return vaex.array_types.full(n, self.fill_value, dtype=dtype)

    def is_masked(self, column):
        return self.original.is_masked(column)

    def shape(self, column):
        return self.original.shape(column)

    def close(self):
        self.original.close()

    def slice(self, start, end):
        if start == 0 and end == self.row_count:
            return self
        return type(self)(self.original.slice(start, end), column_mapping=self.column_mapping, n=self.n, fill_value=self.fill_value)

    def hashed(self):
        if set(self._ids) == set(self):
            return self
        return type(self)(self.original.hashed(), column_mapping=self.column_mapping, n=self.n, fill_value=self.fill_value)
