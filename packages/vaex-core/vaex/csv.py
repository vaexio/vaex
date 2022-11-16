import collections
import mmap
from typing import Any, Dict, List
import numpy as np

import pyarrow as pa
import pyarrow.csv
from dask.utils import parse_bytes
from frozendict import frozendict

import vaex.dataset
import vaex.settings
import vaex.file
from vaex.dataset import Dataset, DatasetFile
from .itertools import pmap, pwait, buffer, consume, filter_none
from .multithreading import get_main_io_pool


MB = 1024**2


def file_chunks(file, chunk_size, newline_readahead, fs=None, fs_options=None):
    """Bytes chunks, split by chunk_size bytes, on newline boundaries"""
    offset = 0
    # with open(file, 'rb') as file:\
    file_size = vaex.file.size(file, fs=fs, fs_options=fs_options)
    leftover = b''
    leftover_previous = b''

    with vaex.file.open(file, fs=fs, fs_options=fs_options) as file:
        begin_offset = 0
    
        done = False
        while not done:
            size_expected = chunk_size + newline_readahead
            data_extra = file.read(size_expected)
            if len(data_extra) == size_expected:
                sample = data_extra[chunk_size:]
                offset = vaex.superutils.find_byte(sample, ord(b'\n'))
                if offset == -1:
                    raise ValueError(f'Expected a newline within {newline_readahead} bytes, but not found, please increase newline_readahead')
                data, leftover = data_extra[:chunk_size+offset+1], data_extra[chunk_size+offset+1:]
            else:
                # read till the end
                data = data_extra
                done = True

            assert len(data) > 0
            def reader(data=data, leftover_previous=leftover_previous):
                chunk = leftover_previous + data
                return memoryview(chunk)
            leftover_previous = leftover
            yield reader


def file_chunks_mmap(file, chunk_size, newline_readahead):
    """Bytes chunks, split by chunk_size bytes, on newline boundaries
    
    Using memory mapping (which avoids a memcpy)
    """
    offset = 0
    with open(file, 'rb') as file:
        file.seek(0, 2)
        file_size = file.tell()
        file.seek(0)
        begin_offset = 0
        kwargs = {}
        if vaex.utils.osname == "windows":
            kwargs["access"] = mmap.ACCESS_READ
        else:
            kwargs["prot"] = mmap.PROT_READ

        file_map = mmap.mmap(file.fileno(), file_size, **kwargs)
        data = memoryview(file_map)
    
        done = False
        while not done:
            # find the next newline boundary
            end_offset = min(file_size, begin_offset + chunk_size)
            end_offset_readahead = min(file_size, begin_offset + chunk_size + newline_readahead)
            if end_offset_readahead < file_size:
                sample = data[end_offset:end_offset_readahead]
                offset = vaex.superutils.find_byte(sample, ord(b'\n'))
                if offset != -1:
                    end_offset += offset + 1 # include the newline
                else:
                    raise ValueError(f'Expected a newline within {newline_readahead} bytes, but not found, please increase newline_readahead')
            else:
                end_offset = file_size
            done = end_offset == file_size

            length = end_offset - begin_offset
            assert length > 0
            def reader(file_offset=begin_offset, length=length):
                return data[file_offset:file_offset+length]
            yield reader
            begin_offset = end_offset


def _row_count(chunk):
    ar = np.frombuffer(chunk, dtype=np.uint8)
    lines = vaex.superutils.count_byte(ar, ord(b'\n'))
    if ar[-1] != ord(b'\n'):
        lines += 1
    return lines


def _copy_or_create(cls, obj, **kwargs):
    if obj is not None:
        # take default from obj, buy kwargs have precendence
        kwargs = kwargs.copy()
        for name in dir(obj):
            value = getattr(obj, name)
            if not name.startswith('__') and not callable(value):
                if name not in kwargs:
                    kwargs[name] = value
    return cls(**kwargs)


def _get_kwargs(obj):
    kwargs = {}
    if obj is not None:
        # take default from obj, buy kwargs have precendence
        kwargs = kwargs.copy()
        for name in dir(obj):
            value = getattr(obj, name)
            if not name.startswith('__') and not callable(value):
                if name not in kwargs:
                    kwargs[name] = value
    return kwargs


@vaex.dataset.register
class DatasetCsvLazy(DatasetFile):
    snake_name = "arrow-csv-lazy"
    def __init__(self, path, chunk_size=10*MB, newline_readahead=1*MB, row_count=None, schema=None, read_options=None, parse_options=None, convert_options=None, schema_infer_fraction=0.001, fs=None, fs_options={}):
        super().__init__(path, fs=fs, fs_options=fs_options)
        try:
            codec = pa.Codec.detect(self.path)
        except Exception:
            codec = None
        if codec:
            raise NotImplementedError("We don't support compressed csv files for lazy reading, cannot read file: %s" % self.path)
        self._given_row_count = row_count
        self._row_count = None
        self._schema = schema
        self.chunk_size = parse_bytes(chunk_size)
        self.newline_readahead = parse_bytes(newline_readahead)


        self.read_options = read_options
        self.parse_options = parse_options
        self.convert_options = convert_options
        self.schema_infer_fraction = schema_infer_fraction
        self._infer_schema()

    @classmethod
    def quick_test(cls, path, fs_options={}, fs=None):
        path, options = vaex.file.split_options(path)
        return path.endswith('.csv')

    @classmethod
    def can_open(cls, path, fs_options={}, fs=None, group=None, **kwargs):
        return cls.quick_test(path, fs_options=fs_options, fs=fs)


    def _encode(self, encoding):
        spec = super()._encode(encoding)
        del spec["write"]
        return spec

    def __getstate__(self):
        state = super().__getstate__()
        state["read_options"] = self.read_options
        state["parse_options"] = self.parse_options
        state["convert_options"] = self.convert_options
        state["_given_row_count"] = self._given_row_count
        state["chunk_size"] = self.chunk_size
        state["newline_readahead"] = self.newline_readahead
        state["_row_count"] = self._row_count
        state["_schema"] = self._schema
        state["schema_infer_fraction"] = self.schema_infer_fraction
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._infer_schema()

    def leafs(self):
        return [self]

    @property
    def _fingerprint(self):
        paths = self.path
        fingerprint_file = vaex.file.fingerprint(self.path, fs_options=self.fs_options, fs=self.fs)
        # TODO: not sure if we need to put in the read_options
        fp = vaex.cache.fingerprint(fingerprint_file)
        return f'dataset-{self.snake_name}-{fp}'

    def _read_table(self, data, first, columns : List[str] = None):
        file_like = pa.input_stream(data)
        use_threads = True
        block_size = len(data)
        if first:
            read_options = _copy_or_create(pyarrow.csv.ReadOptions, self.read_options, use_threads=use_threads, block_size=block_size)
        else:
            read_options = _copy_or_create(pyarrow.csv.ReadOptions, self.read_options, use_threads=use_threads, block_size=block_size, column_names=self._column_names)

        if self._arrow_schema is None:
            convert_options = _copy_or_create(pyarrow.csv.ConvertOptions, self.convert_options, include_columns=columns)
        else:
            schema = pa.schema([(name, self._schema[name]) for name in columns])
            convert_options = _copy_or_create(pyarrow.csv.ConvertOptions, self.convert_options, column_types=schema, include_columns=columns)
        try:
            table = pyarrow.csv.read_csv(file_like, read_options=read_options, convert_options=convert_options)
        except pa.ArrowInvalid as e:
            import tempfile
            f = tempfile.NamedTemporaryFile(mode="wb", suffix=".csv")
            f.write(data)
            raise ValueError(
                f"Error reading csv file {self.path}, write offending chunk to: {f.name} (len={len(data)}, first={first}, columns={columns}, schema={schema}, encoding={read_options.encoding}, schema_infer_fraction={self.schema_infer_fraction}).\n"
                "Possible causes:\n"
                '  * This could be a file encoding error. Consider passing read_options=pyarrow.csv.ReadOptions(encoding="ISO-8859-1") or another encoding as argument.\n'
                "  * We might have inferred the wrong schema:\n"
                '     * Consider giving a schema hint by e.g. passing read_options=pyarrow.csv.ConvertOptions(column_types={"SomeId": pyarrow.string()}).\n'
                "     * Consider increasing schema_infer_fraction (e.g. schema_infer_fraction=1 to parse the whole file to infer the schema).\n"
            ) from e
        return table


    def _infer_schema(self):
        with vaex.file.open(self.path, fs_options=self.fs_options, fs=self.fs) as f:
            data = bytes(f.read(self.newline_readahead*3))
            offset = 0
            offset = data.find(b'\n', offset) + 1
            if offset == 0:
                raise ValueError("Cannot find newline in first %d bytes of file: %s" % (self.newline_readahead*3, self.path))
            offset = data.find(b'\n', offset) + 1
            if offset == 0:
                raise ValueError("Cannot find second newline in first %d bytes of file: %s" % (self.newline_readahead*3, self.path))
            f_first_two_lines = pa.input_stream(memoryview(data[:offset]))
            reader = pyarrow.csv.open_csv(f_first_two_lines, read_options=self.read_options, parse_options=self.parse_options, convert_options=self.convert_options)
            self._arrow_schema = reader.read_next_batch().schema
        self._column_names = list(self._arrow_schema.names)

        self._arrow_schema = None

        pool = get_main_io_pool()
        workers = pool._max_workers
        self._fragment_info = {}
        schemas: Dict[str, List[Any]] = collections.defaultdict(list)
        if self._schema is None and self._given_row_count is None and self._row_count is None:
            if self.fs is None and not self.fs_options:
                chunks = file_chunks_mmap(self.path, self.chunk_size, self.newline_readahead)
            else:
                chunks = file_chunks(self.path, self.chunk_size, self.newline_readahead, fs=self.fs, fs_options=self.fs_options)
            def process(i, chunk_reader):
                data = chunk_reader()
                if (i % int(1/self.schema_infer_fraction)) == 0:
                    table = self._read_table(data, first=i==0, columns=self._column_names)
                    for name, type in zip(table.schema.names, table.schema.types):
                        schemas[name].append(type)
                row_count = _row_count(data)
                if i == 0:
                    row_count -= 1  # we counted the header (TODO: depends on ReadOptions)
                self._fragment_info[i] = dict(row_count=row_count)
            consume(pwait(buffer(pmap(process, enumerate(chunks), pool=pool), workers+3)))
            row_start = 0
            for i in range(len(self._fragment_info)):
                row_end = row_start + self._fragment_info[i]['row_count']
                self._fragment_info[i]['row_start'] = row_start
                self._fragment_info[i]['row_end'] = row_end
                row_start = row_end
            self._row_count = row_start
            pool.map(process, enumerate(chunks))
        else:
            if self._row_count is None:
                self._row_count = self._given_row_count
        if self._schema is None:
            self._schema = {}
            from .schema import resolver_flexible

            for name, types in schemas.items():
                type, shape = resolver_flexible.resolve(types)
                if pa.types.is_null(type):
                    type = pa.int8()
                self._schema[name] = type
        self._arrow_schema = pa.schema([(name, type) for name, type in self._schema.items()])

        self._columns = {name: vaex.dataset.ColumnProxy(self, name, type) for name, type in self._schema.items()}

        self._ids = {}

    def hashed(self):
        raise NotImplementedError

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
        pool = get_main_io_pool()
        
        first = True
        previous = None
        if self.fs is None and not self.fs_options:
            file_chunks_iterator = file_chunks_mmap(self.path, self.chunk_size, self.newline_readahead)
        else:
            file_chunks_iterator = file_chunks(self.path, self.chunk_size, self.newline_readahead, fs=self.fs, fs_options=self.fs_options)
        for i, reader in enumerate(file_chunks_iterator):
            fragment_info = self._fragment_info.get(i)
            # bail out/continue early
            if fragment_info:
                if start >= fragment_info['row_end']:  # we didn't find the beginning yet
                    continue
                # TODO, not triggered, should be <=
                if end < fragment_info['row_start']:  # we are past the end
                    # assert False
                    break

            def chunk_reader(reader=reader, first=first, previous=previous, fragment_info=fragment_info, i=i):
                bytes = reader()
                file_like = pa.input_stream(bytes)
                use_threads = True
                block_size = len(bytes)
                table = self._read_table(bytes, first=i==0, columns=columns)

                row_count = len(table)
                row_start = 0
                if i not in self._fragment_info:
                    if previous:
                        row_start, row_end, chunks = previous.result()
                        row_start = row_end
                    row_end = row_start + len(table)
                    self._fragment_info[i] = dict(
                            # begin_offset=begin_offset,
                            # end_offset=end_offset,
                            row_count=row_count,
                            row_start=row_start,
                            row_end=row_end,
                        )
                else:
                    row_start = self._fragment_info[i]['row_start']
                    row_end = self._fragment_info[i]['row_end']

                # this is the bail out when we didn't have the fragments info cached yet
                fragment_info = self._fragment_info[i]
                if start >= fragment_info['row_end']:  # we didn't find the beginning yet
                    return None
                if end <= fragment_info['row_start']:  # we are past the end
                    return None
                # print(start, end, fragment_info, row_start, row_end)


                if start > row_start:
                    # this means we have to cut off a piece of the beginning
                    if end < row_end:
                        # AND the end
                        length = end - row_start  # without the start cut off
                        length -= start - row_start  # correcting for the start cut off
                        # print(start, end, length, row_start, row_end)
                        table = table.slice(start - row_start, length)
                    else:
                        table = table.slice(start - row_start)
                else:
                    if end < row_end:
                        # we only need to cut off a piece of the end
                        length = end - row_start
                        table = table.slice(0, length)
                
                # table = table.combine_chunks()
                assert len(table)
                chunks = dict(zip(table.column_names, table.columns))
                return row_start, row_end, chunks


            previous = pool.submit(chunk_reader)
            yield previous
            first = False

    def chunk_iterator(self, columns, chunk_size=None, reverse=False, start=0, end=None):
        chunk_size = chunk_size or 1024*1024
        i1 = 0
        chunks_ready_list = []
        i1 = i2 = 0

        # althoug arrow uses threading, we still manage to get some extra performance out of it with more threads
        chunk_generator = self._chunk_producer(columns, chunk_size, start=start, end=end or self._row_count)
        for column in columns:
            if column not in self._columns:
                self._check_existence(column)
        for c1, c2, chunks in filter_none(pwait(buffer(chunk_generator, vaex.settings.main.thread_count_io//2+3))):
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
class DatasetCsv(DatasetFile):
    snake_name = "arrow-csv"

    def __init__(self, path, read_options=None, parse_options=None, convert_options=None, fs=None, fs_options={}):
        super(DatasetCsv, self).__init__(path, fs=fs, fs_options=fs_options)
        self.read_options = read_options
        self.parse_options = parse_options
        self.convert_options = convert_options
        self._read_file()

    @property
    def _fingerprint(self):
        fp = vaex.file.fingerprint(self.path, fs_options=self.fs_options, fs=self.fs)
        return f"dataset-{self.snake_name}-{fp}"

    def _read_file(self):
        import pyarrow.csv

        with vaex.file.open(self.path, fs=self.fs, fs_options=self.fs_options, for_arrow=True) as f:
            try:
                codec = pa.Codec.detect(self.path)
            except Exception:
                codec = None
            if codec:
                f = pa.CompressedInputStream(f, codec.name)
            self._arrow_table = pyarrow.csv.read_csv(f, read_options=self.read_options, parse_options=self.parse_options, convert_options=self.convert_options)
        self._columns = dict(zip(self._arrow_table.schema.names, self._arrow_table.columns))
        self._set_row_count()
        self._ids = frozendict({name: vaex.cache.fingerprint(self._fingerprint, name) for name in self._columns})

    def _encode(self, encoding):
        spec = super()._encode(encoding)
        del spec["write"]
        return spec

    def __getstate__(self):
        state = super().__getstate__()
        state["read_options"] = self.read_options
        state["parse_options"] = self.parse_options
        state["convert_options"] = self.convert_options
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._read_file()

    def close(self):
        pass


import dask.base

@dask.base.normalize_token.register(pyarrow.csv.ConvertOptions)
def _normalize(obj):
    kwargs = _get_kwargs(obj)
    return (pyarrow.csv.ConvertOptions, kwargs)

@dask.base.normalize_token.register(pyarrow.csv.ParseOptions)
def _normalize(obj):
    kwargs = _get_kwargs(obj)
    return (pyarrow.csv.ParseOptions, kwargs)

@dask.base.normalize_token.register(pyarrow.csv.ReadOptions)
def _normalize(obj):
    kwargs = _get_kwargs(obj)
    return (pyarrow.csv.ReadOptions, kwargs)
