import mmap
import logging
import concurrent.futures

import h5py
import numpy as np
import pyarrow as pa
from pyarrow.types import is_temporal

import vaex
import vaex.array_types
import vaex.utils
from .utils import h5mmap
from vaex.column import ColumnStringArrow, _to_string_sequence


USE_MMAP = vaex.utils.get_env_type(bool, 'VAEX_USE_MMAP', True)
logger = logging.getLogger("vaex.hdf5.writer")
max_int32 = 2**31-1


class Writer:
    def __init__(self, path, group="/table", mode="w", byteorder="="):
        self.path = path
        self.byteorder = byteorder
        self.h5 = h5py.File(path, mode)
        self.fs_options = {}
        self.table = self.h5.require_group(group)
        self.table.attrs["type"] = "table"
        self.columns = self.h5.require_group(f"{group}/columns")
        self.mmap = None
        self._layout_called = False

    def close(self):
        # make sure we don't have references to the numpy arrays any more
        self.column_writers = {}
        if self.mmap is not None:
            self.mmap.close()
            self.file.close()
        self.h5.close()
        
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def layout(self, df, progress=None):
        assert not self._layout_called, "Layout called twice"
        N = len(df)
        if N == 0:
            raise ValueError("Cannot layout empty table")
        column_names = df.get_column_names()

        logger.debug("layout columns(hdf5): %r" % column_names)
        progressbar = vaex.utils.progressbars(progress, title="layout(hdf5)")
        progressbar_strings = progressbar.add("storage requirements")
        progressbar_count = progressbar.add("count missing values")
        progressbar_reserve = progressbar.add("reserve disk space")

        self.column_writers = {}
        dtypes = df.schema()
        str_byte_length = {name:df[name].str.byte_length().sum(delay=True, progress=progressbar_strings) for name, dtype in dtypes.items() if dtype.is_string}
        # arrow types can contain nulls even when they don't have a mask
        arrow_null_count = {name: df.count(df[name], delay=True, progress=progressbar_count) for name, dtype in dtypes.items() if dtype.is_arrow or dtype.is_string}
        df.execute()
        progressbar_count(1)
        progressbar_strings(1)

        str_byte_length = {k: v.get() for k, v in str_byte_length.items()}
        has_null = {name: name in arrow_null_count and arrow_null_count[name].get() != N for name, dtype in dtypes.items()}
        has_mask = {name: df.is_masked(name) for name, dtype in dtypes.items() if not dtype.is_string}

        for i, name in enumerate(list(column_names)):
            dtype = dtypes[name]
            shape = (N, ) + df._shape_of(name)[1:]
            if dtype.is_string:
                self.column_writers[name] = ColumnWriterString(self.columns, name, dtypes[name], shape, str_byte_length[name], has_null[name])
            elif dtype.is_primitive or dtype.is_temporal:
                self.column_writers[name] = ColumnWriterPrimitive(self.columns, name, dtypes[name], shape, has_mask[name], has_null[name], self.byteorder)
            elif dtype.is_encoded:
                labels = df.category_labels(name)
                self.column_writers[name] = ColumnWriterDictionaryEncoded(self.columns, name, dtypes[name], labels, shape, has_null[name], self.byteorder, df)
            else:
                raise TypeError(f"Cannot export column of type: {dtype} (column {name})")
            progressbar_reserve((i+1)/len(column_names))
        self.columns.attrs["column_order"] = ",".join(column_names)

        # flush out the content
        self.h5.flush()
        self._layout_called = True

    def write(self, df, chunk_size=int(1e5), parallel=True, progress=None, column_count=1, export_threads=0):
        chunk_size = ((chunk_size + 7) // 8) * 8  # round up to multiple of 8
        assert self._layout_called, "call .layout() first"
        N = len(df)
        if N == 0:
            raise ValueError("Cannot export empty table")

        column_names = list(self.column_writers)
        # now that the file has the final size, we can mmap it
        self.file = open(self.path, "rb+")
        self.fileno = self.file.fileno()
        kwargs = {}
        if vaex.utils.osname == "windows":
            kwargs["access"] = mmap.ACCESS_READ | 0 if not self.write else mmap.ACCESS_WRITE
        else:
            kwargs["prot"] = mmap.PROT_READ | 0 if not self.write else mmap.PROT_WRITE
        self.mmap = mmap.mmap(self.fileno, 0, **kwargs)

        # and have all writers mmap the arrays
        for name in list(column_names):
            self.column_writers[name].mmap(self.mmap, self.file)
            self.column_writers[name].write_extra()

        logger.debug("writing columns(hdf5): %r" % column_names)
        # actual writing part
        progressbar = vaex.utils.progressbars(progress, title="write data")
        with progressbar:
            progressbar_columns = {k: progressbar.add(f"write: {k}") for k in column_names}
            if export_threads:
                pool = concurrent.futures.ThreadPoolExecutor(export_threads)
            for column_names_subgroup in vaex.itertools.chunked(column_names, column_count):
                expressions = [self.column_writers[name].expression for name in column_names_subgroup]
                for _i1, _i2, values in df.evaluate(expressions, chunk_size=chunk_size, filtered=True, parallel=parallel, progress=progressbar.hidden()):
                    pass
                    def write(arg):
                        i, name = arg
                        self.column_writers[name].write(values[i])
                        progressbar_columns[name](self.column_writers[name].progress)
                    if export_threads:
                        list(pool.map(write, enumerate(column_names_subgroup)))
                    else:
                        list(map(write, enumerate(column_names_subgroup)))


class ColumnWriterDictionaryEncoded:
    def __init__(self, h5parent, name, dtype, values, shape, has_null, byteorder="=", df=None):
        if has_null:
            raise ValueError('Encoded index got null values, this is not supported, only support null values in the values')
        self.dtype = dtype
        # make sure it's arrow
        values = self.dtype.value_type.create_array(values)
        # makes dealing with buffers easier
        self.values = vaex.arrow.convert.trim_buffers(values)
        self.byteorder = byteorder
        self.expression = df[name].index_values()
        self.h5group = h5parent.require_group(name)
        self.h5group.attrs["type"] = "dictionary_encoded"
        self.index_writer = ColumnWriterPrimitive(self.h5group, name="indices", dtype=self.dtype.index_type, shape=shape, has_null=has_null, has_mask=False, byteorder=byteorder)
        self._prepare_values()

    def _prepare_values(self):
        dtype_values = self.dtype.value_type
        name = "dictionary"
        shape = (len(self.values),)
        has_null = self.values.null_count > 0
        if dtype_values.is_string:
            str_byte_length = self.values.buffers()[2].size
            self.values_writer = ColumnWriterString(self.h5group, name, dtype_values, shape, str_byte_length, has_null)
        elif dtype_values.is_primitive or dtype_values.is_temporal:
            has_null = False
            self.values_writer = ColumnWriterPrimitive(self.h5group, name, dtype_values, shape, has_null, self.byteorder)
        else:
            raise TypeError(f"Cannot export column of type: {dtype_values}")

    @property
    def progress(self):
        return self.index_writer.progress

    def mmap(self, mmap, file):
        self.index_writer.mmap(mmap, file)
        self.values_writer.mmap(mmap, file)

    def write(self, values):
        self.index_writer.write(values)

    def write_extra(self):
        self.values_writer.write(self.values)


class ColumnWriterPrimitive:
    def __init__(self, h5parent, name, dtype, shape, has_mask, has_null, byteorder="="):
        self.h5parent = h5parent
        self.name = name
        self.shape = shape
        self.count = self.shape[0]
        self.dtype = dtype
        self.to_offset = 0
        self.to_array = None
        self.expression = name
        self.has_mask = has_mask
        self.has_null = has_null
        if self.has_mask and self.has_null:
            raise ValueError("Cannot have both mask and null")

        self.h5group = h5parent.require_group(name)
        if dtype.kind in 'mM':
            self.array = self.h5group.require_dataset('data', shape=shape, dtype=np.int64, track_times=False)
            self.array.attrs["dtype"] = dtype.name
        elif dtype.kind == 'U':
            # numpy uses utf32 for unicode
            char_length = dtype.itemsize // 4
            shape = (N, char_length)
            self.array = self.h5group.require_dataset('data', shape=shape, dtype=np.uint8, track_times=False)
            self.array.attrs["dtype"] = 'utf32'
            self.array.attrs["dlength"] = char_length
        else:
            self.array = self.h5group.require_dataset('data', shape=shape, dtype=dtype.numpy.newbyteorder(byteorder), track_times=False)
        self.array[0] = self.array[0]  # make sure the array really exists

        self.null_bitmap_array = None
        if self.has_null:
            null_shape = ((self.count + 7) // 8,)  # TODO: arrow requires padding right?
            self.null_bitmap_array = self.h5group.require_dataset("null_bitmap", shape=null_shape, dtype="u1", track_times=False)
            self.null_bitmap_array[0] = self.null_bitmap_array[0]  # make sure the array really exists
        if self.has_mask:
            self.mask = self.h5group.require_dataset('mask', shape=shape, dtype=bool, track_times=False)
            self.mask[0] = self.mask[0]  # make sure the array really exists
        else:
            self.mask = None

    @property
    def progress(self):
        return self.to_offset/self.count

    def mmap(self, mmap, file):
        self.to_array = h5mmap(mmap if USE_MMAP else None, file, self.array, self.mask)
        if self.null_bitmap_array is not None:
            self.null_bitmap_array = h5mmap(mmap, file, self.null_bitmap_array)

    def write(self, values):
        no_values = len(values)
        if no_values:
            fill_value = np.nan if self.dtype.kind == "f" else None
            target_set_item = slice(self.to_offset, self.to_offset + no_values)
            if vaex.array_types.is_arrow_array(values):
                values = vaex.arrow.convert.ensure_not_chunked(values)
            if self.dtype.kind in 'mM':
                if vaex.array_types.is_arrow_array(values):
                    values = values.view(pa.int64())
                else:
                    values = values.view(np.int64)
            if self.has_null:
                byte_index1 = self.to_offset // 8  # floor it
                byte_index2 = (self.to_offset + len(values) + 7) // 8  # ceil it
                if self.to_offset != byte_index1 * 8:
                    raise ValueError("Cannot write to non-byte aligned offset")
                null_buffer = values.buffers()[0]
                if null_buffer is not None:
                    self.null_bitmap_array[byte_index1:byte_index2] = memoryview(null_buffer[:byte_index2-byte_index1])
                else:
                    self.null_bitmap_array[byte_index1:byte_index2] = 0xff
            if np.ma.isMaskedArray(self.to_array) and np.ma.isMaskedArray(values):
                assert self.has_mask
                self.to_array.data[target_set_item] = values.filled(fill_value)
                self.to_array.mask[target_set_item] = values.mask
            elif not np.ma.isMaskedArray(self.to_array) and np.ma.isMaskedArray(values):
                self.to_array[target_set_item] = values.filled(fill_value)
            else:
                self.to_array[target_set_item] = values
            self.to_offset += no_values
            assert self.to_offset <= self.count

    def write_extra(self):
        pass


class ColumnWriterString:
    def __init__(self, h5parent, name, dtype, shape, byte_length, has_null):
        self.h5parent = h5parent
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.count = self.shape[0]
        self.h5group = h5parent.require_group(name)
        self.has_null = has_null
        self.to_offset = 0
        self.string_byte_offset = 0
        self.expression = name
        # TODO: if no selection or filter, we could do this
        # if isinstance(column, ColumnStringArrow):
        #     data_shape = column.bytes.shape
        #     indices_shape = column.indices.shape
        # else:

        if byte_length > max_int32:
            dtype_indices = 'i8'
        else:
            dtype_indices = 'i4'

        data_shape = (byte_length, )
        indices_shape = (self.count+1, )

        self.array = self.h5group.require_dataset('data', shape=data_shape, dtype='S1', track_times=False)
        self.array.attrs["dtype"] = 'str'
        if byte_length > 0:
            self.array[0] = self.array[0]  # make sure the array really exists

        self.index_array = self.h5group.require_dataset('indices', shape=indices_shape, dtype=dtype_indices, track_times=False)
        self.index_array[0] = self.index_array[0]  # make sure the array really exists

        if self.has_null > 0:
            null_shape = ((self.count + 7) // 8, )  # TODO: arrow requires padding right?
            self.null_bitmap_array = self.h5group.require_dataset('null_bitmap', shape=null_shape, dtype='u1', track_times=False)
            self.null_bitmap_array[0] = self.null_bitmap_array[0]  # make sure the array really exists
        else:
            self.null_bitmap_array = None
        # TODO: masked support ala arrow?

    @property
    def progress(self):
        return self.to_offset/self.count


    def mmap(self, mmap, file):
        # from now on, we only work with the mmapped array
        # we cannot support USE_MMAP=False for strings yet
        self.array = h5mmap(mmap, file, self.array)
        self.index_array = h5mmap(mmap, file, self.index_array)
        if self.null_bitmap_array is not None:
            self.null_bitmap_array = h5mmap(mmap, file, self.null_bitmap_array)
        if isinstance(self.index_array, np.ndarray):  # this is a real mmappable file
            self.to_array = vaex.arrow.convert.arrow_string_array_from_buffers(self.array, self.index_array, self.null_bitmap_array)
        else:
            self.to_array = ColumnStringArrow(self.index_array, self.array, null_bitmap=self.null_bitmap_array)
        # if not isinstance(to_array, ColumnStringArrow):
        self.to_array = ColumnStringArrow.from_arrow(self.to_array)
        # assert isinstance(to_array, pa.Array)  # we don't support chunked arrays here
        # TODO legacy: we still use ColumnStringArrow to write, find a way to do this with arrow
        # this is the case with hdf5 and remote storage

    def write(self, values):
        no_values = len(values)
        if no_values:
            # to_column = to_array
            from_sequence = _to_string_sequence(values)
            to_sequence = self.to_array.string_sequence.slice(self.to_offset, self.to_offset+no_values, self.string_byte_offset)
            self.string_byte_offset += to_sequence.fill_from(from_sequence)
            self.to_offset += no_values
        if self.to_offset == self.count:
            # last offset
            self.to_array.indices[self.count] = self.string_byte_offset

    def write_extra(self):
        pass
