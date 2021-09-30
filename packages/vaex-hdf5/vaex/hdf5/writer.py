import mmap
import logging
import concurrent.futures

import h5py
import numpy as np

import vaex
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
        if self.mmap is not None:
            self.mmap.close()
            self.file.close()
        self.h5.close()
        
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def layout(self, df):
        assert not self._layout_called, "Layout called twice"
        N = len(df)
        if N == 0:
            raise ValueError("Cannot layout empty table")
        column_names = df.get_column_names()

        logger.debug("layout columns(hdf5): %r" % column_names)

        self.column_writers = {}
        dtypes = df.schema()
        str_byte_length = {name:df[name].str.byte_length().sum(delay=True) for name, dtype in dtypes.items() if dtype.is_string}
        str_count = {name:df.count(df[name], delay=True) for name, dtype in dtypes.items() if dtype.is_string}
        df.execute()

        str_byte_length = {k: v.get() for k, v in str_byte_length.items()}
        has_null_str = {k: N != v.get() for k, v in str_count.items()}
        has_null = {name:df.is_masked(name) for name, dtype in dtypes.items() if not dtype.is_string}

        for name in list(column_names):
            dtype = dtypes[name]

            shape = (N, ) + df._shape_of(name)[1:]
            try:
                if dtype.is_string:
                    self.column_writers[name] = ColumnWriterString(self.columns, name, dtypes[name], shape, str_byte_length[name], has_null_str[name])
                else:
                    self.column_writers[name] = ColumnWriterPrimitive(self.columns, name, dtypes[name], shape, has_null[name], self.byteorder)
            except:
                logger.exception("error creating dataset for %r, with type %r " % (name, dtype))
                del self.columns[name]
                column_names.remove(name)
        self.columns.attrs["column_order"] = ",".join(column_names)

        # flush out the content
        self.h5.flush()
        self._layout_called = True

    def write(self, df, chunk_size=int(1e5), parallel=True, progress=None, column_count=1, export_threads=0):
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

        logger.debug("writing columns(hdf5): %r" % column_names)
        # actual writing part
        progressbar = vaex.utils.progressbars(progress)
        progressbar(0)
        total = N * len(column_names)
        written = 0
        if export_threads:
            pool = concurrent.futures.ThreadPoolExecutor(export_threads)
        for column_names_subgroup in vaex.itertools.chunked(column_names, column_count):
            for i1, i2, values in df.evaluate(column_names_subgroup, chunk_size=chunk_size, filtered=True, parallel=parallel, array_type='numpy-arrow'):
                def write(arg):
                    i, name = arg
                    self.column_writers[name].write(values[i])
                # for i, name in enumerate(column_names_subgroup):
                if export_threads:
                    list(pool.map(write, enumerate(column_names_subgroup)))
                else:
                    list(map(write, enumerate(column_names_subgroup)))
                written += (i2 - i1) * len(column_names_subgroup)
                progressbar(written/total)
        progressbar(1.0)


class ColumnWriterPrimitive:
    def __init__(self, h5parent, name, dtype, shape, has_null, byteorder="="):
        self.h5parent = h5parent
        self.name = name
        self.shape = shape
        self.count = self.shape[0]
        self.dtype = dtype
        self.to_offset = 0
        self.to_array = None

        self.h5group = h5parent.require_group(name)
        if dtype.kind in 'mM':
            self.array = self.h5group.require_dataset('data', shape=shape, dtype=np.int64)
            self.array.attrs["dtype"] = dtype.name
        elif dtype.kind == 'U':
            # numpy uses utf32 for unicode
            char_length = dtype.itemsize // 4
            shape = (N, char_length)
            self.array = self.h5group.require_dataset('data', shape=shape, dtype=np.uint8)
            self.array.attrs["dtype"] = 'utf32'
            self.array.attrs["dlength"] = char_length
        else:
            self.array = self.h5group.require_dataset('data', shape=shape, dtype=dtype.numpy.newbyteorder(byteorder))
        self.array[0] = self.array[0]  # make sure the array really exists

        if has_null:
            self.mask = self.h5group.require_dataset('mask', shape=shape, dtype=np.bool)
            self.mask[0] = self.mask[0]  # make sure the array really exists
        else:
            self.mask = None

    def mmap(self, mmap, file):
        self.to_array = h5mmap(mmap if USE_MMAP else None, file, self.array, self.mask)

    def write(self, values):
        no_values = len(values)
        if no_values:
            fill_value = np.nan if self.dtype.kind == "f" else None
            target_set_item = slice(self.to_offset, self.to_offset + no_values)
            if self.dtype.kind in 'mM':
                values = values.view(np.int64)
            if np.ma.isMaskedArray(self.to_array) and np.ma.isMaskedArray(values):
                self.to_array.data[target_set_item] = values.filled(fill_value)
                self.to_array.mask[target_set_item] = values.mask
            elif not np.ma.isMaskedArray(self.to_array) and np.ma.isMaskedArray(values):
                self.to_array[target_set_item] = values.filled(fill_value)
            else:
                self.to_array[target_set_item] = values
            self.to_offset += no_values
            assert self.to_offset <= self.count


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

        self.array = self.h5group.require_dataset('data', shape=data_shape, dtype='S1')
        self.array.attrs["dtype"] = 'str'
        if byte_length > 0:
            self.array[0] = self.array[0]  # make sure the array really exists

        self.index_array = self.h5group.require_dataset('indices', shape=indices_shape, dtype=dtype_indices)
        self.index_array[0] = self.index_array[0]  # make sure the array really exists

        if self.has_null > 0:
            null_shape = ((self.count + 7) // 8, )  # TODO: arrow requires padding right?
            self.null_bitmap_array = self.h5group.require_dataset('null_bitmap', shape=null_shape, dtype='u1')
            self.null_bitmap_array[0] = self.null_bitmap_array[0]  # make sure the array really exists
        else:
            self.null_bitmap_array = None
        # TODO: masked support ala arrow?


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
