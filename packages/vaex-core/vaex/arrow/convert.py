"""Convert between arrow and vaex/numpy columns/arrays without doing memory copies.

We eventually like to get rid of this, and upstream all to pyarrow.
"""
import pyarrow
import pyarrow as pa
import numpy as np
import vaex.column


def ensure_not_chunked(arrow_array):
    if isinstance(arrow_array, pa.ChunkedArray):
        if len(arrow_array.chunks) == 0:
            return arrow_array.chunks[0]
        table = pa.Table.from_arrays([arrow_array], ["single"])
        table_concat = table.combine_chunks()
        column = table_concat.columns[0]
        assert column.num_chunks == 1
        arrow_array = column.chunk(0)
    return arrow_array


def arrow_array_from_numpy_array(array):  # TODO: -=> rename with ensure
    if isinstance(array, vaex.column.ColumnString):
        # we make sure it's ColumnStringArrow
        array = vaex.column._to_string_column(array)
        if len(array.bytes) < (2**31):  # arrow uses signed ints
            # if possible, downcast (parquet does not support large strings)
            array = array.to_arrow(pa.string())
        else:
            array = array.to_arrow()
    if isinstance(array, (pa.Array, pa.ChunkedArray)):
        return array
    dtype = array.dtype
    mask = None
    if np.ma.isMaskedArray(array):
        mask = array.mask
        # arrow 0.16 behaves weird in this case https://github.com/vaexio/vaex/pull/639
        if mask is np.False_:
            mask = None
        elif mask is np.True_:
            raise ValueError('not sure what pyarrow does with mask=True')
        array = array.data
    if dtype.kind == 'S':
        type = pyarrow.binary(dtype.itemsize)
        arrow_array = pyarrow.array(array, type, mask=mask)
    else:
        if not dtype.isnative:
            array = array.astype(dtype.newbyteorder('='))
        arrow_array = pyarrow.Array.from_pandas(array, mask=mask)
    return arrow_array

from vaex.dataframe import Column


def arrow_string_array_from_buffers(bytes, offsets, null_bitmap):
    if offsets.dtype == np.int32:
        type = pa.string()
    elif offsets.dtype == np.int64:
        type = pa.large_string()
    else:
        raise ValueError(f'Unsupported dtype {offsets.dtype} for string offsets')
    return _arrow_binary_array_from_buffers(bytes, offsets, null_bitmap, type)


def _arrow_binary_array_from_buffers(bytes, offsets, null_bitmap, type):
    length = len(offsets)-1
    offsets = pa.py_buffer(offsets)
    bytes = pa.py_buffer(bytes)
    if null_bitmap is not None:
        null_bitmap = pa.py_buffer(null_bitmap)
    return pa.Array.from_buffers(type, length, [null_bitmap, offsets, bytes])


def column_from_arrow_array(arrow_array):
    # TODO: we may be able to pass chunked arrays
    arrow_array = ensure_not_chunked(arrow_array)
    arrow_type = arrow_array.type
    buffers = arrow_array.buffers()
    if len(buffers) == 2:
        return numpy_array_from_arrow_array(arrow_array)
    elif len(buffers) == 3 and arrow_array.type in [pyarrow.string(), pyarrow.large_string()]:
        bitmap_buffer, offsets, string_bytes = arrow_array.buffers()
        if arrow_array.null_count == 0:
            null_bitmap = None  # we drop any null_bitmap when there are no null counts
        else:
            null_bitmap = np.frombuffer(bitmap_buffer, 'uint8', len(bitmap_buffer))
        if arrow_array.type == pyarrow.string():
            offsets = np.frombuffer(offsets, np.int32, len(offsets)//4)
        else:
            offsets = np.frombuffer(offsets, np.int64, len(offsets)//8)
        if string_bytes is None:
            string_bytes = np.array([], dtype='S1')
        else:
            string_bytes = np.frombuffer(string_bytes, 'S1', len(string_bytes))
        offset = arrow_array.offset
        column = vaex.column.ColumnStringArrow(offsets[offset:], string_bytes, len(arrow_array), null_bitmap=null_bitmap)
        return column
    else:
        raise TypeError('type unsupported: %r' % arrow_type)


def numpy_array_from_arrow_array(arrow_array):
    arrow_array = ensure_not_chunked(arrow_array)
    arrow_type = arrow_array.type
    buffers = arrow_array.buffers()
    assert len(buffers) == 2
    bitmap_buffer, data_buffer = buffers
    offset = arrow_array.offset
    if isinstance(arrow_type, type(pyarrow.binary(1))):  # todo, is there a better way to typecheck?
        # mimics python/pyarrow/array.pxi::Array::to_numpy
        assert len(buffers) == 2
        dtype = "S" + str(arrow_type.byte_width)
        # arrow seems to do padding, check if it is all ok
        expected_length = arrow_type.byte_width * len(arrow_array)
        actual_length = len(buffers[-1])
        if actual_length < expected_length:
            raise ValueError('buffer is smaller (%d) than expected (%d)' % (actual_length, expected_length))
        array = np.frombuffer(buffers[-1], dtype, len(arrow_array))# TODO: deal with offset ? [arrow_array.offset:arrow_array.offset + len(arrow_array)]
    else:
        dtype = arrow_array.type.to_pandas_dtype()
    if np.bool_ == dtype:
        # TODO: this will also be a copy, we probably want to support bitmasks as well
        bitmap = np.frombuffer(data_buffer, np.uint8, len(data_buffer))
        array = numpy_bool_from_arrow_bitmap(bitmap, len(arrow_array) + offset)[offset:]
    else:
        array = np.frombuffer(data_buffer, dtype, len(arrow_array) + offset)[offset:]

    if bitmap_buffer is not None:
        bitmap = np.frombuffer(bitmap_buffer, np.uint8, len(bitmap_buffer))
        mask = numpy_mask_from_arrow_mask(bitmap, len(arrow_array) + offset)[offset:]
        array = np.ma.MaskedArray(array, mask=mask)
    return array


def numpy_mask_from_arrow_mask(bitmap, length):
    # arrow uses a bitmap https://github.com/apache/arrow/blob/master/format/Layout.md
    # we do have to change the ordering of the bits
    return 1-np.unpackbits(bitmap).reshape((len(bitmap),8))[:,::-1].reshape(-1)[:length]


def numpy_bool_from_arrow_bitmap(bitmap, length):
    # arrow uses a bitmap https://github.com/apache/arrow/blob/master/format/Layout.md
    # we do have to change the ordering of the bits
    return np.unpackbits(bitmap).reshape((len(bitmap),8))[:,::-1].reshape(-1)[:length].view(np.bool_)


def arrow_table_from_vaex_df(ds, column_names=None, selection=None, strings=True, virtual=False):
    """Implementation of Dataset.to_arrow_table"""
    names = []
    arrays = []
    for name, array in ds.to_items(column_names=column_names, selection=selection, strings=strings, virtual=virtual):
        names.append(name)
        arrays.append(arrow_array_from_numpy_array(array))
    return pyarrow.Table.from_arrays(arrays, names)


def vaex_df_from_arrow_table(table):
    from .dataset import DatasetArrow
    return DatasetArrow(table=table)
