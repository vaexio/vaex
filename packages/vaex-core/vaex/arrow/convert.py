"""Convert between arrow and vaex/numpy columns/arrays without doing memory copies.

We eventually like to get rid of this, and upstream all to pyarrow.
"""
from typing import Union
import pyarrow
import pyarrow as pa
import numpy as np
import vaex.column


def ensure_not_chunked(arrow_array):
    if isinstance(arrow_array, pa.ChunkedArray):
        if len(arrow_array.chunks) == 1:
            return arrow_array.chunks[0]
        table = pa.Table.from_arrays([arrow_array], ["single"])
        table_concat = table.combine_chunks()
        column = table_concat.columns[0]
        if column.num_chunks == 1:
            arrow_array = column.chunk(0)
        else:
            assert column.num_chunks == 0
            arrow_array = pa.array([], type=arrow_array.type)
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
        column = vaex.column.ColumnStringArrow(offsets, string_bytes, len(arrow_array), offset, null_bitmap=null_bitmap)
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
        dtype = vaex.array_types.to_numpy_type(arrow_array.type)
    if np.bool_ == dtype:
        # TODO: this will also be a copy, we probably want to support bitmasks as well
        bitmap = np.frombuffer(data_buffer, np.uint8, len(data_buffer))
        array = numpy_bool_from_arrow_bitmap(bitmap, len(arrow_array) + offset)[offset:]
    else:
        array = np.frombuffer(data_buffer, dtype, len(arrow_array) + offset)[offset:]

    if bitmap_buffer is not None and arrow_array.null_count > 0:
        bitmap = np.frombuffer(bitmap_buffer, np.uint8, len(bitmap_buffer))
        mask = numpy_mask_from_arrow_mask(bitmap, len(arrow_array) + offset)[offset:]
        array = np.ma.MaskedArray(array, mask=mask)
    assert len(array) == len(arrow_array)
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


def trim_offsets(offset, length, null_buffer, offsets_buffer, large=False):
    if offset == 0:
        return null_buffer, offsets_buffer
    if large:
        offsets = np.frombuffer(offsets_buffer, np.int64, length + 1 + offset)
    else:
        offsets = np.frombuffer(offsets_buffer, np.int32, length + 1 + offset)
    nulls = pa.BooleanArray.from_buffers(pa.bool_(), length, [None, null_buffer], offset=offset)
    nulls = pa.concat_arrays([nulls])
    assert nulls.offset == 0
    assert len(nulls) == length
    offsets = offsets[offset:] - offsets[offset]
    return nulls.buffers()[1], pa.py_buffer(offsets)


def trim_buffers(ar):
    # there are cases where memcopy are made, of modifications are mode (large_string_to_string)
    # in those cases, we don't want to work on the full array, and get rid of the offset if possible
    if ar.type == pa.string() or ar.type == pa.large_string():
        if isinstance(ar, pa.ChunkedArray):
            return ar  # lets assume chunked arrays are fine
        null_bitmap, offsets_buffer, bytes = ar.buffers()
        if ar.type == pa.string():
            offsets = np.frombuffer(offsets_buffer, np.int32, len(ar) + 1 + ar.offset)
        else:
            offsets = np.frombuffer(offsets_buffer, np.int64, len(ar) + 1 + ar.offset)
        # because it is difficult to slice bits
        new_offset = ar.offset % 8
        remove_offset = (ar.offset // 8) * 8
        first_offset = offsets[remove_offset]
        new_offsets = offsets[remove_offset:] - first_offset
        if null_bitmap:
            null_bitmap = null_bitmap.slice(ar.offset // 8)
        new_offsets_buffer = pa.py_buffer(new_offsets)
        bytes = bytes.slice(first_offset)
        ar = pa.Array.from_buffers(ar.type, len(ar), [null_bitmap, new_offsets_buffer, bytes], offset=new_offset)
    return ar


def large_string_to_string(ar):
    if isinstance(ar, pa.ChunkedArray):
        return pa.chunked_array([large_string_to_string(k) for k in ar.chunks], type=pa.string())
    ar = trim_buffers(ar)
    offset = ar.offset
    null_bitmap, offsets_buffer, bytes = ar.buffers()
    offsets = np.frombuffer(offsets_buffer, np.int64, len(ar)+1 + ar.offset)
    if offsets[-1] > (2**31-1):
        raise ValueError('pa.large_string cannot be converted to pa.string')
    offsets = offsets.astype(np.int32)
    offsets_buffer = pa.py_buffer(offsets)
    return pa.Array.from_buffers(pa.string(), len(ar), [null_bitmap, offsets_buffer, bytes], offset=ar.offset)


def trim_buffers_ipc(ar):
    '''
    >>> ar = pa.array([1, 2, 3, 4], pa.int8())
    >>> ar.nbytes
    4
    >>> ar.slice(2, 2) #doctest: +ELLIPSIS
    <pyarrow.lib.Int8Array object at 0x...>
    [
      3,
      4
    ]
    >>> ar.slice(2, 2).nbytes
    4
    >>> trim_buffers_ipc(ar.slice(2, 2)).nbytes  # expected 1
    2
    >>> trim_buffers_ipc(ar.slice(2, 2))#doctest: +ELLIPSIS
    <pyarrow.lib.Int8Array object at 0x...>
    [
      3,
      4
    ]
    '''
    if len(ar) == 0:
        return ar
    schema = pa.schema({'x': ar.type})
    with pa.BufferOutputStream() as sink:
        with pa.ipc.new_stream(sink, schema) as writer:
            writer.write_table(pa.table({'x': ar}))
    with pa.BufferReader(sink.getvalue()) as source:
        with pa.ipc.open_stream(source) as reader:
            table = reader.read_all()
            assert table.num_columns == 1
            assert table.num_rows == len(ar)
            trimmed_ar = table.column(0)
    if isinstance(trimmed_ar, pa.ChunkedArray) and len(trimmed_ar.chunks) == 1:
        trimmed_ar = trimmed_ar.chunks[0]

    return trimmed_ar


def trim_buffers_for_pickle(ar):
    # future version of pyarrow might fix this, so we have a single entry point for this
    return trim_buffers_ipc(ar)


def align(a, b):
    '''Align two arrays/chunked arrays such that they have the same chunk lengths'''
    if len(a) != len(b):
        raise ValueError(f'Length of arrays should be equals ({len(a)} != {len(b)})')
    if isinstance(a, pa.ChunkedArray) and len(a.chunks) == 1:
        a = a.chunks[0]
    if isinstance(b, pa.ChunkedArray) and len(b.chunks) == 1:
        b = b.chunks[0]
    if isinstance(a, pa.ChunkedArray) and isinstance(b, pa.ChunkedArray):
        lengths_a = [len(c) for c in a.chunks]
        lengths_b = [len(c) for c in b.chunks]
        if lengths_a == lengths_b:
            return a, b
        else:
            return ensure_not_chunked(a), ensure_not_chunked(b)
    elif isinstance(a, pa.ChunkedArray) and not isinstance(b, pa.ChunkedArray):
        lengths = [len(c) for c in a.chunks]
        # numpy cannot do an exclusive cumsum
        offsets = np.cumsum([0] + lengths)
        offsets = offsets[:-1]
        return a, pa.chunked_array([b.slice(offset, length) for offset, length in zip(offsets, lengths)])
    elif not isinstance(a, pa.ChunkedArray) and isinstance(b, pa.ChunkedArray):
        b, a = align(b, a)
        return a, b
    else:
        return a, b
    return


def same_type(*arrays):
    types = [ar.type for ar in arrays]
    if any(types[0] != type for type in types):
        if vaex.dtype(types[0]) == str:
            # we have mixed large and normal string
            return [large_string_to_string(ar) if ar.type == pa.large_string() else ar for ar in arrays]
        else:
            raise NotImplementedError
    return arrays


def list_from_arrays(offsets, values) -> Union[pa.LargeListArray, pa.ListArray]:
    import vaex
    import vaex.array_types
    values_arrow = vaex.array_types.to_arrow(values)
    dtype_offsets = vaex.dtype_of(offsets)
    dtype_values = vaex.dtype_of(values_arrow)
    if dtype_offsets.is_integer:
        if dtype_offsets.numpy.itemsize == 4:
            arrow_type = pa.list_(dtype_values.arrow)
        elif dtype_offsets.numpy.itemsize == 8:
            arrow_type = pa.large_list(dtype_values.arrow)
        else:
            raise TypeError('Indices should be int32 or int64, not {dtype_offsets}')
    else:
        raise TypeError('Indices should be integer type, not {dtype_offsets}')

    null_buffer = None
    offsets_buffer = pa.py_buffer(offsets)
    return pa.Array.from_buffers(arrow_type, len(offsets) - 1, [null_buffer, offsets_buffer], offset=0, children=[values_arrow])
