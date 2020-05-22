"""Convert between arrow and vaex/numpy columns/arrays without doing memory copies."""
import pyarrow
import numpy as np
from vaex.column import ColumnStringArrow

def arrow_array_from_numpy_array(array):
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


def column_from_arrow_array(arrow_array):
    arrow_type = arrow_array.type
    buffers = arrow_array.buffers()
    if len(buffers) == 2:
        return numpy_array_from_arrow_array(arrow_array)
    elif len(buffers) == 3 and  isinstance(arrow_array.type, type(pyarrow.string())):
        bitmap_buffer, offsets, string_bytes = arrow_array.buffers()
        if arrow_array.null_count == 0:
            null_bitmap = None  # we drop any null_bitmap when there are no null counts
        else:
            null_bitmap = np.frombuffer(bitmap_buffer, 'uint8', len(bitmap_buffer))
        offsets = np.frombuffer(offsets, np.int32, len(offsets)//4)
        if string_bytes is None:
            string_bytes = np.array([], dtype='S1')
        else:
            string_bytes = np.frombuffer(string_bytes, 'S1', len(string_bytes))
        column = ColumnStringArrow(offsets, string_bytes, len(arrow_array), null_bitmap=null_bitmap)
        return column
    else:
        raise TypeError('type unsupported: %r' % arrow_type)


def numpy_array_from_arrow_array(arrow_array):
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
