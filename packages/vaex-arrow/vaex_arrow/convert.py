import pyarrow
import numpy as np

def arrow_array_from_numpy_array(array):
    dtype = array.dtype
    mask = None
    if np.ma.isMaskedArray(array):
        mask = array.mask
    if dtype.kind == 'S':
        type = pyarrow.binary(dtype.itemsize)
        arrow_array = pyarrow.array(array, type, mask=mask)
    else:
        if dtype.isnative:
            arrow_array = pyarrow.array(array, mask=mask)
        else:
            # TODO: we copy here, but I guess we should not... or give some warning
            arrow_array = pyarrow.array(array.astype(dtype.newbyteorder('=')), mask=mask)
    return arrow_array

def numpy_array_from_arrow_array(arrow_array):
    arrow_type = arrow_array.type
    buffers = arrow_array.buffers()
    assert len(buffers) == 2
    bitmap_buffer = buffers[0]
    data_buffer = buffers[1]
    if isinstance(arrow_type, type(pyarrow.binary(1))):  # todo, is there a better way to typecheck?
        # mimics python/pyarrow/array.pxi::Array::to_numpy
        buffers = arrow_array.buffers()
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
    array = np.frombuffer(data_buffer, dtype, len(arrow_array))
    if bitmap_buffer is not None:
        # arrow uses a bitmap https://github.com/apache/arrow/blob/master/format/Layout.md
        bitmap = np.frombuffer(bitmap_buffer, np.uint8, len(bitmap_buffer))
        # we do have to change the ordering of the bits
        mask = 1-np.unpackbits(bitmap).reshape((len(bitmap),8))[:,::-1].reshape(-1)[:len(arrow_array)]
        array = np.ma.MaskedArray(array, mask=mask)
    return array


def arrow_table_from_vaex_dataset(ds, column_names=None, selection=None, strings=True, virtual=False):
    """Implementation of Dataset.to_arrow_table"""
    names = []
    arrays = []
    for name, array in ds.to_items(column_names=column_names, selection=selection, strings=strings, virtual=virtual):
        names.append(name)
        arrays.append(arrow_array_from_numpy_array(array))
    return pyarrow.Table.from_arrays(arrays, names)

def vaex_dataset_from_arrow_table(table):
    from .dataset import DatasetArrow
    return DatasetArrow(table=table)
