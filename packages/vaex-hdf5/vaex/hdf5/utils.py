import numpy as np

from vaex.column import ColumnNumpyLike


# these helper functions are quite similar to the dataset methods
def mmap_array(mmap, offset, dtype, length):
    return np.frombuffer(mmap, dtype=dtype, count=length, offset=offset)

def h5mmap(mmap, data, mask=None):
    offset = data.id.get_offset()
    if len(data) == 0 and offset is None:
        offset = 0 # we don't care about the offset for empty arrays
    if offset is None:  # non contiguous array, chunked arrays etc
        # we don't support masked in this case
        column = ColumnNumpyLike(data)
        return column
    else:
        shape = data.shape
        dtype = data.dtype
        assert offset + len(data)*dtype.itemsize <= len(mmap)
        if "dtype" in data.attrs:
            # ignore the special str type, which is not a numpy dtype
            if data.attrs["dtype"] != "str":
                dtype = data.attrs["dtype"]
                if dtype == 'utf32':
                    dtype = np.dtype('U' + str(data.attrs['dlength']))
        #self.addColumn(column_name, offset, len(data), dtype=dtype)
        array = mmap_array(mmap, offset, dtype=dtype, length=len(data))
        if mask is not None:
            mask_array = h5mmap(mmap, mask)
            if isinstance(array, np.ndarray):
                ar = np.ma.array(array, mask=mask_array, shrink=False)
                # assert ar.mask is mask_array, "masked array was copied"
            else:
                ar = vaex.column.ColumnMaskedNumpy(array, mask_array)
            return ar
        else:
            return array
