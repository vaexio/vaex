import numpy as np
from vaex.file.column import ColumnFile

from vaex.column import ColumnNumpyLike, ColumnMaskedNumpy


# these helper functions are quite similar to the dataset methods
def mmap_array(mmap, file, offset, dtype, shape):
    length = np.product(shape)
    if mmap is None:
        if len(shape) > 1:
            raise RuntimeError('not supported, high d arrays from non local files')
        return ColumnFile(file, offset, length, dtype, write=True, tls=None)
    else:
        array = np.frombuffer(mmap, dtype=dtype, count=length, offset=offset)
        return array.reshape(shape)

def h5mmap(mmap, file, data, mask=None):
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
        if mmap is not None:
            assert offset + len(data)*dtype.itemsize <= len(mmap)
        if "dtype" in data.attrs:
            # ignore the special str type, which is not a numpy dtype
            if data.attrs["dtype"] != "str":
                dtype = data.attrs["dtype"]
                if dtype == 'utf32':
                    dtype = np.dtype('U' + str(data.attrs['dlength']))
        array = mmap_array(mmap, file, offset, dtype=dtype, shape=shape)
        if mask is not None:
            mask_array = h5mmap(mmap, file, mask)
            if isinstance(array, np.ndarray):
                ar = np.ma.array(array, mask=mask_array, shrink=False)
                # assert ar.mask is mask_array, "masked array was copied"
            else:
                ar = ColumnMaskedNumpy(array, mask_array)
            return ar
        else:
            return array
