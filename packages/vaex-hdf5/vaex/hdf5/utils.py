import os
from typing import List

import h5py
import numpy as np
import vaex
from vaex.file.column import ColumnFile
from vaex.arrow.convert import arrow_string_array_from_buffers as convert_bytes

from vaex.column import ColumnNumpyLike, ColumnMaskedNumpy
from .hdf5_store import HDF5Store, HDF5_STORE


# these helper functions are quite similar to the dataset methods
def mmap_array(mmap, file, offset, dtype, shape):
    if np.lib.NumpyVersion(np.__version__) >= '1.25.0':
        length = np.prod(shape)
    else:
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


def concat_hdf5_files(location: str) -> List[str]:
    """Concatenates all hdf5 in a directory using an HDF5 store

    Vaex stores a dataframe as an hdf5 file in a predictable format using groups

    Each column gets its own group, following "/table/columns/{col}/data

    Create a group per column, appending to each group as we iterate through the files

    :param location: The directory containing the files
    """
    str_cols = []
    stores = {}
    files = os.listdir(location)
    df = vaex.open(f"{location}/{files[0]}")

    # Construct a store (group) per column
    cols = df.get_column_names()
    for col in cols:
        group = f"/table/columns/{col}/data"
        cval = df[col].to_numpy()
        if cval.ndim == 2:
            shape = cval[0].shape
        else:
            shape = ()
        dtype = df[col].dtype.numpy
        if not np.issubdtype(dtype, np.number):
            dtype = h5py.string_dtype(encoding="utf-8")
            str_cols.append(col)
        stores[col] = HDF5Store(f"{location}/{HDF5_STORE}", group, shape, dtype=dtype)

    for file in files:
        fname = f"{location}/{file}"
        with h5py.File(fname, "r") as f:
            dset = f["table"]["columns"]
            keys = dset.keys()
            keys = [key for key in keys if key in cols]
            for key in keys:
                col_data = dset[key]
                # We have a string column, need to parse it
                # TODO: This isn't optimal. We end up creating a bytearray instead
                #  of vaex's string/index structure. Fix this
                if "indices" in col_data.keys():
                    assert key in str_cols, f"Unexpected string column ({key}) found"
                    indcs = col_data["indices"][:]
                    data = col_data["data"][:]
                    d = convert_bytes(data, indcs, None).to_numpy(zero_copy_only=False)
                else:
                    d = col_data["data"][:]
                stores[key].append(d)
        os.remove(fname)
    return str_cols
