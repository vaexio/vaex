from typing import Optional, Tuple, Type

import h5py
import numpy as np

HDF5_STORE = "hdf5_store.hdf5"


class HDF5Store(object):
    """
    Simple class to append value to a hdf5 file on disc.

    Used to concatenate HDF5 files for vaex

    Params:
        datapath: filepath of h5 file
        dataset: dataset name within the file
        shape: dataset shape (not counting main/batch axis)
        dtype: numpy dtype

    Usage:
        hdf5_store = HDF5Store('/tmp/hdf5_store.h5','X', shape=(20,20,3))
        x = np.random.random(hdf5_store.shape)
        hdf5_store.append(x)
        hdf5_store.append(x)

    Adapted From https://gist.github.com/wassname/a0a75f133831eed1113d052c67cf8633
    and modified to work better with Vaex hdf5 structures
    """

    def __init__(
        self,
        datapath: str,
        dataset: str,
        shape: Tuple,
        dtype: Type = np.float32,
        compression: Optional[str] = None,
        chunk_len: int = 1,
    ):
        self.datapath = datapath
        self.dataset = dataset
        self.shape = shape
        self.i = 0

        with h5py.File(self.datapath, mode="a") as h5f:
            self.dset = h5f.create_dataset(
                dataset,
                shape=(0,) + shape,
                maxshape=(None,) + shape,
                dtype=dtype,
                compression=compression,
                chunks=(chunk_len,) + shape,
            )

    def append(self, values: np.ndarray) -> None:
        with h5py.File(self.datapath, mode="a") as h5f:
            dset = h5f[self.dataset]
            dset.resize((self.i + values.shape[0],) + self.shape)
            dset[self.i :] = values
            self.i += values.shape[0]
            h5f.flush()
