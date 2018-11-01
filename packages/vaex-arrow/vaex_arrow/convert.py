import pyarrow
import numpy as np
from .dataset import DatasetArrow

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


def arrow_table_from_vaex_dataset(ds, column_names=None, selection=None, strings=True, virtual=False):
    """Implementation of Dataset.to_arrow_table"""
    names = []
    arrays = []
    for name, array in ds.to_items(column_names=column_names, selection=selection, strings=strings, virtual=virtual):
        names.append(name)
        arrays.append(arrow_array_from_numpy_array(array))
    # import IPython
    # IPython.embed()
    return pyarrow.Table.from_arrays(arrays, names)

def vaex_dataset_from_arrow_table(table):
    return DatasetArrow(table=table)