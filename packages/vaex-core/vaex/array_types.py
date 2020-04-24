"""Conversion between different types of arrays"""
import numpy as np


def to_array_type(x, array_type):
    if array_type is None:
        return x
    elif array_type == "list":
        return to_array_type(x, 'numpy').tolist()
    elif array_type == "numpy":
        return to_numpy(x)
    elif array_type == "xarray":
        import xarray
        return xarray.DataArray(x)
    else:
        raise ValueError(f'Unknown/unsupported array_type {array_type}')


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "to_numpy"):
        x = x.to_numpy()
    return x
