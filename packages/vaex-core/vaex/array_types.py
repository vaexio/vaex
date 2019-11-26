"""Conversion between different types of arrays"""
import numpy as np


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "to_numpy"):
        x = x.to_numpy()
    return x
