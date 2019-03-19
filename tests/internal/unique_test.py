import vaex.superutils
import numpy as np


def test_unique_nanfix():
    x = np.array([0, np.nan, 0, 1, np.nan, 2, np.nan], dtype=np.float64)
    u = vaex.superutils.unique(x)
    assert u[1:].tolist() == [np.nan, 0, 1, 2][1:]
    u, indices = vaex.superutils.unique(x, return_inverse=True)
    assert indices.tolist() == [1, 0, 1, 2, 0, 3, 0]

def test_unique_nanfix():
    x = np.array([0, np.nan, 0, 1, np.nan, 2, np.nan], dtype=np.float32)
    u = vaex.superutils.unique(x)
    assert u[1:].tolist() == [np.nan, 0, 1, 2][1:]
    u, indices = vaex.superutils.unique(x, return_inverse=True)
    assert indices.tolist() == [1, 0, 1, 2, 0, 3, 0]

def test_unique():
    x = np.array([0, 1, 2, 0, 1, 0], dtype=np.float64)
    u = vaex.superutils.unique(x)
    assert u.tolist() == [0, 1, 2]
    u, indices = vaex.superutils.unique(x, return_inverse=True)
    assert u[indices].tolist() == x.tolist()