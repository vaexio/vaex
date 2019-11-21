import numpy as np
import pyarrow as pa
import pytest
from vaex import array_types 


bools = [False, True, True]


def test_bool():
    b = np.array(bools)
    b_arrow = array_types.to_arrow(b)
    assert b_arrow.to_pylist() == bools
    b = array_types.to_numpy(b_arrow)
    assert b.tolist() == bools


def test_bool_sliced():
    b = np.array(bools)
    b_arrow = array_types.to_arrow(b)
    assert b_arrow.to_pylist() == bools
    b_arrow = b_arrow[1:]
    b = array_types.to_numpy(b_arrow)
    assert b.tolist() == bools[1:]


def test_float_sliced():
    x_original = np.arange(10)
    x = x_original
    x_arrow = array_types.to_arrow(x)
    assert x_arrow.to_pylist() == x_original.tolist()
    x_arrow = x_arrow[3:]
    assert x_arrow.to_pylist() == x_original[3:].tolist()
    x = array_types.to_numpy(x_arrow)
    assert x.tolist() == x_original[3:].tolist()


def test_float_sliced_masked():
    x_original = np.arange(5)
    mask = x_original > 2
    x_original = np.ma.array(x_original, mask=mask)
    x = x_original
    x_arrow = array_types.to_arrow(x)
    assert x_arrow.to_pylist() == x_original.tolist()
    x_arrow = x_arrow[2:]
    assert x_arrow.to_pylist() == x_original[2:].tolist()
    x = array_types.to_numpy(x_arrow)
    assert x.tolist() == x_original[2:].tolist()


def test_keep_masked_data_values():
    x_original = np.arange(5, dtype='f8')
    mask = x_original > 2
    xm = np.ma.array(x_original, mask=mask)
    assert xm[-1] is np.ma.masked
    assert xm.data[-1] == 4
    x_arrow = array_types.to_arrow(xm)
    assert x_arrow.to_pylist()[-1] == None
    data_buffer = x_arrow.buffers()[1]
    x_arrow_data = np.frombuffer(data_buffer, dtype='f8')
    assert x_arrow_data[-1] == 4
