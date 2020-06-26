import numpy as np
from vaex.arrow import convert

bools = [False, True, True]


def test_bool():
    b = np.array(bools)
    b_arrow = convert.arrow_array_from_numpy_array(b)
    assert b_arrow.to_pylist() == bools
    b = convert.numpy_array_from_arrow_array(b_arrow)
    assert b.tolist() == bools


def test_bool_sliced():
    b = np.array(bools)
    b_arrow = convert.arrow_array_from_numpy_array(b)
    assert b_arrow.to_pylist() == bools
    b_arrow = b_arrow[1:]
    b = convert.numpy_array_from_arrow_array(b_arrow)
    assert b.tolist() == bools[1:]