import vaex
import numpy as np


def test_expression_slicing():
    x = np.array([-3, -13,   5,   4,  10,   7,  13,  15, -12, -14])
    df = vaex.from_arrays(x=x)
    assert x[:5].tolist() == df.x[:5].values.tolist
    assert x[-5:].tolist() == df.x[-5:].values.tolist


def test_arg_max():
    x = np.array([-3, -13,   5,   4,  10,   7,  13,  15, -12, -14])
    df = vaex.from_arrays(x=x)
    assert x.argmax() == df.x.argmax()


def test_arg_min():
    x = np.array([-3, -13,   5,   4,  10,   7,  13,  15, -12, -14])
    df = vaex.from_arrays(x=x)
    assert x.argmin() == df.x.argmin()