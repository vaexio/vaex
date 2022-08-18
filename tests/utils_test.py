import vaex.utils
import numpy as np
import pytest


def test_required_dtype_for_max():
    assert vaex.utils.required_dtype_for_max(127, signed=True) == np.int8
    assert vaex.utils.required_dtype_for_max(128, signed=True) == np.int16
    assert vaex.utils.required_dtype_for_max(127, signed=False) == np.uint8
    assert vaex.utils.required_dtype_for_max(128, signed=False) == np.uint8

    assert vaex.utils.required_dtype_for_max(2**63-1, signed=True) == np.int64
    assert vaex.utils.required_dtype_for_max(2**63, signed=False) == np.uint64
    with pytest.raises(ValueError):
        assert vaex.utils.required_dtype_for_max(2**63, signed=True) == np.int64



def test_required_dtype_for_range():
    assert vaex.utils.required_dtype_for_range(-100, 127, signed=True) == np.int8
    assert vaex.utils.required_dtype_for_range(-129, 127, signed=True) == np.int16
    assert vaex.utils.required_dtype_for_range(0, 128, signed=True) == np.int16
    assert vaex.utils.required_dtype_for_range(0, 127, signed=False) == np.uint8
    assert vaex.utils.required_dtype_for_range(0, 128, signed=False) == np.uint8
    with pytest.raises(ValueError):
        assert vaex.utils.required_dtype_for_range(-1, 128, signed=False) == np.int16
    assert vaex.utils.required_dtype_for_range(-1, 128, signed=True) == np.int16

    assert vaex.utils.required_dtype_for_range(0, 2**63-1, signed=True) == np.int64
    assert vaex.utils.required_dtype_for_range(-1, 2**63-1, signed=True) == np.int64
    assert vaex.utils.required_dtype_for_range(-2**63, 0, signed=True) == np.int64
    assert vaex.utils.required_dtype_for_range(0, 2**63, signed=False) == np.uint64
    with pytest.raises(ValueError):
        assert vaex.utils.required_dtype_for_range(-1, 2**63, signed=True) == np.int64
    with pytest.raises(ValueError):
        assert vaex.utils.required_dtype_for_range(-2**64-1, 0, signed=True) == np.int64


def test_dict_replace_key():
    d = {'a': 1, 'b': 2}
    result = vaex.utils.dict_replace_key(d, 'a', 'z')
    assert list(result.items()) == [('z', 1), ('b', 2)]
