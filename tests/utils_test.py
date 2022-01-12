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


def test_dict_replace_key():
    d = {'a': 1, 'b': 2}
    result = vaex.utils.dict_replace_key(d, 'a', 'z')
    assert list(result.items()) == [('z', 1), ('b', 2)]
