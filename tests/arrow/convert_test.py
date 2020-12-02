import pytest
import numpy as np
from vaex.arrow import convert
import pyarrow as pa

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


@pytest.mark.skipif(pa.__version__.split(".")[0] == '1', reason="segfaults in arrow v1")
@pytest.mark.parametrize("offset", list(range(1, 17)))
@pytest.mark.parametrize("chunked", [True, False])
def test_large_string_to_string(offset, chunked):
    s = pa.array(['aap', 'noot', None, 'mies'] * 3, type=pa.large_string())
    if chunked:
        s = pa.chunked_array([s.slice(0, 5), s.slice(5)])
    ns = convert.large_string_to_string(s)
    ns.validate()
    assert s.type != ns.type
    assert s.to_pylist() == ns.to_pylist()

    s = s.slice(offset)
    ns = convert.large_string_to_string(s)
    if not chunked:
        assert ns.offset < 8
    assert s.to_pylist() == ns.to_pylist()
