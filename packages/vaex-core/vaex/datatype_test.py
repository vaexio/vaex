import pytest
import pyarrow as pa
import numpy as np
from vaex.datatype import DataType


def test_primitive():
    t1 = DataType(np.dtype('f8'))
    t2 = DataType(np.dtype('f8'))
    assert t1 == t2
    assert t1 == np.dtype('f8')
    # assert np.dtype('f8') == t2  data type not understood


def test_timedelta64():
    t1 = DataType(np.dtype('timedelta64'))
    assert t1 == 'timedelta'
    assert t1 == 'timedelta64'
    assert t1.is_timedelta


@pytest.mark.parametrize('type', [pa.string(), pa.large_string()])
def test_string(type):
    t1 = DataType(type)
    assert t1 == 'string'
    assert t1 == str
    assert not t1.is_float
