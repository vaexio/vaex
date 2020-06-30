# these tests test assumptions made in vaex regarding arrow
import numpy as np
import pyarrow as pa
import pytest
import vaex


def test_non_native():
    # in vaex.array_types we convert to native format
    # if arrow supports this in the future, we can get rid of that 
    x = np.arange(10, dtype='>f4')
    with pytest.raises(pa.lib.ArrowNotImplementedError):
        pa.array(x)

@pytest.mark.skip(reason="Only the case for Arrow 0.17, leaving this is as 'documentation'")
def test_null_behaviour():
    assert pa.NULL in ['Confused'], "Arrow 0.17 "


def test_in_pylist():
    # this reflects the to_pylist in DataFrame.ordinal_encode
    ar = pa.array(['red', 'green'])
    assert ar[0] not in ar.to_pylist(), "Arrow 1.0.1 says no"


def test_cannot_convert_nulls_to_masked():
    x = np.arange(4)
    mask = x < 1
    xm = np.ma.array(x, mask=mask)
    xa = pa.array(xm)
    assert xa.to_pylist() == [None, 1, 2, 3]

    # arrow converts missing to nan
    xmc = xa.to_numpy(zero_copy_only=False)
    assert np.isnan(xmc[0])
    assert xmc.tolist() != xm.tolist()

    # so we roll out our own
    assert vaex.array_types.to_numpy(xa).tolist() == xm.tolist()


def test_filter_does_not_copy_null_value_data():
    x = np.arange(4)
    mask = x > 1
    x_numpy_masked = np.ma.array(x, mask=mask)
    assert x_numpy_masked.data[-1] == 3
    # we expect arrow to use the same underlying buffer
    x_arrow = pa.array(x_numpy_masked)
    x_arrow_data = np.frombuffer(x_arrow.buffers()[1])
    # but for some reason the data is not the same
    assert x_arrow_data[-1] != 3

    alltrue = pa.array([True, True, True, True])
    # same with using filter
    x_arrow_filtered = x_arrow.filter(alltrue)
    x_arrow_filtered_data = np.frombuffer(x_arrow_filtered.buffers()[1])
    # this is not the case, the missing data is filled with 0 or noise
    assert x_arrow_filtered_data[-1] != 3
    # we have to work around this in values_test.py::test_values
