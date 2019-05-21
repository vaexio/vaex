import vaex
import os
import pytest


@pytest.mark.skipif(vaex.utils.devmode, reason='runs too slow when developing')
def test_s3():
    df = vaex.open('s3://vaex/testing/xy.hdf5?cache=false&anon=true')
    assert df.x.tolist() == [1, 2]
    assert df.y.tolist() == [2, 3]

    df = vaex.open('s3://vaex/testing/xy.hdf5?cache=true&anon=true')
    assert df.x.tolist() == [1, 2]
    assert df.y.tolist() == [2, 3]
