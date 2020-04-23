import vaex
import os
import pytest


@pytest.mark.skipif(vaex.utils.devmode, reason='runs too slow when developing')
def test_s3():
    df = vaex.open('s3://vaex/testing/xys.hdf5?cache=false&anon=true')
    assert df.x.tolist() == [1, 2]
<<<<<<< HEAD
    assert df.y.tolist() == [3, 4]
    assert df.s.tolist() == ['5', '6']

    df = vaex.open('s3://vaex/testing/xys.hdf5?cache=true&anon=true')
    assert df.x.tolist() == [1, 2]
    assert df.y.tolist() == [3, 4]
    assert df.s.tolist() == ['5', '6']
=======
    assert df.y.tolist() == [2, 3]
    assert df.s.tolist() == ["4", "5"]

    df = vaex.open('s3://vaex/testing/xys.hdf5?cache=true&anon=true')
    assert df.x.tolist() == [1, 2]
    assert df.y.tolist() == [2, 3]
    assert df.s.tolist() == ["4", "5"]
>>>>>>> remotes/origin/fix_s3_string_support
