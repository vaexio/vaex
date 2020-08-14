import vaex
import pytest


@pytest.mark.skipif(vaex.utils.devmode, reason='runs too slow when developing')
def test_gcs():
    df = vaex.open('gs://vaex-data/testing/xys.hdf5?cache=false&token=anon')
    assert df.x.tolist() == [1, 2]
    assert df.y.tolist() == [3, 4]
    assert df.s.tolist() == ['5', '6']

    df = vaex.open('gs://vaex-data/testing/xys.hdf5?cache=true&token=anon')
    assert df.x.tolist() == [1, 2]
    assert df.y.tolist() == [3, 4]
    assert df.s.tolist() == ['5', '6']


@pytest.mark.skipif(vaex.utils.devmode, reason='runs too slow when developing')
def test_gcs_masked():
    df = vaex.open('gs://vaex-data/testing/xys-masked.hdf5?cache=false&token=anon')
    assert df.x.tolist() == [1, None]
    assert df.y.tolist() == [None, 4]
    assert df.s.tolist() == ['5', None]
