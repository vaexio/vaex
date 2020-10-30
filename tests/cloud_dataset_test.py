import vaex
import pytest


@pytest.mark.skipif(vaex.utils.devmode, reason='runs too slow when developing')
@pytest.mark.parametrize("base_url", ["gs://vaex-data", "s3://vaex"])
@pytest.mark.parametrize("cache", ["true", "false"])
def test_cloud_dataset_basics(base_url, cache):
    df = vaex.open(f'{base_url}/testing/xys.hdf5?cache={cache}', anonymous=True)
    assert df.x.tolist() == [1, 2]
    assert df.y.tolist() == [3, 4]
    assert df.s.tolist() == ['5', '6']

    assert df.x.count() == 2
    assert df.x.sum() == 3


@pytest.mark.skipif(vaex.utils.devmode, reason='runs too slow when developing')
@pytest.mark.parametrize("base_url", ["gs://vaex-data", "s3://vaex"])
@pytest.mark.parametrize("cache", ["true", "false"])
def test_cloud_dataset_masked(base_url, cache):
    df = vaex.open(f'{base_url}/testing/xys-masked.hdf5?cache={cache}', anonymous=True)
    assert df.x.tolist() == [1, None]
    assert df.y.tolist() == [None, 4]
    assert df.s.tolist() == ['5', None]

    assert df.x.count() == 1
    assert df.s.count() == 1
    assert df.x.sum() == 1



@pytest.mark.skipif(vaex.utils.devmode, reason='runs too slow when developing')
@pytest.mark.parametrize("base_url", ["gs://vaex-data", "s3://vaex"])
def test_cloud_glob(base_url):
    assert set(vaex.file.glob(f'{base_url}/testing/*.hdf5', anonymous=True)) >= ({f'{base_url}/testing/xys-masked.hdf5', f'{base_url}/testing/xys.hdf5'})
    assert set(vaex.file.glob(f'{base_url}/testing/*.hdf5?anonymous=true')) >= ({f'{base_url}/testing/xys-masked.hdf5?anonymous=true', f'{base_url}/testing/xys.hdf5?anonymous=true'})
