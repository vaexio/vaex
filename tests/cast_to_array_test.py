from common import *
import pytest

def test_preserve_int64(ds_local):
    ds = ds_local
    assert np.array(ds[['ints']], dtype=np.int64).dtype.kind == 'i', "expected int type precision"
    assert np.array(ds[['ints']], dtype=np.int64)[0][0] == -2**62-1, "lost precision"


def test_safe_casting(ds_local):
    ds = ds_local
    # with pytest.raises(ValueError, match='.*Cannot cast.*', message='Should use safe casting rules (no precision loss)'):
    np.array(ds[['ints']])
    with pytest.raises(ValueError, match='.*Cannot cast.*'):
        np.array(ds[['ints', 'x']], dtype=np.int64)
        pytest.fail('Should use safe casting rules (no precision loss)')


def test_default_float64(ds_local):
    ds = ds_local
    assert np.array(ds[['x']]).dtype == np.dtype('f8'), "expected float precision"


