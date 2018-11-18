import pytest
from common import *


def test_values(ds_local):
    ds = ds_local

    np.testing.assert_array_equal(ds['x'].values, np.array(ds.evaluate('x')))
    np.testing.assert_array_equal(ds['name'].values, np.array(ds.evaluate('name')))

    np.testing.assert_array_equal(ds['obj'].values.mask, ds.evaluate('obj').mask)
    ind = ds['obj'].values.data == ds.evaluate('obj').data
    assert np.concatenate((ind[:5], ind[6:])).all()
    assert np.isnan(ds['obj'].values.data[5])

    np.testing.assert_array_equal(ds[['x', 'y']].values, np.array([ds.evaluate('x'), ds.evaluate('y')]).T)
    # The missing values are included. This may not be the correct behaviour
    np.testing.assert_array_equal(ds[['x', 'y', 'nm']].values, np.array([ds.evaluate('x'), ds.evaluate('y'), ds.evaluate('nm')]).T)

def test_object_column_values(ds_local):
    ds = ds_local
    with pytest.raises(Exception) as e_info:
        ds[['x', 'name', 'nm', 'obj']].values