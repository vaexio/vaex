from common import *


def test_values(ds_local):
    ds = ds_local

    np.testing.assert_array_equal(ds['x'].values, np.array(ds.evaluate('x')).T)
    np.testing.assert_array_equal(ds['name'].values, np.array(ds.evaluate('name')).T)

    np.testing.assert_array_equal(ds['obj'].values.mask, ds.evaluate('obj').mask)
    ind = ds['obj'].values.data == ds.evaluate('obj').data
    assert np.concatenate((ind[:5], ind[6:])).all()
    assert np.isnan(ds['obj'].values.data[5])

    np.testing.assert_array_equal(ds[['x', 'y']].values, np.array(ds.evaluate('x, y')).T)
    # The missing values should be converted to nan, on the whole ndarray should be case to masked array
    np.testing.assert_array_equal(ds[['x', 'y', 'nm']].values, np.array(ds.evaluate('x, y, nm')).T)
    np.testing.assert_array_equal(ds[['x', 'name', 'nm', 'obj']].values, np.array(ds.evaluate('x, name, nm, obj')).T)