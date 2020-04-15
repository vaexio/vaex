import pytest
from common import *


def test_values(ds_local):
    ds = ds_local

    assert ds['x'].values.tolist() == ds.evaluate('x').tolist()
    assert ds['name'].values.tolist() == ds.evaluate('name').tolist()
    assert ds['obj'].values.tolist() == ds.evaluate('obj').tolist()
    assert ds[['x', 'y']].values.tolist() == np.array([ds.evaluate('x'), ds.evaluate('y')]).T.tolist()
    assert ds[['x', 'y']].values.shape == (len(ds), 2)
    assert ds[['m']].values[9][0] == 77777.0
    assert ds[['m','x']].values[9][0] == 77777
    # The missing values are included. This may not be the correct behaviour
    assert ds[['x', 'y', 'nm']].values.tolist(), np.array([ds.evaluate('x'), ds.evaluate('y'), ds.evaluate('nm')]).T.tolist()

@pytest.mark.skip(reason='TOFIX: obj is now recognized as str')
def test_object_column_values(ds_local):
    ds = ds_local
    with pytest.raises(ValueError):
        ds[['x', 'name', 'nm', 'obj']].values

def test_values_masked():
    x = np.ma.MaskedArray(data=[1, 2, 3], mask=[False, False, True])
    y = [10, 20, 30]
    df = vaex.from_arrays(x=x, y=y)

    values = df[['x', 'y']].values
    comparison = np.array([[1., 10.],
                           [2., 20.],
                           [np.nan, 30.]])

    np.testing.assert_array_equal(values, comparison)
