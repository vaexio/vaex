import pytest
from common import *


def test_values(ds_local):
    ds = ds_local

    assert ds['x'].values.tolist() == ds.evaluate('x').tolist()
    assert ds['name'].values.tolist() == ds.evaluate('name').tolist()
    assert ds['obj'].values.data.tolist() == ds.evaluate('obj').data.tolist()
    assert ds[['x', 'y']].values.tolist() == np.array([ds.evaluate('x'), ds.evaluate('y')]).T.tolist()
    assert ds[['x', 'y']].values.shape == (len(ds), 2)
    assert ds[['m']].values[7][0] == 77777.0
    assert ds[['m','x']].values[7][0] == 77777
    # The missing values are included. This may not be the correct behaviour
    assert ds[['x', 'y', 'nm']].values.tolist(), np.array([ds.evaluate('x'), ds.evaluate('y'), ds.evaluate('nm')]).T.tolist()

def test_object_column_values(ds_local):
    ds = ds_local
    with pytest.raises(ValueError):
        ds[['x', 'name', 'nm', 'obj']].values