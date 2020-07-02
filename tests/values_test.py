from common import *
import pytest


def test_values(ds_local):
    ds = ds_local

    assert ds['x'].values.tolist() == ds.evaluate('x').tolist()
    assert ds['name'].tolist() == ds.evaluate('name', array_type='arrow').to_pylist()
    if 'obj' in ds:   # df_arrow does not have it
        assert ds['obj'].tolist() == ds.evaluate('obj').tolist()
    assert ds[['x', 'y']].values.tolist() == np.array([ds.x.to_numpy(), ds.y.to_numpy()]).T.tolist()
    assert ds[['x', 'y']].values.shape == (len(ds), 2)
    assert ds[['m']].values[9].tolist()[0] == None
    assert ds[['m', 'x']].values[9].tolist()[0] == None
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

    assert df[['x', 'y']].values.tolist() ==  [[1., 10.], [2., 20.], [None, 30.]]
