import vaex
from common import *


x = np.arange(10)


def test_column_names(ds_local):
    ds = ds_local
    columns_names = ds.get_column_names(virtual=True)
    ds['__x'] = ds.x
    assert columns_names == ds.get_column_names(virtual=True)
    assert '__x' in ds.get_column_names(virtual=True, hidden=True)
    assert len(columns_names) == len(ds.get_column_names(virtual=True, hidden=True))-1

    ds = vaex.example()
    ds['__x'] = ds['x'] + 1
    assert 'FeH' in ds.get_column_names(regex='e*')
    assert 'FeH' not in ds.get_column_names(regex='e')
    assert '__x' not in ds.get_column_names(regex='__x')
    assert '__x' in ds.get_column_names(regex='__x', hidden=True)


def test_add_invalid_name(tmpdir):
    # support invalid names and keywords
    df = vaex.from_dict({'X!1': x, 'class': x*2})
    assert df['X!1'].tolist() == x.tolist()
    assert (df['X!1']*2).tolist() == (x*2).tolist()
    assert (df['class']).tolist() == (x*2).tolist()
    assert 'X!1' in df._column_aliases
    assert (df.copy()['X!1']*2).tolist() == (x*2).tolist()

    path = str(tmpdir.join('test.hdf5'))
    df.export(path)
    df = vaex.open(path)
    assert df['X!1'].tolist() == x.tolist()
    assert (df.copy()['X!1']*2).tolist() == (x*2).tolist()
