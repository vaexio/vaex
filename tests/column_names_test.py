import vaex
from common import *
import h5py

x = np.arange(10)


def test_column_names(df_arrow):
    ds = df_arrow
    columns_names = ds.get_column_names(virtual=True)
    ds['__x2'] = ds.x
    assert columns_names == ds.get_column_names(virtual=True)
    assert '__x2' in ds.get_column_names(virtual=True, hidden=True)

    ds = vaex.example()
    ds['__x'] = ds['x'] + 1
    assert 'FeH' in ds.get_column_names(regex='e*')
    assert 'FeH' not in ds.get_column_names(regex='e')
    assert '__x' not in ds.get_column_names(regex='__x')
    assert '__x' in ds.get_column_names(regex='__x', hidden=True)


def test_add_invalid_name(tmpdir):
    # support invalid names and keywords
    df = vaex.from_dict({'X!1': x, 'class': x*2})
    assert str(df['X!1']) != 'X!1', "invalid identifier cannot be an expression"
    assert str(df['class']) != 'class', "keyword cannot be an expression"
    assert df.get_column_names() == ['X!1', 'class']
    assert df['X!1'].tolist() == x.tolist()
    assert (df['X!1']*2).tolist() == (x*2).tolist()
    assert (df['class']).tolist() == (x*2).tolist()
    assert (df.copy()['X!1']*2).tolist() == (x*2).tolist()

    path = str(tmpdir.join('test.hdf5'))
    df.export(path)
    df = vaex.open(path)
    assert df['X!1'].tolist() == x.tolist()
    assert (df.copy()['X!1']*2).tolist() == (x*2).tolist()
    assert (df[['X!1']]['X!1']*2).tolist() == (x*2).tolist()

    df_concat = vaex.concat([df, df])
    assert (df_concat[['X!1']]['X!1']*2).tolist() == ((x*2).tolist() + (x*2).tolist())



def test_add_invalid_virtual_columns(df_local):
    df = df_local
     # TODO: support for remote
    df['1'] = df.x
    assert df['1'].expression != '1'
    df['2'] = df.y
    assert df['1'].tolist() == df.x.tolist()
    assert df['2'].tolist() == df.y.tolist()


def test_not_hide_invalid_name():
    x = np.arange(10)
    df = vaex.from_dict({'./bla': x})
    assert len(df.get_column_names()) == 1
    assert df['./bla'].tolist() == x.tolist()


def test_unicode_names():
    x = np.arange(10)
    df = vaex.from_dict({'远': x})
    assert df.远.tolist() == x.tolist()


def test_invalid_name_read(tmpdir):
    # earlier version of vaex could write invalid names, check if we can read those
    df = vaex.from_dict({'x': x})
    # df.columns['1'] = df.columns.pop('x')
    # df.column_names = ['1']
    path = str(tmpdir.join('test.hdf5'))
    df.export(path)

    h5 = h5py.File(path, mode='r+')
    h5['/table/columns']['1'] = h5['/table/columns']['x']
    del h5['/table/columns']['x']

    df = vaex.open(path)
    assert df['1'].tolist() == x.tolist()
    assert (df.copy()['1']*2).tolist() == (x*2).tolist()
