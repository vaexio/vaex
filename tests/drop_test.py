from common import *


def test_drop(ds_local):
    ds = ds_local
    dsd = ds.drop(ds.x)
    assert 'x' not in dsd.get_column_names()
    dsd = ds.drop([ds.x, 'y'])
    assert 'x' not in dsd.get_column_names()
    assert 'y' not in dsd.get_column_names()

    ds.drop([ds.x, 'y'], inplace=True)
    assert 'x' not in ds.get_column_names()
    assert 'y' not in ds.get_column_names()


def test_drop_depending(df_local_non_arrow):
    ds = df_local_non_arrow
    ds['r'] = ds.x + ds.y
    ds.drop(ds.x, inplace=True)
    assert 'x' not in ds.get_column_names()
    assert '__x' in ds.get_column_names(hidden=True)

    ds.drop(ds.y, inplace=True, check=False)
    assert 'y' not in ds.get_column_names()
    assert '__y' not in ds.get_column_names(hidden=True)


def test_drop_depending_filtered(ds_filtered):
    ds = ds_filtered
    ds.drop(ds.x, inplace=True)
    assert 'x' not in ds.get_column_names()
    assert '__x' in ds.get_column_names(hidden=True)
    ds.y.values  # make sure we can evaluate y


def test_drop_autocomplete(ds_local):
    ds = ds_local
    ds['drop'] = ds.m
    ds['columns'] = ds.m

    ds.drop('x', inplace=True)
    assert not hasattr(ds, 'x')

    ds.drop('y', inplace=True)
    assert not hasattr(ds, 'y')

    # make sure we can also remove with name issues
    del ds['drop']
    assert callable(ds.drop)
    del ds['columns']
    assert 'm' in ds.columns or 'm' in ds.virtual_columns


def test_drop_invalid_identifier():
    df = vaex.from_scalars(x=1, y=2)
    df['bla()'] = df.x + df.y
    df.drop(['x', 'y'], inplace=True)
    assert df['bla()'].tolist() == [3]


def test_drop_double():
    df = vaex.from_scalars(x=1, y=2, z=3)
    assert isinstance(df.dataset, vaex.dataset.DatasetArrays)
    df = df.drop('x')
    assert isinstance(df.dataset.original, vaex.dataset.DatasetArrays)
    assert df.dataset._dropped_names == ('x',)
    df = df.drop('y')
    assert isinstance(df.dataset.original, vaex.dataset.DatasetArrays)
    assert df.dataset._dropped_names == ('x', 'y')
