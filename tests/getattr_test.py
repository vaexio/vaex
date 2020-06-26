from common import *

def test_column_subset(ds_local):
    ds = ds_local
    dss = ds[['x', 'y']]
    assert dss.get_column_names() == ['x', 'y']
    np.array(dss)  # test if all columns can be put in arrays

def test_column_subset_virtual(ds_local):
    ds = ds_local
    ds['r'] = ds.x + ds.y
    dss = ds[['r']]
    assert dss.get_column_names() == ['r']
    assert set(dss.get_column_names(hidden=True)) == set(['__x', '__y', 'r'])
    np.array(dss)  # test if all columns can be put in arrays

def test_column_subset_virtual_recursive(df_local_non_arrow):
    df = df_local_non_arrow
    df['r'] = df.x + df.y
    df['q'] = df.r/2
    dfs = df[['q']]
    assert dfs.get_column_names() == ['q']
    all_columns = set(dfs.get_column_names(hidden=True))
    assert all_columns == set(['__x', '__y', '__r', 'q'])
    np.array(dfs)  # test if all columns can be put in arrays

def test_column_subset_virtual(ds_filtered):
    ds = ds_filtered
    dss = ds[['y']]
    assert dss.get_column_names() == ['y']
    all_columns = set(dss.get_column_names(hidden=True))
    assert all_columns == set(['__x', 'y'])
    np.array(dss)  # test if all columns can be put in arrays, with the possible filter copied as hidden

    # 'nested' filter
    ds = ds[ds.y > 2]
    dss = ds[['m']]
    assert dss.get_column_names() == ['m']
    assert set(dss.get_column_names(hidden=True)) == set(['__x', '__y', 'm'])

def test_column_order(ds_local):
    ds = ds_local
    dss = ds[['x', 'y']]
    assert dss.get_column_names() == ['x', 'y']
    assert np.array(dss).T.tolist() == [ds.x.values.tolist(), ds.y.values.tolist()]

    dss = ds[['y', 'x']]
    assert dss.get_column_names() == ['y', 'x']
    assert np.array(dss).T.tolist() == [ds.y.values.tolist(), ds.x.values.tolist()]

def test_column_order_virtual(ds_local):
    ds = ds_local
    # this will do some name mangling, but we don't care about the names
    ds['r'] = ds.y + 10
    ds = ds_local
    dss = ds[['x', 'r']]
    assert dss.get_column_names() == ['x', 'r']
    assert np.array(dss).T.tolist() == [ds.x.values.tolist(), ds.r.values.tolist()]

    dss = ds[['r', 'x']]
    assert dss.get_column_names() == ['r', 'x']
    assert np.array(dss).T.tolist() == [ds.r.values.tolist(), ds.x.values.tolist()]

def test_expression(ds_local):
    ds = ds_local
    # this will do some name mangling, but we don't care about the names
    dss = ds[['y/10', 'x/5']]
    assert 'y' in dss.get_column_names()[0]
    assert 'x' in dss.get_column_names()[1]
    assert np.array(dss).T.tolist() == [(ds.y/10).values.tolist(), (ds.x/5).values.tolist()]


@pytest.mark.skip(reason="Not implemented yet, should work, might need refactoring of copy")
def test_expression_virtual(ds_local):
    ds = ds_local
    # this will do some name mangling, but we don't care about the names
    ds['r'] = ds.y + 10
    dss = ds[['r/10', 'x/5']]
    assert 'r' in dss.get_column_names()[0]
    assert 'x' in dss.get_column_names()[1]
    assert np.array(dss).T.tolist() == [(ds.r/10).values.tolist(), (ds.x/5).values.tolist()]

    dss = ds[['x/5', 'r/10']]
    assert 'r' in dss.get_column_names()[0]
    assert 'x' in dss.get_column_names()[1]
    assert np.array(dss).T.tolist() == [(ds.x/5).values.tolist(), (ds.r/10).values.tolist()]

def test_access_data_after_virtual_column_creation(ds_local):
    ds = ds_local
    # we can access the x column
    assert ds[['x']].values[:,0].tolist() == ds.x.values.tolist()
    ds['virtual'] = ds.x * 2
    # it should also work after we added a virtual column
    assert ds[['x']].values[:,0].tolist() == ds.x.values.tolist()

def test_non_existing_column(df_local):
    df = df_local
    with pytest.raises(NameError, match='.*Did you.*'):
        df['x_']


def test_alias(df_local):
    df = df_local
    df2 = df[['123456']]
    assert '123456' in df2
