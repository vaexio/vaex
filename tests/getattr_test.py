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

def test_column_subset_virtual(ds_filtered):
    ds = ds_filtered
    dss = ds[['y']]
    assert dss.get_column_names() == ['y']
    assert set(dss.get_column_names(hidden=True)) == set(['__x', 'y'])
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

def test_expression(ds_local):
    ds = ds_local
    # this will do some name mangling, but we don't care about the names
    dss = ds[['y/10', 'x/5']]
    assert np.array(dss).T.tolist() == [(ds.y/10).values.tolist(), (ds.x/5).values.tolist()]
