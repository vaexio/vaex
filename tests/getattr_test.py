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
