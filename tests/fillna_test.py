from common import *


def test_fillna_column(ds_local):
    ds = ds_local
    ds['ok'] = ds['obj'].fillna(value='NA')
    assert ds.ok.values[5] == 'NA'


def test_fillna(ds_local):
    ds = ds_local
    ds_copy = ds.copy()

    ds_string_filled = ds.fillna(value='NA')
    assert ds_string_filled.obj.values[5] == 'NA'

    ds_filled = ds.fillna(value=0)
    assert ds_filled.obj.values[5] == 0

    assert ds_filled.to_pandas_df(virtual=True).isna().any().any() == False
    assert ds_filled.to_pandas_df(virtual=True).isna().any().any() == False

    ds_filled = ds.fillna(value=10, fill_masked=False)
    assert ds_filled.n.values[6] == 10.
    assert ds_filled.nm.values[6] == 10.

    ds_filled = ds.fillna(value=-15, fill_nan=False)
    assert ds_filled.m.values[7] == -15.
    assert ds_filled.nm.values[7] == -15.
    assert ds_filled.mi.values[7] == -15.

    ds_filled = ds.fillna(value=-11, column_names=['nm', 'mi'])
    assert ds_filled.to_pandas_df(virtual=True).isna().any().any() == True
    assert ds_filled.to_pandas_df(column_names=['nm', 'mi']).isna().any().any() == False

    state = ds_filled.state_get()
    ds_copy.state_set(state)
    np.testing.assert_array_equal(ds_copy['nm'].values, ds_filled['nm'].values)
    np.testing.assert_array_equal(ds_copy['mi'].values, ds_filled['mi'].values)
