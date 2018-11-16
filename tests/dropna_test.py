from common import *


def test_dropna_objects(ds_local):
    ds = ds_local
    ds_dropped = ds.dropna(column_names=['obj'])
    assert ds_dropped['obj'].values.mask.any() == False
    float_elements = np.array([element for element in ds_dropped['obj'].values.data if isinstance(element, float)])
    assert np.isnan(float_elements).any() == False, 'np.nan still exists in column'


def test_dropna(ds_local):
    ds = ds_local

    ds_dropped = ds.dropna()
    assert len(ds_dropped) == 6

    ds_dropped = ds.dropna(drop_masked=False)
    assert len(ds_dropped) == 8
    assert np.isnan(ds_dropped['n'].values).any() == False
    assert np.isnan(ds_dropped['nm'].values).any() == False

    ds_dropped = ds.dropna(drop_nan=False)
    assert len(ds_dropped) == 8
    assert ds_dropped['m'].values.mask.any() == False
    assert ds_dropped['nm'].values.mask.any() == False
    assert ds_dropped['mi'].values.mask.any() == False
    assert ds_dropped['obj'].values.mask.any() == False

    ds_dropped = ds.dropna(column_names=['nm', 'mi'])
    assert len(ds_dropped) == 8
    assert ds_dropped['nm'].values.mask.any() == False
    assert np.isnan(ds_dropped['nm'].values).any() == False

    ds_dropped = ds.dropna(column_names=['obj'])
    assert len(ds_dropped) == 8
    assert ds_dropped['obj'].values.mask.any() == False
    float_elements = np.array([element for element in ds_dropped['obj'].values.data if isinstance(element, float)])
    assert np.isnan(float_elements).any() == False, 'np.nan still exists in column'

    ds_dropped = ds.dropna(column_names=['nm', 'mi', 'obj'])
    state = ds_dropped.state_get()
    ds_copy.state_set(state)
    assert len(ds_copy) == len(ds_dropped)
    assert len(ds_copy) == 6
