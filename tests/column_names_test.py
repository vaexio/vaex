import vaex
from common import *

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
