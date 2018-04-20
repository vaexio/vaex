from common import *

def test_column_names(ds_local):
	ds = ds_local
	columns_names = ds.get_column_names(virtual=True)
	ds['__x'] = ds.x
	assert columns_names == ds.get_column_names(virtual=True)
	assert '__x' in ds.get_column_names(virtual=True, hidden=True)
	assert len(columns_names) == len(ds.get_column_names(virtual=True, hidden=True))-1
