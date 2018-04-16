from common import *

def test_rename(ds_local):
	ds = ds_local
	ds['r'] = ds.x
	oldx = ds.x.tolist()
	ds['x'] = ds.y # now r should still point to x
	assert ds.r.tolist() == oldx