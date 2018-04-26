from common import *

def test_row():
	ds = vaex.from_items(('x', [1,2,3]), ('y', [4,5,6]))
	assert ds[0] == [1, 4]
	ds['r'] = ds.x + ds.y
	assert ds[0] == [1, 4, 5]
