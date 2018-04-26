from common import *

def test_rename(ds_local):
	ds = ds_local
	ds['r'] = ds.x
	ds['q'] = np.exp(ds.x)
	qvalues = ds.q.values.tolist()
	xvalues = ds.x.tolist()
	qexpr = ds.q.expand().expression
	ds['x'] = ds.y # now r should still point to x
	assert ds.r.values.tolist() == xvalues
	assert ds.q.values.tolist() == qvalues
