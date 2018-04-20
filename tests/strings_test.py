import vaex
import numpy as np

def test_concat():
	ds1 = vaex.from_arrays(names=['hi', 'is', 'l2'])
	ds2 = vaex.from_arrays(names=['hello', 'this', 'is', 'long'])
	ds = ds1.concat(ds2)
	assert len(ds) == len(ds1) + len(ds2)
	assert ds.dtype('names') == ds2.data.names.dtype
	assert ds.dtype('names') != np.object