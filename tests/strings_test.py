import vaex
import numpy as np

def test_concat():
	ds1 = vaex.from_arrays(names=['hi', 'is', 'l2'])
	ds2 = vaex.from_arrays(names=['hello', 'this', 'is', 'long'])
	ds = ds1.concat(ds2)
	assert len(ds) == len(ds1) + len(ds2)
	assert ds.dtype('names') == ds2.data.names.dtype
	assert ds.dtype('names') != np.object

def test_unicode():
	ds = vaex.from_arrays(names=['bla\u1234'])
	assert ds.names.dtype.kind == 'U'
	ds = vaex.from_arrays(names=['bla'])
	assert ds.names.dtype.kind == 'S'

def test_concat_mixed():
	# this can happen when you want to concat multiple csv files
	# and pandas makes one have nans, since they all have missing values
	# and the other string
	ds1 = vaex.from_arrays(names=['not', 'missing'])
	ds2 = vaex.from_arrays(names=[np.nan, np.nan])
	assert ds1.dtype(ds1.names).kind in 'S'
	assert ds2.dtype(ds2.names) == np.float64
	ds = ds1.concat(ds2)
	assert len(ds) == len(ds1) + len(ds2)
	assert ds.dtype(ds.names) == ds1.names.dtype

def test_strip():
	ds = vaex.from_arrays(names=['this ', ' has', ' space'])
	ds['stripped'] = ds.names.str.strip()
	ds.stripped.tolist() == ['this', 'has', 'space']
