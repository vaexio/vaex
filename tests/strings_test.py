# -*- coding: utf-8 -*-
import vaex
import numpy as np
import pytest
import sys

def test_concat():
	ds1 = vaex.from_arrays(names=['hi', 'is', 'l2'])
	ds2 = vaex.from_arrays(names=['hello', 'this', 'is', 'long'])
	ds = ds1.concat(ds2)
	assert len(ds) == len(ds1) + len(ds2)
	assert ds.dtype('names') == ds2.data.names.dtype
	assert ds.dtype('names') != np.object

def test_string_count():
	ds = vaex.from_arrays(names=['hello', 'this', 'is', 'long'])
	assert ds.count(ds.names) == 4
	ds = vaex.from_arrays(names=np.ma.array(['hello', 'this', 'is', 'long'], mask=[0, 0, 1, 0]))
	assert ds.count(ds.names) == 3

@pytest.mark.skip
def test_string_dtype_with_none():
	ds = vaex.from_arrays(names=['hello', 'this', 'is', None])
	assert ds.count(ds.names) == 3

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

@pytest.mark.skipif(sys.version_info < (3,3),
                    reason="requires python3.4 or higher")
def test_unicode(tmpdir):
	path = str(tmpdir.join('utf32.hdf5'))
	ds = vaex.from_arrays(names=["vaex", "or", "væx!"])
	assert str(ds.names.dtype) == '<U4'
	ds.export_hdf5(path)
	ds = vaex.open(path)
	assert str(ds.names.dtype) == '<U4'
	assert ds.names.tolist() == ["vaex", "or", "væx!"]
