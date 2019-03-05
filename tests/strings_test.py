# -*- coding: utf-8 -*-
import vaex
import numpy as np
import pytest
import sys

try:
	unicode
	str_kind = 'S'
except:
	str_kind = 'U'


@pytest.mark.skipif(sys.version_info < (3,3),
                    reason="requires python3.4 or higher")
def test_dtype_object_string(tmpdir):
	x = np.arange(8,12)
	s = np.array(list(map(str, x)), dtype='O')
	print(s, s.dtype)
	df = vaex.from_arrays(x=x, s=s)
	assert df.columns['s'].dtype.kind == 'O'
	path = str(tmpdir.join('test.arrow'))
	df.export(path)
	df_read = vaex.open(path)
	# the data type of s can be different
	assert df_read.compare(df) == ([], [], [], [])


def test_export_arrow_strings_to_hdf5(tmpdir):
	df = vaex.from_arrays(names=['hi', 'is', 'l2'])
	path = str(tmpdir.join('test.arrow'))
	df.export(path)
	df_read_arrow = vaex.open(path)
	path = str(tmpdir.join('test.hdf5'))
	df_read_arrow.export(path)
	df_read_hdf5 = vaex.open(path)
	assert df_read_hdf5.compare(df_read_arrow) == ([], [], [], [])

def test_arrow_strings_concat(tmpdir):
	df = vaex.from_arrays(names=['hi', 'is', 'l2'])
	path = str(tmpdir.join('test.arrow'))
	df.export(path)
	df_read_arrow = vaex.open(path)
	path = str(tmpdir.join('test.hdf5'))
	df_read_arrow.export(path)
	df_read_hdf5 = vaex.open(path)
	assert df_read_hdf5.compare(df_read_arrow) == ([], [], [], [])


def test_concat():
	ds1 = vaex.from_arrays(names=['hi', 'is', 'l2'])
	ds2 = vaex.from_arrays(names=['hello', 'this', 'is', 'long'])
	ds = ds1.concat(ds2)
	assert len(ds) == len(ds1) + len(ds2)
	assert ds.dtype('names') == vaex.column.str_type
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
	assert ds.names.dtype.kind == 'U'

@pytest.mark.skipif(sys.version_info < (3,3),
                    reason="requires python3.4 or higher")
def test_concat_mixed():
	# this can happen when you want to concat multiple csv files
	# and pandas makes one have nans, since they all have missing values
	# and the other string
	ds1 = vaex.from_arrays(names=['not', 'missing'])
	ds2 = vaex.from_arrays(names=[np.nan, np.nan])
	assert ds1.dtype(ds1.names) == str
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
	assert ds.names.dtype == vaex.column.str_type
	ds.export_hdf5(path)
	ds = vaex.open(path)
	assert ds.names.dtype == vaex.column.str_type
	assert ds.names.tolist() == ["vaex", "or", "væx!"]


@pytest.fixture(params=['dfs_arrow', 'dfs_array'])
def dfs(request, dfs_arrow, dfs_array):
    named = dict(dfs_arrow=dfs_arrow, dfs_array=dfs_array)
    return named[request.param]

@pytest.fixture()
def dfs_arrow():
    return vaex.from_arrays(s=vaex.string_column(["vaex", "or", "væx!"]))

@pytest.fixture()
def dfs_array():
    return vaex.from_arrays(s=np.array(["vaex", "or", "væx!"], dtype='O'))

def test_string_contains(dfs):
	assert dfs.s.str.contains('v', regex=False).tolist() == [True, False, True]
