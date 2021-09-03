from common import *
import vaex
import numpy as np
import pyarrow as pa
from vaex.dataframe_protocol import _from_dataframe_to_vaex, _DtypeKind


def test_float_only(df_factory):
	df = df_factory(x=[1.5, 2.5, 3.5], y=[9.2, 10.5, 11.8])
	df2 = _from_dataframe_to_vaex(df.__dataframe__())
	assert  df2.x.tolist() == df.x.tolist()
	assert  df2.y.tolist() == df.y.tolist()
	assert df2.__dataframe__().get_column_by_name('x').null_count == 0
	assert df2.__dataframe__().get_column_by_name('y').null_count == 0

def test_mixed_intfloat(df_factory):
	df = df_factory(x=[1, 2, 0], y=[9.2, 10.5, 11.8])
	df2 = _from_dataframe_to_vaex(df.__dataframe__())
	assert  df2.x.tolist() == df.x.tolist()
	assert  df2.y.tolist() == df.y.tolist()
	assert df2.__dataframe__().get_column_by_name('x').null_count == 0
	assert df2.__dataframe__().get_column_by_name('y').null_count == 0
	
def test_mixed_intfloatbool(df_factory):
	df = df_factory(
		x=np.array([True, True, False]),
		y=np.array([1, 2, 0]),
		z=np.array([9.2, 10.5, 11.8]))
	df2 = _from_dataframe_to_vaex(df.__dataframe__())
	assert  df2.x.tolist() == df.x.tolist()
	assert  df2.y.tolist() == df.y.tolist()
	assert  df2.z.tolist() == df.z.tolist()
	assert df2.__dataframe__().get_column_by_name('x').null_count == 0
	assert df2.__dataframe__().get_column_by_name('y').null_count == 0
	assert df2.__dataframe__().get_column_by_name('z').null_count == 0

def test_mixed_missing(df_factory_arrow):
	df = df_factory_arrow(
		x=np.array([True, None, False, None, True]),
		y=np.array([None, 2, 0, 1, 2]),
		z=np.array([9.2, 10.5, None, 11.8, None]))

	df2 = _from_dataframe_to_vaex(df.__dataframe__())

	assert df.__dataframe__().metadata == df2.__dataframe__().metadata

	assert df['x'].tolist() == df2['x'].tolist()
	assert not df2['x'].is_masked
	assert df2.__dataframe__().get_column_by_name('x').null_count == 2
	assert df['x'].dtype == df2['x'].dtype

	assert df['y'].tolist() == df2['y'].tolist()
	assert not df2['y'].is_masked
	assert df2.__dataframe__().get_column_by_name('y').null_count == 1
	assert df['y'].dtype == df2['y'].dtype

	assert df['z'].tolist() == df2['z'].tolist()
	assert not df2['z'].is_masked
	assert df2.__dataframe__().get_column_by_name('z').null_count == 2
	assert df['z'].dtype == df2['z'].dtype

def test_missing_from_masked(df_factory_numpy):
	df = df_factory_numpy(
		x=np.ma.array([1, 2, 3, 4, 0], mask=[0, 0, 0, 1, 1], dtype=int),
    	y=np.ma.array([1.5, 2.5, 3.5, 4.5, 0], mask=[False, True, True, True, False], dtype=float),
    	z=np.ma.array([True, False, True, True, True], mask=[1, 0, 0, 1, 0], dtype=bool))
	
	df2 = _from_dataframe_to_vaex(df.__dataframe__())

	assert df.__dataframe__().metadata == df2.__dataframe__().metadata

	assert df['x'].tolist() == df2['x'].tolist()
	assert not df2['x'].is_masked
	assert df2.__dataframe__().get_column_by_name('x').null_count == 2
	assert df['x'].dtype == df2['x'].dtype

	assert df['y'].tolist() == df2['y'].tolist()
	assert not df2['y'].is_masked
	assert df2.__dataframe__().get_column_by_name('y').null_count == 3
	assert df['y'].dtype == df2['y'].dtype

	assert df['z'].tolist() == df2['z'].tolist()
	assert not df2['z'].is_masked
	assert df2.__dataframe__().get_column_by_name('z').null_count == 2
	assert df['z'].dtype == df2['z'].dtype

def test_categorical_ordinal():
	colors = ['red', 'blue', 'green', 'blue']
	ds = vaex.from_arrays(
	    colors=colors, 
	    year=[2012, 2013, 2015, 2019], 
	    weekday=[0, 1, 4, 6])
	df = ds.ordinal_encode('colors', ['red', 'green', 'blue'])
	df = df.categorize('year', min_value=2012, max_value=2019)
	df = df.categorize('weekday', labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

	# Some detailed testing for correctness of dtype and null handling:
	col = df.__dataframe__().get_column_by_name('weekday')
	assert col.dtype[0] == _DtypeKind.CATEGORICAL
	assert col.describe_categorical == (False, True, {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'})
	col2 = df.__dataframe__().get_column_by_name('colors')
	assert col2.dtype[0] == _DtypeKind.CATEGORICAL
	assert col2.describe_categorical == (False, True, {0: 'red', 1: 'green', 2: 'blue'})

	df2 = _from_dataframe_to_vaex(df.__dataframe__())
	assert  df2['colors'].tolist() == ['red', 'blue', 'green', 'blue']
	assert  df2['year'].tolist() == [2012, 2013, 2015, 2019]
	assert  df2['weekday'].tolist() == ['Mon', 'Tue', 'Fri', 'Sun']

def test_arrow_dictionary():
	indices = pa.array([0, 1, 0, 1, 2, 0, 1, 2])
	dictionary = pa.array(['foo', 'bar', 'baz'])
	dict_array = pa.DictionaryArray.from_arrays(indices, dictionary)
	df = vaex.from_arrays(x = dict_array)

	# Some detailed testing for correctness of dtype and null handling:
	col = df.__dataframe__().get_column_by_name('x')
	assert col.dtype[0] == _DtypeKind.CATEGORICAL
	assert col.describe_categorical == (False, True, {0: 'foo', 1: 'bar', 2: 'baz'})

	df2 = _from_dataframe_to_vaex(df.__dataframe__())
	assert  df2.x.tolist() == df.x.tolist()
	assert df2.__dataframe__().get_column_by_name('x').null_count == 0

def test_arrow_dictionary_missing():
	indices = pa.array([0, 1, 2, 0, 1], mask=np.array([0, 1, 1, 0, 0], dtype=bool))
	dictionary = pa.array(['aap', 'noot', 'mies'])
	dict_array = pa.DictionaryArray.from_arrays(indices, dictionary)
	df = vaex.from_arrays(x = dict_array)

	# Some detailed testing for correctness of dtype and null handling:
	col = df.__dataframe__().get_column_by_name('x')
	assert col.dtype[0] == _DtypeKind.CATEGORICAL
	assert col.describe_categorical == (False, True, {0: 'aap', 1: 'noot', 2: 'mies'})

	df2 = _from_dataframe_to_vaex(df.__dataframe__())
	assert  df2.x.tolist() == df.x.tolist()
	assert df2.__dataframe__().get_column_by_name('x').null_count == 2
	assert df['x'].dtype.index_type == df2['x'].dtype.index_type

def test_virtual_column():
	df = vaex.from_arrays(
		x=np.array([True, True, False]),
		y=np.array([1, 2, 0]),
		z=np.array([9.2, 10.5, 11.8]))
	df.add_virtual_column("r", "sqrt(y**2 + z**2)")
	df2 = _from_dataframe_to_vaex(df.__dataframe__())
	assert  df2.r.tolist() == df.r.tolist()

def test_select_columns():
	df = vaex.from_arrays(
		x=np.array([True, True, False]),
		y=np.array([1, 2, 0]),
		z=np.array([9.2, 10.5, 11.8]))

	df2 = df.__dataframe__()
	assert df2.select_columns((0,2))._df[:,0].tolist() == df2.select_columns_by_name(('x','z'))._df[:,0].tolist()
	assert df2.select_columns((0,2))._df[:,1].tolist() == df2.select_columns_by_name(('x','z'))._df[:,1].tolist()
