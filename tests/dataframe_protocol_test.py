import vaex
import numpy as np
import pyarrow as pa
from vaex.dataframe_protocol import from_dataframe_to_vaex, _DtypeKind

def test_float_only():
	df = vaex.from_arrays(x=np.array([1.5, 2.5, 3.5]), y=np.array([9.2, 10.5, 11.8]))
	df2 = from_dataframe_to_vaex(df)
	assert  df2.x.tolist() == df.x.tolist()
	assert  df2.y.tolist() == df.y.tolist()

def test_mixed_intfloat():
	df = vaex.from_arrays(x=np.array([1, 2, 0]), y=np.array([9.2, 10.5, 11.8]))
	df2 = from_dataframe_to_vaex(df)
	assert  df2.x.tolist() == df.x.tolist()
	assert  df2.y.tolist() == df.y.tolist()
	
def test_mixed_intfloatbool():
	df = vaex.from_arrays(
		x=np.array([True, True, False]),
		y=np.array([1, 2, 0]),
		z=np.array([9.2, 10.5, 11.8]))
	df2 = from_dataframe_to_vaex(df)
	assert  df2.x.tolist() == df.x.tolist()
	assert  df2.y.tolist() == df.y.tolist()
	assert  df2.z.tolist() == df.z.tolist()

def test_categorical():
	df = vaex.from_arrays(year=[2012, 2015, 2019], weekday=[0, 4, 6])
	df = df.categorize('year', min_value=2012, max_value=2019)
	df = df.categorize('weekday', labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

	# Some detailed testing for correctness of dtype and null handling:
	col = df.__dataframe__().get_column_by_name('weekday')
	assert col.dtype[0] == _DtypeKind.CATEGORICAL
	assert col.describe_categorical == (False, True, {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'})

	df2 = from_dataframe_to_vaex(df)
	assert  df2['year'].tolist() == df['year'].tolist()
	assert  df2['weekday'].tolist() == df['weekday'].tolist()

def test_virtual_column():
	df = vaex.from_arrays(
		x=np.array([True, True, False]),
		y=np.array([1, 2, 0]),
		z=np.array([9.2, 10.5, 11.8]))
	df.add_virtual_column("r", "sqrt(y**2 + z**2)")
	df2 = from_dataframe_to_vaex(df)
	assert  df2.x.tolist() == df.x.tolist()
	assert  df2.y.tolist() == df.y.tolist()
	assert  df2.z.tolist() == df.z.tolist()
	assert  df2.r.tolist() == df.r.tolist()

def test_arrow_dictionary():
	# ERRORS!
	indices = pa.array([0, 1, 0, 1, 2, 0, None, 2])
	dictionary = pa.array(['foo', 'bar', 'baz'])
	dict_array = pa.DictionaryArray.from_arrays(indices, dictionary)
	df = vaex.from_arrays(x = dict_array)

	# Some detailed testing for correctness of dtype and null handling:
	col = df.__dataframe__().get_column_by_name('x')
	assert col.dtype[0] == _DtypeKind.CATEGORICAL
	assert col.describe_categorical == (False, True, {0: 'foo', 1: 'bar', 2: 'baz'})

	df2 = from_dataframe_to_vaex(df)
	assert  df2.x.tolist() == df.x.tolist()