from common import *
import numpy as np

def test_dt():
	t = np.array(['2007-07-13', '2006-01-13', '2010-08-13'], dtype='datetime64')
	ds = vaex.from_arrays(t=t)
	df = ds.to_pandas_df()
	assert ds.t.dt.year.values.tolist() == df.t.dt.year.values.tolist()
	assert ds.t.dt.dayofweek.values.tolist() == df.t.dt.dayofweek.values.tolist()
	assert ds.t.dt.hour.values.tolist() == df.t.dt.hour.values.tolist()
