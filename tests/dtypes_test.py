from commont import *


def test_dtype(ds_local):
  ds = ds_local
  assert ds.dtype(ds.x) == np.float64
  assert ds.dtype(ds.y) == np.float64
  assert ds.dtype(ds.m) == np.float64
  assert ds.dtype(ds.nm) == np.float64
  assert ds.dtype(ds.mi) == np.int64
  assert ds.dtype('ints') == np.int64
  assert ds.dtype('name') == 'S25'
  assert ds.dtype('obj') == np.object

def test_dtypes(ds_local):
  ds = ds_local
  all_dtypes = np.array([np.float64, np.float64, np.float64, np.float64, np.int64, np.int64, 'S25', np.object])
  np.testing.assert_array_equal(ds.dtypes(columns=None), all_dtypes)
  some_dtypes = np.array([np.float64, np.int64, 'S25', np.object])
  np.testing.assert_array_equal(ds.dtypes(columns=['x', 'mi', 'name', 'obj']), some_dtypes)
