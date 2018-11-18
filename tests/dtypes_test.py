from common import *


def test_dtype(ds_local):
  ds = ds_local
  for name in ds.column_names:
      assert ds[name].values.dtype == ds.dtype(ds[name])


def test_dtypes(ds_local):
  ds = ds_local
  all_dtypes = [np.float64, np.float64, np.float64, np.float64, np.int64, np.int64, 'S25', np.object]
  np.testing.assert_array_equal(ds.dtypes(columns=None), all_dtypes)
  some_dtypes = [np.float64, np.int64, 'S25', np.object]
  np.testing.assert_array_equal(ds.dtypes(columns=['x', 'mi', 'name', 'obj']), some_dtypes)
