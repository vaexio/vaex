from common import *


def test_dtype(ds_local):
  ds = ds_local
  for name in ds.column_names:
    assert ds[name].values.dtype == ds.dtype(ds[name])


def test_dtypes(ds_local):
  ds = ds_local
  assert (ds.dtypes.values == [ds[name].dtype for name in ds.column_names]).all()
