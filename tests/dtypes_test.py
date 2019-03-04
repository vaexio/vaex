from common import *
from vaex.column import str_type


def test_dtype(ds_local):
  ds = ds_local
  for name in ds.column_names:
    if ds.dtype(name) == str_type:
      assert ds[name].values.dtype.kind == 'O'
    else:
      assert ds[name].values.dtype == ds.dtype(ds[name])


def test_dtypes(ds_local):
  ds = ds_local
  assert [ds.dtypes[name] for name in ds.column_names] == [ds[name].dtype for name in ds.column_names]
