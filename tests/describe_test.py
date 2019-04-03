from common import *

def test_describe(ds_local):
  ds = ds_local
  pdf = ds.describe()
  assert 'class \'str\'' not in str(pdf)
