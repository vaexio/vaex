from common import *

def test_describe(ds_local):
  ds = ds_local
  pdf = ds.describe()
  assert 'class \'str\'' not in str(pdf)

def test_describe_missing():
    x = np.array([5, '', 1, 4, None, 6 , np.nan, np.nan, 10, '', 0, 0, -13.5])
    y = np.array([5, -2, 1, 4, 0, 6 , np.nan, np.nan, 10, 42, 0, np.nan, -13.5])
    df = vaex.from_arrays(x=x, y=y)

    desc_df = df.describe()

    assert desc_df.x.loc['missing'] == 2
    assert desc_df.y.loc['missing'] == 3
