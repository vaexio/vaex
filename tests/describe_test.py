from common import *

import pandas as pd


def test_describe(ds_local):
    ds = ds_local
    pdf = ds.describe()
    assert 'class \'str\'' not in str(pdf)


def test_describe_NA():
    x = np.array([5, '', -1, 4.5, None, np.nan, np.nan])
    y = np.array([0, 6, np.nan, np.nan, -13.13, -0.5, np.nan])
    df = vaex.from_arrays(x=x, y=y)
    desc_df = df.describe()
    assert desc_df.x.loc['NA'] == 3
    assert desc_df.y.loc['NA'] == 3


def test_describe_nat_in_dtype_object():
    t = np.array([np.datetime64('2001'), pd.NaT, np.datetime64('2005'), np.nan])
    df = vaex.from_arrays(t=t)
    desc_df = df.describe()
    assert desc_df.t.loc['NA'] == 2


def test_describe_nat():
    t = np.array([np.datetime64('2001'), np.datetime64('NaT'), np.datetime64('2005')], dtype=np.datetime64)
    df = vaex.from_arrays(t=t)
    desc_df = df.describe()
    assert desc_df.t.loc['NA'] == 1
