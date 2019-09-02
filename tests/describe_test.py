from common import *
import pandas as pd


def test_describe(ds_local):
    ds = ds_local
    pdf = ds.describe()
    assert 'class \'str\'' not in str(pdf)


def test_describe_NA():
    x = np.array([5, '', 1, 4, None, 6, np.nan, np.nan, 10, '', 0, 0, -13.5])
    y = np.array([5, -2, 1, 4, 0, 6, np.nan, np.nan, 10, 42, 0, np.nan, -13.5])
    t = np.array([np.datetime64('2001'), np.datetime64('2005'), np.datetime64('2001'),
                  np.datetime64('2002'), np.datetime64('2006'), np.datetime64('2009'),
                  np.datetime64('2003'), np.datetime64('2007'), np.datetime64('2011'),
                  np.datetime64('2004'), np.datetime64('2008'), np.nan,
                  pd.NaT])

    df = vaex.from_arrays(x=x, y=y, t=t)

    desc_df = df.describe()

    assert desc_df.x.loc['NA'] == 3
    assert desc_df.y.loc['NA'] == 3
    assert desc_df.t.loc['NA'] == 2
