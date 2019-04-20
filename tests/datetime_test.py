from common import *
import numpy as np
import pytest

def test_datetime_operations():
    date = np.array([np.datetime64('2009-10-12T03:31:00'),
        np.datetime64('2016-02-11T10:17:34'),
        np.datetime64('2015-11-12T11:34:22'),
        np.datetime64('2003-03-03T00:33:15'),
        np.datetime64('2014-07-23T15:08:05'),
        np.datetime64('2011-01-01T07:02:01')], dtype='<M8[ns]')

    df = vaex.from_arrays(date=date)._readonly()
    pandas_df = df.to_pandas_df()

    assert df.date.dt.hour.values.tolist() == pandas_df.date.dt.hour.values.tolist()
    assert df.date.dt.minute.values.tolist() == pandas_df.date.dt.minute.values.tolist()
    assert df.date.dt.second.values.tolist() == pandas_df.date.dt.second.values.tolist()
    assert df.date.dt.day.values.tolist() == pandas_df.date.dt.day.values.tolist()
    assert df.date.dt.day_name.values.tolist() == pandas_df.date.dt.day_name().values.tolist()
    assert df.date.dt.month.values.tolist() == pandas_df.date.dt.month.values.tolist()
    assert df.date.dt.month_name.values.tolist() == pandas_df.date.dt.month_name().values.tolist()
    assert df.date.dt.year.values.tolist() == pandas_df.date.dt.year.values.tolist()
    assert df.date.dt.is_leap_year.values.tolist() == pandas_df.date.dt.is_leap_year.values.tolist()
    assert any(df.date.dt.is_leap_year.values.tolist())
    assert df.date.dt.weekofyear.values.tolist() == pandas_df.date.dt.weekofyear.values.tolist()
    assert df.date.dt.dayofyear.values.tolist() == pandas_df.date.dt.dayofyear.values.tolist()
    assert df.date.dt.dayofweek.values.tolist() == pandas_df.date.dt.dayofweek.values.tolist()

def test_datetime_agg():
    date = [np.datetime64('2009-10-12T03:31:00'),
        np.datetime64('2016-02-11T10:17:34'),
        np.datetime64('2015-11-12T11:34:22'),
        np.datetime64('2003-03-03T00:33:15'),
        np.datetime64('2014-07-23T15:08:05'),
        np.datetime64('2011-01-01T07:02:01')]

    df = vaex.from_arrays(date=date)
    assert df.count(df.date) == len(date)
    assert df.max(df.date) == np.datetime64('2016-02-11T10:17:34')

def test_datetime_stats():
    x1 = np.datetime64('2005-01-01')
    x2 = np.datetime64('2015-02-01')
    x = np.arange(x1, x2, dtype=np.datetime64)
    y = np.arange(len(x))
    df = vaex.from_arrays(x=x, y=y)
    d1, d2 = df.x.minmax()
    assert d1 == x1
    assert d2 == x[-1]

    # TODO: we may want to support storing objects in the variables automatically
    # df['deltax'] = df.x - x1
    # assert df['deltax'].astype('datetime64[D]') == []
    # print(repr(df['deltax']))  # coverage
