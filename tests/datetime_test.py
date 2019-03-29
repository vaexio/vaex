from common import *
import numpy as np


def test_datetime_operations():
    date = [np.datetime64('2009-10-12T03:31:00'),
            np.datetime64('2016-02-11T10:17:34'),
            np.datetime64('2015-11-12T11:34:22'),
            np.datetime64('2003-03-03T00:33:15'),
            np.datetime64('2014-07-23T15:08:05'),
            np.datetime64('2011-01-01T07:02:01')]

    df = vaex.from_arrays(date=date)
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
    assert df.date.dt.weekofyear.values.tolist() == pandas_df.date.dt.weekofyear.values.tolist()
    assert df.date.dt.dayofyear.values.tolist() == pandas_df.date.dt.dayofyear.values.tolist()
    assert df.date.dt.dayofweek.values.tolist() == pandas_df.date.dt.dayofweek.values.tolist()
