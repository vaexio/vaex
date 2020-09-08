from common import *
import numpy as np


def test_datetime_operations():
    date = np.array([np.datetime64('2009-10-12T03:31:00'),
                    np.datetime64('2016-02-11T10:17:34'),
                    np.datetime64('2015-11-12T11:34:22'),
                    np.datetime64('2003-03-03T00:33:15'),
                    np.datetime64('2014-07-23T15:08:05'),
                    np.datetime64('2011-01-01T07:02:01')], dtype='<M8[ns]')

    df = vaex.from_arrays(date=date)._readonly()
    pandas_df = df.to_pandas_df()

    assert df.date.dt.hour.tolist() == pandas_df.date.dt.hour.values.tolist()
    assert df.date.dt.minute.tolist() == pandas_df.date.dt.minute.values.tolist()
    assert df.date.dt.second.tolist() == pandas_df.date.dt.second.values.tolist()
    assert df.date.dt.day.tolist() == pandas_df.date.dt.day.values.tolist()
    assert df.date.dt.day_name.tolist() == pandas_df.date.dt.day_name().values.tolist()
    assert df.date.dt.month.tolist() == pandas_df.date.dt.month.values.tolist()
    assert df.date.dt.month_name.tolist() == pandas_df.date.dt.month_name().values.tolist()
    assert df.date.dt.quarter.tolist() == pandas_df.date.dt.quarter.values.tolist()
    assert df.date.dt.year.tolist() == pandas_df.date.dt.year.values.tolist()
    assert df.date.dt.is_leap_year.tolist() == pandas_df.date.dt.is_leap_year.values.tolist()
    assert any(df.date.dt.is_leap_year.tolist())
    assert df.date.dt.weekofyear.tolist() == pandas_df.date.dt.weekofyear.values.tolist()
    assert df.date.dt.dayofyear.tolist() == pandas_df.date.dt.dayofyear.values.tolist()
    assert df.date.dt.dayofweek.tolist() == pandas_df.date.dt.dayofweek.values.tolist()
    assert df.date.dt.floor('H').tolist() == pandas_df.date.dt.floor('H').values.tolist()
    assert df.date.dt.date.tolist() == pandas_df.date.dt.date.values.tolist()


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
    assert df.mean(df.date) < np.datetime64('2016-02-11T10:17:34')
    assert df.mean(df.date) > date[0]


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


def test_timedelta_arithmetics():
    x = np.array(['2019-01-04T21:23:00', '2019-02-04T05:00:10', '2019-03-04T15:15:15', '2019-06-21T10:31:15'],
                 dtype=np.datetime64)
    y = np.array(['2018-06-14T12:11:00', '2019-02-02T22:19:00', '2017-11-18T10:11:19', '2019-07-12T11:00:00'],
                 dtype=np.datetime64)

    df = vaex.from_arrays(x=x, y=y)
    df['diff'] = df.x-df.y
    df['diff_dev_hours'] = df.diff / np.timedelta64(1, 'h')
    df['diff_add_days'] = df.diff + np.timedelta64(5, 'D')

    # normal numerical/numpy values
    diff = df.x.values-df.y.values
    diff_dev_hours = diff / np.timedelta64(1, 'h')
    diff_add_days = diff + np.timedelta64(5, 'D')

    # compare vaex to numerical results
    assert diff_dev_hours.tolist() == df['diff_dev_hours'].values.tolist()
    assert diff_add_days.tolist() == df['diff_add_days'].values.tolist()

    # check the min/max values for the TimeDelta column
    assert df.diff.min() == df.diff.values.min()
    assert df.diff.max() == df.diff.values.max()


def test_datetime_binary_operations():
    x = np.array(['2019-01-04T21:23:00', '2019-02-04T05:00:10', '2019-03-04T15:15:15', '2019-06-21T10:31:15'],
                 dtype=np.datetime64)
    y = np.array(['2018-06-14T12:11:00', '2019-02-02T22:19:00', '2017-11-18T10:11:19', '2019-07-12T11:00:00'],
                 dtype=np.datetime64)

    sample_date = np.datetime64('2019-03-15')
    df = vaex.from_arrays(x=x, y=y)

    # Try simple binary operations
    assert (df.x > sample_date).tolist() == list(df.x.values > sample_date)
    assert (df.x <= sample_date).tolist() == list(df.x.values <= sample_date)
    assert (df.x > df.y).tolist() == list(df.x.values > df.y.values)


@pytest.mark.skipif(vaex.utils.osname == 'windows',
                    reason="windows' snprintf seems buggy")
def test_create_datetime64_column_from_ints():
    year = np.array([2015, 2015, 2017])
    month = np.array([1, 2, 10])
    day = np.array([1, 3, 22])
    time = np.array([945, 1015, 30])
    df = vaex.from_arrays(year=year, month=month, day=day, time=time)

    df['hour'] = (df.time // 100 % 24).format('%02d')
    df['minute'] = (df.time % 100).format('%02d')

    expr = df.year.format('%4d') + '-' + df.month.format('%02d') + '-' + df.day.format('%02d') + 'T' + df.hour + ':' + df.minute
    assert expr.values.astype(np.datetime64).tolist() == expr.astype('datetime64').tolist()


def test_create_datetime64_column_from_str():
    year = np.array(['2015', '2015', '2017'])
    month = np.array(['01', '02', '10'])
    day = np.array(['01', '03', '22'])
    hour = np.array(['09', '10', '00'])
    minute = np.array(['45', '15', '30'])
    df = vaex.from_arrays(year=year, month=month, day=day, hour=hour, minute=minute)

    expr = df.year + '-' + df.month + '-' + df.day + 'T' + df.hour + ':' + df.minute
    assert expr.values.astype(np.datetime64).tolist() == expr.astype('datetime64').tolist()
    assert expr.values.astype('datetime64[ns]').tolist() == expr.astype('datetime64[ns]').tolist()

def test_create_str_column_from_datetime64():
    date = np.array([np.datetime64('2009-10-12T03:31:00'),
                np.datetime64('2016-02-11T10:17:34'),
                np.datetime64('2015-11-12T11:34:22'),
                np.datetime64('2003-03-03T00:33:15'),
                np.datetime64('2014-07-23T15:08:05'),
                np.datetime64('2011-01-01T07:02:01')], dtype='<M8[ns]')

    df = vaex.from_arrays(date=date)
    pandas_df = df.to_pandas_df()

    date_format = "%Y/%m/%d"

    assert df.date.dt.strftime(date_format).values.tolist() == pandas_df.date.dt.strftime(date_format).values.tolist()