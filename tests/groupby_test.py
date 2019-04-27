from common import *
import collections
import numpy as np
import vaex
import pytest


def test_groupby_options():
    t = np.arange('2015-01-01', '2015-02-01', dtype=np.datetime64)
    y = np.arange(len(t))
    sum_answer = [y[k*7:(k+1)*7].sum() for k in range(5)]
    mean_answer = [y[k*7:(k+1)*7].mean() for k in range(5)]

    df = vaex.from_arrays(t=t, y=y)
    by = vaex.BinnerTime.per_week(df.t)

    dfg = df.groupby(by, agg={'y': 'sum'})
    assert dfg.y.tolist() == sum_answer
    dfg = df.groupby(by, agg={'y': vaex.agg.sum})
    assert dfg.y.tolist() == sum_answer

    dfg = df.groupby(by, agg={'z': vaex.agg.sum('y')})
    assert dfg.z.tolist() == sum_answer

    dfg = df.groupby(by, agg=[vaex.agg.sum('y')])
    assert dfg.y_sum.tolist() == sum_answer


    dfg = df.groupby(by, agg=[vaex.agg.sum('y'), vaex.agg.mean('y')])
    assert dfg.y_sum.tolist() == sum_answer
    assert dfg.y_mean.tolist() == mean_answer

    dfg = df.groupby(by, agg={'z': [vaex.agg.sum('y'), vaex.agg.mean('y')]})
    assert dfg.z_sum.tolist() == sum_answer
    assert dfg.z_mean.tolist() == mean_answer


    # default is to do all columns
    dfg = df.groupby(by, agg=[vaex.agg.sum, vaex.agg.mean])
    assert dfg.y_sum.tolist() == sum_answer
    assert dfg.y_mean.tolist() == mean_answer

    dfg = df.groupby(by, agg=vaex.agg.sum)
    assert dfg.y_sum.tolist() == sum_answer
    assert "t_sum" not in dfg.get_column_names()

    dfg = df.groupby(by).agg({'y': 'sum'})
    assert dfg.y.tolist() == [y[k*7:(k+1)*7].sum() for k in range(5)]

    dfg = df.groupby(by).agg({'y': 'sum'})
    assert dfg.y.tolist() == [y[k*7:(k+1)*7].sum() for k in range(5)]

    dfg = df.groupby(by, 'sum')
    assert dfg.y_sum.tolist() == sum_answer


def test_groupby_1d(ds_local):
    ds = ds_local.extract()
    g = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    ds.add_column('g', g)
    dfg = ds.groupby(by=ds.g, agg={'count': vaex.agg.count()}).sort('g')
    assert dfg.g.tolist() == [0, 1, 2]
    assert dfg['count'].tolist() == [4, 4, 2]


def test_binby_1d(ds_local):
    ds = ds_local.extract()
    g = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    ds.add_column('g', g)
    ar = ds.binby(by=ds.g, agg={'count': vaex.agg.count()})

    assert ar.coords['g'].values.tolist() == [0, 1, 2]
    assert ar.coords['statistic'].values.tolist() == ["count"]
    assert ar.dims == ('statistic', 'g')
    assert ar.data.tolist() == [[4, 4, 2]]


def test_binby_2d(ds_local):
    ds = ds_local.extract()
    g = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    h = np.array([5, 5, 5, 6, 5, 5, 5, 5, 6, 6])
    ds['g'] = g
    ds['h'] = h
    ar = ds.binby(by=[ds.g, ds.h], agg={'count': vaex.agg.count()})
    assert ar.coords['g'].values.tolist() == [0, 1, 2]
    assert ar.coords['h'].values.tolist() == [5, 6]
    assert ar.coords['statistic'].values.tolist() == ["count"]
    assert ar.dims == ('statistic', 'g', 'h')
    assert ar.data.tolist() == [[[3, 1], [4, 0], [0, 2]]]

    ar = ds.binby(by=[ds.g, ds.h], agg=vaex.agg.count())
    assert ar.dims == ('g', 'h')
    assert ar.data.tolist() == [[3, 1], [4, 0], [0, 2]]


def test_groupby_2d(ds_local):
    ds = ds_local.extract()
    g = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    h = np.array([5, 5, 5, 6, 5, 5, 5, 5, 6, 6])
    ds['g'] = g
    ds['h'] = h
    dfg = ds.groupby(by=[ds.g, ds.h], agg={'count': vaex.agg.count()}).sort('g')
    assert dfg.g.tolist() == [0, 0, 1, 2]
    assert dfg['count'].tolist() == [3, 1, 4, 2]


def test_groupby_datetime():
    t = np.arange('2015-01-01', '2015-02-01', dtype=np.datetime64)
    y = np.arange(len(t))

    df = vaex.from_arrays(t=t, y=y)

    dfg = df.groupby(vaex.BinnerTime.per_week(df.t), agg={'y': 'sum'})
    assert dfg.y.tolist() == [y[k*7:(k+1)*7].sum() for k in range(5)]

    # other syntax
    dfg = df.groupby(vaex.BinnerTime.per_week(df.t)).agg({'y': 'sum'})
    assert dfg.y.tolist() == [y[k*7:(k+1)*7].sum() for k in range(5)]


def test_groupby_datetime_quarter():
    t = np.arange('2015-01-01', '2016-01-02', dtype=np.datetime64)
    y = np.arange(len(t))

    df = vaex.from_arrays(t=t, y=y)
    dfg = df.groupby(vaex.BinnerTime.per_quarter(df.t)).agg({'y': 'sum'})

    values = dfg.y.tolist()
    assert len(values) == 5
    assert sum(values) == sum(y)


def test_groupby_count():
    # ds = ds_local.extract()
    g = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2], dtype='int32')
    s = np.array(list(map(str,[0, 0, 0, 0, 1, 1, 1, 1, 2, 2])))
    df = vaex.from_arrays(g=g, s=s)
    groupby = df.groupby('s')
    dfg = groupby.agg({'g': 'mean'}).sort('s')
    assert dfg.s.tolist() == ['0', '1', '2']
    assert dfg.g.tolist() == [0, 1, 2]

    dfg2 = df.groupby('s', {'g': 'mean'}).sort('s')
    assert dfg._equals(dfg2)

def test_groupby_count_string():
    g = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    s = np.array(list(map(str,[0, 0, 0, 0, 1, 1, 1, 1, 2, 2])))
    df = vaex.from_arrays(g=g, s=s)
    groupby = df.groupby('s')
    dfg = groupby.agg({'c': vaex.agg.count('s')})
    assert dfg.s.tolist() == ['0', '1', '2']
    assert dfg.c.tolist() == [4, 4, 2]

def test_groupby_datatime_buffer():
    # Generate data datetime data
    h = np.array([17, 12, 20,  7, 11, 21,  9,  9, 19,  1,  2,  1, 19,  9,  6, 10, 23,
                   8,  4, 13,  4, 10, 20, 15, 20,  4,  0,  6, 17, 14, 17,  5,  6,  4,
                   3, 21, 10, 20, 23,  5,  4, 19, 23,  8, 10,  1, 13,  3, 21, 18,  9,
                   3,  8, 14,  2, 20, 20, 15, 14, 18,  6, 19, 16,  3,  2,  4, 15, 23,
                   3, 16, 14,  6,  3, 18,  6,  5,  4, 15,  7,  8,  5, 22,  0,  1, 16,
                   7,  7, 18,  3, 14,  3,  1,  6,  3, 22,  0,  5,  0,  7,  0])
    time = np.zeros_like(h, dtype=np.object)
    for i, v in enumerate(h):
        tmp = np.datetime64(str('2007-01-01T%02i:00:00' % (v)))
        time[i] = tmp
    time  = time.astype(np.datetime64)

    # Create a DataFrame and set a small executor buffer
    df = vaex.from_arrays(time=time)
    df.executor.buffer_size=3
    # Extract a hour column from the datetime column
    df['h'] = df.time.dt.hour

    # Compare value_counts with the groupby counts for the hour column
    vc = df.h.value_counts()
    # Do this repeatedly, since each new execution to expose potential suffling between the rows
    for i in range(10):
        group = df.groupby(by=df.h).agg({'h': 'count'})
        group = group.sort(by='count', ascending=False)

    assert vc.values.tolist() == group['count'].values.tolist(), 'counts are not correct.'
    assert vc.index.tolist() == group['h'].values.tolist(), 'the indices of the counts are not correct.'
