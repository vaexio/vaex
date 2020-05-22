from common import *
import numpy as np
import vaex


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

    dfg = df.groupby(by, agg={'z': vaex.agg.sum(df.y)})
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

    dfg = df.groupby(by, agg=vaex.agg.sum('y'))
    assert dfg.y_sum.tolist() == sum_answer
    assert "t_sum" not in dfg.get_column_names()

    dfg = df.groupby(by, agg=vaex.agg.sum(df.y))
    assert dfg.y_sum.tolist() == sum_answer
    assert "t_sum" not in dfg.get_column_names()

    # coverage only
    dfg = df.groupby(df.y, agg=vaex.agg.mean(df.y))

    dfg = df.groupby(by).agg({'y': 'sum'})
    assert dfg.y.tolist() == [y[k*7:(k+1)*7].sum() for k in range(5)]

    dfg = df.groupby(by).agg({'y': 'sum'})
    assert dfg.y.tolist() == [y[k*7:(k+1)*7].sum() for k in range(5)]

    dfg = df.groupby(by, 'sum')
    assert dfg.y_sum.tolist() == sum_answer


def test_groupby_long_name(df_local):
    df = df_local.extract()
    g = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    df.add_column('g', g)
    df['long_name'] = df.x
    dfg = df.groupby(by=df.g, agg=[vaex.agg.mean(df.long_name)]).sort('g')
    # bugfix check for mixing up the name
    assert 'long_name_mean' in dfg


def test_groupby_1d(ds_local):
    ds = ds_local.extract()
    g = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    ds.add_column('g', g)
    dfg = ds.groupby(by=ds.g, agg={'count': vaex.agg.count()}).sort('g')
    assert dfg.g.tolist() == [0, 1, 2]
    assert dfg['count'].tolist() == [4, 4, 2]

def test_groupby_1d_cat(ds_local):
    ds = ds_local.extract()
    g = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    ds.add_column('g', g)
    ds.categorize('g', labels=['cat', 'dog', 'snake'], inplace=True)
    dfg = ds.groupby(by=ds.g, agg='count')

    assert dfg.g.tolist() == ['cat', 'dog', 'snake']
    assert dfg['count'].tolist() == [4, 4, 2]



def test_groupby_1d_nan(ds_local):
    ds = ds_local.extract()
    g = np.array([0, 0, 0, 0, 1, 1, 1, np.nan, 2, 2])
    ds.add_column('g', g)
    dfg = ds.groupby(by=ds.g, agg={'count': vaex.agg.count()}).sort('g')
    assert dfg.g.tolist()[:-1] == [0, 1, 2]  # last item is nan
    assert dfg['count'].tolist() == [4, 3, 2, 1]


def test_binby_1d(ds_local):
    ds = ds_local.extract()
    g = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    ds.add_column('g', g)
    ar = ds.binby(by=ds.g, agg={'count': vaex.agg.count()})

    assert ar.coords['g'].values.tolist() == [0, 1, 2]
    assert ar.coords['statistic'].values.tolist() == ["count"]
    assert ar.dims == ('statistic', 'g')
    assert ar.data.tolist() == [[4, 4, 2]]


def test_binby_1d_cat(ds_local):
    ds = ds_local.extract()
    g = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    ds.add_column('g', g)
    ds.categorize('g', labels=['cat', 'dog', 'snake'], inplace=True)
    ar = ds.binby(by=ds.g, agg=vaex.agg.count())

    assert ar.coords['g'].values.tolist() == ['cat', 'dog', 'snake']
    assert ar.data.tolist() == [4, 4, 2]


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
    g = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1], dtype='int32')
    s = np.array(list(map(str, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2])))
    df = vaex.from_arrays(g=g, s=s)
    groupby = df.groupby('s')
    dfg = groupby.agg({'g': 'mean'}).sort('s')
    assert dfg.s.tolist() == ['0', '1', '2']
    assert dfg.g.tolist() == [0, 1, 0.5]

    dfg2 = df.groupby('s', {'g': 'mean'}).sort('s')
    assert dfg._equals(dfg2)


def test_groupby_std():
    g = np.array([9, 2, 3, 4, 0, 1, 2, 3, 2, 5], dtype='int32')
    s = np.array(list(map(str, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2])))
    df = vaex.from_arrays(g=g, s=s)
    groupby = df.groupby('s')
    dfg = groupby.agg({'g': 'std'})
    assert dfg.s.tolist() == ['0', '1', '2']
    pandas_g = df.to_pandas_df().groupby('s').std(ddof=0).g.tolist()
    np.testing.assert_array_almost_equal(dfg.g.tolist(), pandas_g)


def test_groupby_count_string():
    g = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    s = np.array(list(map(str, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2])))
    df = vaex.from_arrays(g=g, s=s)
    groupby = df.groupby('s')
    dfg = groupby.agg({'m': vaex.agg.count('s')})
    assert dfg.s.tolist() == ['0', '1', '2']
    assert dfg.m.tolist() == [4, 4, 2]


@pytest.mark.skip(reason='not yet supported')
def test_groupby_mode():
    animals = ['dog', 'dog', 'cat', 'cat', 'dog', 'mouse', 'mouse', 'cat', 'cat', 'dog']
    nums = [1, 2, 2, 1, 2, 2, 3, 3, 3, 1]
    vehicles = ['car', 'bus', 'car', 'bus', 'car', 'bus', 'plane', 'bus', 'plane', 'car']
    df = vaex.from_arrays(animals=animals, nums=nums, vehicles=vehicles)
    groupby = df.groupby('nums')
    dfg = groupby.agg({'animals': 'mode',
                       'vehicles': 'mode'})
    # Simple case
    assert dfg.animals.tolist() == ['dog', 'dog', 'cat']
    # Case when there is no clear mode in one sample
    grouped_vehicles = dfg.vehicles.tolist()
    assert grouped_vehicles[0] == 'car'
    assert set(grouped_vehicles[1]) == set({'bus', 'car'})
    assert grouped_vehicles[2] == 'plane'


@pytest.mark.skip(reason='not yet supported')
def test_grouby_mode_string():
    animals = ['dog', 'dog', 'cat', 'cat', 'dog', 'mouse', 'mouse', 'cat', 'cat', 'dog']
    nums = [1, 2, 2, 1, 2, 2, 3, 3, 3, 1]
    vehicles = ['car', 'bus', 'car', 'bus', 'car', 'bus', 'plane', 'bus', 'plane', 'car']
    df = vaex.from_arrays(animals=animals, nums=nums, vehicles=vehicles)
    groupby = df.groupby('vehicles')
    dfg = groupby.agg({'animals': 'mode',
                       'nums': 'mode'})

    grouped_animals = dfg.animals.tolist()
    assert grouped_animals[0] == 'cat'
    assert grouped_animals[1] == 'dog'
    assert set(grouped_animals[2]) == set({'cat', 'mouse'})   # Special case when no mode is found

    grouped_nums = dfg.nums.tolist()
    assert grouped_nums[0] == 2
    assert set(grouped_nums[1]) == set({1, 2})
    assert grouped_nums[2] == 3  # Special case when no mode is found


def test_groupby_same_result():
    h = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], dtype=int)
    df = vaex.from_arrays(h=h)

    # Compare value_counts with the groupby counts for the hour column
    vc = df.h.value_counts()

    with small_buffer(df):
        group = df.groupby(by=df.h).agg({'h': 'count'})
        # second time it uses a new set, this caused a bug
        # see https://github.com/vaexio/vaex/pull/233
        group = df.groupby(by=df.h).agg({'h': 'count'})
        group_sort = group.sort(by='count', ascending=False)

        assert vc.values.tolist() == group_sort['count'].values.tolist(), 'counts are not correct.'
        assert vc.index.tolist() == group_sort['h'].values.tolist(), 'the indices of the counts are not correct.'
