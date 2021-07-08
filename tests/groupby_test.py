import pytest
from common import *
import numpy as np
import vaex
import datetime


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


def test_groupby_space_in_name(df_local):
    df = df_local.extract()
    g = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    df.add_column('g with space', g)
    df['long name'] = df.x
    dfg = df.groupby(by='g with space', agg=[vaex.agg.sum(df['long name'])]).sort(df['g with space'])
    assert dfg.get_column_names() == ['g with space', 'long name_sum']


def test_groupby_space_in_agg(df_local):
    df = df_local.extract()
    g = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    df.add_column('g with space', g)
    df['long_name'] = df.x
    dfg = df.groupby(by='g with space', agg=[vaex.agg.mean(df.long_name)]).sort(df['g with space'])
    assert dfg.get_column_names() == ['g with space', 'long_name_mean']


def test_groupby_1d(ds_local):
    ds = ds_local.extract()
    g = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    ds.add_column('g', g)
    dfg = ds.groupby(by=ds.g, agg={'count': vaex.agg.count()}).sort('g')
    assert dfg.g.tolist() == [0, 1, 2]
    assert dfg['count'].tolist() == [4, 4, 2]


@pytest.mark.parametrize("as_category", [False, True])
@pytest.mark.parametrize("pre_sort", [False, True])
def test_groupby_sort_primitive(df_factory, as_category, pre_sort):
    df = df_factory(g=[1, 1, 1, 1, 0, 0, 0, 0, 2, 2])
    if as_category:
        df = df.ordinal_encode('g')
    dfg = df.groupby(by=vaex.groupby.Grouper(df.g, sort=True, pre_sort=pre_sort), agg={'count': vaex.agg.count()})
    assert dfg.g.tolist() == [0, 1, 2]
    assert dfg['count'].tolist() == [4, 4, 2]


@pytest.mark.parametrize("as_category", [False, True])
def test_groupby_sort_string(df_factory, as_category):
    df = df_factory(g=['a', None, 'c', 'c', 'a', 'a', 'b', None, None, None])
    if as_category:
        df = df.ordinal_encode('g')
    dfg = df.groupby(by='g', sort=True, agg={'count': vaex.agg.count()})
    assert dfg.g.tolist() == [None, 'a', 'b', 'c']
    assert dfg['count'].tolist() == [4, 3, 1, 2]


@pytest.mark.parametrize("auto_encode", [False, True])
def test_groupby_1d_cat(ds_local, auto_encode):
    df = ds_local.extract()
    g = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    df.add_column('g', g)
    df.categorize('g', labels=['cat', 'dog', 'snake'], inplace=True)
    df = df._future() if auto_encode else df
    dfg = df.groupby(by=df.g, agg='count')

    assert dfg.g.tolist() == ['cat', 'dog', 'snake']
    assert dfg['count'].tolist() == [4, 4, 2]

    with pytest.raises(vaex.RowLimitException, match='.*Resulting grouper.*'):
        df.groupby(df.g, row_limit=1)


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



@pytest.mark.parametrize("assume_sparse", [True, False])
def test_groupby_2d_cat(df_factory, assume_sparse):
    g = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
    h = [5, 5, 5, 6, 5, 5, 5, 5, 6, 6]
    df = df_factory(g=g, h=h)
    df.categorize('g', inplace=True)
    df.categorize('h', inplace=True)
    dfg = df.groupby(by=[df.g, df.h], agg={'count': vaex.agg.count()}, sort=True)
    assert dfg.g.tolist() == [0, 0, 1, 2]
    assert dfg['count'].tolist() == [3, 1, 4, 2]

    dff = df[df.g != 2]
    dfg = dff.groupby(by=[dff.g, dff.h], agg={'count': vaex.agg.count()}, sort=True)
    assert dfg.g.tolist() == [0, 0, 1]
    assert dfg['count'].tolist() == [3, 1, 4]


def test_combined_grouper_over64bit():
    bits = [15, 16, 17] * 2
    assert sum(bits) > 64
    N = 2**max(bits)
    def unique_ints(offset, bit):
        # create 2**bits unique ints
        ar = np.full(N, offset, dtype='int32')
        n = 2**bit
        ar[:n] = np.arange(offset, offset + n)
        return ar
    arrays = {f'x_{i}': unique_ints(i, bit) for i, bit in enumerate(bits)}
    names = list(arrays)
    df = vaex.from_dict(arrays)
    grouper = df.groupby(names)
    dfg = grouper.agg('count')
    for i, bit in enumerate(bits):
        xi = dfg[f'x_{i}'].to_numpy()
        assert len(xi) == N
        xiu = np.unique(xi)
        Ni = 2**bits[i]
        assert len(xiu) == Ni
    assert dfg['count'].sum() == N
    with pytest.raises(vaex.RowLimitException, match='.* >= 2 .*'):
        df.groupby(names, row_limit=2)
    with pytest.raises(vaex.RowLimitException):
        df.groupby([names[0]], row_limit=2**bits[0]-1)



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
    pandas_g = df.to_pandas_df(array_type='numpy').groupby('s').std(ddof=0).g.tolist()
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
        group = df.groupby(by=df.h).agg({'h_count': 'count'})
        # second time it uses a new set, this caused a bug
        # see https://github.com/vaexio/vaex/pull/233
        group = df.groupby(by=df.h).agg({'h_count': 'count'})
        group_sort = group.sort(by='h_count', ascending=False)

        assert vc.values.tolist() == group_sort['h_count'].values.tolist(), 'counts are not correct.'
        assert vc.index.tolist() == group_sort['h'].values.tolist(), 'the indices of the counts are not correct.'


@pytest.mark.parametrize("assume_sparse", [True, False])
def test_groupby_iter(assume_sparse):
    # ds = ds_local.extract()
    g = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1], dtype='int32')
    s = np.array(list(map(str, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2])))
    df = vaex.from_arrays(g=g, s=s)
    groupby = df.groupby('g')
    assert set(groupby.groups) == {(0, ), (1, )}
    dfs = list(groupby)
    assert dfs[0][0] == (0, )
    assert dfs[0][1].g.tolist() == [0] * 5
    assert dfs[1][0] == (1, )
    assert dfs[1][1].g.tolist() == [1] * 5

    groupby = df.groupby(['g', 's'], sort=True, assume_sparse=assume_sparse)
    assert set(groupby.groups) == {(0, '0'), (1, '1'), (0, '2'), (1, '2')}
    dfs = list(groupby)
    assert dfs[0][0] == (0, '0')
    assert dfs[0][1].g.tolist() == [0] * 4
    assert dfs[1][0] == (0, '2')
    assert dfs[1][1].g.tolist() == [0] * 1


def test_groupby_datetime():
    data = {'z': [2, 4, 8, 10],
            't': [np.datetime64('2020-01-01'),
                  np.datetime64('2020-01-01'),
                  np.datetime64('2020-02-01'),
                  np.datetime64('2020-02-01')]
            }

    df = vaex.from_dict(data)
    dfg = df.groupby(by='t').agg({'z': 'mean'})

    assert dfg.column_count() == 2
    assert dfg.z.tolist() == [3, 9]
    assert dfg.t.dtype.is_datetime
    assert set(dfg.t.tolist()) == {datetime.date(2020, 1, 1), datetime.date(2020, 2, 1)}


def test_groupby_state(df_factory, rebuild_dataframe):
    df = df_factory(g=[0, 0, 0, 1, 1, 2], x=[1, 2, 3, 4, 5, 6])._future()
    dfg = df.groupby(by=df.g, agg={'count': vaex.agg.count(), 'sum': vaex.agg.sum('x')})
    dfg.sort('g')  # TODO: sort it not yet implemented in the dataset state
    assert dfg.g.tolist() == [0, 1, 2]
    assert dfg['count'].tolist() == [3, 2, 1]
    assert dfg['sum'].tolist() == [1+2+3, 4+5, 6]

    dfg = dfg._future()  # to support rebuilding

    assert rebuild_dataframe(dfg.hashed()).dataset.hashed() == dfg.dataset.hashed()
    dfg = dfg.hashed()

    df = df_factory(g=[0, 0, 0, 1, 1, 2], x=[2, 3, 4, 5, 6, 7])._future()
    df.state_set(dfg.state_get())
    # import pdb; pdb.set_trace()
    assert df.g.tolist() == [0, 1, 2]
    assert df['count'].tolist() == [3, 2, 1]
    assert df['sum'].tolist() == [1+2+3, 4+5, 6]
