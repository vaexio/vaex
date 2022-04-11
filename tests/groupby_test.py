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


def test_groupby_empty_1d(df_local):
    df = df_local
    dff = df[df.x > 1000]
    # floats
    assert dff.groupby('x', agg='count')['count'].tolist() == []
    # ints
    dff.groupby('mi', agg='count')['count'].tolist() == []
    # using groupers
    dff.groupby("name", agg="count")['count'].tolist() == []


@pytest.mark.parametrize("assume_sparse", ['auto', True, False])
def test_groupby_empty_combine(df_local, assume_sparse):
    df = df_local
    dff = df[df.x > 1000]
    assert dff.groupby(['x', 'y'], agg='count', assume_sparse=assume_sparse)['count'].tolist() == []
    assert dff.groupby(["name", "y"], agg="count", assume_sparse=assume_sparse)['count'].tolist() == []


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
    dfg = ds.groupby(by=ds.g, agg={'count': vaex.agg.count()}, sort=True)
    assert dfg.g.tolist() == [0, 1, 2]
    assert dfg['count'].tolist() == [4, 4, 2]


@pytest.mark.parametrize("as_category", [False, True])
@pytest.mark.parametrize("pre_sort", [False, True])
@pytest.mark.parametrize("ascending", [False, True])
def test_groupby_sort_primitive(df_factory, as_category, pre_sort, ascending):
    df = df_factory(g=[1, 1, 1, 1, 0, 0, 0, 0, 2, 2])
    if as_category:
        df = df.ordinal_encode("g")
    dfg = df.groupby(by=vaex.groupby.Grouper(df.g, sort=True, ascending=ascending, pre_sort=pre_sort), agg={"count": vaex.agg.count()})
    direction = 1 if ascending else -1
    assert dfg.g.tolist() == [0, 1, 2][::direction]
    assert dfg["count"].tolist() == [4, 4, 2][::direction]


@pytest.mark.parametrize("as_category", [False, True])
@pytest.mark.parametrize("ascending", [False, True])
def test_groupby_sort_string(df_factory, as_category, ascending):
    df = df_factory(g=['a', None, 'c', 'c', 'a', 'a', 'b', None, None, None])
    if as_category:
        df = df.ordinal_encode("g")
    dfg = df.groupby(by="g", sort=True, agg={"count": vaex.agg.count()}, ascending=ascending)
    expected = ["a", "b", "c", None] if ascending else ["c", "b", "a", None]
    assert dfg.g.tolist() == expected
    expected = [3, 1, 2, 4] if ascending else [2, 1, 3, 4]
    assert dfg["count"].tolist() == expected


@pytest.mark.parametrize("auto_encode", [False, True])
@pytest.mark.parametrize("pre_sort", [False, True])
def test_groupby_1d_cat(ds_local, auto_encode, pre_sort):
    df = ds_local.extract()
    g = np.array([0, 0, 0, 0, 2, 2, 1, 1, 1, 1, ])
    df.add_column('g', g)
    df.categorize('g', labels=['cat', 'dog', 'snake'], inplace=True)
    df = df._future() if auto_encode else df
    grouper = vaex.groupby.GrouperCategory(df.g, sort=True, pre_sort=pre_sort)
    dfg = df.groupby(by=grouper, agg='count')

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
    ar = ds.binby(by=ds.g, agg={'count': vaex.agg.count()}, sort=True)

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
    ar = ds.binby(by=[ds.g, ds.h], agg={'count': vaex.agg.count()}, sort=True)
    assert ar.coords['g'].values.tolist() == [0, 1, 2]
    assert ar.coords['h'].values.tolist() == [5, 6]
    assert ar.coords['statistic'].values.tolist() == ["count"]
    assert ar.dims == ('statistic', 'g', 'h')
    assert ar.data.tolist() == [[[3, 1], [4, 0], [0, 2]]]

    ar = ds.binby(by=[ds.g, ds.h], agg=vaex.agg.count(), sort=True)
    assert ar.dims == ('g', 'h')
    assert ar.data.tolist() == [[3, 1], [4, 0], [0, 2]]


@pytest.mark.parametrize("assume_sparse", [True, False])
def test_groupby_2d_full(df_factory, assume_sparse):
    g = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    h = np.array([5, 5, 5, 6, 5, 5, 5, 6, 5, 6])
    df = df_factory(g=g, h=h)
    dfg = df.groupby(by=[df.g, df.h], agg={'count': vaex.agg.count()}, sort=True, assume_sparse=assume_sparse)
    assert dfg.g.tolist() == [0, 0, 1, 1, 2, 2]
    assert dfg.h.tolist() == [5, 6, 5, 6, 5, 6]
    assert dfg['count'].tolist() == [3, 1, 3, 1, 1, 1]


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


@pytest.mark.parametrize("pre_sort", [False, True])
def test_groupby_with_missing(df_factory, pre_sort):
    df = df_factory(g=[0, 0, 1, 1, 1, None, None, 2])
    grouper = vaex.groupby.Grouper(df.g, pre_sort=pre_sort, sort=True)
    dfg = df.groupby(grouper, agg='count', sort=True, copy=False)
    assert dfg['g'].tolist() == [0, 1, 2, None]
    assert dfg['count'].tolist() == [2, 3, 1, 2]

    df = df_factory(g=[0, 0, 1, 1, 1, None, None], x=[3, None, 4, 5, None, 6, 7])
    grouper = vaex.groupby.Grouper(df.g, pre_sort=pre_sort, sort=True)
    dfg = df.groupby(grouper, agg={'sum': vaex.agg.sum('x')})
    assert dfg['g'].tolist() == [0, 1, None]
    assert dfg['sum'].tolist() == [3, 4+5, 6+7]



# we don't support pre_sort=False currently (not sure if possible)
#@pytest.mark.parametrize("pre_sort", [False, True])
@pytest.mark.parametrize("pre_sort", [True])
@pytest.mark.parametrize("assume_sparse", [True, False])
def test_groupby_with_missing_combine(df_factory, pre_sort, assume_sparse):
    df = df_factory(g1=[0, 0, 1, 1, 1, None, None, 2],
                    g2=[0, 1, 0, 1, 1, 0,    1,    0],
    )
    grouper1 = vaex.groupby.Grouper(df.g1, pre_sort=pre_sort, sort=True)
    grouper2 = vaex.groupby.Grouper(df.g2, pre_sort=pre_sort, sort=True)
    groupers = [grouper1, grouper2]
    dfg = df.groupby(groupers, agg='count', sort=True, copy=False, assume_sparse=assume_sparse)
    assert dfg['g1'].tolist() == [0, 0, 1, 1, 2, None, None]
    assert dfg['count'].tolist() == [1, 1, 1, 2, 1, 1, 1]

    df = df_factory(g=[0, 0, 1, 1, 1, None, None], x=[3, None, 4, 5, None, 6, 7])
    grouper = vaex.groupby.Grouper(df.g, pre_sort=pre_sort, sort=True)
    dfg = df.groupby(grouper, agg={'sum': vaex.agg.sum('x')})
    assert dfg['g'].tolist() == [0, 1, None]
    assert dfg['sum'].tolist() == [3, 4+5, 6+7]


def test_groupby_boolean_without_null(df_factory):
    df = df_factory(g=[False, False, True, True, True], x=[3, None, 4, 5, None])
    dfg = df.groupby('g', agg={'sum': vaex.agg.sum('x')}, sort=True)
    assert dfg['g'].tolist() == [False, True]
    assert dfg['sum'].tolist() == [3, 4+5]


def test_groupby_boolean_with_null(df_factory):
    df = df_factory(g=[False, False, True, True, True, None, None], x=[3, None, 4, 5, None, 6, 7])
    dfg = df.groupby('g', agg={'sum': vaex.agg.sum('x')}, sort=True)
    assert dfg['g'].tolist() == [False, True, None]
    assert dfg['sum'].tolist() == [3, 4+5, 6+7]


def test_groupby_uint8_with_null(df_factory):
    df = df_factory(g=[0, 1, 1, 255, None, None], x=[3, None, 4, 5, None, 6])
    df['g'] = df['g'].astype('uint8')
    dfg = df.groupby('g', agg={'sum': vaex.agg.sum('x')}, sort=True)
    assert dfg['g'].tolist() == [0, 1, 255, None]
    assert dfg['sum'].tolist() == [3, 4, 5, 6]


def test_groupby_int8_with_null(df_factory):
    df = df_factory(g=[-10, 1, 1, 127, None, None], x=[3, None, 4, 5, None, 6])
    df['g'] = df['g'].astype('int8')
    dfg = df.groupby('g', agg={'sum': vaex.agg.sum('x')}, sort=True)
    assert dfg['g'].tolist() == [-10, 1, 127, None]
    assert dfg['sum'].tolist() == [3, 4, 5, 6]

    dfg = df.groupby("g", agg={"sum": vaex.agg.sum("x")}, sort=True, ascending=False)
    assert dfg["g"].tolist() == [127, 1, -10, None]
    assert dfg["sum"].tolist() == [5, 4, 3, 6]

    dfg = df.groupby('g', agg={'sum': vaex.agg.sum('x')}, sort=False)
    assert set(dfg['g'].tolist()) == set([-10, 1, 127, None])
    assert set(dfg['sum'].tolist()) == set([3, 4, 5, 6])

    for sort in [True, False]:
        binner = vaex.groupby.BinnerInteger(df.g, dropmissing=True)
        dfg = df.groupby(binner, agg={'sum': vaex.agg.sum('x')}, sort=sort)
        assert set(dfg['g'].tolist()) == set([-10, 1, 127])
        assert set(dfg['sum'].tolist()) == set([3, 4, 5])


def test_groupby_int_binner_with_null(df_factory):
    large = 2 ** 16
    df = df_factory(g=[-10, 1, 1, 127, large, None], x=[3, None, 4, 5, 99, 6])

    binner = vaex.groupby.BinnerInteger(df.g, min_value=-10, max_value=large, sort=True, ascending=True)
    dfg = df.groupby(binner, agg={"sum": vaex.agg.sum("x")})
    assert dfg["g"].tolist() == [-10, 1, 127, large, None]
    assert dfg["sum"].tolist() == [3, 4, 5, 99, 6]

    binner = vaex.groupby.BinnerInteger(df.g, min_value=-10, max_value=large, sort=True, ascending=False)
    dfg = df.groupby(binner, agg={"sum": vaex.agg.sum("x")})
    assert dfg["g"].tolist() == [large, 127, 1, -10, None]
    assert dfg["sum"].tolist() == [99, 5, 4, 3, 6]


def test_groupby_int_binner_with_offset_combine(df_factory):
    offset = 2**16
    df = df_factory(x=np.array([0, 1, 1, 2], dtype='int32') + offset, y=np.array([0, 0, 0, 1], dtype='int32'))

    binner_x = vaex.groupby.BinnerInteger(df.x, min_value=offset, max_value=offset+2, sort=True, ascending=True)
    binner_y = vaex.groupby.BinnerInteger(df.y, min_value=0, max_value=1, sort=True, ascending=True)
    dfg = df.groupby([binner_x, binner_y], agg={"sum": vaex.agg.sum("x")}, assume_sparse=True, sort=True)
    assert dfg["x"].tolist() == [offset+0, offset+1, offset+2]
    assert dfg["y"].tolist() == [0,     0,   1]
    assert dfg["sum"].tolist() ==[offset, (offset+1)*2, offset+2]



def test_groupby_simplify_to_int_binner():
    x = np.arange(1024)
    x[:200] = 0  # <20% the same
    df = vaex.from_arrays(x=x)
    grouper = df.groupby("x")
    assert isinstance(grouper.by[0], vaex.groupby.BinnerInteger)

def test_groupby_std():
    g = np.array([9, 2, 3, 4, 0, 1, 2, 3, 2, 5], dtype='int32')
    s = np.array(list(map(str, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2])))
    df = vaex.from_arrays(g=g, s=s)
    groupby = df.groupby('s', sort=True)
    dfg = groupby.agg({'g': 'std'})
    assert dfg.s.tolist() == ['0', '1', '2']
    pandas_g = df.to_pandas_df(array_type='numpy').groupby('s').std(ddof=0).g.tolist()
    np.testing.assert_array_almost_equal(dfg.g.tolist(), pandas_g)


def test_groupby_count_string():
    g = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    s = np.array(list(map(str, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2])))
    df = vaex.from_arrays(g=g, s=s)
    groupby = df.groupby('s', sort=True)
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
    dfg = df.groupby(by='t', sort=True).agg({'z': 'mean'})

    assert dfg.column_count() == 2
    assert dfg.z.tolist() == [3, 9]
    assert dfg.t.dtype.is_datetime
    assert set(dfg.t.tolist()) == {datetime.date(2020, 1, 1), datetime.date(2020, 2, 1)}


def test_groupby_state(df_factory, rebuild_dataframe):
    df = df_factory(g=[0, 0, 0, 1, 1, 2], x=[1, 2, 3, 4, 5, 6])._future()
    dfg = df.groupby(by=df.g, agg={'count': vaex.agg.count(), 'sum': vaex.agg.sum('x')}, sort=True)
    assert dfg.g.tolist() == [0, 1, 2]
    assert dfg['count'].tolist() == [3, 2, 1]
    assert dfg['sum'].tolist() == [1+2+3, 4+5, 6]

    dfg = dfg._future()  # to support rebuilding

    assert rebuild_dataframe(dfg.hashed()).dataset.hashed() == dfg.dataset.hashed()
    dfg = dfg.hashed()

    df = df_factory(g=[0, 0, 0, 1, 1, 2], x=[2, 3, 4, 5, 6, 7])._future()
    df.state_set(dfg.state_get())
    assert df.g.tolist() == [0, 1, 2]
    assert df['count'].tolist() == [3, 2, 1]
    assert df['sum'].tolist() == [1+2+3, 4+5, 6]


def test_old_years():
    df = vaex.from_arrays(t=[np.datetime64('1900-01-01'), np.datetime64('1945-01-01'), np.datetime64('2020-02-01')])
    assert df.groupby(df.t.astype('datetime64[Y]'), 'count')['count'].tolist() == [1, 1, 1]


@pytest.mark.parametrize("binby", [False, True])
def test_delay_ordinal(binby):
    df = vaex.from_arrays(x=[1, 2, 2, 3, 3, 3], s=["aap", "aap", "aap", "noot", "noot", "mies"])
    df.ordinal_encode("x", inplace=True)
    df.ordinal_encode("s", inplace=True)
    df.executor.passes = 0
    if binby:
        ar1 = df.binby('x', agg='count', delay=True)
        ar2 = df.binby('s', agg='count', delay=True)
    else:
        df1 = df.groupby('x', agg='count', delay=True)
        df2 = df.groupby('s', agg='count', delay=True)
    df.execute()
    assert df.executor.passes == 1


def test_delay_non_ordinal_1d():
    df = vaex.from_arrays(s=["aap", "aap", "aap", "noot", "noot", "mies"])
    df.executor.passes = 0
    df2 = df.groupby('s').agg('count', delay=True)
    df.execute()
    assert df.executor.passes == 2


def test_delay_non_ordinal_2d():
    df = vaex.from_arrays(x=[1, 2, 2, 3, 3, 3], s=["aap", "aap", "aap", "noot", "noot", "mies"])
    df.executor.passes = 0
    df1 = df.groupby('x', agg='count', delay=True)
    df2 = df.groupby('s', agg='count', delay=True)
    df.execute()
    assert df.executor.passes == 2


def test_binner_1d(df_factory):
    df = df_factory(x=[0.1, 1.1, 1.2, 2.2, 2.5, 2.7])
    binner = vaex.groupby.Binner(df.x, 0, 3, bins=3)
    dfg = df.groupby(binner, agg='count')

    assert dfg.x.tolist() == [0.5, 1.5, 2.5]
    assert dfg['count'].tolist() == [1, 2, 3]
    xar = df.binby(binner, agg='count')
    assert xar.data.tolist() == [1, 2, 3]


def test_binner_2d(df_factory):
    df = df_factory(x=[0.1, 1.1, 1.2, 2.2, 2.5, 2.7, 100], g=[0, 0, 1, 0, 1, 1, 1])
    binner = vaex.groupby.Binner(df.x, 0, 3, bins=3)
    grouper = vaex.groupby.Grouper(df.g, sort=True)
    dfg = df.groupby([binner, grouper], agg='count', assume_sparse=False)
    assert dfg.x.tolist() == [0.5, 1.5, 1.5, 2.5, 2.5]
    assert dfg.g.tolist() == [0, 0, 1, 0, 1]
    assert dfg['count'].tolist() == [1, 1, 1, 1, 2]

    with pytest.raises(NotImplementedError):
        dfg = df.groupby([binner, grouper], agg='count', assume_sparse=True)
    assert dfg['count'].tolist() == [1, 1, 1, 1, 2]

    xar = df.binby([binner, grouper], agg='count')
    assert xar.coords['x'].data.tolist() == [0.5, 1.5, 2.5]
    assert xar.coords['g'].data.tolist() == [0, 1]
    assert xar.data.tolist() == [[1, 0], [1, 1], [1, 2]]


def test_binby_non_identifiers():
    df = vaex.from_dict({"#": [1, 2, 3]})
    binner = vaex.groupby.Binner(df['#'], 0, 3, bins=3)
    xar = df.binby(binner, agg="count")
    assert xar.coords['#'].data.tolist() == [0.5, 1.5, 2.5]
    assert xar.data.tolist() == [0, 1, 1]


def test_groupby_limited_plain(df_factory):
    df = df_factory(x=[1, 2, 2, 3, 3, 4], s=["aap", "aap", "aap", "noot", "noot", "mies"])
    g = vaex.groupby.GrouperLimited(df.s, values=["noot", "aap"], keep_other=True, other_value="others", label="type", sort=True)
    dfg = df.groupby(g, agg={"sum": vaex.agg.sum("x")})
    assert dfg["type"].tolist() == ["aap", "noot", "others"]
    assert dfg["sum"].tolist() == [1 + 2 + 2, 3 + 3, 4]

    g = vaex.groupby.GrouperLimited(df.s, values=["aap", "noot"], keep_other=True, other_value="others", label="type", sort=True, ascending=False)
    dfg = df.groupby(g, agg={"sum": vaex.agg.sum("x")})
    assert dfg["type"].tolist() == ["noot", "aap", "others"]
    assert dfg["sum"].tolist() == [3 + 3, 1 + 2 + 2, 4]

    # not supported yet
    # g = vaex.groupby.GrouperLimited(df.s, values=['aap', 'noot'], keep_other=False, other_value='others', label="type")
    # dfg = df.groupby(g, agg={'sum': vaex.agg.sum('x')})
    # assert dfg['type'].tolist() == ['aap', 'noot']
    # assert dfg['sum'].tolist() == [1+2+2, 3+3]

def test_groupby_limited_with_missing(df_factory):
    df = df_factory(x=[1, 2, 2, 3, 3, 4, 9, 9], s=["aap", "aap", "aap", "noot", "noot", "mies", None, None])
    g = vaex.groupby.GrouperLimited(df.s, values=['aap', 'noot', None], keep_other=True, other_value='others', label="type")
    dfg = df.groupby(g, agg={'sum': vaex.agg.sum('x')})
    assert dfg['type'].tolist() == ['aap', 'noot', None, 'others']
    assert dfg['sum'].tolist() == [1+2+2, 3+3, 9+9, 4]


def test_groupby_limited_with_nan(df_factory):
    a = 1.2
    b = np.nan
    c = 3.4
    others = 42.
    df = df_factory(x=[1, 2, 2, 3, 3, 4, 9, 9], s=[a, a, a, b, b, c, None, None])
    g = vaex.groupby.GrouperLimited(df.s, values=[a, b, None], keep_other=True, other_value=others, label="type")
    dfg = df.groupby(g, agg={'sum': vaex.agg.sum('x')})
    # we don't check, because comparing nan is always false
    # assert dfg['type'].tolist() == [a, b,  None, others]
    assert dfg['sum'].tolist() == [1+2+2, 3+3, 9+9, 4]


@pytest.mark.parametrize("assume_sparse", [True, False])
@pytest.mark.parametrize("ascending1", [False, True])
@pytest.mark.parametrize("ascending2", [False, True])
def test_binner2d_limited_combine(df_factory, assume_sparse, ascending1, ascending2):
    df = df_factory(x=[1, 2, 2, 3, 3, 4], s=["aap", "aap", "aap", "noot", "noot", "mies"])
    g1 = vaex.groupby.GrouperLimited(df.s, values=["aap", "noot"], keep_other=True, other_value="others", label="type", sort=True, ascending=ascending1)
    g2 = vaex.groupby.Grouper(df.x, sort=True, ascending=ascending2)
    g = df.groupby([g1, g2], assume_sparse=assume_sparse, sort=True)
    dfg = g.agg({'sum': vaex.agg.sum('x')})
    if ascending1:
        if ascending2:
            indices = [0, 1, 2, 3]
        else:
            indices = [1, 0, 2, 3]
    else:
        if ascending2:
            indices = [2, 0, 1, 3]
        else:
            indices = [2, 1, 0, 3]

    def take(ar, indices):
        return np.take(ar, indices).tolist()

    assert dfg["type"].tolist() == take(["aap", "aap", "noot", "others"], indices)
    assert dfg["x"].tolist() == take([1, 2, 3, 4], indices)
    assert dfg["sum"].tolist() == take([1, 4, 6, 4], indices)


@pytest.mark.parametrize("assume_sparse", [True, False])
@pytest.mark.parametrize("ascending1", [False, True])
@pytest.mark.parametrize("ascending2", [False, True])
def test_groupby_sort_ascending(df_factory, ascending1, ascending2, assume_sparse):
    df = df_factory(x=[1, 2, 2, 3, 3, 4], s=["aap", "aap", "aap", "noot", "noot", "mies"])
    g = df.groupby(["s", "x"], sort=True, ascending=[ascending1, ascending2], assume_sparse=assume_sparse)
    dfg = g.agg({"sum": vaex.agg.sum("x")})
    if ascending1:
        if ascending2:
            indices = [0, 1, 2, 3]
        else:
            indices = [1, 0, 2, 3]
    else:
        if ascending2:
            indices = [3, 2, 0, 1]
        else:
            indices = [3, 2, 1, 0]

    def take(ar, indices):
        return np.take(ar, indices).tolist()

    assert dfg["s"].tolist() == take(["aap", "aap", "mies", "noot"], indices)
    assert dfg["x"].tolist() == take([1, 2, 4, 3], indices)
    assert dfg["sum"].tolist() == take([1, 4, 4, 6], indices)


def test_row_limit_sparse():
    x = np.arange(100) % 10
    y = np.arange(100) % 11
    # > 11 combinations, but not x and y seperately
    # we should force a 'compress' phase to detect the row_limit
    df = vaex.from_arrays(x=x, y=y)
    with pytest.raises(vaex.RowLimitException, match='.* would have >= 11 unique combinations.*'):
        df.groupby(['x', 'y'], assume_sparse=False, row_limit=11)


def test_describe_agg():
    df = vaex.datasets.titanic()
    res = df.groupby('pclass').describe(['age', df.sex])
    assert res.shape == (3, 9)
    assert res.get_column_names() == ['pclass',
                                      'age_count',
                                      'age_count_na',
                                      'age_mean',
                                      'age_std',
                                      'age_min',
                                      'age_max',
                                      'sex_count',
                                      'sex_count_na']
    assert res.age_count_na.tolist() == [39, 16, 208]
    assert res.age_max.tolist() == [80, 70, 74]

    # make it work without args
    res = df.groupby().describe(['age', df.sex])


def test_groupby_empty(df_factory):
    df = df_factory(x=[1, 2, 2, 3, 3, 4], s=["aap", "aap", "aap", "noot", "noot", "mies"])
    dfg = df.groupby(agg={"count": vaex.agg.count(), "first_x": vaex.agg.first("x"), "s": vaex.agg.list("s")})
    assert dfg["count"].tolist() == [6]
    assert dfg["first_x"].tolist() == [1]
    assert dfg["s"].tolist() == [df.s.tolist()]
