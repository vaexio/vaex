import vaex
import numpy as np
from common import *

# def test_count_multiple_selections():

def test_sum(df, ds_trimmed):
    df.select("x < 5")
    np.testing.assert_array_almost_equal(df.sum("x", selection=None), np.nansum(ds_trimmed.data.x))
    np.testing.assert_array_almost_equal(df.sum("x", selection=True), np.nansum(ds_trimmed.data.x[:5]))
    np.testing.assert_array_almost_equal(df.sum(df.x, selection=None), np.nansum(ds_trimmed.data.x))
    np.testing.assert_array_almost_equal(df.sum(df.x, selection=True), np.nansum(ds_trimmed.data.x[:5]))

    df.select("x > 5")
    np.testing.assert_array_almost_equal(df.sum("m", selection=None), np.nansum(ds_trimmed.data.m))
    np.testing.assert_array_almost_equal(df.sum("m", selection=True), np.nansum(ds_trimmed.data.m[6:]))
    np.testing.assert_array_almost_equal(df.m.sum(selection=True), np.nansum(ds_trimmed.data.m[6:]))

    df.select("x < 5")
    # convert to float
    x = ds_trimmed.x.to_numpy()
    y = ds_trimmed.data.y
    x_with_nan = x * 1
    x_with_nan[0] = np.nan
    ds_trimmed.columns["x"] = x_with_nan
    np.testing.assert_array_almost_equal(df.sum("x", selection=None), np.nansum(x))
    np.testing.assert_array_almost_equal(df.sum("x", selection=True), np.nansum(x[:5]))

    task = df.sum("x", selection=True, delay=True)
    df.execute()
    np.testing.assert_array_almost_equal(task.get(), np.nansum(x[:5]))


    np.testing.assert_array_almost_equal(df.sum("x", selection=None, binby=["x"], limits=[0, 10], shape=1), [np.nansum(x)])
    np.testing.assert_array_almost_equal(df.sum("x", selection=True, binby=["x"], limits=[0, 10], shape=1), [np.nansum(x[:5])])
    np.testing.assert_array_almost_equal(df.sum("x", selection=None, binby=["y"], limits=[0, 9**2+1], shape=1), [np.nansum(x)])
    np.testing.assert_array_almost_equal(df.sum("x", selection=True, binby=["y"], limits=[0, 9**2+1], shape=1), [np.nansum(x[:5])])
    np.testing.assert_array_almost_equal(df.sum("x", selection=None, binby=["x"], limits=[0, 10], shape=2), [np.nansum(x[:5]), np.nansum(x[5:])])
    np.testing.assert_array_almost_equal(df.sum("x", selection=True, binby=["x"], limits=[0, 10], shape=2), [np.nansum(x[:5]), 0])

    i = 7
    np.testing.assert_array_almost_equal(df.sum("x", selection=None, binby=["y"], limits=[0, 9**2+1], shape=2), [np.nansum(x[:i]), np.nansum(x[i:])])
    np.testing.assert_array_almost_equal(df.sum("x", selection=True, binby=["y"], limits=[0, 9**2+1], shape=2), [np.nansum(x[:5]), 0])

    i = 5
    np.testing.assert_array_almost_equal(df.sum("y", selection=None, binby=["x"], limits=[0, 10], shape=2), [np.nansum(y[:i]), np.nansum(y[i:])])
    np.testing.assert_array_almost_equal(df.sum("y", selection=True, binby=["x"], limits=[0, 10], shape=2), [np.nansum(y[:5]), 0])


def test_correlation_basics(df_local):
    df = df_local  # TODO: why does this not work with remote?
    correlation = df.correlation(df.y, df.y)
    np.testing.assert_array_almost_equal(correlation, 1.0)

    correlation = df.correlation(df.y, -df.y)
    np.testing.assert_array_almost_equal(correlation, -1.0)


def test_correlation(df_local):
    df = df_local  # TODO: why does this not work with remote?

    # convert to float
    x = df.x.to_numpy() #self.dataset_local.columns["x"][self.zero_index:10] = self.dataset_local.columns["x"][:10] * 1.
    y = df.y.to_numpy()
    def correlation(x, y):
        c = np.cov([x, y], bias=1)
        return c[0,1] / (c[0,0] * c[1,1])**0.5

    np.testing.assert_array_almost_equal(df.correlation([["x", "y"], ["x", "x**2"]], selection=None), [correlation(x, y), correlation(x, x**2)])

    df.select("x < 5")
    np.testing.assert_array_almost_equal(df.correlation("x", "y", selection=None), correlation(x, y))
    np.testing.assert_array_almost_equal(df.correlation("x", "y", selection=True), correlation(x[:5], y[:5]))

    np.testing.assert_array_almost_equal(df.correlation("x", "y", selection=None), correlation(x, y))
    np.testing.assert_array_almost_equal(df.correlation("x", "y", selection=True), correlation(x[:5], y[:5]))

    task = df.correlation("x", "y", selection=True, delay=True)
    df.execute()
    np.testing.assert_array_almost_equal(task.get(), correlation(x[:5], y[:5]))

    np.testing.assert_array_almost_equal(df.correlation("x", "y", selection=None, binby=["x"], limits=[0, 10], shape=1), [correlation(x, y)])
    np.testing.assert_array_almost_equal(df.correlation("x", "y", selection=True, binby=["x"], limits=[0, 10], shape=1), [correlation(x[:5], y[:5])])
    np.testing.assert_array_almost_equal(df.correlation("x", "y", selection=None, binby=["y"], limits=[0, 9**2+1], shape=1), [correlation(x, y)])
    np.testing.assert_array_almost_equal(df.correlation("x", "y", selection=True, binby=["y"], limits=[0, 9**2+1], shape=1), [correlation(x[:5], y[:5])])
    np.testing.assert_array_almost_equal(df.correlation("x", "y", selection=None, binby=["x"], limits=[0, 10], shape=2), [correlation(x[:5], y[:5]), correlation(x[5:], y[5:])])
    np.testing.assert_array_almost_equal(df.correlation("x", "y", selection=True, binby=["x"], limits=[0, 10], shape=2), [correlation(x[:5], y[:5]), np.nan])

    i = 7
    np.testing.assert_array_almost_equal(df.correlation("x", "y", selection=None, binby=["y"], limits=[0, 9**2+1], shape=2), [correlation(x[:i], y[:i]), correlation(x[i:], y[i:])])
    np.testing.assert_array_almost_equal(df.correlation("x", "y", selection=True, binby=["y"], limits=[0, 9**2+1], shape=2), [correlation(x[:5], y[:5]), np.nan])

    i = 5
    np.testing.assert_array_almost_equal(df.correlation("x", "y", selection=None, binby=["x"], limits=[0, 10], shape=2), [correlation(x[:i], y[:i]), correlation(x[i:], y[i:])])
    np.testing.assert_array_almost_equal(df.correlation("x", "y", selection=True, binby=["x"], limits=[0, 10], shape=2), [correlation(x[:i], y[:i]), np.nan])
    np.testing.assert_array_almost_equal(df.correlation("x", "y", selection=True, binby=["x"], limits=[[0, 10]], shape=2), [correlation(x[:i], y[:i]), np.nan])

    assert df.correlation("x", "y", selection=None, binby=["x"], shape=1) > 0
    assert  df.correlation("x", "y", selection=None, binby=["x"], limits="90%", shape=1) > 0
    assert  df.correlation("x", "y", selection=None, binby=["x"], limits=["90%"], shape=1) > 0
    assert  df.correlation("x", "y", selection=None, binby=["x"], limits="minmax", shape=1) > 0


def test_count_basics(df):
    # df = df_l
    y = df.y.to_numpy()
    x = df.x.to_numpy()
    counts = df.count(binby=df.x, limits=[0,10], shape=10)
    assert len(counts) == 10
    assert all(counts == 1), "counts is %r" % counts

    sums = df["y"].sum(binby=df.x, limits=[0,10], shape=10)
    assert len(sums) == 10
    assert(all(sums == y))

    df.select("x < 5")
    mask = x < 5

    counts = df["x"].count(binby=df.x, limits=[0,10], shape=10, selection=True)
    mod_counts = counts * 1.
    mod_counts[~mask] = 0
    assert(all(counts == mod_counts))

    mod_sums = y * 1.
    mod_sums[~mask] = 0
    sums = df["y"].sum(binby=df.x, limits=[0,10], shape=10, selection=True)
    assert(all(sums == mod_sums))

    # TODO: we may want to test this for a remote df
    # 2d
    x = np.array([0, 1, 0, 1])
    y = np.array([0, 0, 1, 1])
    df = vaex.from_arrays(x=x, y=y)
    counts = df.count(binby=[df.x, df.y], limits=[[0.,2.], [0.,2.]], shape=2)
    assert np.all(counts == 1)

    # 3d
    x = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    z = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    df = vaex.from_arrays(x=x, y=y, z=z)
    counts = df.count(binby=[df.x, df.y, df.z], limits=[[0.,2.], [0.,2.], [0.,2.]], shape=2)
    assert np.all(counts == 1)


def test_count_1d():
    x = np.array([-1, -2, 0.5, 1.5, 4.5, 5], dtype='f8')
    df = vaex.from_arrays(x=x)

    bins = 5
    binner = df._binner_scalar('x', [0, 5], bins)
    grid = vaex.superagg.Grid([binner])
    agg = vaex.agg.count(edges=True)
    grid = df._agg(agg, grid)
    assert grid.tolist() == [0, 2, 1, 1, 0, 0, 1, 1]


def test_count_types(ds_local):
    df = ds_local
    assert df.count(df.x) is not None
    assert df.count(df.datetime) is not None
    assert df.min(df.datetime) is not None
    assert df.max(df.datetime) is not None
    assert df.minmax(df.datetime) is not None
    assert df.std(df.datetime) is not None


def test_count_1d_ordinal():
    x = np.array([-1, -2, 0, 1, 4, 5], dtype='i8')
    df = vaex.from_arrays(x=x)

    bins = 5
    binner = df._binner_ordinal('x', 5)
    grid = vaex.superagg.Grid([binner])
    agg = vaex.agg.count(edges=True)
    grid = df._agg(agg, grid)
    assert grid.tolist() == [0, 2, 1, 1, 0, 0, 1, 1]



def test_mean_basics(df):
    x, y = df.mean([df.x, df.y])
    assert x == 4.5
    assert y == 28.5

    df.select("x < 3")
    x, y = df.mean([df.x, df.y], selection=True)
    assert x == 1
    assert y == 5/3


def test_minmax_local():
    x = np.arange(1, 10, 1)
    df = vaex.from_arrays(x=x)
    assert df.x.min() == 1
    assert df.x.max() == 9

    assert df[(df.x > 3) & (df.x < 7)]['x'].min() == (4)
    assert df[(df.x > 3) & (df.x < 7)]['x'].max() == (6)

    df = vaex.from_arrays(x=-x)
    assert df.x.max() == -1
    assert df.x.min() == -9



def test_minmax_basics(df):
    xmin, xmax = df["x"].minmax()
    np.testing.assert_array_almost_equal(xmin, 0)
    np.testing.assert_array_almost_equal(xmax, 9)

    np.testing.assert_array_almost_equal(df.minmax("x"), [0, 9.])
    np.testing.assert_array_almost_equal(df.minmax("y"), [0, 9.**2])
    np.testing.assert_array_almost_equal(df.minmax(["x", "y"]), [[0, 9.], [0, 9.**2]])

    df.select("x < 5")
    xmin2, xmax2 = df["x"].minmax(selection=True)
    np.testing.assert_array_almost_equal(xmin2, 0)
    np.testing.assert_array_almost_equal(xmax2, 4)

    np.testing.assert_array_almost_equal(df.minmax("x", selection=True), [0, 4])
    np.testing.assert_array_almost_equal(df.minmax("y", selection=True), [0, 4**2])
    np.testing.assert_array_almost_equal(df.minmax(["x", "y"], selection=True), [[0, 4], [0, 4**2]])
    np.testing.assert_array_almost_equal(df.x.minmax(selection=True), [0, 4])
    np.testing.assert_array_almost_equal(df.x.min(selection=True), 0)
    np.testing.assert_array_almost_equal(df.x.max(selection=True), 4)

    task = df.minmax("x", selection=True, delay=True)
    df.execute()
    np.testing.assert_array_almost_equal(task.get(), [0, 4])

    return  # TODO: below fails for remote dataframes
    np.testing.assert_array_almost_equal(df.minmax("x", selection=None, binby=["x"], limits="minmax", shape=1), [[0, 8]])
    np.testing.assert_array_almost_equal(df.minmax("x", selection=True, binby=["x"], limits="minmax", shape=1), [[0, 3]])

    np.testing.assert_array_almost_equal(df.minmax("x", selection=None, binby=["x"], limits="minmax", shape=2), [[0, 4], [5, 8]])
    np.testing.assert_array_almost_equal(df.minmax("x", selection=True, binby=["x"], limits="minmax", shape=2), [[0, 1], [2, 3]])


def test_minmax_all_dfs(df):
    vmin, vmax = df.minmax(df.x)
    assert df.min(df.x) == vmin
    assert df.max(df.x) == vmax


def test_minmax_mixed_types():
    x = np.array([1, 0], dtype=np.int)
    y = np.array([0.5, 1.5], dtype=np.float)
    df = vaex.from_arrays(x=x, y=y)
    with pytest.raises(TypeError):
        df.minmax(['x', 'y'])


def test_big_endian_binning():
    x = np.arange(10, dtype='>f8')
    y = np.zeros(10, dtype='>f8')
    ds = vaex.from_arrays(x=x, y=y)
    counts = ds.count(binby=[ds.x, ds.y], limits=[[-0.5, 9.5], [-0.5, 0.5]], shape=[10, 1])
    assert counts.ravel().tolist() == np.ones(10).tolist()


def test_big_endian_binning_non_contiguous():
    x = np.arange(20, dtype='>f8')[::2]
    x[:] = np.arange(10, dtype='>f8')
    y = np.arange(20, dtype='>f8')[::2]
    y[:] = np.arange(10, dtype='>f8')
    ds = vaex.from_arrays(x=x, y=y)
    counts = ds.count(binby=[ds.x, ds.y], limits=[[-0.5, 9.5], [-0.5, 9.5]], shape=[10, 10])
    assert np.diagonal(counts).tolist() == np.ones(10).tolist()


def test_strides():
    ar = np.zeros((10, 2)).reshape(20)
    x = ar[::2]
    x[:] = np.arange(10)
    ds = vaex.from_arrays(x=x)
    counts = ds.count(binby=ds.x, limits=[-0.5, 9.5], shape=10)
    assert counts.tolist() == np.ones(10).tolist()


def test_expr():
    ar = np.zeros((10, 2)).reshape(20)
    x = ar[::2]
    x[:] = np.arange(10)
    ds = vaex.from_arrays(x=x)
    counts = ds.count('x*2', binby='x*2', limits=[-0.5, 19.5], shape=10)
    assert counts.tolist() == np.ones(10).tolist()


def test_nunique():
    s = ['aap', 'aap', 'noot', 'mies', None, 'mies', 'kees', 'mies', 'aap']
    x = [0,     0,     0,      0,      0,     1,      1,     1,      2]
    df = vaex.from_arrays(x=x, s=s)
    dfg = df.groupby(df.x, agg={'nunique': vaex.agg.nunique(df.s)}).sort(df.x)
    items = list(zip(dfg.x.values, dfg.nunique.values))
    assert items == [(0, 4), (1, 2), (2, 1)]

    dfg = df.groupby(df.x, agg={'nunique': vaex.agg.nunique(df.s, dropmissing=True)}).sort(df.x)
    items = list(zip(dfg.x.values, dfg.nunique.values))
    assert items == [(0, 3), (1, 2), (2, 1)]

    # we just map the strings to floats, to have the same test for floats/primitives
    mapping = {'aap': 1.2, 'noot': 2.5, 'mies': 3.7, 'kees': 4.8, None: np.nan}
    s = np.array([mapping[k] for k in s], dtype=np.float64)
    df = vaex.from_arrays(x=x, s=s)
    dfg = df.groupby(df.x, agg={'nunique': vaex.agg.nunique(df.s)}).sort(df.x)
    items = list(zip(dfg.x.values, dfg.nunique.values))
    assert items == [(0, 4), (1, 2), (2, 1)]

    dfg = df.groupby(df.x, agg={'nunique': vaex.agg.nunique(df.s, dropnan=True)}).sort(df.x)
    items = list(zip(dfg.x.values, dfg.nunique.values))
    assert items == [(0, 3), (1, 2), (2, 1)]


def test_nunique_filtered():
    s = ['aap', 'aap', 'noot', 'mies', None, 'mies', 'kees', 'mies', 'aap']
    x = [0,     0,     0,      0,      0,     1,      1,     1,      2]
    y = [1,     1,     0,      1,      0,     0,      0,     1,      1]
    df = vaex.from_arrays(x=x, s=s, y=y)
    dfg = df[df.y==0].groupby(df.x, agg={'nunique': vaex.agg.nunique(df.s)}).sort(df.x)
    items = list(zip(dfg.x.values, dfg.nunique.values))
    assert items == [(0, 2), (1, 2)]

    # we just map the strings to floats, to have the same test for floats/primitives
    mapping = {'aap': 1.2, 'noot': 2.5, 'mies': 3.7, 'kees': 4.8, None: np.nan}
    s = np.array([mapping[k] for k in s], dtype=np.float64)
    df = vaex.from_arrays(x=x, s=s, y=y)
    dfg = df[df.y==0].groupby(df.x, agg={'nunique': vaex.agg.nunique(df.s)}).sort(df.x)
    items = list(zip(dfg.x.values, dfg.nunique.values))
    assert items == [(0, 2), (1, 2)]


def test_unique_missing_groupby():
    s = ['aap', 'aap', 'noot', 'mies', None, 'mies', 'kees', 'mies', 'aap']
    x = [0,     0,     0,      np.nan,      np.nan,     1,      1,     np.nan,      2]
    df = vaex.from_arrays(x=x, s=s)
    dfg = df.groupby(df.x, agg={'nunique': vaex.agg.nunique(df.s)}).sort(df.x)
    items = list(zip(dfg.x.values, dfg.nunique.values))
    assert items[:-1] == [(0, 2), (1, 2), (2, 1)]

def test_agg_selections():
    x = np.array([0, 0, 0, 1, 1, 2, 2])
    y = np.array([1, 3, 5, 1, 7, 1, -1])
    z = np.array([0, 2, 3, 4, 5, 6, 7])
    w = np.array(['dog', 'cat', 'mouse', 'dog', 'dog', 'dog', 'cat'])

    df = vaex.from_arrays(x=x, y=y, z=z, w=w)

    df_grouped = df.groupby(df.x).agg({'count': vaex.agg.count(selection='y<=3'),
                                   'z_sum_selected': vaex.agg.sum(expression=df.z, selection='y<=3'),
                                   'z_mean_selected': vaex.agg.mean(expression=df.z, selection=df.y <= 3),
                                   'w_nuniqe_selected': vaex.agg.nunique(expression=df.w, selection=df.y <= 3, dropna=True)
                                  }).sort('x')

    assert df_grouped['count'].tolist() == [2, 1, 2]
    assert df_grouped['z_sum_selected'].tolist() == [2, 4, 13]
    assert df_grouped['z_mean_selected'].tolist() == [1, 4, 6.5]
    assert df_grouped['w_nuniqe_selected'].tolist() == [2, 1, 2]

def test_agg_selections_equal():
    x = np.array([0, 0, 0, 1, 1, 2, 2])
    y = np.array([1, 3, 5, 1, 7, 1, -1])
    z = np.array([0, 2, 3, 4, 5, 6, 7])
    w = np.array(['dog', 'cat', 'mouse', 'dog', 'dog', 'mouse', 'cat'])

    df = vaex.from_arrays(x=x, y=y, z=z, w=w)


    df_grouped = df.groupby(df.x).agg({'counts': vaex.agg.count(),
                                      'sel_counts': vaex.agg.count(selection=df.y==1.)
                                      })
    assert df_grouped['counts'].tolist() == [3, 2, 2]
    assert df_grouped['sel_counts'].tolist() == [1, 1, 1]

def test_agg_selection_nodata():
    x = np.array([0, 0, 0, 1, 1, 2, 2])
    y = np.array([1, 3, 5, 1, 7, 1, -1])
    z = np.array([0, 2, 3, 4, 5, 6, 7])
    w = np.array(['dog', 'cat', 'mouse', 'dog', 'dog', 'mouse', 'cat'])

    df = vaex.from_arrays(x=x, y=y, z=z, w=w)

    df_grouped = df.groupby(df.x).agg({'counts': vaex.agg.count(),
                                      'dog_counts': vaex.agg.count(selection=df.w == 'dog')
                                      })

    assert len(df_grouped) == 3
    assert df_grouped['counts'].tolist() == [3, 2, 2]
    assert df_grouped['dog_counts'].tolist() == [1, 2, 0]

def test_upcast():
    df = vaex.from_arrays(b=[False, True, True], i8=np.array([120, 121, 122], dtype=np.int8),
        f4=np.array([1, 1e-13, 1], dtype=np.float32))
    assert df.b.sum() == 2
    assert df.i8.sum() == 120*3 + 3
    assert df.f4.sum() == (2 + 1e-13)

    assert abs(df.b.var() - (0.2222)) < 0.01


def test_agg_filtered_df_invalid_data():
    # Custom function to be applied to a filtered DataFrame
    def custom_func(x):
        assert 4 not in x; return x**2

    df = vaex.from_arrays(x=np.arange(10))
    df_filtered = df[df.x!=4]
    df_filtered.add_function('custom_function', custom_func)
    df_filtered['y'] = df_filtered.func.custom_function(df_filtered.x)
    # assert df_filtered.y.tolist() == [0, 1, 4, 9, 25, 36, 49, 64, 81]
    assert df_filtered.count(df_filtered.y) == 9



def test_var_and_std(df):
    x = df.x.to_numpy()
    y = df.y.to_numpy()

    vx, vy = df.var([df.x, df.y])
    assert vx == np.var(x)
    assert vy == np.var(y)

    sx, sy = df.std(["x", "y"])
    assert sx == np.std(x)
    assert sy == np.std(y)

    df.select("x < 5")
    vx, vy = df.var([df.x, df.y], selection=True)
    assert vx == np.var(x[:5])
    assert vy == np.var(y[:5])
    assert np.var(y[:5]), df.y.var(selection=True)

    sx, sy = df.std(["x", "y"], selection=True)
    assert sx == np.std(x[:5])
    assert sy == np.std(y[:5])


# TODO: does this not work on the remote dataframe?
def test_mutual_information(df_local):
    df = df_local
    limits = df.limits(["x", "y"], "minmax")
    mi2 = df.mutual_information("x", "y", mi_limits=limits, mi_shape=32)

    np.testing.assert_array_almost_equal(2.043192, mi2)

    # no test, just for coverage
    mi1d = df.mutual_information("x", "y", mi_limits=limits, mi_shape=32, binby="x", limits=[0, 10], shape=2)
    assert mi1d.shape == (2,)

    mi2d = df.mutual_information("x", "y", mi_limits=limits, mi_shape=32, binby=["x", "y"], limits=[[0, 10], [0, 100]], shape=(2, 3))
    assert mi2d.shape == (2,3)

    mi3d = df.mutual_information("x", "y", mi_limits=limits, mi_shape=32, binby=["x", "y", "z"], limits=[[0, 10], [0, 100], [-100, 100]], shape=(2, 3, 4))
    assert mi3d.shape == (2,3,4)

    mi_list, subspaces = df.mutual_information([["x", "y"], ["x", "z"]], sort=True)
    mi1 = df.mutual_information("x", "y")
    mi2 = df.mutual_information("x", "z")
    assert mi_list.tolist() == list(sorted([mi1, mi2]))


def test_format_xarray_and_list(df_local):
    df = df_local
    count = df.count(binby='x', limits=[-0.5, 9.5], shape=5, array_type='xarray')
    assert count.coords['x'].data.tolist() == [0.5, 2.5, 4.5, 6.5, 8.5]
    count = df.count(binby='x', limits=[-0.5, 9.5], shape=5, array_type='list')
    assert count == [2] * 5
    assert isinstance(count, list)

    df = df[:3]
    df['g'] = df.x
    df.categorize(df.g, labels=['aap', 'noot', 'mies'], inplace=True)
    count = df.count(binby='g', array_type='xarray')
    assert count.coords['g'].data.tolist() == ['aap', 'noot', 'mies']

    count = df.sum([df.x, df.g], binby='g', array_type='xarray')
    assert count.coords['expression'].data.tolist() == ['x', 'g']
    assert count.coords['g'].data.tolist() == ['aap', 'noot', 'mies']
