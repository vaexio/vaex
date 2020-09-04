from common import *


def test_dropna_objects(df_local_non_arrow):
    ds = df_local_non_arrow
    ds_dropped = ds.dropna(column_names=['obj'])
    assert ds_dropped['obj'].values.mask.any() == False
    float_elements = np.array([element for element in ds_dropped['obj'].values.data if isinstance(element, float)])
    assert np.isnan(float_elements).any() == False, 'np.nan still exists in column'


def test_dropna_cache_bug():
    # tests https://github.com/vaexio/vaex/pull/874
    # where repeated dropna would use a cached length
    df = vaex.from_arrays(
        x=[1, None, 2],
        y=[3, 4, None],
    )
    df1 = df.dropna('x')
    assert len(df1) == 2

    df2 = df1.dropna('y')
    assert len(df2) == 1


def test_dropna(ds_local):
    ds = ds_local
    ds_copy = ds.copy()

    ds_dropped = ds.dropna()
    assert len(ds_dropped) == 6

    ds_dropped = ds.dropna(drop_masked=False)
    assert len(ds_dropped) == 8
    assert np.isnan(ds_dropped['n'].values).any() == False
    assert np.isnan(ds_dropped['nm'].values).any() == False

    ds_dropped = ds.dropna(drop_nan=False)
    assert len(ds_dropped) == 8
    assert ds_dropped['m'].values.mask.any() == False
    assert ds_dropped['nm'].values.mask.any() == False
    assert ds_dropped['mi'].values.mask.any() == False
    assert ds_dropped['obj'].values.mask.any() == False

    ds_dropped = ds.dropna(column_names=['nm', 'mi'])
    assert len(ds_dropped) == 8
    assert ds_dropped['nm'].values.mask.any() == False
    assert np.isnan(ds_dropped['nm'].values).any() == False

    ds_dropped = ds.dropna(column_names=['obj'])
    assert len(ds_dropped) == 8
    assert ds_dropped['obj'].values.mask.any() == False
    float_elements = np.array([element for element in ds_dropped['obj'].values.data if isinstance(element, float)])
    assert np.isnan(float_elements).any() == False, 'np.nan still exists in column'

    ds_dropped = ds.dropna(column_names=['nm', 'mi', 'obj'])
    state = ds_dropped.state_get()
    ds_copy.state_set(state)
    assert len(ds_copy) == len(ds_dropped)
    assert len(ds_copy) == 6

# equivalent of isna_test
def test_dropmissing():
    s = vaex.string_column(["aap", None, "noot", "mies"])
    o = ["aap", None, "noot", np.nan]
    x = np.arange(4, dtype=np.float64)
    x[2] = x[3] = np.nan
    m = np.ma.array(x, mask=[0, 1, 0, 1])
    df = vaex.from_arrays(x=x, m=m, s=s, o=o)
    x = df.x.dropmissing().tolist()
    assert (9 not in x)
    assert np.any(np.isnan(x)), "nan is not a missing value"
    m = df.m.dropmissing().tolist()
    assert (m[:1] == [0])
    assert np.isnan(m[1])
    assert len(m) == 2
    assert (df.s.dropmissing().tolist() == ["aap", "noot", "mies"])
    assert (df.o.dropmissing().tolist()[:2] == ["aap", "noot"])
    # this changed in vaex 4, since the np.nan is considered missing, the whole
    # columns is seen as string
    # assert np.isnan(df.o.dropmissing().tolist()[2])


# equivalent of isna_test
def test_dropnan():
    s = vaex.string_column(["aap", None, "noot", "mies"])
    o = ["aap", None, "noot", np.nan]
    x = np.arange(4, dtype=np.float64)
    x[2] = x[3] = np.nan
    m = np.ma.array(x, mask=[0, 1, 0, 1])
    df = vaex.from_arrays(x=x, m=m, s=s, o=o)
    x = df.x.dropnan().tolist()
    assert x == [0, 1]
    m = df.m.dropnan().tolist()
    assert m == [0, None, None]
    assert (df.s.dropnan().tolist() == ["aap", None, "noot", "mies"])
    # this changed in vaex 4, since the np.nan is considered missing, the whole
    # columns is seen as string
    assert (df.o.dropnan().tolist() == ["aap", None, "noot", None])

def test_dropna():
    s = vaex.string_column(["aap", None, "noot", "mies"])
    o = ["aap", None, "noot", np.nan]
    x = np.arange(4, dtype=np.float64)
    x[2] = x[3] = np.nan
    m = np.ma.array(x, mask=[0, 1, 0, 1])
    df = vaex.from_arrays(x=x, m=m, s=s, o=o)
    x = df.x.dropna().tolist()
    assert x == [0, 1]
    m = df.m.dropna().tolist()
    assert m == [0]
    assert (df.s.dropna().tolist() == ["aap", "noot", "mies"])
    assert (df.o.dropna().tolist() == ["aap", "noot"])


def test_dropna_all_columns():
    x = [1, 2, 3, 4, 5]
    y = ['dog', 'dog', None, 'cat', None]
    df = vaex.from_arrays(x=x, y=y)

    df_dropped = df.dropna()
    assert df_dropped.x.tolist() == [1, 2, 4]
    assert df_dropped.y.tolist() == ['dog', 'dog', 'cat']
