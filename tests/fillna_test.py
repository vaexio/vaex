from common import *


def test_fillna_column(df_local_non_arrow):
    if isinstance(df_local_non_arrow.dataset['obj'], vaex.column.ColumnConcatenatedLazy):
        # TODO: vaex.column.ColumnConcatenatedLazy is too eager to cast to string
        # we need the same behaviour as in vaex.dataset.to_supported_array
        return
    df = df_local_non_arrow
    df['ok'] = df['obj'].fillna(value='NA')
    assert df.ok.values[5] == 'NA'
    df['obj'] = df['obj'].fillna(value='NA')
    assert df.obj.values[5] == 'NA'


def test_fillna(ds_local):
    df = ds_local
    df_copy = df.copy()

    df_string_filled = df.fillna(value='NA')
    assert df_string_filled.obj.values[5] == 'NA'

    df_filled = df.fillna(value=0)
    assert df_filled.obj.values[5] == 0

    assert df_filled.to_pandas_df(virtual=True).isna().any().any() == False
    assert df_filled.to_pandas_df(virtual=True).isna().any().any() == False

    df_filled = df.fillna(value=10, fill_masked=False)
    assert df_filled.n.values[6] == 10.
    assert df_filled.nm.values[6] == 10.

    df_filled = df.fillna(value=-15, fill_nan=False)
    assert df_filled.m.values[7] == -15.
    assert df_filled.nm.values[7] == -15.
    assert df_filled.mi.values[7] == -15.

    df_filled = df.fillna(value=-11, column_names=['nm', 'mi'])
    assert df_filled.to_pandas_df(virtual=True).isna().any().any() == True
    assert df_filled.to_pandas_df(column_names=['nm', 'mi']).isna().any().any() == False

    state = df_filled.state_get()
    df_copy.state_set(state)
    np.testing.assert_array_equal(df_copy['nm'].values, df_filled['nm'].values)
    np.testing.assert_array_equal(df_copy['mi'].values, df_filled['mi'].values)


def test_fillna_virtual():
    # this might be a duplicate or rename_test.py/test_reassign_virtual
    x = [1, 2, 3, 5, np.nan, -1, -7, 10]
    df = vaex.from_arrays(x=x)
    # create a virtual column that will have nans due to the calculations
    df['r'] = np.log(df.x)
    df['r'] = df.r.fillna(value=0xdeadbeef)
    assert df.r.tolist()[:4] == [0.0, 0.6931471805599453, 1.0986122886681098, 1.6094379124341003]
    assert df.r.tolist()[4:7] == [0xdeadbeef, 0xdeadbeef, 0xdeadbeef]

def test_fillna_missing():
    # Create test data
    x = np.array(['A', 'B', -1, 0, 2, '', '', None, None, None, np.nan, np.nan, np.nan, np.nan])
    df = vaex.from_arrays(x=x)
    # Assert the correctness of the fillna
    assert df.x.fillna(value=-5).tolist() == ['A', 'B', -1, 0, 2, '', '', -5, -5, -5, -5, -5, -5, -5]

# equivalent of isna_test
def test_fillmissing():
    s = vaex.string_column(["aap", None, "noot", "mies"])
    o = ["aap", None, False, np.nan]
    x = np.arange(4, dtype=np.float64)
    x[2] = x[3] = np.nan
    m = np.ma.array(x, mask=[0, 1, 0, 1])
    df = vaex.from_arrays(x=x, m=m, s=s, o=o)
    x = df.x.fillmissing(9).tolist()
    assert (9 not in x)
    assert np.any(np.isnan(x)), "nan is not a missing value"
    m = df.m.fillmissing(9).tolist()
    assert (m[:2] == [0, 9])
    assert np.isnan(m[2])
    assert m[3] == 9
    assert (df.s.fillmissing('kees').tolist() == ["aap", "kees", "noot", "mies"])
    assert (df.o.fillmissing({'a':1}).tolist()[:3] == ["aap", {'a':1}, False])
    assert np.isnan(df.o.fillmissing([1]).tolist()[3])

# equivalent of isna_test
def test_fillnan():
    s = vaex.string_column(["aap", None, "noot", "mies"])
    o = ["aap", None, False, np.nan]
    x = np.arange(4, dtype=np.float64)
    x[2] = x[3] = np.nan
    m = np.ma.array(x, mask=[0, 1, 0, 1])
    df = vaex.from_arrays(x=x, m=m, s=s, o=o)
    x = df.x.fillnan(9).tolist()
    assert x == [0, 1, 9, 9]
    m = df.m.fillnan(9).tolist()
    assert m == [0, None, 9, None]
    assert (df.s.fillnan('kees').tolist() == ["aap", None, "noot", "mies"])
    assert (df.o.fillnan({'a':1}).tolist() == ["aap", None, False, {'a':1}])

def test_fillna():
    s = vaex.string_column(["aap", None, "noot", "mies"])
    o = ["aap", None, False, np.nan]
    x = np.arange(4, dtype=np.float64)
    x[2] = x[3] = np.nan
    m = np.ma.array(x, mask=[0, 1, 0, 1])
    df = vaex.from_arrays(x=x, m=m, s=s, o=o)
    x = df.x.fillna(9).tolist()
    assert x == [0, 1, 9, 9]
    m = df.m.fillna(9).tolist()
    assert m == [0, 9, 9, 9]
    assert (df.s.fillna('kees').tolist() == ["aap", "kees", "noot", "mies"])
    assert (df.o.fillna({'a':1}).tolist() == ["aap", {'a': 1}, False, {'a':1}])

def test_fillna_array():
    x = np.array([1, 2, 3, np.nan])
    df = vaex.from_arrays(x=x)

    # fillna should take scalar arrays, so you can pass directly the result of df.x.mean() for instance
    df['x_2'] = df.x.fillna(np.array(2.0))
    assert df.x_2.tolist() == [1, 2, 3, 2]

def test_fillna_dataframe(df_factory):
    x = np.array([3, 1, np.nan, 10, np.nan])
    y = np.array([None, 1, True, '10street', np.nan], dtype='object')
    z = np.ma.array(data=[5, 7, 3, 1, -10], mask=[False, False, True, False, True])
    df = df_factory(x=x, y=y, z=z)

    df_filled = df.fillna(value=-1)

    assert df_filled.x.tolist() == [3, 1, -1, 10, -1]
    assert df_filled.y.tolist() == [-1, 1, True, '10street', -1]
    assert df_filled.z.tolist() == [5, 7, -1, 1, -1]

def test_fillna_string_dtype():
    name = ['Maria', 'Adam', None, None, 'Dan']
    age = [28, 15, 34, 55, 41]
    weight = [np.nan, np.nan, 77.5, 65, 95]
    df = vaex.from_arrays(name=name, age=age, weight=weight)

    # Originally - the column "name" is string
    assert df['name'].is_string()

    df['name'] = df['name'].fillna('missing')

    # Confirm that the column "name" is still of type string after fillna
    assert df['name'].is_string()
