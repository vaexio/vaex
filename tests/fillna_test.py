from common import *


def test_fillna_column(ds_local):
    df = ds_local
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
