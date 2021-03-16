from common import *
import vaex.ml


def test_state_get_set(ds_local):
    ds = ds_local

    ds_copy = ds.copy()

    ds['v'] = ds.x + 1

    state = ds.state_get()
    ds_copy.state_set(state)
    assert ds_copy.v.values.tolist() == ds.v.values.tolist()

    # making a copy when the state is set should work as well
    assert ds_copy.copy().v.values.tolist() == ds.v.values.tolist()
    assert 'v' in ds_copy.get_column_names()


def test_state_mem_waste(df_trimmed):
    df = df_trimmed
    assert df._selection_masks == {}
    state = df.state_get()
    df.state_set(state)
    assert df._selection_masks == {}


def test_state_variables(df_local_non_arrow, tmpdir):
    filename = str(tmpdir.join('state.json'))
    df = df_local_non_arrow
    df_copy = df.copy()
    t_test = np.datetime64('2005-01-01')
    df.add_variable('dt_var', t_test)
    variables = df.variables.copy()

    # this virtual column will add a variable (the timedelta)
    df['seconds'] = df.timedelta / np.timedelta64(1, 's')
    assert len(df.variables) == len(variables) + 1
    var_name = list(set(df.variables) - set(variables))[0]

    df.state_write(filename)

    df_copy.state_load(filename)
    assert isinstance(df_copy.variables[var_name], np.timedelta64)
    assert df.seconds.tolist() == df_copy.seconds.tolist()
    assert df_copy.variables['dt_var'] == t_test


def test_state_transfer_reassign(df):
    df_original = df.copy()

    df['new_x'] = df.x + 1
    df['new_x'] = df.x + 1

    # State transfer
    df_original.state_set(df.state_get())
    assert df_original.new_x.tolist() == df.new_x.tolist()


def test_state_keep_column():
    df1 = vaex.from_scalars(x=1, y=2, extra=3)
    df2 = vaex.from_scalars(x=10, y=20)
    df2['z'] = df1.x + df1.y
    df1_copy = df1.copy()

    df1.state_set(df2.state_get(), keep_columns=['extra'])
    assert df1.z.tolist() == [3]
    assert df1.extra.tolist() == [3]

    with pytest.raises(KeyError):
        df1_copy.state_set(df2.state_get(), keep_columns=['doesnotexis'])


def test_state_skip_filter():
    df1 = vaex.from_arrays(x=[1,2], y=[2,3])
    df2 = df1.copy()
    df2['z'] = df1.x + df1.y
    df2 = df2[df2.x > 1]
    assert len(df2) == 1
    df1.state_set(df2.state_get(), set_filter=False)
    assert df1.z.tolist() == [3, 5]


def test_filter_rename_column():
    df = vaex.from_dict({'feat1':[1, 2, 3],
                         'feat2': [10, 20, 30],
                         'y': ['word', None, 'Place']})
    df = df.dropna(column_names=['y'])

    # Now we want only to transfer the stuff done on the features
    state = df[['feat1', 'feat2']].state_get()

    # And apply it to a test dataframe
    df_test = vaex.from_scalars(feat1=5, feat2=10)
    df_test.state_set(state, set_filter=False)

    assert df_test.shape == (1, 2)
    assert df_test.get_column_names() == ['feat1', 'feat2']
    assert df_test.feat1.tolist() == [5]
    assert df_test.feat2.tolist() == [10]


def test_state_load_gcs():
    df = vaex.ml.datasets.load_iris()
    f = vaex.file.open('gs://vaex-data/testing/test_iris_state.json', fs_options={'token': 'anon', 'cache': True})
    import io
    f = io.TextIOWrapper(f, encoding='utf8')
    f.read()
    df.state_load('gs://vaex-data/testing/test_iris_state.json', fs_options={'token': 'anon', 'cache': True})

    assert df.column_count() == 7
    assert 'norm_sepal_length' in df.column_names
    assert 'minmax_petal_width' in df.column_names
    assert df.minmax_petal_width.minmax().tolist() == [0, 1]
    assert df.norm_sepal_length.mean().round(decimals=5) == 0
    assert df.norm_sepal_length.std().round(decimals=5) == 1