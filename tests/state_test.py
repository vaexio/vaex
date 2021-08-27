from common import *
import vaex.ml


def test_state_skip_main(df_factory):
    df = df_factory(x=[1, 2]).hashed()._future()
    dff = df[df.x > 1]
    dff._push_down_filter()
    state = dff.state_get()
    assert dff.dataset.id in state['objects']
    assert df.dataset.id not in state['objects']


def test_state_skip_slice(df_factory):
    df = df_factory(x=[1, 2, 2]).hashed()._future()
    dfs = df[:2]
    state = dfs.state_get()
    assert dfs.dataset.id in state['objects']
    # assert df.dataset.id not in state['objects']


def test_apply_state():
    x = [1, 2, 3, 4, 5]
    df = vaex.from_arrays(x=x)
    dfc = df.copy()
    df['y'] = df.x.apply(lambda x: x**2)
    dfc.state_set(df.state_get())
    assert df.y.tolist() == dfc.y.tolist()


def test_isin(tmpdir):
    df = vaex.from_arrays(x=np.array(['a', 'b', 'c'], dtype='O'),
                          y=np.array([1, 2, 3], dtype='O'))._future()

    df2 = df.copy()
    df2['test'] = df2.x.isin(['a'])
    df2.state_write(tmpdir / 'state.json')
    # import pdb; pdb.set_trace()
    df.state_load(tmpdir / 'state.json')
    assert df.test.tolist() == df2.test.tolist()


@pytest.mark.parametrize("future", [False, True])
def test_state_get_set(df_local, future):
    df = df_local
    df = df._future() if future else df

    df_copy = df.copy()

    df['v'] = df.x + 1

    state = df.state_get()
    df_copy.state_set(state)
    assert df_copy.v.values.tolist() == df.v.values.tolist()

    # making a copy when the state is set should work as well
    assert df_copy.copy().v.values.tolist() == df.v.values.tolist()
    assert 'v' in df_copy.get_column_names()


@pytest.mark.parametrize("future", [False, True])
def test_state_rename(df_factory, future):
    df = df_factory(x=[1])
    df = df._future() if future else df
    dfc = df.copy()
    df['y'] = df.x + 1
    df.rename('x', 'a')
    df.rename('y', 'b')
    df.a.tolist() == [1]
    state = df.state_get()
    dfc.state_set(state)
    assert dfc.a.tolist() == [1]
    assert dfc.b.tolist() == [2]


def test_state_mem_waste(df_trimmed):
    df = df_trimmed
    assert df._selection_masks == {}
    state = df.state_get()
    df.state_set(state)
    assert df._selection_masks == {}


def test_state_variables(df_local_non_arrow, tmpdir):
    filename = str(tmpdir.join('state.json'))
    df = df_local_non_arrow._future()
    df_copy = df.copy()
    t_test = np.datetime64('2005-01-01')
    df.add_variable('dt_var', t_test)
    variables = df.variables.copy()

    # this virtual column will add a variable (the timedelta)
    df['seconds'] = df.timedelta / np.timedelta64(1, 's')
    assert len(df.variables) == len(variables) + 1
    var_name = list(set(df.variables) - set(variables))[0]

    # an array
    df.add_variable('some_array', np.arange(10))

    df.state_write(filename)

    df_copy.state_load(filename)
    assert isinstance(df_copy.variables[var_name], np.timedelta64)
    assert df.seconds.tolist() == df_copy.seconds.tolist()
    assert df_copy.variables['dt_var'] == t_test
    assert df_copy.variables['some_array'].tolist() == df.variables['some_array'].tolist()


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


def test_state_drop():
    df = vaex.from_scalars(x=1, y=2)
    dfc = df.copy()
    df = df.drop('x')
    dfc.state_set(df.state_get())
    assert 'x' not in dfc
    assert 'x' not in dfc.dataset


def test_transform(df_factory_numpy, tmpdir):
    df = df_factory_numpy(x=[1, 1, 2, 2, 2, 3], y=[0, 1, 2, 3, 4, 5])
    df['y'] = df.y + 1
    def add(df, col, value):
        df[col] = df[col] + value
        return df

    df2 = df.transform(add, 'y', 2)
    assert df2['y'].tolist() == [3, 4, 5, 6, 7, 8]


def test_notebook():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

    import numpy as np

    import pylab as p
    import seaborn as sns

    import pandas as pd 

    import shap

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # import tensorflow as tf
    # from tensorflow import keras as K

    from tqdm.notebook import tqdm

    import vaex
    import vaex.ml

    import warnings
    df_train_full = vaex.open('/home/maartenbreddels/github/JovanVeljanoski/vaex-notebooks/dash/Predictive-maintenance/data/data_train.hdf5')._future()
    val_index = np.array([34, 70,  4, 71, 96])  # Fixed to ensure reproducibility

    print('units part of the validation set:', val_index)

    # Split on train and validation sets based on the val_index
    df_train = df_train_full[~df_train_full.unit_number.isin(val_index)].extract()
    df_val = df_train_full[df_train_full.unit_number.isin(val_index)].extract()
    df_train_copy = df_train.copy()


    # Get the sensor names
    sensors = df_train.get_column_names()[5:-1]

    # Curate a list of columns to drop
    # Drop the settings - as they are not sensors to be monitored
    cols_to_drop = ['setting_1', 'setting_2', 'setting_3']

    # Remove sensors with constant or near constant values
    cols_const = [s for s in sensors if df_train[s].nunique() < 5]
    print('Columns with constant values:', cols_const)
    cols_to_drop += cols_const

    # Find Sensors that too weakly correlate with the RUL
    cols_weak_corr_target = np.array(sensors)[np.array([np.abs(df_train.correlation(s, 'RUL')) for s in sensors]) < 0.01].tolist()
    print('Columns too weakly correlating with the RUL:', cols_weak_corr_target)
    cols_to_drop += cols_weak_corr_target

    # Find highly correlated columns
    # TODO - depending on the "new" correlation and mutual information API
    cols_high_corr = ['NRc']
    print('Highly correlated (> 0.95) columns:', cols_high_corr)
    cols_to_drop += cols_high_corr

    # Remove duplicates from the list of columns to drop
    cols_to_drop = sorted(list(set(cols_to_drop)))

    print()
    print('Final list of columns to drop:', cols_to_drop)

    # Drop columns that are not needed
    df_train = df_train.drop(columns=cols_to_drop)

    # Columns to MinMax scale
    cols_to_scale = df_train.get_column_names()[2:-1]

    # Normalize in range (0, 1)
    df_train = df_train.ml.minmax_scaler(features=cols_to_scale)

    # Get the scaled columns and from them create columns to be transformed into features
    scaled_cols = df_train.get_column_names(regex='^minmax')
    for col in scaled_cols:
        feat_name = col.replace('minmax_scaled', 'feat')
        df_train[feat_name] = df_train[col].copy()
    
    features = df_train.get_column_names(regex='^minmax_')
    # features = df_train.get_column_names(regex='^minmax_')
    features_to_reshape = features + ['RUL']
    target = 'RUL_target'
    sequence_length = 50
    batch_size = 192
    

    def shift_and_concat(df, split_column, shift_columns, sequence_length):
        df = vaex.concat([df_tmp.shift(periods=(0, sequence_length), column=shift_columns, trim=True) for _, df_tmp in df.groupby(split_column)])
        return df

    df_train = df_train.transform(shift_and_concat, 'unit_number', features_to_reshape, sequence_length)
    df_train['RUL_target'] = df_train.RUL[:, -1]

    df_val_copy = df_val.copy()
    df_val = df_train.pipeline.transform(df_val)
    df_train.pipeline.save('pipeline.json')

    df_val = df_val_copy.copy()
    df_val = df_val.pipeline.load_and_transform('pipeline.json', trusted=True)




    #df_val.pipeline._apply(.pipeline_get(), trusted=True)
    assert len(df_val.RUL_target.tolist()) == len(df_val)
    # import pdb; pdb.set_trace()

