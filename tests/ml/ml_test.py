import pytest
import numpy as np
import vaex
import vaex.ml
pytest.importorskip("sklearn")
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler


np_version = tuple(map(int, np.__version__.split('.')))


def data_maker(n_rows, n_cols):
    return {f'feat_{i}': np.random.normal(loc=np.random.uniform(-100, 100),
                                          scale=np.random.uniform(0.5, 25),
                                          size=n_rows) for i in range(n_cols)}


def test_pca(df_iris):
    ds = df_iris
    pca1 = ds.ml.pca(features=[ds.petal_width, ds.petal_length], n_components=2, transform=False)
    pca2 = ds.ml.pca(features=[ds.sepal_width, ds.sepal_length, ds.petal_length], n_components=3, transform=False)
    ds = pca1.transform(ds)
    print(ds.virtual_columns.keys())
    virtual_column_count1 = len(ds.virtual_columns.keys())
    ds = pca2.transform(ds)
    print(ds.virtual_columns.keys())
    virtual_column_count2 = len(ds.virtual_columns.keys())
    assert virtual_column_count2 == virtual_column_count1 + 3, 'make sure we don\'t overwrite column names'
    # Fit-transform
    pca = vaex.ml.PCA(features=['sepal_width', 'petal_length', 'sepal_length', 'petal_width'], n_components=2)
    ds = pca.fit_transform(ds)


def test_valid_sklearn_pca(df_iris):
    ds = df_iris
    features = ['sepal_width', 'petal_length', 'sepal_length', 'petal_width']
    # sklearn approach
    pca = PCA(n_components=3, random_state=33, svd_solver='full', whiten=False)
    pca.fit(ds[features])
    sklearn_trans = pca.transform(ds[features])
    # vaex-ml approach
    ds_pca = ds.ml.pca(n_components=3, features=features)
    # Compare the two approaches
    np.testing.assert_almost_equal(ds_pca.evaluate('PCA_0'), sklearn_trans[:, 0])


def test_pca_incremental(df_iris):
    df = df_iris
    features = ['sepal_width', 'petal_length', 'sepal_length', 'petal_width']
    pca = vaex.ml.PCAIncremental(features=features, n_components=2)
    df_transformed = pca.fit_transform(df)
    assert pca.n_samples_seen_ == 150
    assert len(pca.eigen_values_) == 2
    assert len(pca.explained_variance_) == 2
    assert len(df_transformed.get_column_names()) == 7
    assert len(df_transformed.get_column_names(regex='^PCA_')) == 2


def test_valid_sklearn_pca_incremental(df_iris):
    df = df_iris
    features = ['sepal_width', 'petal_length', 'sepal_length', 'petal_width']
    # scikit-learn
    sk_pca = IncrementalPCA(batch_size=10)
    sk_result = sk_pca.fit_transform(df[features].values)
    # vaex-ml
    vaex_pca = vaex.ml.PCAIncremental(features=features, batch_size=10)
    df_transformed = vaex_pca.fit_transform(df)
    # Compare the results
    np.testing.assert_almost_equal(df_transformed.evaluate('PCA_0'), sk_result[:, 0])
    np.testing.assert_almost_equal(df_transformed.evaluate('PCA_1'), sk_result[:, 1])


@pytest.mark.parametrize("n_components", [5, 15, 150])
@pytest.mark.parametrize("matrix_type", ['gaussian', 'sparse'])
def test_random_projections(n_components, matrix_type):
    df = vaex.from_dict(data=data_maker(n_rows=100_000, n_cols=31))
    features = df.get_column_names()

    rand_proj = df.ml.random_projections(features=features, n_components=n_components, matrix_type=matrix_type, transform=False)
    df_trans = rand_proj.fit_transform(df)

    assert np.array(rand_proj.random_matrix_).shape == (n_components, 31)
    assert len(df_trans.get_column_names(regex='^rand')) == n_components


@pytest.mark.parametrize("matrix_type", ['gaussian', 'sparse'])
def test_valid_sklearn_random_projections(df_iris, matrix_type):
    df = df_iris
    features = df.get_column_names()[:4]
    random_state=42

    rand_proj = vaex.ml.RandomProjections(features=features, n_components=3, matrix_type=matrix_type, random_state=random_state)
    df_trans = rand_proj.fit_transform(df)
    result = df_trans[df_trans.get_column_names(regex='^rand*')].values

    if matrix_type == 'gaussian':
        sk_gaus = GaussianRandomProjection(n_components=3, random_state=random_state)
        sk_trans = sk_gaus.fit_transform(df[features].values)

    else:
        sk_sparse = SparseRandomProjection(n_components=3, random_state=random_state, dense_output=True)
        sk_trans = sk_sparse.fit_transform(df[features].values)

    np.testing.assert_almost_equal(result, sk_trans)


def test_standard_scaler(df_iris):
    ds = df_iris
    ss1 = ds.ml.standard_scaler(features=[ds.petal_width, ds.petal_length], with_mean=True, with_std=True, transform=False)
    ss2 = ds.ml.standard_scaler(features=[ds.petal_width, ds.petal_length], with_mean=True, with_std=False, transform=False)
    ss3 = ds.ml.standard_scaler(features=[ds.petal_width, ds.petal_length], with_mean=False, with_std=True, transform=False)
    ss4 = ds.ml.standard_scaler(features=[ds.petal_width, ds.petal_length], with_mean=False, with_std=False, transform=False)
    ds1 = ss1.transform(ds)
    print(ds.virtual_columns.keys())
    ds2 = ss2.transform(ds)
    print(ds.virtual_columns.keys())
    ds3 = ss3.transform(ds)
    print(ds.virtual_columns.keys())
    ds4 = ss4.transform(ds)
    print(ds.virtual_columns.keys())
    assert (ds1.virtual_columns.keys() == ds2.virtual_columns.keys() ==
            ds3.virtual_columns.keys() == ds4.virtual_columns.keys()), 'output columns do not match'
    # Compare against sklearn
    features = ['petal_width', 'petal_length']
    skl_scaler = StandardScaler(with_mean=True, with_std=True)
    skl_scaler.fit(ds[features])
    sk_vals = skl_scaler.transform(ds[features])
    # assert
    np.testing.assert_almost_equal(sk_vals[:, 0], ds1.evaluate('standard_scaled_petal_width'),
                                   err_msg='vaex and sklearn results do not match')
    np.testing.assert_almost_equal(sk_vals[:, 1], ds1.evaluate('standard_scaled_petal_length'),
                                   err_msg='vaex and sklearn results do not match')
    # Fit-transform
    scaler = vaex.ml.StandardScaler(features=['petal_width', 'petal_length'])
    ds = scaler.fit_transform(ds)


def test_minmax_scaler(df_iris):
    ds = df_iris
    mms1 = ds.ml.minmax_scaler(features=[ds.petal_width, ds.petal_length], transform=False)
    mms2 = ds.ml.minmax_scaler(features=[ds.petal_width, ds.petal_length], feature_range=(-5, 2), transform=False)
    ds1 = mms1.transform(ds)
    print(ds.virtual_columns.keys())
    ds2 = mms2.transform(ds)
    print(ds.virtual_columns.keys())
    assert ds1.virtual_columns.keys() == ds2.virtual_columns.keys(), 'output columns do not match'
    # Compare against sklearn (default minmax range)
    features = ['petal_width', 'petal_length']
    skl_scaler = MinMaxScaler()
    skl_scaler.fit(ds[features])
    sk_vals = skl_scaler.transform(ds[features])
    # assert
    np.testing.assert_almost_equal(sk_vals[:, 0], ds1.evaluate('minmax_scaled_petal_width'),
                                   err_msg='vaex and sklearn results do not match')
    np.testing.assert_almost_equal(sk_vals[:, 1], ds1.evaluate('minmax_scaled_petal_length'),
                                   err_msg='vaex and sklearn results do not match')
    # Compare against sklearn (custom minmax range)
    skl_scaler = MinMaxScaler(feature_range=(-5, 2))
    skl_scaler.fit(ds[features])
    sk_vals = skl_scaler.transform(ds[features])
    # assert
    np.testing.assert_almost_equal(sk_vals[:, 0], ds2.evaluate('minmax_scaled_petal_width'),
                                   err_msg='vaex and sklearn results do not match')
    np.testing.assert_almost_equal(sk_vals[:, 1], ds2.evaluate('minmax_scaled_petal_length'),
                                   err_msg='vaex and sklearn results do not match')
    # Fit-transform
    scaler = vaex.ml.MinMaxScaler(features=['petal_width', 'petal_length'])
    ds = scaler.fit_transform(ds)


def test_train_test_split_values(df_factory):
    # Create a dataset
    ds = df_factory(x=np.arange(100))
    # Set active range to begin with
    ds.set_active_range(50, 100)
    # do the splitting
    train, test = ds.ml.train_test_split(verbose=False)
    # assert whether the split is correct
    np.testing.assert_equal(np.arange(50, 60), test.evaluate('x'))
    np.testing.assert_equal(np.arange(60, 100), train.evaluate('x'))


def test_frequency_encoder(df_factory):
    animals = np.array(['dog', 'dog', 'cat', 'dog', 'mouse', 'mouse'])
    numbers = np.array([1, 2, 3, 1, 1, np.nan])
    train = df_factory(animals=animals, numbers=numbers)
    animals = np.array(['dog', 'cat', 'mouse', 'ant', np.nan])
    numbers = np.array([2, 1, np.nan, np.nan, 5])
    test = df_factory(animals=animals, numbers=numbers)
    features = ['animals', 'numbers']

    fe = train.ml.frequency_encoder(features=features, unseen='nan', transform=False)
    fe.fit(train)
    test_a = fe.transform(test)
    np.testing.assert_almost_equal(test_a.frequency_encoded_animals.values,
                               [0.5, 0.166, 0.333, np.nan, np.nan],
                               decimal=3)
    np.testing.assert_almost_equal(test_a.frequency_encoded_numbers.values,
                                   [0.166, 0.5, 0.166, 0.166, np.nan],
                                   decimal=3)

    fe = vaex.ml.FrequencyEncoder(features=features, unseen='zero')
    fe.fit(train)
    test_b = fe.transform(test)
    np.testing.assert_almost_equal(test_b.frequency_encoded_animals.values,
                               [0.5, 0.166, 0.333, 0, 0],
                               decimal=3)
    np.testing.assert_almost_equal(test_b.frequency_encoded_numbers.values,
                                   [0.166, 0.5, 0.166, 0.166, 0.],
                                   decimal=3)


def test_label_encoder(df_factory):
    # Create sample data
    x1 = np.array(['dog', 'cat', 'mouse', 'mouse', 'dog', 'dog', 'dog', 'cat', 'cat', 'mouse', 'dog'])
    x2 = np.array(['dog', 'dog', 'cat', 'cat', 'mouse'])
    x3 = np.array(['mouse', 'dragon', 'dog', 'dragon'])  # unseen value 'dragon'
    y1 = np.array([1, 2, 2, 2, 3, 1, 2, 3, 5, 5, 1])
    y2 = np.array([3, 3, 1, 3, 2])
    y3 = np.array([3, 2, 1, 4])  # unseen value 4

    # Create
    df_train = df_factory(x=x1, y=y1)
    df_test = df_factory(x=x2, y=y2)
    df_unseen = df_factory(x=x3, y=y3)

    # # Label Encode with vaex.ml
    label_encoder = df_train.ml.label_encoder(features=['x', 'y'], prefix='mypref_', transform=False)

    # Assertions: makes sure that the categories are correctly identified:
    assert set(list(label_encoder.labels_['x'].keys())) == set(np.unique(x1))
    assert set(list(label_encoder.labels_['y'].keys())) == set(np.unique(y1))

    # Transform
    df_train = label_encoder.transform(df_train)
    df_test = label_encoder.transform(df_test)

    # Make asserssions on the "correctness" of the implementation by "manually" applying the labels to the categories
    assert df_test.x.apply(lambda elem: label_encoder.labels_['x'][elem]).tolist() == df_test.mypref_x.tolist()
    assert df_test.y.apply(lambda elem: label_encoder.labels_['y'][elem]).tolist() == df_test.mypref_y.tolist()

    # Try to get labels from the dd dataset unseen categories
    with pytest.raises(ValueError):
        label_encoder.transform(df_unseen)

    # Now try again, but allow for unseen categories
    label_encoder = df_train.ml.label_encoder(features=['x', 'y'], prefix='mypref_', allow_unseen=True, transform=False)
    df_unseen = label_encoder.transform(df_unseen)
    assert set(df_unseen[df_unseen.x == 'dragon'].mypref_x.tolist()) == {-1}
    assert set(df_unseen[df_unseen.y == 4].mypref_x.tolist()) == {-1}


def test_one_hot_encoding(df_factory):
    # Categories
    a = ['cat', 'dog', 'mouse']
    b = ['boy', 'girl']
    c = [0, 1]
    # Generate data
    x = np.random.choice(a, size=100, replace=True)
    y = np.random.choice(b, size=100, replace=True)
    z = np.random.choice(c, size=100, replace=True)
    # Create dataset
    ds = df_factory(animals=x, kids=y, numbers=z)
    # First try to one-hot encode without specifying features: this should raise an exception
    # TODO: should we do this?
    # with pytest.raises(ValueError):
    #     onehot = ds.ml.one_hot_encoder(features=None)
    # split in train and test
    train, test = ds.ml.train_test_split(test_size=.25, verbose=False)
    # fit onehot encoder on the train set
    onehot = train.ml.one_hot_encoder(features=['kids', 'animals', 'numbers'], prefix='', transform=False)
    # transform the test set
    test = onehot.transform(test)
    # asses the success of the test
    np.testing.assert_equal(test.kids_boy.tolist(), np.array([1 if i == 'boy' else 0 for i in test.kids.tolist()]))
    np.testing.assert_equal(test.kids_girl.tolist(), np.array([0 if i == 'boy' else 1 for i in test.kids.tolist()]))
    np.testing.assert_equal(test.animals_dog.tolist(), np.array([1 if i == 'dog' else 0 for i in test.animals.tolist()]))
    np.testing.assert_equal(test.animals_cat.tolist(), np.array([1 if i == 'cat' else 0 for i in test.animals.tolist()]))
    np.testing.assert_equal(test.animals_mouse.tolist(), np.array([1 if i == 'mouse' else 0 for i in test.animals.tolist()]))
    np.testing.assert_equal(test.numbers_0.tolist(), np.array([1 if i == 0 else 0 for i in test.numbers.tolist()]))
    np.testing.assert_equal(test.numbers_1.tolist(), np.array([0 if i == 0 else 1 for i in test.numbers.tolist()]))
    # Fit-transform
    ohe = vaex.ml.OneHotEncoder(features=['kids', 'animals', 'numbers'])
    ohe.fit_transform(ds)

def test_one_hot_encoding_with_na(df_factory):
    x = ['Reggie', 'Michael', None, 'Reggie']
    y = [31, 23, np.nan, 31]
    df_train = df_factory(x=x, y=y)

    x = ['Michael', 'Reggie', None, None]
    y = [23, 31, np.nan, np.nan]
    df_test = df_factory(x=x, y=y)


    enc = vaex.ml.OneHotEncoder(features=['x', 'y'])
    enc.fit(df_train)

    assert enc.uniques_[0] == [None, 'Michael', 'Reggie']
    np.testing.assert_array_equal(enc.uniques_[1], [np.nan, 23.0, 31.0])

    df_train = enc.transform(df_train)
    assert df_train.x_missing.tolist() == [0, 0, 1, 0]
    assert df_train.x_Michael.tolist() == [0, 1, 0, 0]
    assert df_train.x_Reggie.tolist() == [1, 0, 0, 1]
    assert df_train['y_23.0'].tolist() == [0, 1, 0, 0]
    assert df_train['y_31.0'].tolist() == [1, 0, 0, 1]
    assert df_train['y_nan'].tolist() == [0, 0, 1, 0]

    assert(str(df_train.x_missing.dtype) == 'uint8')
    assert(str(df_train.x_Michael.dtype) == 'uint8')
    assert(str(df_train.x_Reggie.dtype) == 'uint8')
    assert(str(df_train['y_23.0'].dtype) == 'uint8')
    assert(str(df_train['y_31.0'].dtype) == 'uint8')
    assert(str(df_train['y_nan'].dtype) == 'uint8')

    df_test = enc.transform(df_test)
    assert df_test.x_missing.tolist() == [0, 0, 1, 1]
    assert df_test.x_Michael.tolist() == [1, 0, 0, 0]
    assert df_test.x_Reggie.tolist() == [0, 1, 0, 0]
    assert df_test['y_23.0'].tolist() == [1, 0, 0, 0]
    assert df_test['y_31.0'].tolist() == [0, 1, 0, 0]
    assert df_test['y_nan'].tolist() == [0, 0, 1, 1]

    assert(str(df_test.x_missing.dtype) == 'uint8')
    assert(str(df_test.x_Michael.dtype) == 'uint8')
    assert(str(df_test.x_Reggie.dtype) == 'uint8')
    assert(str(df_test['y_23.0'].dtype) == 'uint8')
    assert(str(df_test['y_31.0'].dtype) == 'uint8')
    assert(str(df_test['y_nan'].dtype) == 'uint8')


def test_maxabs_scaler(df_factory):
    x = np.array([-2.65395789, -7.97116295, -4.76729177, -0.76885033, -6.45609635])
    y = np.array([-8.9480332, -4.81582449, -3.73537263, -3.46051912,  1.35137275])
    z = np.array([-0.47827432, -2.26208059, -3.75151683, -1.90862151, -1.87541903])
    w = np.zeros_like(x)

    ds = df_factory(x=x, y=y, z=z, w=w)
    df = ds.to_pandas_df(array_type='numpy')

    features = ['x', 'y', 'w']

    scaler_skl = MaxAbsScaler()
    result_skl = scaler_skl.fit_transform(df[features])
    scaler_vaex = vaex.ml.MaxAbsScaler(features=features)
    result_vaex = scaler_vaex.fit_transform(ds)

    assert result_vaex.absmax_scaled_x.values.tolist() == result_skl[:, 0].tolist(), "scikit-learn and vaex results do not match"
    assert result_vaex.absmax_scaled_y.values.tolist() == result_skl[:, 1].tolist(), "scikit-learn and vaex results do not match"
    assert result_vaex.absmax_scaled_w.values.tolist() == result_skl[:, 2].tolist(), "scikit-learn and vaex results do not match"


import numpy
import sys
import platform
version = tuple(map(int, numpy.__version__.split('.')))

@pytest.mark.skipif((np_version[0] == 1) & (np_version[1] < 21), reason="strange ref count issue with numpy")
def test_robust_scaler(df_factory):
    x = np.array([-2.65395789, -7.97116295, -4.76729177, -0.76885033, -6.45609635])
    y = np.array([-8.9480332, -4.81582449, -3.73537263, -3.46051912,  1.35137275])
    z = np.array([-0.47827432, -2.26208059, -3.75151683, -1.90862151, -1.87541903])
    w = np.zeros_like(x)

    ds = df_factory(x=x, y=y, z=z, w=w)
    df = ds.to_pandas_df(array_type='numpy')

    features = ['x', 'y']

    scaler_skl = RobustScaler()
    result_skl = scaler_skl.fit_transform(df[features])
    scaler_vaex = vaex.ml.RobustScaler(features=features)
    result_vaex = scaler_vaex.fit_transform(ds)

    np.testing.assert_array_almost_equal(scaler_vaex.center_, scaler_skl.center_, decimal=0.2)

    # check that an exception is rased for invalid percentile range
    scaler_vaex = vaex.ml.RobustScaler(features=features, percentile_range=(12, 175))
    with pytest.raises(Exception):
        result_vaex = scaler_vaex.fit_transform(ds)


def test_cyclical_transformer(tmpdir, df_factory):
    df_train = df_factory(hour=[0, 3, 6])
    df_test = df_factory(hour=[12, 24, 21, 15])

    trans = vaex.ml.CycleTransformer(n=24, features=['hour'], prefix_x='pref_', prefix_y='pref_')
    df_train = trans.fit_transform(df_train)
    np.testing.assert_array_almost_equal(df_train.pref_hour_x.values, [1, 0.707107, 0])
    np.testing.assert_array_almost_equal(df_train.pref_hour_y.values, [0, 0.707107, 1])

    state_path = str(tmpdir.join('state.json'))
    df_train.state_write(state_path)
    df_test.state_load(state_path)
    np.testing.assert_array_almost_equal(df_test.pref_hour_x.values, [-1, 1, 0.707107, -0.707107])
    np.testing.assert_array_almost_equal(df_test.pref_hour_y.values, [0, 0, -0.707107, -0.707107])


def test_bayesian_target_encoder(tmpdir, df_factory):
    df_train = df_factory(x1=['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b'],
                                x2=['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'q', 'q'],
                                y=[1, 1, 1, 1, 0, 0, 0, 0, 0, 1])
    df_test = df_factory(x1=['a', 'b', 'c'],
                               x2=['p', 'q', 'w'])

    target_encoder = vaex.ml.BayesianTargetEncoder(target='y', features=['x1', 'x2'], unseen='zero', prefix='enc_', weight=10)
    df_train = target_encoder.fit_transform(df_train)

    assert df_train.enc_x1.tolist() == [0.6, 0.6, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 0.4, 0.4]
    assert df_train.enc_x2.tolist() == [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    assert target_encoder.mappings_ == {'x1': {'a': 0.6, 'b': 0.4}, 'x2': {'p': 0.5, 'q': 0.5}}

    state_path = str(tmpdir.join('state.json'))
    df_train.state_write(state_path)
    df_test.state_load(state_path)

    df_test.enc_x1.tolist() == [0.6, 0.4, 0.0]
    df_test.enc_x2.tolist() == [0.5, 0.5, 0.0]

@pytest.mark.parametrize("as_bool", [False, True])
def test_weight_of_evidence_encoder(tmpdir, as_bool, df_factory):
    y = [1, 1, 1, 1, 1, 0, 0, 1]
    if as_bool:
        y = [bool(k) for k in y]
    df_train = df_factory(x=['a', 'a',  'b', 'b', 'b', 'b', 'c', 'c'],
                                y=y)
    df_test = df_factory(x=['a', 'b', 'c', 'd'])

    trans = vaex.ml.WeightOfEvidenceEncoder(target='y', features=['x'])
    df_train = trans.fit_transform(df_train)
    np.testing.assert_almost_equal(list(trans.mappings_['x'].values()),
                               [13.815510557964274, 1.0986122886681098, 0.0],
                               decimal=10)
    np.testing.assert_array_almost_equal(df_train.woe_encoded_x.values,
                                         [13.815510, 13.815510, 1.098612, 1.098612, 1.098612, 1.098612, 0., 0.])

    state_path = str(tmpdir.join('state.json'))
    df_train.state_write(state_path)
    df_test.state_load(state_path)
    np.testing.assert_array_almost_equal(df_test.woe_encoded_x.values, [13.815510, 1.098612, 0, np.nan])

def test_weight_of_evidence_encoder_bad_values(df_factory):
    y = [1, 2]
    df = df_factory(x=['a', 'b'], y=y)
    trans = vaex.ml.WeightOfEvidenceEncoder(target='y', features=['x'])
    with pytest.raises(ValueError):
        trans.fit_transform(df)

    # masked values in target should be ignored
    y = np.ma.array([1, 0], mask=[False, True])
    df = df_factory(x=['a', 'b'], y=y)
    trans.fit_transform(df)

    # nan values in target should be ignored
    y = [1, np.nan]
    df = df_factory(x=['a', 'b'], y=y)
    trans.fit_transform(df)

    # all 1's
    y = [1, 1]
    df = df_factory(x=['a', 'b'], y=y)
    trans.fit_transform(df)

    # all 0's
    y = [0, 0]
    df = df_factory(x=['a', 'b'], y=y)
    trans.fit_transform(df)

    # all nan
    y = [np.nan, np.nan]
    df = df_factory(x=['a', 'b'], y=y)
    trans.fit_transform(df)

    # masked values in target should be ignored
    y = np.ma.array([1, 0], mask=[True, True])
    df = df_factory(x=['a', 'b'], y=y)
    trans.fit_transform(df)

def test_weight_of_evidence_encoder_edge_cases(df_factory):
    y = [1, 0, 1, 0, 1, 0, 0, 0, 1, 1]
    x = ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'd', 'd']
    df = df_factory(x=x, y=y)

    woe_encoder = vaex.ml.WeightOfEvidenceEncoder(features=['x'], target='y', unseen='zero')
    df = woe_encoder.fit_transform(df)

    expected_values = [0.69314, 0.69314, 0.69314, -0.69314, -0.69314,
                      -0.69314, -13.81550, -13.81550, 13.81551, 13.81551]
    np.testing.assert_array_almost_equal(df.woe_encoded_x.tolist(),
                                         expected_values,
                                         decimal=5)

def test_groupby_transformer_basics(df_factory):
    df_train = df_factory(x=['dog', 'dog', 'dog', 'cat', 'cat'], y=[2, 3, 4, 10, 20])
    df_test = df_factory(x=['dog', 'cat', 'dog', 'mouse'], y=[5, 5, 5, 5])

    group_trans = vaex.ml.GroupByTransformer(by='x', agg={'mean_y': vaex.agg.mean('y')}, rsuffix='_agg')
    df_train_trans = group_trans.fit_transform(df_train)
    df_test_trans = group_trans.transform(df_test)

    assert df_train_trans.mean_y.tolist() == [3.0, 3.0, 3.0, 15.0, 15.0]
    assert df_test_trans.mean_y.tolist() == [3.0, 15, 3.0, None]
    assert df_test_trans.x.tolist() == ['dog', 'cat', 'dog', 'mouse']
    assert df_test_trans.y.tolist() == [5, 5, 5, 5]

    # Alternative API
    trans = df_train.ml.groupby_transformer(by='x', agg={'mean_y': vaex.agg.mean('y')}, rsuffix='_agg', transform=False)
    df_test_trans_2 = trans.transform(df_test)
    assert df_test_trans.mean_y.tolist() == df_test_trans_2.mean_y.tolist()
    assert df_test_trans.x.tolist() == df_test_trans_2.x.tolist()
    assert df_test_trans.y.tolist() == df_test_trans_2.y.tolist()


def test_groupby_transformer_serialization(df_factory):
    df_train = df_factory(x=['dog', 'dog', 'dog', 'cat', 'cat'], y=[2, 3, 4, 10, 20])
    df_test = df_factory(x=['dog', 'cat', 'dog', 'mouse'], y=[5, 5, 5, 5])

    group_trans = vaex.ml.GroupByTransformer(by='x', agg={'mean_y': vaex.agg.mean('y')}, rsuffix='_agg')
    df_train_trans = group_trans.fit_transform(df_train)

    state = df_train_trans.state_get()
    df_test.state_set(state)

    assert df_train_trans.mean_y.tolist() == [3.0, 3.0, 3.0, 15.0, 15.0]
    assert df_test.mean_y.tolist() == [3.0, 15, 3.0, None]
    assert df_test.x.tolist() == ['dog', 'cat', 'dog', 'mouse']
    assert df_test.y.tolist() == [5, 5, 5, 5]

#
# @pytest.mark.skipif(platform.system().lower() == 'windows', reason="strange ref count issue with numpy")
# @pytest.mark.parametrize('strategy', ['uniform', 'quantile', 'kmeans'])
# def test_kbinsdiscretizer(tmpdir, strategy):
#     df_train = vaex.from_arrays(x=[0, 2.5, 5, 7.5, 10, 12.5, 15],
#                                 y=[0, 0, 5, 5, 5, 9, 9])
#     df_test = vaex.from_arrays(x=[1, 4, 8, 9, 20, -2],
#                                y=[1, 2, 5, 6, 10, 9])

#     trans = vaex.ml.KBinsDiscretizer(features=['x', 'y'], n_bins=3, strategy=strategy)
#     df_train_trans = trans.fit_transform(df_train)
#     df_test_trans = trans.transform(df_test)

#     if strategy == 'quantile':
#         expected_result_train_x = [0, 0, 1, 1, 1, 2, 2]
#     else:
#         expected_result_train_x = [0, 0, 0, 1, 1, 2, 2]
#     expected_result_train_y = [0, 0, 1, 1, 1, 2, 2]
#     expected_result_test_x = [0, 0, 1, 1, 2, 0]
#     expected_result_test_y = [0, 0, 1, 1, 2, 2]

#     assert df_train_trans.shape == (7, 4)
#     assert df_test_trans.shape == (6, 4)
#     assert df_train_trans.binned_x.tolist() == expected_result_train_x
#     assert df_train_trans.binned_y.tolist() == expected_result_train_y
#     assert df_test_trans.binned_x.tolist() == expected_result_test_x
#     assert df_test_trans.binned_y.tolist() == expected_result_test_y

#     # Test serialization
#     df_train_trans.state_write(str(tmpdir.join('test.json')))
#     df_test.state_load(str(tmpdir.join('test.json')))
#     assert df_test.shape == (6, 4)
#     assert df_test.binned_x.tolist() == expected_result_test_x
#     assert df_test.binned_y.tolist() == expected_result_test_y


def test_multihot_encoder(tmpdir, df_factory):
    a = ['cat', 'dog', 'mouse']
    b = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'violet', 'magenta', 'lime', 'grey', 'white', 'black']

    np.random.seed(42)
    x_train = np.random.choice(a, size=100, replace=True)
    y_train = np.random.choice(b, size=100, replace=True)

    x_test = np.random.choice(a, size=100, replace=True)
    y_test = np.random.choice(b, size=100, replace=True)

    df_train = df_factory(animals=x_train, colors=y_train)
    df_test = df_factory(animals=x_test, colors=y_test)

    encoder = vaex.ml.MultiHotEncoder(features=['animals', 'colors'])
    encoder.fit(df_train)
    df_train = encoder.transform(df_train)

    df_train.state_write(str(tmpdir.join('test.json')))
    df_test.state_load(str(tmpdir.join('test.json')))

    # Verify dogs
    # 'dog' binary code is '010'
    dogs = df_train[df_train.animals == 'dog'][df_train.get_column_names(regex='^animals_')]
    assert dogs.animals_0.unique() == [0]
    assert dogs.animals_1.unique() == [1]
    assert dogs.animals_2.unique() == [0]
    dogs = df_test[df_test.animals == 'dog'][df_test.get_column_names(regex='^animals_')]
    assert dogs.animals_0.unique() == [0]
    assert dogs.animals_1.unique() == [1]
    assert dogs.animals_2.unique() == [0]

    # Verify cats
    # 'cat' binary code is '001'
    cats = df_train[df_train.animals == 'cat'][df_train.get_column_names(regex='^animals_')]
    assert cats.animals_0.unique() == [0]
    assert cats.animals_1.unique() == [0]
    assert cats.animals_2.unique() == [1]
    cats = df_test[df_test.animals == 'cat'][df_test.get_column_names(regex='^animals_')]
    assert cats.animals_0.unique() == [0]
    assert cats.animals_1.unique() == [0]
    assert cats.animals_2.unique() == [1]

    # Verify mice
    # mouse binary code is '011'
    mice = df_train[df_train.animals == 'mouse'][df_train.get_column_names(regex='^animals_')]
    assert mice.animals_0.unique() == [0]
    assert mice.animals_1.unique() == [1]
    assert mice.animals_2.unique() == [1]
    mice = df_test[df_test.animals == 'mouse'][df_test.get_column_names(regex='^animals_')]
    assert mice.animals_0.unique() == [0]
    assert mice.animals_1.unique() == [1]
    assert mice.animals_2.unique() == [1]


    # Verify Red
    # 'red' binary code is '1001'
    clr = df_test[df_test.colors == 'red'][df_test.get_column_names(regex='^colors_')]
    assert clr.colors_0.unique() == [1]
    assert clr.colors_1.unique() == [0]
    assert clr.colors_2.unique() == [0]
    assert clr.colors_3.unique() == [1]

    # Verity white
    # 'white' binary code is '1011'
    clr = df_test[df_test.colors == 'white'][df_test.get_column_names(regex='^colors_')]
    assert clr.colors_0.unique() == [1]
    assert clr.colors_1.unique() == [0]
    assert clr.colors_2.unique() == [1]
    assert clr.colors_3.unique() == [1]

    # Verify blue
    # 'blue' binary code is '0010'
    clr = df_test[df_test.colors == 'blue'][df_test.get_column_names(regex='^colors_')]
    assert clr.colors_0.unique() == [0]
    assert clr.colors_1.unique() == [0]
    assert clr.colors_2.unique() == [1]
    assert clr.colors_3.unique() == [0]
