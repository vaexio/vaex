import pytest
import numpy as np
import vaex
import vaex.ml
import vaex.ml.datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, MaxAbsScaler, RobustScaler


def test_pca():
    ds = vaex.ml.datasets.load_iris()
    pca1 = ds.ml.pca(features=[ds.petal_width, ds.petal_length], n_components=2)
    pca2 = ds.ml.pca(features=[ds.sepal_width, ds.sepal_length, ds.petal_length], n_components=3)
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


def test_valid_sklearn_pca():
    ds = vaex.ml.datasets.load_iris()
    features = ['sepal_width', 'petal_length', 'sepal_length', 'petal_width']
    # sklearn approach
    pca = PCA(n_components=3, random_state=33, svd_solver='full', whiten=False)
    pca.fit(ds[features])
    sklearn_trans = pca.transform(ds[features])
    # vaex-ml approach
    vaexpca = ds.ml.pca(n_components=3, features=features)
    vpca = vaexpca.transform(ds[features])
    # Compare the two approaches
    np.testing.assert_almost_equal(vpca.evaluate('PCA_0'), sklearn_trans[:, 0])


def test_standard_scaler():
    ds = vaex.ml.datasets.load_iris()
    ss1 = ds.ml.standard_scaler(features=[ds.petal_width, ds.petal_length], with_mean=True, with_std=True)
    ss2 = ds.ml.standard_scaler(features=[ds.petal_width, ds.petal_length], with_mean=True, with_std=False)
    ss3 = ds.ml.standard_scaler(features=[ds.petal_width, ds.petal_length], with_mean=False, with_std=True)
    ss4 = ds.ml.standard_scaler(features=[ds.petal_width, ds.petal_length], with_mean=False, with_std=False)
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


def test_minmax_scaler():
    ds = vaex.ml.datasets.load_iris()
    mms1 = ds.ml.minmax_scaler(features=[ds.petal_width, ds.petal_length])
    mms2 = ds.ml.minmax_scaler(features=[ds.petal_width, ds.petal_length], feature_range=(-5, 2))
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


def test_train_test_split_values():
    # Create a dataset
    ds = vaex.from_arrays(x=np.arange(100))
    # Set active range to begin with
    ds.set_active_range(50, 100)
    # do the splitting
    train, test = ds.ml.train_test_split(verbose=False)
    # assert whether the split is correct
    np.testing.assert_equal(np.arange(50, 60), test.evaluate('x'))
    np.testing.assert_equal(np.arange(60, 100), train.evaluate('x'))


def test_frequency_encoder():
    animals = np.array(['dog', 'dog', 'cat', 'dog', 'mouse', 'mouse'])
    train = vaex.from_arrays(animals=animals)
    animals = np.array(['dog', 'cat', 'mouse', 'ant', np.nan], dtype=np.object)
    test = vaex.from_arrays(animals=animals)
    features = ['animals']

    fe = train.ml.frequency_encoder(features=features, unseen='nan')
    fe.fit(train)
    test_a = fe.transform(test)
    np.testing.assert_almost_equal(test_a.frequency_encoded_animals.values,
                                   [0.5, 0.166, 0.333, np.nan, np.nan],
                                   decimal=3)
    fe = vaex.ml.FrequencyEncoder(features=features, unseen='zero')
    fe.fit(train)
    test_b = fe.transform(test)
    np.testing.assert_almost_equal(test_b.frequency_encoded_animals.values,
                                   [0.5, 0.166, 0.333, 0, 0],
                                   decimal=3)


def test_label_encoder():
    # Options to choose from
    m = np.array(['dog', 'cat', 'mouse', 'horse', 'pig'])
    n = np.array(['yacht', 'boat', 'ship', 'submarine', 'dinghy', 'catamaran'])
    o = np.array([0, 1, 2, 3, 4])
    q = np.array(['dog', 'cat', 'mouse', 'dragon'])
    x = np.random.choice(m, size=40)
    y = np.random.choice(n, size=40)
    z = np.random.choice(o, size=40)
    # Create s vaex dataset
    ds = vaex.from_arrays(x=x, y=y, z=z)
    dd = vaex.from_arrays(x=np.random.choice(q, size=40), y=y)  # this is to test the unseen categories
    # Split in train and test sets
    train, test = ds.ml.train_test_split(test_size=0.25, verbose=False)
    # Label Encode with vaex
    le_vaex = train.ml.label_encoder(features=['x', 'y', 'z'], prefix='mypref_')
    # Transfrom
    train = le_vaex.transform(train)
    test = le_vaex.transform(test)
    # Label Encode with scikit-learn
    le_skl_x = LabelEncoder().fit(train.evaluate('x'))
    le_skl_y = LabelEncoder().fit(train.evaluate('y'))
    le_skl_z = LabelEncoder().fit(train.evaluate('z'))
    # Compare the test set only
    np.testing.assert_equal(le_skl_x.transform(test.evaluate('x')), test.evaluate('mypref_x'))
    np.testing.assert_almost_equal(le_skl_y.transform(test.evaluate('y')), test.evaluate('mypref_y'))
    np.testing.assert_almost_equal(le_skl_z.transform(test.evaluate('z')), test.evaluate('mypref_z'))
    # Try to get labels from the dd dataset unseen categories
    with pytest.raises(ValueError):
        le_vaex.transform(dd)
    # Fit-transform
    le = vaex.ml.LabelEncoder(features=['x', 'y', 'z'], prefix='mypref_')
    le.fit_transform(ds)


def test_one_hot_encoding():
    # Categories
    a = ['cat', 'dog', 'mouse']
    b = ['boy', 'girl']
    c = [0, 1]
    # Generate data
    x = np.random.choice(a, size=100, replace=True)
    y = np.random.choice(b, size=100, replace=True)
    z = np.random.choice(c, size=100, replace=True)
    # Create dataset
    ds = vaex.from_arrays(animals=x, kids=y, numbers=z)
    # First try to one-hot encode without specifying features: this should raise an exception
    with pytest.raises(ValueError):
        onehot = ds.ml.one_hot_encoder(features=None)
    # split in train and test
    train, test = ds.ml.train_test_split(test_size=.25, verbose=False)
    # fit onehot encoder on the train set
    onehot = train.ml.one_hot_encoder(features=['kids', 'animals', 'numbers'], prefix='')
    # transform the test set
    test = onehot.transform(test)
    # asses the success of the test
    np.testing.assert_equal(test.kids_boy.values, np.array([1 if i == 'boy' else 0 for i in test.kids.values]))
    np.testing.assert_equal(test.kids_girl.values, np.array([0 if i == 'boy' else 1 for i in test.kids.values]))
    np.testing.assert_equal(test.animals_dog.values, np.array([1 if i == 'dog' else 0 for i in test.animals.values]))
    np.testing.assert_equal(test.animals_cat.values, np.array([1 if i == 'cat' else 0 for i in test.animals.values]))
    np.testing.assert_equal(test.animals_mouse.values, np.array([1 if i == 'mouse' else 0 for i in test.animals.values]))
    np.testing.assert_equal(test.numbers_0.values, np.array([1 if i == 0 else 0 for i in test.numbers.values]))
    np.testing.assert_equal(test.numbers_1.values, np.array([0 if i == 0 else 1 for i in test.numbers.values]))
    # Fit-transform
    ohe = vaex.ml.OneHotEncoder(features=['kids', 'animals', 'numbers'])
    ohe.fit_transform(ds)


def test_maxabs_scaler():
    x = np.array([-2.65395789, -7.97116295, -4.76729177, -0.76885033, -6.45609635])
    y = np.array([-8.9480332, -4.81582449, -3.73537263, -3.46051912,  1.35137275])
    z = np.array([-0.47827432, -2.26208059, -3.75151683, -1.90862151, -1.87541903])
    w = np.zeros_like(x)

    ds = vaex.from_arrays(x=x, y=y, z=z, w=w)
    df = ds.to_pandas_df()

    features = ['x', 'y', 'w']

    scaler_skl = MaxAbsScaler()
    result_skl = scaler_skl.fit_transform(df[features])
    scaler_vaex = vaex.ml.MaxAbsScaler(features=features)
    result_vaex = scaler_vaex.fit_transform(ds)

    assert result_vaex.absmax_scaled_x.values.tolist() == result_skl[:, 0].tolist(), "scikit-learn and vaex results do not match"
    assert result_vaex.absmax_scaled_y.values.tolist() == result_skl[:, 1].tolist(), "scikit-learn and vaex results do not match"
    assert result_vaex.absmax_scaled_w.values.tolist() == result_skl[:, 2].tolist(), "scikit-learn and vaex results do not match"


def test_robust_scaler():
    x = np.array([-2.65395789, -7.97116295, -4.76729177, -0.76885033, -6.45609635])
    y = np.array([-8.9480332, -4.81582449, -3.73537263, -3.46051912,  1.35137275])
    z = np.array([-0.47827432, -2.26208059, -3.75151683, -1.90862151, -1.87541903])
    w = np.zeros_like(x)

    ds = vaex.from_arrays(x=x, y=y, z=z, w=w)
    df = ds.to_pandas_df()

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
