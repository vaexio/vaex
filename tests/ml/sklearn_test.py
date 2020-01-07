import pytest
import vaex
pytest.importorskip("sklearn")
from vaex.ml.sklearn import Predictor, IncrementalPredictor

import numpy as np

# Regressions
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDClassifier, SGDRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier


models_regression = [LinearRegression(),
                     Ridge(random_state=42, max_iter=100),
                     Lasso(random_state=42, max_iter=100),
                     SVR(gamma='scale'),
                     AdaBoostRegressor(random_state=42, n_estimators=10),
                     GradientBoostingRegressor(random_state=42, max_depth=3, n_estimators=10),
                     RandomForestRegressor(n_estimators=10, random_state=42, max_depth=3)]

models_classification = [LogisticRegression(solver='lbfgs', max_iter=100, random_state=42),
                         SVC(gamma='scale', max_iter=100),
                         AdaBoostClassifier(random_state=42, n_estimators=10),
                         GradientBoostingClassifier(random_state=42, max_depth=3, n_estimators=10),
                         RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)]


def test_sklearn_estimator():
    ds = vaex.ml.datasets.load_iris()
    features = ['sepal_length', 'sepal_width', 'petal_length']

    train, test = ds.ml.train_test_split(verbose=False)

    model = Predictor(model=LinearRegression(), features=features, prediction_name='pred')
    model.fit(train, train.petal_width)
    prediction = model.predict(test)
    test = model.transform(test)
    np.testing.assert_array_almost_equal(test.pred.values, prediction, decimal=5)

    # Transfer the state of train to ds
    train = model.transform(train)
    state = train.state_get()
    ds.state_set(state)
    assert ds.pred.values.shape == (150,)


def test_sklearn_estimator_virtual_columns():
    ds = vaex.ml.datasets.load_iris()
    ds['x'] = ds.sepal_length * 1
    ds['y'] = ds.sepal_width * 1
    ds['w'] = ds.petal_length * 1
    ds['z'] = ds.petal_width * 1
    train, test = ds.ml.train_test_split(test_size=0.2, verbose=False)
    features = ['x', 'y', 'z']
    model = Predictor(model=LinearRegression(), features=features, prediction_name='pred')
    model.fit(ds, ds.w)
    ds = model.transform(ds)
    assert ds.pred.values.shape == (150,)


def test_sklearn_estimator_serialize(tmpdir):
    ds = vaex.ml.datasets.load_iris()
    features = ['sepal_length', 'sepal_width', 'petal_length']

    model = Predictor(model=LinearRegression(), features=features, prediction_name='pred')
    model.fit(ds, ds.petal_width)

    pipeline = vaex.ml.Pipeline([model])
    pipeline.save(str(tmpdir.join('test.json')))
    pipeline.load(str(tmpdir.join('test.json')))

    model = Predictor(model=LinearRegression(), features=features, prediction_name='pred')
    model.fit(ds, ds.petal_width)

    model.state_set(model.state_get())
    pipeline = vaex.ml.Pipeline([model])
    pipeline.save(str(tmpdir.join('test.json')))
    pipeline.load(str(tmpdir.join('test.json')))


def test_sklearn_estimator_regression_validation():
    ds = vaex.ml.datasets.load_iris()
    train, test = ds.ml.train_test_split(verbose=False)
    features = ['sepal_length', 'sepal_width', 'petal_length']

    # Dense features
    Xtrain = train[features].values
    Xtest = test[features].values
    ytrain = train.petal_width.values

    for model in models_regression:

        # vaex
        vaex_model = Predictor(model=model, features=features, prediction_name='pred')
        vaex_model.fit(train, train.petal_width)
        test = vaex_model.transform(test)

        # sklearn
        model.fit(Xtrain, ytrain)
        skl_pred = model.predict(Xtest)

        np.testing.assert_array_almost_equal(test.pred.values, skl_pred, decimal=5)


def test_sklearn_estimator_pipeline():
    ds = vaex.ml.datasets.load_iris()
    train, test = ds.ml.train_test_split(verbose=False)
    # Add virtual columns
    train['sepal_virtual'] = np.sqrt(train.sepal_length**2 + train.sepal_width**2)
    train['petal_scaled'] = train.petal_length * 0.2
    # Do a pca
    features = ['sepal_virtual', 'petal_scaled']
    pca = train.ml.pca(n_components=2, features=features)
    train = pca.transform(train)
    # Do state transfer
    st = train.ml.state_transfer()
    # now apply the model
    features = ['sepal_virtual', 'petal_scaled']
    model = Predictor(model=LinearRegression(), features=features, prediction_name='pred')
    model.fit(train, train.petal_width)
    # Create a pipeline
    pipeline = vaex.ml.Pipeline([st, model])
    # Use the pipeline
    pred = pipeline.predict(test)
    df_trans = pipeline.transform(test)

    # WARNING: on windows/appveyor this gives slightly different results
    # do we fully understand why? I also have the same results on my osx laptop
    # sklearn 0.21.1 (scikit-learn-0.21.2 is installed on windows) so it might be a
    # version related thing
    np.testing.assert_array_almost_equal(pred, df_trans.pred.values)


def test_sklearn_estimator_classification_validation():
    ds = vaex.ml.datasets.load_titanic()

    train, test = ds.ml.train_test_split(verbose=False)
    features = ['pclass', 'parch', 'sibsp']

    # Dense features
    Xtrain = train[features].values
    Xtest = test[features].values
    ytrain = train.survived.values

    for model in models_classification:

        # vaex
        vaex_model = Predictor(model=model, features=features, prediction_name='pred')
        vaex_model.fit(train, train.survived)
        test = vaex_model.transform(test)

        # scikit-learn
        model.fit(Xtrain, ytrain)
        skl_pred = model.predict(Xtest)

        assert np.all(skl_pred == test.pred.values)


def test_sklearn_incremental_predictor_regression():
    df = vaex.example()
    df_train, df_test = df.ml.train_test_split(test_size=0.1, verbose=False)

    features = df_train.column_names[:6]
    target = 'FeH'

    incremental = IncrementalPredictor(model=SGDRegressor(),
                                       features=features,
                                       batch_size=10*1000,
                                       num_epochs=5,
                                       shuffle=True,
                                       prediction_name='pred')
    incremental.fit(df=df_train, target=target)
    df_train = incremental.transform(df_train)

    # State transfer
    state = df_train.state_get()
    df_test.state_set(state)

    assert df_train.column_count() == df_test.column_count()
    assert df_test.pred.values.shape == (33000,)

    pred_in_memory = incremental.predict(df_test)
    np.testing.assert_array_almost_equal(pred_in_memory, df_test.pred.values, decimal=1)


def test_sklearn_incremental_predictor_classification():
    df = vaex.ml.datasets.load_iris_1e5()
    df_train, df_test = df.ml.train_test_split(test_size=0.1, verbose=False)

    features = df_train.column_names[:4]
    target = 'class_'

    incremental = IncrementalPredictor(model=SGDClassifier(learning_rate='constant', eta0=0.01),
                                       features=features,
                                       batch_size=10_000,
                                       num_epochs=3,
                                       shuffle=False,
                                       prediction_name='pred',
                                       partial_fit_kwargs={'classes':[0, 1, 2]})

    incremental.fit(df=df_train, target=target)
    df_train = incremental.transform(df_train)

    # State transfer
    state = df_train.state_get()
    df_test.state_set(state)

    assert df_test.column_count() == 6
    assert df_test.pred.values.shape == (10050,)

    pred_in_memory = incremental.predict(df_test)
    np.testing.assert_array_equal(pred_in_memory, df_test.pred.values)


def test_sklearn_incremental_predictor_serialize(tmpdir):
    df = vaex.example()
    df_train, df_test = df.ml.train_test_split(test_size=0.1, verbose=False)

    features = df_train.column_names[:6]
    target = 'FeH'

    incremental = IncrementalPredictor(model=SGDRegressor(),
                                       features=features,
                                       batch_size=10*1000,
                                       num_epochs=5,
                                       shuffle=True,
                                       prediction_name='pred')
    incremental.fit(df=df_train, target=target)
    df_train = incremental.transform(df_train)

    # State transfer - serialization
    df_train.state_write(str(tmpdir.join('test.json')))
    df_test.state_load(str(tmpdir.join('test.json')))

    assert df_train.column_count() == df_test.column_count()
    assert df_test.pred.values.shape == (33000,)

    pred_in_memory = incremental.predict(df_test)
    np.testing.assert_array_almost_equal(pred_in_memory, df_test.pred.values, decimal=1)


@pytest.mark.parametrize("batch_size", [6789, 10*1000])
@pytest.mark.parametrize("num_epochs", [1, 5])
def test_sklearn_incremental_predictor_partial_fit_calls(batch_size, num_epochs):
    df = vaex.example()
    df_train, df_test = df.ml.train_test_split(test_size=0.1, verbose=False)

    features = df_train.column_names[:6]
    target = 'FeH'

    N_total = len(df_train)
    num_batches = (N_total + batch_size - 1) // batch_size

    # Create a mock model for counting the number of samples seen and partial_fit calls
    class MockModel():
        def __init__(self):
            self.n_samples_ = 0
            self.n_partial_fit_calls_ = 0

        def partial_fit(self, X, y):
            self.n_samples_ += X.shape[0]
            self.n_partial_fit_calls_ += 1

    incremental = IncrementalPredictor(model=MockModel(),
                                       features=features,
                                       batch_size=batch_size,
                                       num_epochs=num_epochs,
                                       shuffle=False,
                                       prediction_name='pred')

    incremental.fit(df=df_train, target=target)
    assert incremental.model.n_samples_ == N_total * num_epochs
    assert incremental.model.n_partial_fit_calls_ == num_batches * num_epochs
