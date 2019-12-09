import pytest
import vaex
pytest.importorskip("sklearn")
from vaex.ml.sklearn import SKLearnPredictor

import numpy as np

# Regressions
from sklearn.linear_model import LinearRegression, Ridge, Lasso
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

    model = SKLearnPredictor(model=LinearRegression(), features=features, prediction_name='pred')
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
    model = SKLearnPredictor(model=LinearRegression(), features=features, prediction_name='pred')
    ds = model.fit_transform(ds, ds.w)
    assert ds.pred.values.shape == (150,)


def test_sklearn_estimator_serialize(tmpdir):
    ds = vaex.ml.datasets.load_iris()
    features = ['sepal_length', 'sepal_width', 'petal_length']

    model = SKLearnPredictor(model=LinearRegression(), features=features, prediction_name='pred')
    model.fit(ds, ds.petal_width)

    pipeline = vaex.ml.Pipeline([model])
    pipeline.save(str(tmpdir.join('test.json')))
    pipeline.load(str(tmpdir.join('test.json')))

    model = SKLearnPredictor(model=LinearRegression(), features=features, prediction_name='pred')
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
        vaex_model = SKLearnPredictor(model=model, features=features, prediction_name='pred')
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
    model = SKLearnPredictor(model=LinearRegression(), features=features, prediction_name='pred')
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
        vaex_model = SKLearnPredictor(model=model, features=features, prediction_name='pred')
        vaex_model.fit(train, train.survived)
        test = vaex_model.transform(test)

        # scikit-learn
        model.fit(Xtrain, ytrain)
        skl_pred = model.predict(Xtest)

        assert np.all(skl_pred == test.pred.values)
