import pytest
pytest.importorskip("lightgbm")

import numpy as np
import lightgbm as lgb
import vaex.ml.lightgbm
import vaex.ml.datasets
from vaex.utils import _ensure_strings_from_expressions


# the parameters of the model
params = {
    'learning_rate': 0.1,     # learning rate
    'max_depth': 1,           # max depth of the tree
    'colsample_bytree': 0.8,  # subsample ratio of columns when constructing each tree
    'subsample': 0.8,         # subsample ratio of the training instance
    'reg_lambda': 1,          # L2 regularisation
    'reg_alpha': 0,           # L1 regularisation
    'min_child_weight': 1,    # minimum sum of instance weight (hessian) needed in a child
    'objective': 'softmax',   # learning task objective
    'num_class': 3,           # number of target classes (if classification)
    'random_state': 42,       # fixes the seed, for reproducibility
    'n_jobs': -1}             # cpu cores used

params_reg = {
    'learning_rate': 0.1,     # learning rate
    'max_depth': 3,             # max depth of the tree
    'colsample_bytree': 0.8,    # subsample ratio of columns when con$
    'subsample': 0.8,           # subsample ratio of the training ins$
    'reg_lambda': 1,            # L2 regularisation
    'reg_alpha': 0,             # L1 regularisation
    'min_child_weight': 1,      # minimum sum of instance weight (hes$
    'objective': 'regression',  # learning task objective
    'random_state': 42,         # fixes the seed, for reproducibility
    'n_jobs': -1}               # cpu cores used


def test_light_gbm_virtual_columns():
    ds = vaex.ml.datasets.load_iris()
    ds['x'] = ds.sepal_length * 1
    ds['y'] = ds.sepal_width * 1
    ds['w'] = ds.petal_length * 1
    ds['z'] = ds.petal_width * 1
    ds_train, ds_test = ds.ml.train_test_split(test_size=0.2, verbose=False)
    features = ['x', 'y', 'z', 'w']
    booster = vaex.ml.lightgbm.LightGBMModel(num_boost_round=10,
                                             params=params,
                                             features=features)
    booster.fit(ds_train, ds.class_, copy=False)


def test_lightgbm():
    ds = vaex.ml.datasets.load_iris()
    ds_train, ds_test = ds.ml.train_test_split(test_size=0.2, verbose=False)
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    booster = vaex.ml.lightgbm.LightGBMModel(num_boost_round=10, params=params,
                                             features=_ensure_strings_from_expressions(features))
    booster.fit(ds_train, ds.class_, copy=True)    # for coverage
    class_predict = booster.predict(ds_test, copy=True)  # for coverage
    booster.fit(ds_train, ds.class_)
    class_predict = booster.predict(ds_test)
    assert np.all(ds.col.class_ == class_predict)

    ds = booster.transform(ds)   # this will add the lightgbm_prediction column
    state = ds.state_get()
    ds = vaex.ml.datasets.load_iris()
    ds.state_set(state)
    assert np.all(ds.col.class_ == ds.evaluate(ds.lightgbm_prediction))


def test_lightgbm_serialize(tmpdir):
    ds = vaex.ml.datasets.load_iris()
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    target = 'class_'

    gbm = ds.ml_lightgbm_model(target, 20, features=features, params=params)
    pl = vaex.ml.Pipeline([gbm])
    pl.save(str(tmpdir.join('test.json')))
    pl.load(str(tmpdir.join('test.json')))

    gbm = ds.ml_lightgbm_model(target, 20, features=features, params=params)
    gbm.state_set(gbm.state_get())
    pl = vaex.ml.Pipeline([gbm])
    pl.save(str(tmpdir.join('test.json')))
    pl.load(str(tmpdir.join('test.json')))


def test_lightgbm_numerical_validation():
    ds = vaex.ml.datasets.load_iris()
    features = ['sepal_width', 'petal_length', 'sepal_length', 'petal_width']

    # Vanilla lightgbm
    X = np.array(ds[features])
    dtrain = lgb.Dataset(X, label=ds.class_.values)
    lgb_bst = lgb.train(params, dtrain, 3)
    lgb_pred = lgb_bst.predict(X)

    # Vaex.ml.lightgbm
    booster = ds.ml.lightgbm_model(target=ds.class_, num_boost_round=3, features=features, params=params, copy=False)
    vaex_pred = booster.predict(ds)

    # Comparing the the predictions of lightgbm vs vaex.ml
    np.testing.assert_equal(vaex_pred, lgb_pred, verbose=True, err_msg='The predictions of vaex.ml do not match those of lightgbm')


def test_lightgbm_validation_set():
    # read data
    ds = vaex.example()
    # Train and test split
    train, test = ds.ml.train_test_split(verbose=False)
    # Define the training featuress
    features = ['vx', 'vy', 'vz', 'Lz', 'L']
    # history of the booster (evaluations of the train and validation sets)
    history = {}
    # instantiate the booster model
    booster = vaex.ml.lightgbm.LightGBMModel(features=features, num_boost_round=10, params=params_reg)
    # fit the booster - including saving the history of the validation sets
    with pytest.warns(UserWarning):
        booster.fit(train, 'E', valid_sets=[train, test], valid_names=['train', 'test'],
                    early_stopping_rounds=2, evals_result=history, copy=False)
    assert booster.booster.best_iteration == 10
    assert len(history['train']['l2']) == 10
    assert len(history['test']['l2']) == 10
    booster.fit(train, 'E', valid_sets=[train, test], valid_names=['train', 'test'],
                early_stopping_rounds=2, evals_result=history, copy=True)
    assert booster.booster.best_iteration == 10
    assert len(history['train']['l2']) == 10
    assert len(history['test']['l2']) == 10


def test_lightgbm_pipeline():
    # read data
    ds = vaex.example()
    # train test splot
    train, test = ds.ml.train_test_split(verbose=False)
    # add virtual columns
    train['r'] = np.sqrt(train.x**2 + train.y**2 + train.z**2)
    # Do a pca
    features = ['vx', 'vy', 'vz', 'Lz', 'L']
    pca = train.ml.pca(n_components=3, features=features)
    train = pca.transform(train)
    # Do state transfer
    st = vaex.ml.state_transfer(train)
    # now the lightgbm model thingy
    features = ['r', 'PCA_0', 'PCA_1', 'PCA_2']
    # The booster model from vaex.ml
    booster = train.ml.lightgbm_model(target='E', num_boost_round=10, features=features, params=params_reg)
    # Create a pipeline
    pp = vaex.ml.Pipeline([st, booster])
    # Use the pipeline
    pred = pp.predict(test)                  # This works
    trans = pp.transform(test)               # This will crash (softly)
    # trans.evaluate('lightgbm_prediction')   # This is where the problem happens
    np.testing.assert_equal(pred, trans.evaluate('lightgbm_prediction'),
                            verbose=True, err_msg='The predictions from the fit and transform method do not match')
