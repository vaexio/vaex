import pytest
pytest.importorskip("pygbm")

import os
import numpy as np
import pygbm as lgb
import vaex.ml.incubator.pygbm
import vaex.ml.datasets
from vaex.utils import _ensure_strings_from_expressions
import test_utils


# the parameters of the model
param = {'learning_rate': 0.1,     # learning rate
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


@test_utils.skip_incubator
def test_py_gbm_virtual_columns():
    ds = vaex.ml.datasets.load_iris()
    ds['x'] = ds.sepal_length * 1
    ds['y'] = ds.sepal_width * 1
    ds['w'] = ds.petal_length * 1
    ds['z'] = ds.petal_width * 1
    ds_train, ds_test = ds.ml.train_test_split(test_size=0.2, verbose=False)
    features = ['x', 'y', 'z', 'w']
    booster = vaex.ml.incubator.pygbm.PyGBMModel(num_round=10, param=param,
                                       features=_ensure_strings_from_expressions(features))
    booster.fit(ds_train, ds_train.class_)

@test_utils.skip_incubator
def test_pygbm():
    for filename in 'blah.col.meta blah.col.page'.split():
        if os.path.exists(filename):
            os.remove(filename)
    ds = vaex.ml.datasets.load_iris()
    ds_train, ds_test = ds.ml.train_test_split(test_size=0.2, verbose=False)
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    booster = vaex.ml.incubator.pygbm.PyGBMModel(num_round=10, param=param,
                                       features=_ensure_strings_from_expressions(features))
    booster.fit(ds_train, ds_train.class_)
    class_predict = booster.predict(ds_test)
    booster.fit(ds_train, ds_train.class_)
    class_predict = booster.predict(ds_test)
    assert np.all(ds.col.class_ == class_predict)

    ds = booster.transform(ds)   # this will add the pygbm_prediction column
    state = ds.state_get()
    ds = vaex.ml.datasets.load_iris()
    ds.state_set(state)
    assert np.all(ds.col.class_ == ds.evaluate(ds.pygbm_prediction))

@test_utils.skip_incubator
def test_pygbm_serialize(tmpdir):
    ds = vaex.ml.datasets.load_iris()
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    target = 'class_'

    gbm = ds.ml_pygbm_model(target, 20, features=features, param=param, classifier=False)
    pl = vaex.ml.Pipeline([gbm])
    pl.save(str(tmpdir.join('test.json')))
    pl.load(str(tmpdir.join('test.json')))

    gbm = ds.ml_pygbm_model(target, 20, features=features, param=param, classifier=True)
    gbm.state_set(gbm.state_get())
    pl = vaex.ml.Pipeline([gbm])
    pl.save(str(tmpdir.join('test.json')))
    pl.load(str(tmpdir.join('test.json')))


@test_utils.skip_incubator
def test_pygbm_invalid():
    for filename in 'blah.col.meta blah.col.page'.split():
        if os.path.exists(filename):
            os.remove(filename)
    ds = vaex.ml.iris()
    features = ['sepal_length', 'sepal_width', 'petal_length', 'wrong']
    booster = vaex.ml.pygbm.XGBModel(num_round=10, param=param,
                                       features=vaex.dataset._ensure_strings_from_expressions(features))
    booster.fit(ds, ds.class_)

@test_utils.skip_incubator
def test_pygbm_validation():
    ds = vaex.ml.iris()
    features = ['sepal_width', 'petal_length', 'sepal_length', 'petal_width']

    # Vanilla pygbm
    X = np.array(ds[features])
    # dtrain = lgb.Dataset(X, label=ds.data.class_)
    lgb_bst = lgb.train(param, dtrain, 3)
    lgb_pred = np.argmax(lgb_bst.predict(X), axis=1)

    # Vaex.ml.pygbm
    booster = ds.ml.pygbm_model(label='class_', num_round=3, features=features, param=param, classifier=True)
    vaex_pred = booster.predict(ds)

    # Comparing the the predictions of pygbm vs vaex.ml
    np.testing.assert_equal(vaex_pred, lgb_pred, verbose=True, err_msg='The predictions of vaex.ml do not match those of pygbm')


@test_utils.skip_incubator
def test_pygbm_pipeline():
    param = {'learning_rate': 0.1,     # learning rate
             'max_depth': 5,             # max depth of the tree
             'colsample_bytree': 0.8,    # subsample ratio of columns when con$
             'subsample': 0.8,           # subsample ratio of the training ins$
             'reg_lambda': 1,            # L2 regularisation
             'reg_alpha': 0,             # L1 regularisation
             'min_child_weight': 1,      # minimum sum of instance weight (hes$
             'objective': 'regression',  # learning task objective
             'random_state': 42,         # fixes the seed, for reproducibility
             'silent': 1,                # silent mode
             'n_jobs': -1}               # cpu cores used

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
    # now the pygbm model thingy
    features = ['r', 'PCA_0', 'PCA_1', 'PCA_2']
    # The booster model from vaex.ml
    booster = train.ml.pygbm_model(label='E', max_iter=10, features=features, param=param)
    # Create a pipeline
    pp = vaex.ml.Pipeline([st, booster])
    # Use the pipeline
    pred = pp.predict(test)                  # This works
    trans = pp.transform(test)               # This will crash (softly)
    # trans.evaluate('pygbm_prediction')   # This is where the problem happens
    np.testing.assert_equal(pred, trans.evaluate('pygbm_prediction'), verbose=True, err_msg='The predictions from the fit and transform method do not match')
