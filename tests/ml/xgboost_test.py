import numpy as np
import xgboost as xgb
import vaex.ml.xgboost
import vaex.ml.datasets


# the parameters of the model
params_multiclass = {
    'learning_rate': 0.1,     # learning rate
    'max_depth': 3,           # max depth of the tree
    'colsample_bytree': 0.8,  # subsample ratio of columns when constructing each tree
    'subsample': 0.8,         # subsample ratio of the training instance
    'reg_lambda': 1,          # L2 regularisation
    'reg_alpha': 0,           # L1 regularisation
    'min_child_weight': 1,    # minimum sum of instance weight (hessian) needed in a child
    'objective': 'multi:softmax',  # learning task objective
    'num_class': 3,           # number of target classes (if classification)
    'random_state': 42,       # fixes the seed, for reproducibility
    'silent': 1,              # silent mode
    'n_jobs': -1              # cpu cores used
    }

# xgboost params
params_reg = {
    'learning_rate': 0.1,     # learning rate
    'max_depth': 3,             # max depth of the tree
    'colsample_bytree': 0.8,    # subsample ratio of columns when constructing each tree
    'subsample': 0.8,           # subsample ratio of the training instance
    'reg_lambda': 1,            # L2 regularisation
    'reg_alpha': 0,             # L1 regularisation
    'min_child_weight': 1,      # minimum sum of instance weight (hessian) needed in a child
    'objective': 'reg:linear',  # learning task objective
    'random_state': 42,         # fixes the seed, for reproducibility
    'silent': 1,                # silent mode
    'n_jobs': -1                # cpu cores used
    }


def test_xgboost():
    ds = vaex.ml.datasets.load_iris()
    ds_train, ds_test = ds.ml.train_test_split(test_size=0.2, verbose=False)
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    booster = vaex.ml.xgboost.XGBoostModel(num_boost_round=10,
                                           params=params_multiclass,
                                           features=features)
    booster.fit(ds_train, ds_train.class_)
    class_predict = booster.predict(ds_test)
    assert np.all(ds.col.class_ == class_predict)

    ds = booster.transform(ds)   # this will add the xgboost_prediction column
    state = ds.state_get()
    ds = vaex.ml.datasets.load_iris()
    ds.state_set(state)
    assert np.all(ds.col.class_ == ds.evaluate(ds.xgboost_prediction))


def test_xgboost_numerical_validation():
    ds = vaex.ml.datasets.load_iris()
    features = ['sepal_width', 'petal_length', 'sepal_length', 'petal_width']

    # Vanilla xgboost
    dtrain = xgb.DMatrix(ds[features], label=ds.data.class_)
    xgb_bst = xgb.train(params=params_multiclass, dtrain=dtrain, num_boost_round=3)
    xgb_pred = xgb_bst.predict(dtrain)

    # xgboost through vaex
    booster = vaex.ml.xgboost.XGBoostModel(features=features, params=params_multiclass, num_boost_round=3)
    booster.fit(ds, ds.class_)
    vaex_pred = booster.predict(ds)

    # Comparing the the predictions of xgboost vs vaex.ml
    np.testing.assert_equal(vaex_pred, xgb_pred, verbose=True,
                            err_msg='The predictions of vaex.ml.xboost do not match those of pure xgboost')


def test_xgboost_serialize(tmpdir):
    ds = vaex.ml.datasets.load_iris()
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    target = 'class_'

    gbm = ds.ml.xgboost_model(target, 20, features=features, params=params_multiclass)
    pl = vaex.ml.Pipeline([gbm])
    pl.save(str(tmpdir.join('test.json')))
    pl.load(str(tmpdir.join('test.json')))

    gbm = ds.ml.xgboost_model(target, 20, features=features, params=params_multiclass)
    gbm.state_set(gbm.state_get())
    pl = vaex.ml.Pipeline([gbm])
    pl.save(str(tmpdir.join('test.json')))
    pl.load(str(tmpdir.join('test.json')))


def test_xgboost_validation_set():
    # read data
    ds = vaex.example()
    # Train and test split
    train, test = ds.ml.train_test_split(verbose=False)
    # Define the training featuress
    features = ['vx', 'vy', 'vz', 'Lz', 'L']
    # history of the booster (evaluations of the train and validation sets)
    history = {}
    # instantiate the booster model
    booster = vaex.ml.xgboost.XGBoostModel(features=features, num_boost_round=10, params=params_reg)
    # fit the booster - including saving the history of the validation sets
    booster.fit(train, 'E', evals=[(train, 'train'), (test, 'test')],
                early_stopping_rounds=2, evals_result=history)
    assert booster.booster.best_ntree_limit == 10
    assert booster.booster.best_iteration == 9
    assert len(history['train']['rmse']) == 10
    assert len(history['test']['rmse']) == 10


def test_xgboost_pipeline():
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
    # now the xgboost model thingy
    features = ['r', 'PCA_0', 'PCA_1', 'PCA_2']
    # define the boosting model
    booster = train.ml.xgboost_model(target='E', num_boost_round=10, features=features, params=params_reg)
    # Create a pipeline
    pp = vaex.ml.Pipeline([st, booster])
    # Use the pipeline
    pred = pp.predict(test)                  # This works
    trans = pp.transform(test)               # This will crash (softly)
    # trans.evaluate('xgboost_prediction')   # This is where the problem happens
    np.testing.assert_equal(pred,
                            trans.evaluate('xgboost_prediction'),
                            verbose=True,
                            err_msg='The predictions from the predict and transform method do not match')
