import numpy as np
import catboost as cb
import vaex.ml.catboost
import vaex.ml.datasets


# the parameters of the model
params_multiclass = {
    'leaf_estimation_method': 'Gradient',
    'learning_rate': 0.1,
    'max_depth': 3,
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.8,
    'sampling_frequency': 'PerTree',
    'colsample_bylevel': 0.8,
    'reg_lambda': 1,
    'objective': 'MultiClass',
    'eval_metric': 'MultiClass',
    'random_state': 42,
    'verbose': 0,
}

# catboost params
params_reg = {
    'leaf_estimation_method': 'Gradient',
    'learning_rate': 0.1,
    'max_depth': 3,
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.8,
    'sampling_frequency': 'PerTree',
    'colsample_bylevel': 0.8,
    'reg_lambda': 1,
    'objective': 'MAE',
    'eval_metric': 'R2',
    'random_state': 42,
    'verbose': 0,
}


def test_catboost():
    ds = vaex.ml.datasets.load_iris()
    ds_train, ds_test = ds.ml.train_test_split(test_size=0.2, verbose=False)
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    booster = vaex.ml.catboost.CatBoostModel(num_boost_round=10,
                                             params=params_multiclass,
                                             features=features)
    booster.fit(ds_train, ds_train.class_)
    class_predict = booster.predict(ds_test)
    assert np.all(ds.col.class_ == class_predict)

    ds = booster.transform(ds)   # this will add the catboost_prediction column
    state = ds.state_get()
    ds = vaex.ml.datasets.load_iris()
    ds.state_set(state)
    assert np.all(ds.col.class_ == ds.evaluate(ds.catboost_prediction))


def test_catboost_numerical_validation():
    ds = vaex.ml.datasets.load_iris()
    features = ['sepal_width', 'petal_length', 'sepal_length', 'petal_width']

    # Vanilla catboost
    dtrain = cb.Pool(ds[features].values, label=ds.data.class_)
    cb_bst = cb.train(params=params_multiclass, dtrain=dtrain, num_boost_round=3)
    cb_pred = cb_bst.predict(dtrain, prediction_type='Probability')

    # catboost through vaex
    booster = vaex.ml.catboost.CatBoostModel(features=features, params=params_multiclass, num_boost_round=3)
    booster.fit(ds, ds.class_)
    vaex_pred = booster.predict(ds)

    # Comparing the the predictions of catboost vs vaex.ml
    np.testing.assert_equal(vaex_pred, cb_pred, verbose=True,
                            err_msg='The predictions of vaex.ml.catboost do not match those of pure catboost')


def test_lightgbm_serialize(tmpdir):
    ds = vaex.ml.datasets.load_iris()
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    target = 'class_'

    gbm = ds.ml.catboost_model(target, 20, features=features, params=params_multiclass)
    pl = vaex.ml.Pipeline([gbm])
    pl.save(str(tmpdir.join('test.json')))
    pl.load(str(tmpdir.join('test.json')))

    gbm = ds.ml.catboost_model(target, 20, features=features, params=params_multiclass,)
    gbm.state_set(gbm.state_get())
    pl = vaex.ml.Pipeline([gbm])
    pl.save(str(tmpdir.join('test.json')))
    pl.load(str(tmpdir.join('test.json')))


def test_catboost_validation_set():
    # read data
    ds = vaex.example()
    # Train and test split
    train, test = ds.ml.train_test_split(verbose=False)
    # Define the training featuress
    features = ['vx', 'vy', 'vz', 'Lz', 'L']
    # instantiate the booster model
    booster = vaex.ml.catboost.CatBoostModel(features=features, num_boost_round=10, params=params_reg)
    # fit the booster - including saving the history of the validation sets
    booster.fit(train, 'E', evals=[train, test], early_stopping_rounds=2)
    assert len(booster.booster.evals_result_['learn']['MAE']) == 5
    assert len(booster.booster.evals_result_['learn']['R2']) == 5
    assert len(booster.booster.evals_result_['validation_0']['MAE']) == 5
    assert len(booster.booster.evals_result_['validation_0']['R2']) == 5
    assert booster.booster.best_iteration_ == 2


def test_catboost_pipeline():
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
    # now the catboost model thingy
    features = ['r', 'PCA_0', 'PCA_1', 'PCA_2']
    # define the boosting model
    booster = train.ml.catboost_model(target='E', num_boost_round=10, features=features, params=params_reg)
    # Create a pipeline
    pp = vaex.ml.Pipeline([st, booster])
    # Use the pipeline
    pred = pp.predict(test)                  # This works
    trans = pp.transform(test)               # This will crash (softly)
    # trans.evaluate('catboost_prediction')   # This is where the problem happens
    np.testing.assert_equal(pred,
                            trans.evaluate('catboost_prediction'),
                            verbose=True,
                            err_msg='The predictions from the predict and transform method do not match')
