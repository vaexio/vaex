import catboost as cb
import numpy as np
import vaex.ml.catboost
import vaex.ml.datasets
from sklearn.metrics import roc_auc_score, accuracy_score

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
                                             features=features,
                                             target='class_',
                                             prediction_type='Probability')
    # Predict in memory
    booster.fit(ds_train)
    class_predict = booster.predict(ds_test)
    assert np.all(ds_test.col.class_.values == np.argmax(class_predict, axis=1))

    # Transform
    ds_train = booster.transform(ds_train)   # this will add the catboost_prediction column
    state = ds_train.state_get()
    ds_test.state_set(state)
    assert np.all(ds_test.col.class_.values == np.argmax(ds_test.catboost_prediction.values, axis=1))


def test_catboost_batch_training():
    """
    We train three models. One on 10 samples. the second on 100 samples with batches of 10,
    and the third too on 100 samples with batches of 10, but we weight the models as if only the first batch matters.
    A model trained on more data, should do better than the model who only trained on 10 samples,
    and the weighted model will do exactly as good as the one who trained on 10 samples as it ignore the rest by weighting.
    """
    ds = vaex.ml.datasets.load_iris()
    ds_train, ds_test = ds.ml.train_test_split(test_size=0.2, verbose=False)
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    target = 'class_'
    prediction_type = 'Class'
    vanilla = vaex.ml.catboost.CatBoostModel(num_boost_round=1,
                                             params=params_multiclass,
                                             features=features,
                                             target=target,
                                             prediction_type=prediction_type)
    batch_booster = vaex.ml.catboost.CatBoostModel(num_boost_round=1,
                                                   params=params_multiclass,
                                                   features=features,
                                                   target=target,
                                                   prediction_type=prediction_type,
                                                   batch_size=10)
    weights = [1.0] + [0.0] * 9
    weights_booster = vaex.ml.catboost.CatBoostModel(num_boost_round=1,
                                                     params=params_multiclass,
                                                     features=features,
                                                     target=target,
                                                     prediction_type=prediction_type,
                                                     batch_size=10,
                                                     batch_weights=weights)

    vanilla.fit(ds_train.head(10), evals=[ds_test])
    batch_booster.fit(ds_train.head(100), evals=[ds_test])
    weights_booster.fit(ds_train.head(100), evals=[ds_test])

    ground_truth = ds_test[target].values
    vanilla_accuracy = accuracy_score(ground_truth, vanilla.predict(ds_test))
    batch_accuracy = accuracy_score(ground_truth, batch_booster.predict(ds_test))
    weighted_accuracy = accuracy_score(ground_truth, weights_booster.predict(ds_test))
    assert vanilla_accuracy == weighted_accuracy
    assert vanilla_accuracy < batch_accuracy

    assert list(weights_booster.booster.get_feature_importance()) == list(vanilla.booster.get_feature_importance())
    assert list(weights_booster.booster.get_feature_importance()) != list(batch_booster.booster.get_feature_importance())


def test_catboost_numerical_validation():
    ds = vaex.ml.datasets.load_iris()
    features = ['sepal_width', 'petal_length', 'sepal_length', 'petal_width']

    # Vanilla catboost
    dtrain = cb.Pool(ds[features].values, label=ds.class_.values)
    cb_bst = cb.train(params=params_multiclass, dtrain=dtrain, num_boost_round=3)
    cb_pred = cb_bst.predict(dtrain, prediction_type='Probability')

    # catboost through vaex
    booster = vaex.ml.catboost.CatBoostModel(features=features, target='class_', params=params_multiclass, num_boost_round=3)
    booster.fit(ds)
    vaex_pred = booster.predict(ds)

    # Comparing the the predictions of catboost vs vaex.ml
    np.testing.assert_equal(vaex_pred, cb_pred, verbose=True,
                            err_msg='The predictions of vaex.ml.catboost do not match those of pure catboost')


def test_lightgbm_serialize(tmpdir):
    ds = vaex.ml.datasets.load_iris()
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    target = 'class_'

    gbm = ds.ml.catboost_model(target=target, features=features, num_boost_round=20, params=params_multiclass, transform=False)
    pl = vaex.ml.Pipeline([gbm])
    pl.save(str(tmpdir.join('test.json')))
    pl.load(str(tmpdir.join('test.json')))

    gbm = ds.ml.catboost_model(target=target, features=features, num_boost_round=20, params=params_multiclass, transform=False)
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
    booster = vaex.ml.catboost.CatBoostModel(features=features, target='E', num_boost_round=10, params=params_reg)
    # fit the booster - including saving the history of the validation sets
    booster.fit(train, evals=[train, test])
    assert hasattr(booster, 'booster')
    assert len(booster.booster.evals_result_['learn']['MAE']) == 10
    assert len(booster.booster.evals_result_['learn']['R2']) == 10
    assert len(booster.booster.evals_result_['validation_0']['MAE']) == 10
    assert len(booster.booster.evals_result_['validation_0']['R2']) == 10
    assert hasattr(booster.booster, 'best_iteration_')
    assert booster.booster.best_iteration_ is not None


def test_catboost_pipeline():
    # read data
    ds = vaex.example()
    # train test splot
    train, test = ds.ml.train_test_split(verbose=False)
    # add virtual columns
    train['r'] = np.sqrt(train.x**2 + train.y**2 + train.z**2)
    # Do a pca
    features = ['vx', 'vy', 'vz', 'Lz', 'L']
    pca = train.ml.pca(n_components=3, features=features, transform=False)
    train = pca.transform(train)
    # Do state transfer
    st = train.ml.state_transfer()
    # now the catboost model thingy
    features = ['r', 'PCA_0', 'PCA_1', 'PCA_2']
    # define the boosting model
    booster = train.ml.catboost_model(target='E', num_boost_round=10, features=features, params=params_reg, transform=False)
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
