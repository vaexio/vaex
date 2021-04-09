import pytest
import vaex
pytest.importorskip('river')
from vaex.ml.incubator.river import RiverModel

import numpy as np

from river.linear_model import LinearRegression, LogisticRegression
import river.optim


models_regression = [LinearRegression()]
models_classification = [LogisticRegression()]


def test_river_regression(df_example):
    df = df_example
    df_train, df_test = df.ml.train_test_split(test_size=0.1, verbose=False)

    features = df_train.column_names[:6]
    target = 'FeH'

    river_model = RiverModel(model=LinearRegression(),
                             features=features,
                             target=target,
                             batch_size=50_000,
                             num_epochs=3,
                             shuffle=True,
                             prediction_name='pred')
    river_model.fit(df=df_train)
    df_train = river_model.transform(df_train)

    # State transfer
    state = df_train.state_get()
    df_test.state_set(state)

    assert df_train.column_count() == df_test.column_count()
    assert df_test.pred.values.shape == (33000,)

    pred_in_memory = river_model.predict(df_test)
    np.testing.assert_array_almost_equal(pred_in_memory, df_test.pred.values, decimal=1)

@pytest.mark.parametrize("prediction_type", ['predict', 'predict_proba'])
def test_river_classification(df_iris_1e5, prediction_type):
    df = df_iris_1e5
    # Convert to a binary problem
    df['target'] = (df['class_'] == 2).astype('int')
    df_train, df_test = df.ml.train_test_split(test_size=0.1, verbose=False)

    features = df_train.column_names[:4]
    target = 'target'

    river_model = RiverModel(model=LogisticRegression(),
                             features=features,
                             target=target,
                             batch_size=50_000,
                             num_epochs=3,
                             shuffle=True,
                             prediction_name='pred',
                             prediction_type=prediction_type)
    river_model.fit(df=df_train)
    df_train = river_model.transform(df_train)

    # State transfer
    state = df_train.state_get()
    df_test.state_set(state)

    assert df_train.column_count() == df_test.column_count()
    if prediction_type == 'predict':
        assert df_test.pred.values.shape == (10050,)
    else:
        assert df_test.pred.values.shape == (10050, 2)

    pred_in_memory = river_model.predict(df_test)
    np.testing.assert_array_almost_equal(pred_in_memory, df_test.pred.values, decimal=1)


def test_river_sertialize(tmpdir, df_example):
    df = df_example
    df_train, df_test = df.ml.train_test_split(test_size=0.1, verbose=False)

    features = df_train.column_names[:6]
    target = 'FeH'

    river_model = RiverModel(model=LinearRegression(),
                             features=features,
                             target=target,
                             batch_size=50_000,
                             num_epochs=3,
                             shuffle=True,
                             prediction_name='pred')
    river_model.fit(df=df_train)
    df_train = river_model.transform(df_train)

    # State transfer - serialization
    df_train.state_write(str(tmpdir.join('test.json')))
    df_test.state_load(str(tmpdir.join('test.json')))

    assert df_train.column_count() == df_test.column_count()
    assert df_test.pred.values.shape == (33000,)

    pred_in_memory = river_model.predict(df_test)
    np.testing.assert_array_almost_equal(pred_in_memory, df_test.pred.values, decimal=1)

@pytest.mark.parametrize("batch_size", [6789, 10_000])
@pytest.mark.parametrize("num_epochs", [1, 5])
def test_river_learn_many_calls(batch_size, num_epochs, df_example):
    df = df_example
    df_train, df_test = df.ml.train_test_split(test_size=0.1, verbose=False)

    features = df_train.column_names[:6]
    target = 'FeH'

    N_total = len(df_train)
    num_batches = (N_total + batch_size - 1) // batch_size

    # Create a mock model for counting the number of samples seen and partial_fit calls
    class MockModel():
        def __init__(self):
            self.n_samples_ = 0
            self.n_learn_many_calls_ = 0

        def learn_many(self, X, y):
            self.n_samples_ += X.shape[0]
            self.n_learn_many_calls_ += 1

    river_model = RiverModel(model=MockModel(),
                             features=features,
                             target=target,
                             num_epochs=num_epochs,
                             batch_size=batch_size,
                             shuffle=False,
                             prediction_name='pred')

    river_model.fit(df=df_train)
    assert river_model.model.n_samples_ == N_total * num_epochs
    assert river_model.model.n_learn_many_calls_ == num_batches * num_epochs
