import logging

import pandas as pd

import pytest

from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, mean_absolute_error

import tensorflow as tf

import vaex


logger = tf.get_logger()
logger.setLevel(logging.ERROR)


def make_binary_classification_data():
    X, y = make_classification(n_samples=1000,
                               n_features=5,
                               n_informative=4,
                               n_repeated=0,
                               n_redundant=1,
                               n_classes=2,
                               class_sep=1,
                               random_state=42)
    df = pd.DataFrame(data=X, columns=['feat' + str(i) for i in range(X.shape[1])])
    df['target'] = y
    df = vaex.from_pandas(df, copy_index=False)
    df_train, df_val, df_test = df.split_random(frac=[0.8, 0.1, 0.1], random_state=42)
    features = df.get_column_names(regex='^feat')
    target = 'target'
    return df_train, df_val, df_test, features, target


def make_multiclass_classification_data():
    X, y = make_classification(n_samples=1000,
                               n_features=10,
                               n_informative=8,
                               n_repeated=0,
                               n_redundant=2,
                               n_classes=4,
                               class_sep=1,
                               random_state=42)
    df = pd.DataFrame(data=X, columns=['feat' + str(i) for i in range(X.shape[1])])
    df['target'] = y
    df = vaex.from_pandas(df, copy_index=False)
    df = df.ml.one_hot_encoder(features=['target']).transform(df)
    df_train, df_val, df_test = df.split_random(frac=[0.8, 0.1, 0.1], random_state=42)
    features = df.get_column_names(regex='^feat')
    target = 'target'
    targets = df.get_column_names(regex='target_')
    return df_train, df_val, df_test, features, target, targets


def make_regression_data():
    X, y = make_regression(n_samples=1000,
                           n_features=5,
                           n_targets=1,
                           n_informative=4,
                           noise=0.01,
                           tail_strength=0.1,
                           random_state=42)
    df = pd.DataFrame(data=X, columns=['feat' + str(i) for i in range(X.shape[1])])
    df['target'] = y
    df = vaex.from_pandas(df, copy_index=False)
    df_train, df_val, df_test = df.split_random(frac=[0.8, 0.1, 0.1], random_state=42)
    features = df.get_column_names(regex='^feat')
    target = 'target'
    return df_train, df_val, df_test, features, target


@pytest.mark.parametrize("as_dict", [True, False])
def test_to_dataset_tensorflow(as_dict):
    chunk_size = 100
    df_train, df_val, df_test, features, target = make_binary_classification_data()

    ds = df_train.ml.tensorflow.to_dataset(target=target, chunk_size=chunk_size, as_dict=as_dict)
    list_ds = list(ds)
    assert len(list_ds) == 8  # The number of "batches" in the iterable, as defined by the chunk_size arg

    idx_min = 0
    idx_max = chunk_size
    for batch in list_ds:
        assert len(batch) == 2
        assert batch[1].numpy().tolist() == df_train[target][idx_min:idx_max].tolist()
        if as_dict:
            for feat in batch[0]:
                assert batch[0][feat].numpy().tolist() == df_train[feat][idx_min:idx_max].tolist()
        else:
            assert batch[0].numpy().tolist() == df_train[features][idx_min:idx_max].values.tolist()

        idx_min += chunk_size
        idx_max += chunk_size



@pytest.mark.parametrize("repeat", [1, 3, 10])
@pytest.mark.parametrize("shuffle", [False, True])
def test_make_input_function_options(repeat, shuffle):
    df_train, df_val, df_test, features, target = make_binary_classification_data()

    num_batches = repeat * 10

    input_function = df_test.ml.tensorflow.make_input_function(features=features,
                                                               target=target,
                                                               chunk_size=10,
                                                               repeat=repeat,
                                                               shuffle=shuffle)
    ds = input_function()
    list_ds = list(ds)
    assert len(list_ds) == num_batches  # The number of "batches" in the iterable, as defined by the chunk_size arg


@pytest.mark.parametrize("parallel", [False, True])
def test_to_keras_generator(parallel):
    df_train, df_val, df_test, features, target = make_binary_classification_data()

    train_gen_1 = df_train.ml.tensorflow.to_keras_generator(features=features, target=target, chunk_size=100, parallel=parallel)
    train_gen_2 = df_train.ml.tensorflow.to_keras_generator(features=features, target=target, chunk_size=100, parallel=parallel)

    for i in range(24):
        batch_1 = next(train_gen_1)
        assert len(batch_1) == 2
        assert batch_1[0].shape == (100, 5)
        assert batch_1[1].shape == (100, 1)
        if i > 7:
            batch_2 = next(train_gen_2)
            assert batch_1[0].tolist() == batch_2[0].tolist()
            assert batch_1[1].tolist() == batch_2[1].tolist()


def test_tf_estimator_binary_classification_booster_trees(tmpdir):
    df_train, df_val, df_test, features, target = make_binary_classification_data()

    feature_colums = []
    for feat in features:
        feature_colums.append(tf.feature_column.numeric_column(feat))

    train_fn = df_train.ml.tensorflow.make_input_function(features=features, target=target, repeat=10, shuffle=True)
    val_fn = df_val.ml.tensorflow.make_input_function(features=features, target=target)
    test_fn = df_test.ml.tensorflow.make_input_function(features=features, repeat=10)

    est = tf.estimator.BoostedTreesClassifier(feature_columns=feature_colums, n_batches_per_layer=2, n_classes=2)
    est.train(train_fn)
    val_result = est.evaluate(val_fn)
    assert val_result['accuracy'] > 0.80
    assert val_result['recall'] > 0.80
    assert val_result['precision'] > 0.80
    assert list(val_result.keys()) == ['accuracy', 'accuracy_baseline', 'auc', 'auc_precision_recall',
                                       'average_loss', 'label/mean', 'loss', 'precision', 'prediction/mean',
                                       'recall', 'global_step']
    pred_result = list(est.predict(test_fn, yield_single_examples=False))[0]
    assert list(pred_result.keys()) == ['logits', 'logistic', 'probabilities', 'class_ids',
                                        'classes', 'all_class_ids', 'all_classes']
    assert pred_result['class_ids'].shape == (100, 1)
    assert pred_result['probabilities'].shape == (100, 2)
    acc = accuracy_score(df_test.target.values, pred_result['class_ids'].flatten())
    assert acc > 0.8

    model = vaex.ml.tensorflow.Model(model=est, features=features, target=target)
    model.init(df_test)
    df = model.transform(df_test)
    # for debugging purposes, single threaded evaluate:
    # values = [df[k].values for k in features]
    # probabilities = model('probabilities', *values)
    # assert probabilities.tolist() == pred_result['probabilities'].tolist()
    assert df['probabilities'].tolist() == pred_result['probabilities'].tolist()

    # seems impossible to serialize a tensorflow model
    # state_path = str(tmpdir.join('state.json'))
    # df.state_write(state_path)
    # df_test.state_load(state_path)
    # so we only test state transfer (in memory)
    state = df.state_get()
    df_test.state_set(state)
    assert df_test['probabilities'].tolist() == pred_result['probabilities'].tolist()



def test_keras_binary_classification():
    df_train, df_val, df_test, features, target = make_binary_classification_data()
    train_gen = df_train.ml.tensorflow.to_keras_generator(features=features, target=target, chunk_size=100, parallel=True)
    val_gen = df_val.ml.tensorflow.to_keras_generator(features=features, target=target, chunk_size=100, parallel=False)
    test_gen = df_test.ml.tensorflow.to_keras_generator(features=features, target=target, chunk_size=100, parallel=True)

    def _make_keras_model():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                      loss=tf.keras.losses.binary_crossentropy,
                      metrics=['accuracy'])

        return model

    model = _make_keras_model()
    model.fit(train_gen, epochs=300, steps_per_epoch=8, verbose=0, validation_data=val_gen, validation_steps=1)
    pred = model.predict(test_gen, steps=1)
    assert pred.shape == (100, 1)  # Returns probabilities
    acc = accuracy_score(df_test.target.values, pred.round().flatten())
    assert acc > 0.8


@pytest.mark.parametrize("as_dict", [True, False])
def test_keras_binary_classification_tf_dataset_input(as_dict):
    df_train, df_val, df_test, features, target = make_binary_classification_data()

    train_ds = df_train.ml.tensorflow.to_dataset(features=features, target=target, chunk_size=100, parallel=True, as_dict=as_dict)
    train_ds = train_ds.repeat(30)
    val_ds = df_val.ml.tensorflow.to_dataset(features=features, target=target, chunk_size=100, parallel=False, as_dict=as_dict)
    test_ds = df_test.ml.tensorflow.to_dataset(features=features, target=target, chunk_size=100, parallel=True, as_dict=as_dict)

    feature_colums = []
    for feat in features:
        feature_colums.append(tf.feature_column.numeric_column(feat))

    def _make_keras_model(as_dict):
        if as_dict:
            model = tf.keras.Sequential([
                tf.keras.layers.DenseFeatures(feature_columns=feature_colums),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        else:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    model = _make_keras_model(as_dict=as_dict)
    model.fit(train_ds, epochs=30, verbose=0, validation_data=val_ds, validation_steps=1)
    pred = model.predict(test_ds, steps=1)
    assert pred.shape == (100, 1)  # Returns probabilities
    acc = accuracy_score(df_test.target.values, pred.round().flatten())
    assert acc > 0.8


def test_tf_estimator_multiclass_classification_deep_neural_network():
    df_train, df_val, df_test, features, target, targets = make_multiclass_classification_data()

    feature_colums = []
    for feat in features:
        feature_colums.append(tf.feature_column.numeric_column(feat))

    train_fn = df_train.ml.tensorflow.make_input_function(features=features, target=target, repeat=500, shuffle=True)
    val_fn = df_val.ml.tensorflow.make_input_function(features=features, target=target)
    test_fn = df_test.ml.tensorflow.make_input_function(features=features)

    est = tf.estimator.DNNClassifier(feature_columns=feature_colums, hidden_units=[32, 64, 32], n_classes=4, optimizer='RMSProp')
    est.train(train_fn)
    val_result = est.evaluate(val_fn)
    assert val_result['accuracy'] > 0.65
    assert list(val_result.keys()) == ['accuracy', 'average_loss', 'loss', 'global_step']
    pred_result = list(est.predict(test_fn, yield_single_examples=False))[0]
    assert list(pred_result.keys()) == ['logits', 'probabilities', 'class_ids', 'classes', 'all_class_ids', 'all_classes']
    assert pred_result['class_ids'].shape == (100, 1)
    assert pred_result['probabilities'].shape == (100, 4)
    acc = accuracy_score(df_test.target.values, pred_result['class_ids'].flatten())
    assert acc > 0.65


def test_keras_multiclass_classification():
    df_train, df_val, df_test, features, target, targets = make_multiclass_classification_data()

    train_gen = df_train.ml.tensorflow.to_keras_generator(features=features, target=targets, chunk_size=100, parallel=True)
    val_gen = df_val.ml.tensorflow.to_keras_generator(features=features, target=targets, chunk_size=100, parallel=False)
    test_gen = df_test.ml.tensorflow.to_keras_generator(features=features, target=targets, chunk_size=100, parallel=True)

    def _make_keras_model():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(4, activation='softmax')
        ])

        model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=['accuracy'])

        return model

    model = _make_keras_model()
    model.fit(train_gen, epochs=300, steps_per_epoch=8, verbose=0, validation_data=val_gen, validation_steps=1)
    pred = model.predict(test_gen, steps=1)
    assert pred.shape == (100, 4)  # Returns probabilities per class
    acc = accuracy_score(df_test.target.values, pred.argmax(axis=1))
    assert acc > 0.65


@pytest.mark.parametrize("as_dict", [True, False])
def test_keras_multiclass_classification_tf_dataset_input(as_dict):
    df_train, df_val, df_test, features, target, targets = make_multiclass_classification_data()

    train_ds = df_train.ml.tensorflow.to_dataset(features=features, target=targets, chunk_size=100, parallel=True, as_dict=as_dict)
    train_ds = train_ds.repeat(30)
    val_ds = df_val.ml.tensorflow.to_dataset(features=features, target=targets, chunk_size=100, parallel=False, as_dict=as_dict)
    test_ds = df_test.ml.tensorflow.to_dataset(features=features, target=targets, chunk_size=100, parallel=True, as_dict=as_dict)

    feature_colums = []
    for feat in features:
        feature_colums.append(tf.feature_column.numeric_column(feat))

    def _make_keras_model(as_dict):
        if as_dict:
            model = tf.keras.Sequential([
                tf.keras.layers.DenseFeatures(feature_columns=feature_colums),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(4, activation='softmax')
            ])
        else:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(4, activation='softmax'),
            ])
        model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    model = _make_keras_model(as_dict=as_dict)
    model.fit(train_ds, epochs=30, verbose=0, validation_data=val_ds, validation_steps=1)
    pred = model.predict(test_ds, steps=1)
    assert pred.shape == (100, 4)  # Returns probabilities
    acc = accuracy_score(df_test.target.values, pred.argmax(axis=1))
    assert acc > 0.75


def test_tf_estimator_regression_booster_trees():
    df_train, df_val, df_test, features, target = make_regression_data()

    feature_colums = []
    for feat in features:
        feature_colums.append(tf.feature_column.numeric_column(feat))

    train_fn = df_train.ml.tensorflow.make_input_function(features=features, target=target, repeat=30, shuffle=True)
    val_fn = df_val.ml.tensorflow.make_input_function(features=features, target=target)
    test_fn = df_test.ml.tensorflow.make_input_function(features=features)

    est = tf.estimator.BoostedTreesRegressor(feature_columns=feature_colums, n_batches_per_layer=1, learning_rate=0.5)
    est.train(train_fn)
    val_result = est.evaluate(val_fn)
    assert list(val_result.keys()) == ['average_loss', 'label/mean', 'loss', 'prediction/mean', 'global_step']
    pred_result = list(est.predict(test_fn, yield_single_examples=False))[0]
    assert list(pred_result.keys()) == ['predictions']
    assert pred_result['predictions'].shape == (100, 1)
    mae = mean_absolute_error(df_test.target.values, pred_result['predictions'].flatten())
    assert mae < 20


def test_keras_regression():
    df_train, df_val, df_test, features, target = make_regression_data()

    train_gen = df_train.ml.tensorflow.to_keras_generator(features=features, target=target, chunk_size=100, parallel=True)
    val_gen = df_val.ml.tensorflow.to_keras_generator(features=features, target=target, chunk_size=100, parallel=False)
    test_gen = df_test.ml.tensorflow.to_keras_generator(features=features, target=target, chunk_size=100, parallel=True)

    def _make_keras_model():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])

        model.compile(optimizer=tf.keras.optimizers.SGD(),
                      loss=tf.keras.losses.mean_absolute_error,
                      metrics=['mean_absolute_error'])

        return model

    model = _make_keras_model()
    model.fit(train_gen, epochs=100, steps_per_epoch=8, verbose=0, validation_data=val_gen, validation_steps=1)
    pred = model.predict(test_gen, steps=1)
    assert pred.shape == (100, 1)  # Returns probabilities
    mae = mean_absolute_error(df_test.target.values, pred.flatten())
    assert mae < 10


@pytest.mark.parametrize("as_dict", [True, False])
def test_keras_regression_tf_dataset_input(as_dict):
    df_train, df_val, df_test, features, target = make_regression_data()

    train_ds = df_train.ml.tensorflow.to_dataset(features=features, target=target, chunk_size=100, parallel=True, as_dict=as_dict)
    train_ds = train_ds.repeat(10)
    val_ds = df_val.ml.tensorflow.to_dataset(features=features, target=target, chunk_size=100, parallel=False, as_dict=as_dict)
    test_ds = df_test.ml.tensorflow.to_dataset(features=features, target=target, chunk_size=100, parallel=True, as_dict=as_dict)

    feature_colums = []
    for feat in features:
        feature_colums.append(tf.feature_column.numeric_column(feat))

    def _make_keras_model(as_dict):
        if as_dict:
            model = tf.keras.Sequential([
                tf.keras.layers.DenseFeatures(feature_columns=feature_colums),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1, activation='linear')
            ])
        else:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1, activation='linear')
            ])
        model.compile(optimizer=tf.keras.optimizers.SGD(),
                      loss=tf.keras.losses.mean_absolute_error,
                      metrics=['mean_absolute_error'])
        return model

    model = _make_keras_model(as_dict=as_dict)
    model.fit(train_ds, epochs=100, verbose=0, validation_data=val_ds, validation_steps=1)
    pred = model.predict(test_ds, steps=1)
    assert pred.shape == (100, 1)  # Returns probabilities
    mae = mean_absolute_error(df_test.target.values, pred.flatten())
    assert mae < 10
