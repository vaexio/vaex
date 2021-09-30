import pytest
pytest.importorskip("tensorflow")

import tensorflow.keras as K

import vaex
import vaex.ml
import vaex.ml.tensorflow


# Custom metric used in some of the tests
def r2_keras(y_true, y_pred):
    SS_res =  K.backend.sum(K.backend.square(y_true - y_pred))
    SS_tot = K.backend.sum(K.backend.square(y_true - K.backend.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.backend.epsilon()))


def test_keras_model_classification(tmpdir, df_iris):
    df = df_iris
    copy = df.copy()
    features = ['sepal_width', 'petal_length', 'sepal_length', 'petal_width']
    target = 'class_'

    # Preprocessing
    df = df.ml.minmax_scaler(features)
    df = df.ml.one_hot_encoder(features=[target])
    features = df.get_column_names(regex='^minmax')
    targets = df.get_column_names(regex='^class__')

    # Create the neural netwrok model
    nn_model = K.Sequential()
    nn_model.add(K.layers.Dense(4, activation='tanh'))
    nn_model.add(K.layers.Dense(3, activation='softmax'))
    nn_model.compile(optimizer=K.optimizers.RMSprop(learning_rate=0.01),
                    loss=K.losses.categorical_crossentropy,
                    metrics='accuracy')

    X = df[features].values
    y = df[targets].values
    nn_model.fit(x=X, y=y, validation_split=0.05, epochs=11, verbose=0)

    keras_model = vaex.ml.tensorflow.KerasModel(features=features, prediction_name='pred', model=nn_model)
    df_trans = keras_model.transform(df)
    assert df_trans.pred.shape == (150, 3)

    state_path = str(tmpdir.join('state.json'))
    df_trans.state_write(state_path)
    copy.state_load(state_path)

    assert copy.pred.shape == (150, 3)


def test_keras_model_regression(df_example):
    df = df_example
    df = df[:1_000]  # To make the tests run faster
    df_train, df_valid, df_test = df.split_random([0.8, 0.1, 0.1], random_state=42)
    features = ['vx', 'vy', 'vz']
    target = 'FeH'

    # Scaling the features
    df_train = df_train.ml.minmax_scaler(features=features)
    features = df_train.get_column_names(regex='^minmax_')

    # Apply preprocessing to the validation
    state_prep = df_train.state_get()
    df_valid.state_set(state_prep)

    train_gen = df_train.ml.tensorflow.to_keras_generator(features=features, target=target, batch_size=128)
    valid_gen = df_valid.ml.tensorflow.to_keras_generator(features=features, target=target, batch_size=128)

    # Create the model
    nn_model = K.Sequential()
    nn_model.add(K.layers.Dense(3, activation='tanh'))
    nn_model.add(K.layers.Dense(1, activation='linear'))
    nn_model.compile(optimizer='sgd', loss='mse', metrics=[r2_keras])
    nn_model.fit(x=train_gen, validation_data=valid_gen, epochs=5, steps_per_epoch=7, validation_steps=1, verbose=0)

    keras_model = vaex.ml.tensorflow.KerasModel(features=features, prediction_name='pred', model=nn_model, custom_objects={'r2_keras': r2_keras})
    df_train = keras_model.transform(df_train)

    # The final state transfer
    state_final = df_train.state_get()
    df_valid.state_set(state_final)
    df_test.state_set(state_final)

    assert 'pred' in df_train
    assert 'pred' in df_valid
    assert 'pred' in df_test
    assert df_valid.shape == (100, 15)
    assert df_test.shape == (100, 15)


@pytest.mark.parametrize("parallel", [False, True])
def test_to_keras_generator(df_example, parallel):
    df = df_example
    df = df[:1000]
    features = ['x', 'y', 'z', 'vx', 'vy', 'vx']
    target = 'FeH'

    train_gen_1 = df.ml.tensorflow.to_keras_generator(features=features, target=target, parallel=parallel, batch_size=100, shuffle=False)
    train_gen_2 = df.ml.tensorflow.to_keras_generator(features=features, target=target, parallel=parallel, batch_size=100, shuffle=False)

    for i in range(30):
        batch_1 = next(train_gen_1)
        assert len(batch_1) == 2
        assert batch_1[0].shape == (100, 6)
        assert batch_1[1].shape == (100, 1)
        if i > 9:
            batch_2 = next(train_gen_2)
            assert batch_1[0].tolist() == batch_2[0].tolist()
            assert batch_1[1].tolist() == batch_2[1].tolist()
