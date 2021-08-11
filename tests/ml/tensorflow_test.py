import sys
import pytest
pytest.importorskip("tensorflow")

import tensorflow.keras as K

import vaex
import vaex.ml
import vaex.ml.tensorflow


def test_kears_model(tmpdir, df_iris):
    df = df_iris
    copy = df.copy()
    features = ['sepal_width', 'petal_length', 'sepal_length', 'petal_width']
    target = 'class_'

    def create_neural_network():
        inp = K.layers.Input(shape=(4,), name='input')
        x = K.layers.Dense(4, activation='relu', name='layer')(inp)
        x = K.layers.Dense(3, activation='softmax', name='output')(x)

        nn = K.models.Model(inputs=inp, outputs=x)

        nn.compile(optimizer=K.optimizers.RMSprop(learning_rate=0.01),
                loss=K.losses.categorical_crossentropy,
                metrics='accuracy')

        return nn

    nn = create_neural_network()

    X = df[features].values
    y = df[target].values
    y_enc = K.utils.to_categorical(y)
    nn.fit(x=X, y=y_enc, validation_split=0.05, epochs=11)

    keras_model = vaex.ml.tensorflow.KerasModel(features=features,
                                                prediction_name='pred',
                                                model=nn)
    df_trans = keras_model.transform(df)
    assert df_trans.pred.shape == (150, 3)

    state_path = str(tmpdir.join('state.json'))
    df_trans.state_write(state_path)
    copy.state_load(state_path)

    assert copy.pred.shape == (150, 3)



