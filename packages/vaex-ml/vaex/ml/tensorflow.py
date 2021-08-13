import base64
import tempfile
import shutil

import numpy as np

from tensorflow import keras as K

import traitlets

import vaex
import vaex.serialize
import vaex.ml.state
import vaex.ml.generate


class DataFrameAccessorTensorflow():
    def __init__(self, ml):
        self.ml = ml
        self.df = self.ml.df

    def to_keras_generator(self, features, target=None, batch_size=1024, parallel=True, shuffle=True, infinite=True, verbose=True):
        """Return a batch generator suitable as a Keras datasource.  Note that by default the generator is infinite,
        i.e. it loops continuously ovder the data. Thus you need to specify the "steps_per_epoch" arg when fitting a Keras model,
        the "validation_steps" when using it for validation, and "steps" when calling the "predict" method of a keras model.

        :param features: A list of features.
        :param target: The dependent or target column or a list of columns, if any.
        :param batch_size: Number of samples per chunk of data. This can be thought of as the batch size.
        :param parallel: If True, vaex will evaluate the data chunks in parallel.
        :param shuffle: If True, shuffle the data before every pass.
        :param infinite: If True, the generator is infinite, i.e. it loops continously over the data. If False, the generator does only one pass over the data.
        :parallel verbose: If True, show an info on the recommended "steps_per_epoch" based on the total number of samples and "batch_size".

        Example:

        >>> import vaex
        >>> import vaex.ml
        >>> import tensorflow.keras as K

        >>> df = vaex.example()
        >>> features = ['x', 'y', 'z', 'vx', 'vz', 'vz']
        >>> target = 'FeH'

        >>> # Scaling the features
        >>> df = df.ml.minmax_scaler(features=features)
        >>> features = df.get_column_names(regex='^minmax_')

        >>> # Create a training generator
        >>> train_generator = df.ml.tensorflow.to_keras_generator(features=features, target=target, batch_size=512)
        Recommended "steps_per_epoch" arg: 645.0

        >>> # Build a neural network model
        >>> nn_model = K.Sequential()
        >>> nn_model.add(K.layers.Dense(4, activation='tanh'))
        >>> nn_model.add(K.layers.Dense(4, activation='tanh'))
        >>> nn_model.add(K.layers.Dense(1, activation='linear'))
        >>> nn_model.compile(optimizer='sgd', loss='mse')

        >>> nn_model.fit(x=train_generator, epochs=3, steps_per_epoch=645)
        Epoch 1/3
        645/645 [==============================] - 3s 5ms/step - loss: 0.2068
        Epoch 2/3
        645/645 [==============================] - 3s 5ms/step - loss: 0.1706
        Epoch 3/3
        645/645 [==============================] - 3s 5ms/step - loss: 0.1705
        """

        if verbose:
            steps_per_epoch = np.ceil(len(self.df) / batch_size)
            print(f'Recommended "steps_per_epoch" arg: {steps_per_epoch}')

        def _generator(features, target, chunk_size, parallel, shuffle, infinite):
            if shuffle:
                df = self.df.shuffle().copy()
            else:
                df = self.df.copy()

            if target is not None:
                target = vaex.utils._ensure_list(target)
                target = vaex.utils._ensure_strings_from_expressions(target)
                n_target_cols = len(target)
                column_names = features + target
            else:
                column_names = features

            while True:
                if shuffle:
                    df = self.df.shuffle().copy()
                else:
                    df = self.df.copy()

                if target is not None:
                    for i1, i2, chunk, in df.evaluate_iterator(column_names, chunk_size=chunk_size, parallel=parallel, array_type='numpy'):
                        chunk_shape = len(chunk[0].shape) + 1
                        transpose_order = np.arange(1, chunk_shape).tolist() + [0]
                        X = np.array(chunk[:-n_target_cols]).transpose(transpose_order)
                        y = np.array(chunk[-n_target_cols:], copy=False).T
                        yield (X, y)

                else:
                    for i1, i2, chunk, in df.evaluate_iterator(column_names, chunk_size=chunk_size, parallel=parallel, array_type='numpy'):
                        chunk_shape = len(chunk[0].shape) + 1
                        transpose_order = np.arange(1, chunk_shape).tolist() + [0]
                        X = np.array(chunk).transpose(transpose_order)
                        yield (X, )

                if not infinite:
                    break

        return _generator(features=features, target=target, chunk_size=batch_size, parallel=parallel, shuffle=shuffle, infinite=infinite)


@vaex.serialize.register
@vaex.ml.generate.register
class KerasModel(vaex.ml.state.HasState):
    '''A simple class that makes a Keras model serializable in the Vaex state, as well as enables lazy transformations of the preditions.

    For more infromation on how to use the Keras library, please visit https://keras.io/.
    '''
    features = traitlets.List(traitlets.Unicode(), help='List of features to use when applying the KerasModel.')
    prediction_name = traitlets.Unicode(default_value='keras_prediction', help='The name of the virtual column housing the predictions.')
    model = traitlets.Any(help='A fitted Keras Model')

    def __call__(self, *args):
        data2d = np.stack([np.asarray(arg, np.float64) for arg in args], axis=-1)
        return self.model.predict(data2d)

    def fit(self, df):
        '''Not implemented: A Placeholder method, put here for potential future developement.
        '''
        raise NotImplementedError('The `fit` method is not implemented. To satisfy the large number of use-cases and for maximum flexiblity, please fit the model using the `tensorflow.keras` API.')

    def transform(self, df):
        '''Transform a DataFrame such that it contains the predictions of the KerasModel in form of a virtual column.

        :param df: A vaex DataFrame.

        :return copy: A shallow copy of the DataFrame that includes the KerasModel prediction as a virtual column.
        :rtype: DataFrame
        '''
        copy = df.copy()
        lazy_function = copy.add_function('keras_prediction_function', self, unique=True)
        expression = lazy_function(*self.features)
        copy.add_virtual_column(self.prediction_name, expression, unique=False)
        return copy

    def state_get(self):
        state = super(KerasModel, self).state_get()

        with tempfile.TemporaryDirectory() as directory:
            self.model.save(directory)
            zip_path = tempfile.mktemp(".zip")
            shutil.make_archive(zip_path[:-4], 'zip', directory)
            with open(zip_path, 'rb') as f:
                data = f.read()
            state['model'] = base64.encodebytes(data).decode('ascii')
            return state

    def state_set(self, state, trusted=True):
        state = state.copy()
        model_data = state.pop('model')
        super(KerasModel, self).state_set(state)

        data = base64.decodebytes(model_data.encode('ascii'))
        with tempfile.TemporaryDirectory() as directory:
            zip_path = tempfile.mktemp('.zip')
            with open(zip_path, 'wb') as f:
                f.write(data)
            shutil.unpack_archive(zip_path, directory)
            self.model = K.models.load_model(directory)
