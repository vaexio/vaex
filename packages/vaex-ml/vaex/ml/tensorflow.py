import base64
import tempfile

import numpy as np

from tensorflow import keras as K

import traitlets

import vaex
import vaex.serialize
import vaex.ml.state
import vaex.ml.generate



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

        filename = tempfile.mktemp(".hdf5")
        self.model.save(filename)
        with open(filename, 'rb') as f:
            data = f.read()
        state['model'] = base64.encodebytes(data).decode('ascii')
        return state

    def state_set(self, state, trusted=True):
        model_data = state.pop('model')
        super(KerasModel, self).state_set(state)

        data = base64.decodebytes(model_data.encode('ascii'))
        filename = tempfile.mktemp()
        with open(filename, 'wb') as f:
            f.write(data)
        self.model = K.models.load_model(filename)
