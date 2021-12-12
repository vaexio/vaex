import base64
import tempfile

import numpy as np
import pandas as pd
import traitlets
import vaex.serialize
from sklearn.utils import shuffle
from vowpalwabbit.DFtoVW import DFtoVW
from vowpalwabbit.pyvw import vw

from . import generate
from . import state


@vaex.serialize.register
@generate.register
class VowpalWabbitModel(state.HasState):
    '''The Vowpal Wabbit algorithm.

    This class provides an interface to the Vowpal Wabbit package.

    Vowpal Wabbit provides fast, efficient, and flexible online machine learning
    techniques for reinforcement learning, supervised learning, and more.
    It is influenced by an ecosystem of community contributions, academic research, and proven algorithms.
    Microsoft Research is a major contributor to Vowpal Wabbit.

    For more information, please visit https://vowpalwabbit.org/index.html.

    Example:

    >>> import vaex.ml
    >>> import vaex.ml.vowpalwabbit
    >>> df = vaex.ml.datasets.load_iris()
    >>> df['class_'] = df['class_']+1 # Vowpal Wabbit classification is an int stareting from 1.
    >>> features = ['sepal_width', 'petal_length', 'sepal_length', 'petal_width']
    >>> df_train, df_test = df.ml.train_test_split()
    >>> params = {'oaa': '3', 'P': 1}
    >>> booster = vaex.ml.vowpalwabbit.VowpalWabbitModel(features=features, target='class_', epochs=100, params=params)
    >>> booster.fit(df_train)
    >>> df_train = booster.transform(df_train)
    >>> df_train.head(3)
     #    sepal_width    petal_length    sepal_length    petal_width    class_    vowpalwabbit_prediction
     0            3               4.5             5.4            1.5         2    2
     1            3.4             1.6             4.8            0.2         1    1
     2            3.1             4.9             6.9            1.5         2    2
    >>> df_test = booster.transform(df_test)
    >>> df_test.head(3)
     #    sepal_width    petal_length    sepal_length    petal_width    class_    vowpalwabbit_prediction
     0            3               4.2             5.9            1.5         2    2
     1            3               4.6             6.1            1.4         2    2
     2            2.9             4.6             6.6            1.3         2    2
    '''
    snake_name = 'vowpalwabbit_model'
    features = traitlets.List(traitlets.Unicode(), help='List of features to use when fitting the Vowpal Wabbit.')
    target = traitlets.Unicode(allow_none=False, help='The name of the target column.')
    passes = traitlets.CInt(help='Number of iterations.')
    params = traitlets.Dict(default_value={}, help='parameters to be passed on the to the Vowpal Wabbit model.')
    prediction_name = traitlets.Unicode(default_value='vowpalwabbit_prediction',
                                        help='The name of the virtual column housing the predictions.')

    def __call__(self, *args):
        data2d = np.array(args).T
        return self.predict(data2d)

    def transform(self, df):
        '''Transform a DataFrame such that it contains the predictions of the
        Vowpal Wabbit in form of a virtual column.

        :param df: A vaex DataFrame.

        :return copy: A shallow copy of the DataFrame that includes the Vowpal Wabbit prediction as a virtual column.
        :rtype: DataFrame
        '''
        copy = df.copy()
        lazy_function = copy.add_function('vowpalwabbit_prediction_function', self, unique=True)
        expression = lazy_function(*self.features)
        copy.add_virtual_column(self.prediction_name, expression, unique=False)
        return copy

    def fit(self, df, passes=1, chunk_size=500, partial_fit=False):
        """Fit the VowpalWabbitModel to the DataFrame.
        :param df: A vaex DataFrame containing the features and target on which to train the model.
        :param int passes: Number of passes over the data
        :param int chunk_size: Size of chunks to iterate
        """
        passes = passes or self.passes
        params = {k: v for k, v in self.params.items() if v is not None}
        target = self.target
        features = self.features
        model = self.model if (hasattr(self, 'model') and self.model is not None and partial_fit) else vw(**self.params)
        for n in range(passes):
            for _, _, X in df.to_pandas_df(chunk_size=chunk_size):
                if n > 1:
                    X = shuffle(X)
                for ex in DFtoVW.from_colnames(df=X, y=target, x=features).convert_df():
                    model.learn(ex)
        self.model = model
        return self

    def predict(self, df, **kwargs):
        '''Get an in-memory numpy array with the predictions of the VowpalWabbitModel on a vaex DataFrame.
        This method accepts the key word arguments of the predict method from VowpalWabbit.

        :param df: A vaex DataFrame.

        :returns: A in-memory numpy array containing the VowpalWabbitModel predictions.
        :rtype: numpy.array
        '''
        if self.model is None:
            raise RuntimeError("model is not fitted")
        X = pd.DataFrame(df, columns=self.features) if isinstance(df, np.ndarray) else df[
            self.features].to_pandas_df()
        X[self.target] = 1  # DFtoVW.from_colnames issue - will be ignored in predictions
        examples = DFtoVW.from_colnames(df=X, y=self.target, x=self.features).convert_df()
        return np.array([self.model.predict(ex) for ex in examples])

    def _encode_vw(self):
        if self.model is None:
            return None
        if isinstance(self.model, bytes):
            return self.model
        filename = tempfile.mktemp()
        self.model.save(filename)
        with open(filename, 'rb') as f:
            model_data = f.read()
        return base64.encodebytes(model_data).decode('ascii')

    def _decode_vw(self, encoding):
        if encoding is None:
            return vw(**self.params)
        if isinstance(encoding, str):
            model_data = base64.decodebytes(encoding.encode('ascii'))
            openfilename = tempfile.mktemp()
            with open(openfilename, 'wb') as f:
                f.write(model_data)
            params = self.params.copy()
            params['i'] = openfilename
            return vw(**params)
        else:
            return encoding

    def state_get(self):
        return dict(model_state=self._encode_vw(),
                    substate=super(VowpalWabbitModel, self).state_get())

    def state_set(self, state, trusted=True):
        super(VowpalWabbitModel, self).state_set(state['substate'])
        self.model = self._decode_vw(state['model_state'])
