import pickle
import base64
import tempfile

import traitlets
import numpy as np

import vaex
import vaex.serialize
from vaex.ml import state
from vaex.ml import generate
from vaex.ml.state import serialize_pickle


@vaex.serialize.register
@generate.register
class SKLearnPredictor(state.HasState):
    '''This class wraps any scikit-learn estimator (a.k.a predictor) making it a vaex pipeline object.

    By wrapping any scikit-learn estimators with this class, it becoes a vaex
    pipeline object. Thus, it can take full advantage of the serialization and
    pipeline system of vaex. One can use the `predict` method to get a numpy
    array as an output of a fitted estimator, or the `transform` method do add
    such a prediction to a vaex DataFrame as a virtual column.

    Note that a full memory copy of the data used is created when the `fit` and
    `predict` are called. The `transform` method is evaluated lazily.

    The scikit-learn estimators themselves are not modified at all, they are
    taken from your local installation of scikit-learn.

    Example:

    >>> import vaex.ml
    >>> from vaex.ml.sklearn import SKLearnPredictor
    >>> from sklearn.linear_model import LinearRegression
    >>> df = vaex.ml.datasets.load_iris()
    >>> features = ['sepal_width', 'petal_length', 'sepal_length']
    >>> df_train, df_test = vaex.ml.train_test_split(df)
    >>> model = SKLearnPredictor(model=LinearRegression(), features=features, prediction_name='pred')
    >>> model.fit(df_train, df_train.petal_width)
    >>> df_train = model.transform(df_train)
    >>> df_train.head(3)
     #    sepal_length    sepal_width    petal_length    petal_width    class_      pred
     0             5.4            3               4.5            1.5         1  1.64701
     1             4.8            3.4             1.6            0.2         0  0.352236
     2             6.9            3.1             4.9            1.5         1  1.59336
    >>> df_test = model.transform(df_test)
    >>> df_test.head(3)
     #    sepal_length    sepal_width    petal_length    petal_width    class_     pred
     0             5.9            3               4.2            1.5         1  1.39437
     1             6.1            3               4.6            1.4         1  1.56469
     2             6.6            2.9             4.6            1.3         1  1.44276
    '''

    model = traitlets.Any(default_value=None, allow_none=True, help='A scikit-learn estimator.').tag(**serialize_pickle)
    features = traitlets.List(traitlets.Unicode(), help='List of features to use.')
    prediction_name = traitlets.Unicode(default_value='prediction', help='The name of the virtual column housing the predictions.')

    def __call__(self, *args):
        X = np.vstack([arg.astype(np.float64) for arg in args]).T.copy()
        return self.model.predict(X)

    def predict(self, df):
        '''Get an in-memory numpy array with the predictions of the SKLearnPredictor.self

        :param df: A vaex DataFrame, containing the input features.
        :returns: A in-memory numpy array containing the SKLearnPredictor predictions.
        :rtype: numpy.array
        '''
        data = df[self.features].values
        return self.model.predict(data)

    def transform(self, df):
        '''Transform a DataFrame such that it contains the predictions of the SKLearnPredictor.
        in form of a virtual column.

        :param df: A vaex DataFrame.

        :return copy: A shallow copy of the DataFrame that includes the SKLearnPredictor prediction as a virtual column.
        :rtype: DataFrame
        '''
        copy = df.copy()
        lazy_function = copy.add_function('sklearn_prediction_function', self)
        expression = lazy_function(*self.features)
        copy.add_virtual_column(self.prediction_name, expression, unique=False)
        return copy

    def fit(self, df, target, **kwargs):
        '''Fit the SKLearnPredictor to the DataFrame.

        :param df: A vaex DataFrame containing the features on which to train the model.
        :param target: The name of the column containing the target variable.
        '''
        X = df[self.features].values
        y = df.evaluate(target)
        self.model.fit(X=X, y=y, **kwargs)
