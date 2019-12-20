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
        lazy_function = copy.add_function('sklearn_prediction_function', self, unique=True)
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


@vaex.serialize.register
@generate.register
class IncrementalPredictor(state.HasState):
    '''This class wraps any scikit-learn estimator (a.k.a predictions) that has
    a `.partial_fit` method, and makes it a vaex pipeline object.

    By wrapping "on-line" scikit-learn estimators with this class, they become a vaex
    pipeline object. Thus, they can take full advantage of the serialization and
    pipeline system of vaex. While the underlying estimator need to call the
    `.partial_fit` method, this class contains the standard `.fit` method, and
    the rest happens behind the scenes. One can also iterate over the data
    multiple times (epochs), and optionally shuffle each batch before it is sent
    to the estimator. The `predict` method returns a numpy array, while the `transform`
    method adds the prediction as a virtual column to a vaex DataFrame.

    Note: the `.fit` method will use as much memory as needed to copy one
    batch of data, while the `.predict` method will require as much memory as
    needed to output the predictions as a numpy array. The `transform` method is
    evaluated lazily, and no memory copies are made.

    Note: we are using normal sklearn without modifications here.

    Example:

    >>> import vaex
    >>> import vaex.ml
    >>> from vaex.ml.sklearn import IncrementalPredictor
    >>> from sklearn.linear_model import SGDRegressor
    >>>
    >>> df = vaex.example()
    >>>
    >>> features = df.column_names[:6]
    >>> target = 'FeH'
    >>>
    >>> standard_scaler = vaex.ml.StandardScaler(features=features)
    >>> df = standard_scaler.fit_transform(df)
    >>>
    >>> features = df.get_column_names(regex='^standard')
    >>> model = SGDRegressor(learning_rate='constant', eta0=0.01, random_state=42)
    >>>
    >>> incremental = IncrementalPredictor(model=model,
    ...                                    features=features,
    ...                                    batch_size=10_000,
    ...                                    num_epochs=3,
    ...                                    shuffle=True,
    ...                                    prediction_name='pred_FeH')
    >>> incremental.fit(df=df, target=target)
    >>> df = incremental.transform(df)
    >>> df.head(5)[['FeH', 'pred_FeH']]
      #        FeH    pred_FeH
      0  -2.30923     -1.66226
      1  -1.78874     -1.68218
      2  -0.761811    -1.59562
      3  -1.52088     -1.62225
      4  -2.65534     -1.61991
    '''

    model = traitlets.Any(default_value=None, allow_none=True, help='A scikit-learn estimator with `.fit_predict` method.').tag(**serialize_pickle)
    features = traitlets.List(traitlets.Unicode(), help='List of features to use.')
    batch_size = traitlets.Int(default_value=1_000_000, allow_none=False, help='Number of samples to be sent to the model in each batch.')
    num_epochs = traitlets.Int(default_value=1, allow_none=False, help='Number of times each batch is sent to the model.')
    shuffle = traitlets.Bool(default_value=False, allow_none=False, help='If True, shuffle the samples before sending them to the model.')
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

        return self.transform(df)[self.prediction_name].values

    def transform(self, df):
        '''Transform a DataFrame such that it contains the predictions of the IncrementalPredictor.
        in form of a virtual column.

        :param df: A vaex DataFrame.

        :return copy: A shallow copy of the DataFrame that includes the IncrementalPredictor prediction as a virtual column.
        :rtype: DataFrame
        '''
        copy = df.copy()
        lazy_function = copy.add_function('incremental_prediction_function', self, unique=True)
        expression = lazy_function(*self.features)
        copy.add_virtual_column(self.prediction_name, expression, unique=False)
        return copy

    def fit(self, df, target, progress=None, **kwargs):
        '''Fit the IncrementalPredictor to the DataFrame.

        :param df: A vaex DataFrame containing the features on which to train the model.
        :param target: The name of the column containing the target variable.
        '''
        # Check whether the model is appropriate
        assert hasattr(self.model, 'partial_fit'), 'The model must have a `.partial_fit` method.'

        # Number of batches to send (or training rounds)
        N_total = len(df)

        # For slicing the df
        index_min = 0
        index_max = self.batch_size

        # For reference: tqdm version was - would be nice to have an informative progress bar
        # for i in tqdm(range(num_batches), leave=True, desc='Training in batches...'):

        progressbar = vaex.utils.progressbars(progress)

        # Main loop: pass on batches to the model
        # for i, batch in enumerate(range(num_batches)):
        expressions = self.features + [target]
        for i1, i2, chunks in df.evaluate_iterator(expressions, chunk_size=self.batch_size):
            X = np.array(chunks[:-1]).T  # the most efficient way acctually depends on the algorithm (row of column based access)
            y = np.array(chunks[-1], copy=False)
            # For reference: tqdm version was - would be nice to have an informative progress bar
            # for j in tqdm(range(num_epochs), leave=False, desc='Iterating over batch...'):

            # Loop over for each epoch
            for j, epoch in enumerate(range(self.num_epochs)):
                progressbar((i2 * self.num_epochs + j) / (N_total * self.num_epochs))
                if self.shuffle:
                    shuffle_index = np.arange(len(X))
                    np.random.shuffle(shuffle_index)
                    X = X[shuffle_index]
                    y = y[shuffle_index]

                # train the model
                self.model.partial_fit(X, y)

            # update the slicing indices to be ready for the next batch
            index_min += self.batch_size
            index_max += self.batch_size
            index_max = min(N_total, index_max)  # clip upper value
        progressbar(1.0)
