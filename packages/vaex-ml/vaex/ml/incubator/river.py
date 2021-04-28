import warnings

import numpy as np

import pandas as pd

import traitlets

import vaex
import vaex.serialize
from vaex.ml import generate
from vaex.ml import state
from vaex.ml.state import serialize_pickle

@vaex.serialize.register
@generate.register
class RiverModel(state.HasState):
    '''This class wraps River (github.com/online-ml/river) estimators, making them vaex pipeline objects.

    This class conveniently wraps River models making them vaex pipeline objects.
    Thus they take full advantage of the serialization and pipeline system of vaex.
    Only the River models that implement the `learn_many` are compatible.
    One can also wrap an entire River pipeline, as long as each pipeline step
    implements the `learn_many` method. With the wrapper one can iterate over the
    data multiple times (epochs), and optinally shuffle each batch before it is
    sent to the estimator. The `predict` method wil require as much memory as
    needed to output the predictions as a numpy array, while the `transform`
    method is evaluated lazily, and no memory copies are made.

    Example:
    >>> import vaex
    >>> import vaex.ml
    >>> from vaex.ml.incubator.river import RiverModel
    >>> from river.linear_model import LinearRegression
    >>> from river import optim
    >>>
    >>> df = vaex.example()
    >>>
    >>> features = df.column_names[:6]
    >>> target = 'FeH'
    >>>
    >>> df = df.ml.standard_scaler(features=features, prefix='scaled_')
    >>>
    >>> features = df.get_column_names(regex='^scaled_')
    >>> model = LinearRegression(optimizer=optim.SGD(lr=0.1), intercept_lr=0.1)
    >>>
    >>> river_model = RiverModel(model=model,
                            features=features,
                            target=target,
                            batch_size=10_000,
                            num_epochs=3,
                            shuffle=True,
                            prediction_name='pred_FeH')
    >>>
    >>> river_model.fit(df=df)
    >>> df = river_model.transform(df)
    >>> df.head(5)[['FeH', 'pred_FeH']]
      #       FeH    pred_FeH
      0  -1.00539    -1.6332
      1  -1.70867    -1.56632
      2  -1.83361    -1.55338
      3  -1.47869    -1.60646
      4  -1.85705    -1.5996
    '''
    snake_name = 'river_model'
    model = traitlets.Any(default_value=None, allow_none=True, help='A River model which implements the `learn_many` method.').tag(**serialize_pickle)
    features = traitlets.List(traitlets.Unicode(), help='List of features to use.')
    target = traitlets.Unicode(allow_none=False, help='The name of the target column.')
    batch_size = traitlets.Int(default_value=1_000_000, allow_none=False, help='Number of samples to be sent to the model in each batch.')
    num_epochs = traitlets.Int(default_value=1, allow_none=False, help='Number of times each batch is sent to the model.')
    shuffle = traitlets.Bool(default_value=False, allow_none=False, help='If True, shuffle the samples before sending them to the model.')
    prediction_name = traitlets.Unicode(default_value='prediction', help='The name of the virtual column housing the predictions.')
    prediction_type = traitlets.Enum(values=['predict', 'predict_proba'], default_value='predict',
                                     help='Which method to use to get the predictions. \
                                     Can be "predict" or "predict_proba" which correspond to \
                                     "predict_many" and "predict_proba_many in a River model respectively.')

    def __call__(self, *args):
        X = {feature: np.asarray(arg, np.float64) for arg, feature in zip(args, self.features)}
        X = pd.DataFrame(X)

        if self.prediction_type == 'predict':
            return self.model.predict_many(X).values
        else:
            return self.model.predict_proba_many(X).values


    def predict(self, df):
        '''Get an in memory numpy array with the predictions of the Model

        :param df: A vaex DataFrame containing the input features
        :returns: A in-memory numpy array containing the Model predictions
        :rtype: numpy.array
        '''
        return self.transform(df)[self.prediction_name].values


    def transform(self, df):
        '''
        Transform A DataFrame such that it contains the predictions of the Model in a form of a virtual column.

        :param df: A vaex DataFrame

        :return df: A vaex DataFrame
        :rtype: DataFrame
        '''
        copy = df.copy()
        lazy_function = copy.add_function('river_model_prediction_function', self, unique=True)
        expression = lazy_function(*self.features)
        copy.add_virtual_column(self.prediction_name, expression, unique=False)
        return copy


    def fit(self, df, progress=None):
        '''Fit the RiverModel to the DataFrame.

        :param df: A vaex DataFrame containig the features and target on which to train the model
        :param progress: If True, display a progressbar which tracks the training progress.
        '''

        # Check whether the model is appropriate
        assert hasattr(self.model, 'learn_many'), 'The model must implement the `.learn_many` method.'

        n_samples = len(df)

        progressbar = vaex.utils.progressbars(progress)

        # Portions of the DataFrame to evaluate
        expressions = self.features + [self.target]

        for epoch in range(self.num_epochs):
            for i1, i2, df_tmp in df.to_pandas_df(column_names=expressions, chunk_size=self.batch_size):
                progressbar((n_samples * epoch + i1) / (self.num_epochs * n_samples))
                y_tmp = df_tmp.pop(self.target)

                if self.shuffle:
                    shuffle_index = np.arange(len(df_tmp))
                    np.random.shuffle(shuffle_index)
                    df_tmp = df_tmp.iloc[shuffle_index]
                    y_tmp = y_tmp.iloc[shuffle_index]

                # Train the model
                self.model.learn_many(X=df_tmp, y=y_tmp)  # TODO: We should also support weights
        progressbar(1.0)


