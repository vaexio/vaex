import base64
import tempfile
import traitlets

import vaex
import vaex.serialize
from . import state
from . import generate

import numpy as np
import catboost


@vaex.serialize.register
@generate.register
class CatBoostModel(state.HasState):
    '''The CatBoost algorithm.

    This class provides an interface to the CatBoost aloritham.
    CatBoost is a fast, scalable, high performance Gradient Boosting on
    Decision Trees library, used for ranking, classification, regression and
    other machine learning tasks. For more information please visit
    https://github.com/catboost/catboost

    Example:

    >>> import vaex
    >>> import vaex.ml.catboost
    >>> df = vaex.datasets.iris()
    >>> features = ['sepal_width', 'petal_length', 'sepal_length', 'petal_width']
    >>> df_train, df_test = df.ml.train_test_split()
    >>> params = {
        'leaf_estimation_method': 'Gradient',
        'learning_rate': 0.1,
        'max_depth': 3,
        'bootstrap_type': 'Bernoulli',
        'objective': 'MultiClass',
        'eval_metric': 'MultiClass',
        'subsample': 0.8,
        'random_state': 42,
        'verbose': 0}
    >>> booster = vaex.ml.catboost.CatBoostModel(features=features, target='class_', num_boost_round=100, params=params)
    >>> booster.fit(df_train)
    >>> df_train = booster.transform(df_train)
    >>> df_train.head(3)
    #    sepal_length    sepal_width    petal_length    petal_width    class_  catboost_prediction
    0             5.4            3               4.5            1.5         1  [0.00615039 0.98024259 0.01360702]
    1             4.8            3.4             1.6            0.2         0  [0.99034267 0.00526382 0.0043935 ]
    2             6.9            3.1             4.9            1.5         1  [0.00688241 0.95190908 0.04120851]
    >>> df_test = booster.transform(df_test)
    >>> df_test.head(3)
    #    sepal_length    sepal_width    petal_length    petal_width    class_  catboost_prediction
    0             5.9            3               4.2            1.5         1  [0.00464228 0.98883351 0.00652421]
    1             6.1            3               4.6            1.4         1  [0.00350424 0.9882139  0.00828186]
    2             6.6            2.9             4.6            1.3         1  [0.00325705 0.98891631 0.00782664]
    '''
    snake_name = "catboost_model"
    features = traitlets.List(traitlets.Unicode(), help='List of features to use when fitting the CatBoostModel.')
    target = traitlets.Unicode(allow_none=False, help='The name of the target column.')
    num_boost_round = traitlets.CInt(default_value=None, allow_none=True, help='Number of boosting iterations.')
    params = traitlets.Dict(help='A dictionary of parameters to be passed on to the CatBoostModel model.')
    pool_params = traitlets.Dict(default_value={}, help='A dictionary of parameters to be passed to the Pool data object construction')
    prediction_name = traitlets.Unicode(default_value='catboost_prediction', help='The name of the virtual column housing the predictions.')
    prediction_type = traitlets.Enum(values=['Probability', 'Class', 'RawFormulaVal'], default_value='Probability',
                                     help='The form of the predictions. Can be "RawFormulaVal", "Probability" or "Class".')
    batch_size = traitlets.CInt(default_value=None, allow_none=True, help='If provided, will train in batches of this size.')
    batch_weights = traitlets.List(traitlets.Float(), default_value=[], allow_none=True, help='Weights to sum models at the end of training in batches.')
    evals_result_ = traitlets.List(traitlets.Dict(), default_value=[], help="Evaluation results")
    ctr_merge_policy = traitlets.Enum(values=['FailIfCtrsIntersects', 'LeaveMostDiversifiedTable', 'IntersectingCountersAverage'],
                                      default_value='IntersectingCountersAverage', help="Strategy for summing up models. Only used when training in batches. See the CatBoost documentation for more info.")

    def __call__(self, *args):
        data2d = np.stack([np.asarray(arg, np.float64) for arg in args], axis=1)
        dmatrix = catboost.Pool(data2d, **self.pool_params)
        return self.booster.predict(dmatrix, prediction_type=self.prediction_type)

    def transform(self, df):
        '''Transform a DataFrame such that it contains the predictions of the CatBoostModel in form of a virtual column.

        :param df: A vaex DataFrame. It should have the same columns as the DataFrame used to train the model.

        :return copy: A shallow copy of the DataFrame that includes the CatBoostModel prediction as a virtual column.
        :rtype: DataFrame
        '''
        copy = df.copy()
        lazy_function = copy.add_function('catboost_prediction_function', self, unique=True)
        expression = lazy_function(*self.features)
        copy.add_virtual_column(self.prediction_name, expression, unique=False)
        return copy

    def fit(self, df, evals=None, early_stopping_rounds=None, verbose_eval=None, plot=False, progress=None, **kwargs):
        '''Fit the CatBoostModel model given a DataFrame.
        This method accepts all key word arguments for the catboost.train method.

        :param df: A vaex DataFrame containing the features and target on which to train the model.
        :param evals: A list of DataFrames to be evaluated during training.
            This allows user to watch performance on the validation sets.
        :param int early_stopping_rounds: Activates early stopping.
        :param bool verbose_eval: Requires at least one item in *evals*.
            If *verbose_eval* is True then the evaluation metric on the validation set is printed at each boosting stage.
        :param bool plot: if True, display an interactive widget in the Jupyter
            notebook of how the train and validation sets score on each boosting iteration.
        :param progress: If True display a progressbar when the training is done in batches.
        '''
        self.pool_params['feature_names'] = self.features
        if evals is not None:
            for i, item in enumerate(evals):
                data = item[self.features].values
                target_data = item[self.target].to_numpy()
                evals[i] = catboost.Pool(data=data, label=target_data, **self.pool_params)

        # This does the actual training/fitting of the catboost model
        if self.batch_size is None:
            data = df[self.features].values
            target_data = df[self.target].to_numpy()
            dtrain = catboost.Pool(data=data, label=target_data, **self.pool_params)
            model = catboost.train(params=self.params,
                                   dtrain=dtrain,
                                   num_boost_round=self.num_boost_round,
                                   evals=evals,
                                   early_stopping_rounds=early_stopping_rounds,
                                   verbose_eval=verbose_eval,
                                   plot=plot,
                                   **kwargs)
            self.booster = model
            self.evals_result_ = [model.evals_result_]
            self.feature_importances_ = list(model.feature_importances_)
        else:
            models = []

            # Set up progressbar
            n_samples = len(df)
            progressbar = vaex.utils.progressbars(progress, title="fit(catboost)")

            column_names = self.features + [self.target]
            iterator = df[column_names].to_pandas_df(chunk_size=self.batch_size)
            for i1, i2, chunk in iterator:
                progressbar(i1 / n_samples)
                data = chunk[self.features].values
                target_data = chunk[self.target].values
                dtrain = catboost.Pool(data=data, label=target_data, **self.pool_params)
                model = catboost.train(params=self.params,
                                       dtrain=dtrain,
                                       num_boost_round=self.num_boost_round,
                                       evals=evals,
                                       early_stopping_rounds=early_stopping_rounds,
                                       verbose_eval=verbose_eval,
                                       plot=plot,
                                       **kwargs)
                self.evals_result_.append(model.evals_result_)
                models.append(model)
            progressbar(1.0)

            # Weights are key when summing models
            if len(self.batch_weights) == 0:
                batch_weights = [1/len(models)] * len(models)
            elif self.batch_weights is not None and len(self.batch_weights) != len(models):
                raise ValueError("'batch_weights' must be te same length as the number of models.")
            else:
                batch_weights = self.batch_weights

            # Sum the models
            self.booster = catboost.sum_models(models, weights=batch_weights, ctr_merge_policy=self.ctr_merge_policy)


    def predict(self, df, **kwargs):
        '''Provided a vaex DataFrame, get an in-memory numpy array with the predictions from the CatBoostModel model.
        This method accepts the key word arguments of the predict method from catboost.

        :param df: a vaex DataFrame

        :returns: A in-memory numpy array containing the CatBoostModel predictions.
        :rtype: numpy.array
        '''
        data = df[self.features].values
        dmatrix = catboost.Pool(data, **self.pool_params)
        return self.booster.predict(dmatrix, prediction_type=self.prediction_type, **kwargs)

    def state_get(self):
        filename = tempfile.mktemp()
        self.booster.save_model(filename)
        with open(filename, 'rb') as f:
            data = f.read()
        return dict(tree_state=base64.encodebytes(data).decode('ascii'),
                    substate=super(CatBoostModel, self).state_get())

    def state_set(self, state, trusted=True):
        super(CatBoostModel, self).state_set(state['substate'])
        data = base64.decodebytes(state['tree_state'].encode('ascii'))
        filename = tempfile.mktemp()
        with open(filename, 'wb') as f:
            f.write(data)
        self.booster = catboost.CatBoost().load_model(fname=filename)
