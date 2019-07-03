
import base64
import tempfile
import traitlets

import xgboost
import numpy as np

import vaex
from . import state
from . import generate
import vaex.serialize


@vaex.serialize.register
@generate.register
class XGBoostModel(state.HasState):
    '''The XGBoost algorithm.

    XGBoost is an optimized distributed gradient boosting library designed to be
    highly efficient, flexible and portable. It implements machine learning
    algorithms under the Gradient Boosting framework. XGBoost provides a parallel
    tree boosting (also known as GBDT, GBM) that solve many data science
    problems in a fast and accurate way.
    (https://github.com/dmlc/xgboost)

    Example:

    >>> import vaex
    >>> import vaex.ml.xgboost
    >>> df = vaex.ml.datasets.load_iris()
    >>> features = ['sepal_width', 'petal_length', 'sepal_length', 'petal_width']
    >>> df_train, df_test = vaex.ml.train_test_split(df)
    >>> params = {
        'max_depth': 5,
        'learning_rate': 0.1,
        'objective': 'multi:softmax',
        'num_class': 3,
        'subsample': 0.80,
        'colsample_bytree': 0.80,
        'silent': 1}
    >>> booster = vaex.ml.xgboost.XGBoostModel(features=features, num_boost_round=100, params=params)
    >>> booster.fit(df_train, 'class_')
    >>> df_train = booster.transform(df_train)
    >>> df_train.head(3)
    #    sepal_length    sepal_width    petal_length    petal_width    class_    xgboost_prediction
    0             5.4            3               4.5            1.5         1                     1
    1             4.8            3.4             1.6            0.2         0                     0
    2             6.9            3.1             4.9            1.5         1                     1
    >>> df_test = booster.transform(df_test)
    >>> df_test.head(3)
    #    sepal_length    sepal_width    petal_length    petal_width    class_    xgboost_prediction
    0             5.9            3               4.2            1.5         1                     1
    1             6.1            3               4.6            1.4         1                     1
    2             6.6            2.9             4.6            1.3         1                     1
    '''

    features = traitlets.List(traitlets.Unicode(), help='List of features to use when fitting the XGBoostModel.')
    num_boost_round = traitlets.CInt(help='Number of boosting iterations.')
    params = traitlets.Dict(help='A dictionary of parameters to be passed on to the XGBoost model.')
    prediction_name = traitlets.Unicode(default_value='xgboost_prediction', help='The name of the virtual column housing the predictions.')

    def __call__(self, *args):
        data2d = np.vstack([arg.astype(np.float64) for arg in args]).T.copy()
        dmatrix = xgboost.DMatrix(data2d)
        return self.booster.predict(dmatrix)

    def transform(self, df):
        '''Transform a DataFrame such that it contains the predictions of the XGBoostModel in form of a virtual column.

        :param df: A vaex DataFrame. It should have the same columns as the DataFrame used to train the model.

        :return copy: A shallow copy of the DataFrame that includes the XGBoostModel prediction as a virtual column.
        :rtype: DataFrame
        '''
        copy = df.copy()
        lazy_function = copy.add_function('xgboost_prediction_function', self)
        expression = lazy_function(*self.features)
        copy.add_virtual_column(self.prediction_name, expression, unique=False)
        return copy

    def fit(self, df, target, evals=(), early_stopping_rounds=None,
            evals_result=None, verbose_eval=False, **kwargs):
        '''Fit the XGBoost model given a DataFrame.
        This method accepts all key word arguments for the xgboost.train method.

        :param df: A vaex DataFrame containing the training features.
        :param target: The column name of the target variable.
        :param evals: A list of pairs (DataFrame, string).
        List of items to be evaluated during training, this allows user to watch performance on the validation set.
        :param int early_stopping_rounds: Activates early stopping.
        Validation error needs to decrease at least every *early_stopping_rounds* round(s) to continue training.
        Requires at least one item in *evals*. If there's more than one, will use the last. Returns the model
        from the last iteration (not the best one).
        :param dict evals_result: A dictionary storing the evaluation results of all the items in *evals*.
        :param bool verbose_eval: Requires at least one item in *evals*.
        If *verbose_eval* is True then the evaluation metric on the validation set is printed at each boosting stage.
        '''
        data = df[self.features].values
        target_data = df.evaluate(target)
        dtrain = xgboost.DMatrix(data, target_data)
        if evals is not None:
            evals = [list(elem) for elem in evals]
            for item in evals:
                data = item[0][self.features].values
                target_data = item[0].evaluate(target)
                item[0] = xgboost.DMatrix(data, target_data)
        else:
            evals = ()

        # This does the actual training / fitting of the xgboost model
        self.booster = xgboost.train(params=self.params,
                                     dtrain=dtrain,
                                     num_boost_round=self.num_boost_round,
                                     evals=evals,
                                     early_stopping_rounds=early_stopping_rounds,
                                     evals_result=evals_result,
                                     verbose_eval=verbose_eval,
                                     **kwargs)

    def predict(self, df, **kwargs):
        '''Provided a vaex DataFrame, get an in-memory numpy array with the predictions from the XGBoost model.
        This method accepts the key word arguments of the predict method from XGBoost.

        :returns: A in-memory numpy array containing the XGBoostModel predictions.
        :rtype: numpy.array
        '''
        data = df[self.features].values
        dmatrix = xgboost.DMatrix(data)
        return self.booster.predict(dmatrix, **kwargs)

    def state_get(self):
        filename = tempfile.mktemp()
        self.booster.save_model(filename)
        with open(filename, 'rb') as f:
            data = f.read()
        return dict(tree_state=base64.encodebytes(data).decode('ascii'),
                    substate=super(XGBoostModel, self).state_get())

    def state_set(self, state, trusted=True):
        super(XGBoostModel, self).state_set(state['substate'])
        data = base64.decodebytes(state['tree_state'].encode('ascii'))
        filename = tempfile.mktemp()
        with open(filename, 'wb') as f:
            f.write(data)
        self.booster = xgboost.Booster(model_file=filename)
