import ctypes
import math
import warnings

import vaex
import lightgbm
import numpy as np
import tempfile
import base64
import vaex.serialize
from . import state
import traitlets
from . import generate


lib = lightgbm.basic._LIB


@vaex.serialize.register
@generate.register
class LightGBMModel(state.HasState):
    '''The LightGBM algorithm.

    This class provides an interface to the LightGBM aloritham, with some optimizations
    for better memory efficiency when training large datasets. The algorithm itself is
    not modified at all.

    LightGBM is a fast gradient boosting algorithm based on decision trees and is
    mainly used for classification, regression and ranking tasks. It is under the
    umbrella of the Distributed Machine Learning Toolkit (DMTK) project of Microsoft.
    For more information, please visit https://github.com/Microsoft/LightGBM/.

    Example:

    >>> import vaex.ml
    >>> import vaex.ml.lightgbm
    >>> df = vaex.ml.datasets.load_iris()
    >>> features = ['sepal_width', 'petal_length', 'sepal_length', 'petal_width']
    >>> df_train, df_test = vaex.ml.train_test_split(df)
    >>> params = {
        'boosting': 'gbdt',
        'max_depth': 5,
        'learning_rate': 0.1,
        'application': 'multiclass',
        'num_class': 3,
        'subsample': 0.80,
        'colsample_bytree': 0.80}
    >>> booster = vaex.ml.lightgbm.LightGBMModel(features=features, num_boost_round=100, params=params)
    >>> booster.fit(df_train, 'class_')
    >>> df_train = booster.transform(df_train)
    >>> df_train.head(3)
     #    sepal_width    petal_length    sepal_length    petal_width    class_    lightgbm_prediction
     0            3               4.5             5.4            1.5         1    [0.00165619 0.98097899 0.01736482]
     1            3.4             1.6             4.8            0.2         0    [9.99803930e-01 1.17346471e-04 7.87235133e-05]
     2            3.1             4.9             6.9            1.5         1    [0.00107541 0.9848717  0.01405289]
    >>> df_test = booster.transform(df_test)
    >>> df_test.head(3)
     #    sepal_width    petal_length    sepal_length    petal_width    class_    lightgbm_prediction
     0            3               4.2             5.9            1.5         1    [0.00208904 0.9821348  0.01577616]
     1            3               4.6             6.1            1.4         1    [0.00182039 0.98491357 0.01326604]
     2            2.9             4.6             6.6            1.3         1    [2.50915444e-04 9.98431777e-01 1.31730785e-03]
    '''
    features = traitlets.List(traitlets.Unicode(), help='List of features to use when fitting the LightGBMModel.')
    num_boost_round = traitlets.CInt(help='Number of boosting iterations.')
    params = traitlets.Dict(help='parameters to be passed on the to the LightGBM model.')
    prediction_name = traitlets.Unicode(default_value='lightgbm_prediction', help='The name of the virtual column housing the predictions.')

    def __call__(self, *args):
        data2d = np.vstack([arg.astype(np.float64) for arg in args]).T.copy()
        return self.booster.predict(data2d)

    def transform(self, df):
        '''Transform a DataFrame such that it contains the predictions of the LightGBMModel
        in form of a virtual column.

        :param df: A vaex DataFrame.

        :return copy: A shallow copy of the DataFrame that includes the LightGBMModel prediction as a virtual column.
        :rtype: DataFrame
        '''
        copy = df.copy()
        lazy_function = copy.add_function('lightgbm_prediction_function', self)
        expression = lazy_function(*self.features)
        copy.add_virtual_column(self.prediction_name, expression, unique=False)
        return copy

    def fit(self, df, target, valid_sets=None, valid_names=None,
            early_stopping_rounds=None, evals_result=None, verbose_eval=None,
            copy=False, **kwargs):
        '''Fit the LightGBMModel to the DataFrame.

        :param df: A vaex DataFrame.
        :param target: The name of the column containing the target variable.
        :param list valid_sets: A list of DataFrames to be used for validation.
        :param list valid_names: A list of strings to label the validation sets.
        :param early_stopping_rounds int: Activates early stopping.
        The model will train until the validation score stops improving.
        Validation score needs to improve at least every *early_stopping_rounds* rounds
        to continue training. Requires at least one validation DataFrame, metric
        specified. If there's more than one, will check all of them, but the
        training data is ignored anyway. If early stopping occurs, the model
        will add ``best_iteration`` field to the booster object.
        :param dict evals_result: A dictionary storing the evaluation results of all *valid_sets*.
        :param bool verbose_eval: Requires at least one item in *evals*.
        If *verbose_eval* is True then the evaluation metric on the validation
        set is printed at each boosting stage.
        :param bool copy: (default, False) If True, make an in memory copy of the data before passing it to LightGBMModel.
        '''

        if copy:
            dtrain = lightgbm.Dataset(df[self.features].values, df.evaluate(target))
            if valid_sets is not None:
                for i, item in enumerate(valid_sets):
                    valid_sets[i] = lightgbm.Dataset(item[self.features].values, item[target].values)
            else:
                valid_sets = ()
        else:
            dtrain = VaexDataset(df, target, features=self.features)
            if valid_sets is not None:
                for i, item in enumerate(valid_sets):
                    # valid_sets[i] = VaexDataset(item, target, features=self.features)
                    valid_sets[i] = lightgbm.Dataset(item[self.features].values, item[target].values)
                    warnings.warn('''Validation sets do not obey the `copy=False` argument.
                                  A standard in-memory copy is made for each validation set.''', UserWarning)
            else:
                valid_sets = ()

        self.booster = lightgbm.train(params=self.params,
                                      train_set=dtrain,
                                      num_boost_round=self.num_boost_round,
                                      valid_sets=valid_sets,
                                      valid_names=valid_names,
                                      early_stopping_rounds=early_stopping_rounds,
                                      evals_result=evals_result,
                                      verbose_eval=verbose_eval,
                                      **kwargs)

    def predict(self, df, **kwargs):
        '''Get an in-memory numpy array with the predictions of the LightGBMModel on a vaex DataFrame.
        This method accepts the key word arguments of the predict method from LightGBM.

        :param df: A vaex DataFrame.

        :returns: A in-memory numpy array containing the LightGBMModel predictions.
        :rtype: numpy.array
        '''
        # TODO: we want to go multithreaded/parallel/chunks
        return self.booster.predict(df[self.features].values)

    def state_get(self):
        filename = tempfile.mktemp()
        self.booster.save_model(filename)
        with open(filename, 'rb') as f:
            data = f.read()
        return dict(tree_state=base64.encodebytes(data).decode('ascii'),
                    substate=super(LightGBMModel, self).state_get())

    def state_set(self, state, trusted=True):
        super(LightGBMModel, self).state_set(state['substate'])
        data = base64.decodebytes(state['tree_state'].encode('ascii'))
        filename = tempfile.mktemp()
        with open(filename, 'wb') as f:
            f.write(data)
        self.booster = lightgbm.Booster(model_file=filename)


class VaexDataset(lightgbm.Dataset):
    def __init__(self, df, label=None, features=None, blocksize=100*1000, sample_count=10*100, params={}):
        super(VaexDataset, self).__init__(None)
        self.df = df
        self.features = features or self.df.get_column_names(virtual=True)
        assert len(set(self.features)) == len(self.features), "using duplicate features"
        self.blocksize = blocksize

        self.data_references = {}

        self.c_features = (ctypes.c_char_p * len(self.features))()
        self.c_features[:] = [k.encode() for k in self.features]

        self.handle = ctypes.c_void_p()

        row_count = len(self.df)
        ncol = len(self.features)
        num_sample_row = min(sample_count, row_count)

        data_list = [self.df.evaluate(k, i1=0, i2=num_sample_row).astype(np.float64) for k in self.features]
        data_pointers = [k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for k in data_list]
        sample_data = (ctypes.POINTER(ctypes.c_double) * ncol)(*data_pointers)

        indices = np.arange(num_sample_row, dtype=np.int32)
        indices_pointers = [indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)) for k in range(ncol)]
        sample_indices = (ctypes.POINTER(ctypes.c_int) * ncol)(*indices_pointers)
        num_per_col = (ctypes.c_int * ncol)(*((num_sample_row,) * ncol))
        parameters = ctypes.c_char_p(lightgbm.basic.param_dict_to_str(params).encode())

        lightgbm.basic._safe_call(lib.LGBM_DatasetCreateFromSampledColumn(sample_data,
                                                                          sample_indices,
                                                                          ctypes.c_uint(ncol),
                                                                          num_per_col,
                                                                          ctypes.c_uint(num_sample_row),
                                                                          ctypes.c_uint(row_count),
                                                                          parameters,
                                                                          ctypes.byref(self.handle)))

        blocks = int(math.ceil(row_count / blocksize))
        dtype = np.float64
        for i in range(blocks):
            i1 = i * blocksize
            i2 = min(row_count, (i+1) * blocksize)
            data = np.array([df.evaluate(k, i1=i1, i2=i2).astype(dtype) for k in self.features]).T.copy()
            ctypemap = {np.float64: ctypes.c_double, np.float32: ctypes.c_float}
            capi_typemap = {np.float64: lightgbm.basic.C_API_DTYPE_FLOAT64, np.float32: lightgbm.basic.C_API_DTYPE_FLOAT32}
            lightgbm.basic._safe_call(lib.LGBM_DatasetPushRows(self.handle,
                                                               data.ctypes.data_as(ctypes.POINTER(ctypemap[dtype])),
                                                               ctypes.c_uint(capi_typemap[dtype]),
                                                               ctypes.c_uint32(i2-i1),
                                                               ctypes.c_uint32(ncol),
                                                               ctypes.c_uint32(i1),
                                                               ))

        if label is not None:
            self.label_data = self.df.evaluate(label)
            self.set_label(self.label_data)


if __name__ == "__main__":
    df = vaex.ml.iris()
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    # ldf = VaexDataset(df, 'class_', features=features)
    ldf = lightgbm.Dataset(np.array(df[features]), df.data.class_)
    param = {'num_leaves': 31, 'num_trees': 100, 'objective': 'softmax', 'num_class': 3}
    num_boost_round = 30
    booster = lightgbm.train(param, ldf, num_boost_round)
    print(booster.predict(np.array(df[features])))
