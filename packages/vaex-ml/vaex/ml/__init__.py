import warnings

import vaex
import vaex.dataframe
from . import datasets
from .pipeline import Pipeline

from vaex.utils import InnerNamespace
from vaex.utils import _ensure_strings_from_expressions

def pca(self, n_components=2, features=None, prefix='PCA_', progress=False):
    '''Requires vaex.ml: Create :class:`vaex.ml.transformations.PCA` and fit it.

    :param n_components: Number of components to retain. If None, all the components will be retained.
    :param features: List of features to transform.
    :param prefix: Prefix for the names of the transformed features.
    :param progress: If True, display a progressbar of the PCA fitting process.
    '''
    features = features or self.get_column_names()
    features = _ensure_strings_from_expressions(features)
    pca = PCA(n_components=n_components, features=features, progress=progress)
    pca.fit(self)
    return pca


def label_encoder(self, features=None, prefix='label_encoded_', allow_unseen=False):
    '''Requires vaex.ml: Create :class:`vaex.ml.transformations.LabelEncoder` and fit it.

    :param features: List of features to encode.
    :param prefix: Prefix for the names of the encoded features.
    :param allow_unseen: If True, encode unseen value as -1, otherwise an error is raised.
    '''
    features = features or self.get_column_names()
    features = _ensure_strings_from_expressions(features)
    label_encoder = LabelEncoder(features=features, prefix=prefix, allow_unseen=allow_unseen)
    label_encoder.fit(self)
    return label_encoder


def one_hot_encoder(self, features=None, one=1, zero=0, prefix=''):
    '''Requires vaex.ml: Create :class:`vaex.ml.transformations.OneHotEncoder` and fit it.

    :param features: List of features to encode.
    :param one: What value to use instead of "1".
    :param zero: What value to use instead of "0".
    :param prefix: Prefix for the names of the encoded features.
    :returns one_hot_encoder: vaex.ml.transformations.OneHotEncoder object.
    '''
    if features is None:
        raise ValueError('Please give at least one categorical feature.')
    features = _ensure_strings_from_expressions(features)
    one_hot_encoder = OneHotEncoder(features=features, one=one, zero=zero, prefix=prefix)
    one_hot_encoder.fit(self)
    return one_hot_encoder


def frequency_encoder(self, features=None, unseen='nan', prefix='frequency_encoded_'):
    '''
    Requires vaex.ml: Create :class:`vaex.ml.transformations.FrequencyEncoder` and fit it.

    :param features: List of features to encode.
    :param unseen: Strategy to deal with unseen values. Accepted arguments are "zero" or "nan".
    :param prefix: Prefix for the names of the encoded features.
    '''
    features = features or self.get_column_names()
    features = _ensure_strings_from_expressions(features)
    freq_encoder = FrequencyEncoder(features=features, prefix=prefix)
    freq_encoder.fit(self)
    return freq_encoder


def standard_scaler(self, features=None, with_mean=True, with_std=True, prefix='standard_scaled_'):
    '''Requires vaex.ml: Create :class:`vaex.ml.transformations.StandardScaler` and fit it.

    :param features: List of features to scale.
    :param with_mean: If True, remove the mean from each feature.
    :param with_std: If True, scale each feature to unit variance.
    :param prefix: Prefix for the names of the scaled features.
    '''
    features = features or self.get_column_names()
    features = _ensure_strings_from_expressions(features)
    standard_scaler = StandardScaler(features=features, with_mean=with_mean, with_std=with_std, prefix=prefix)
    standard_scaler.fit(self)
    return standard_scaler


def minmax_scaler(self, features=None, feature_range=[0, 1], prefix='minmax_scaled_'):
    '''Requires vaex.ml: Create :class:`vaex.ml.transformations.MinMaxScaler` and fit it.

    :param features: List of features to scale.
    :param feature_range: The range the features are scaled to.
    :param prefix: Prefix for the names of the scaled features.
    '''
    features = features or self.get_column_names()
    features = _ensure_strings_from_expressions(features)
    minmax_scaler = MinMaxScaler(features=features, feature_range=feature_range, prefix=prefix)
    minmax_scaler.fit(self)
    return minmax_scaler


def xgboost_model(self, target, num_boost_round, features=None, params={}, prediction_name='xgboost_prediction'):
    '''Requires vaex.ml: create a XGBoost model and train/fit it.

    :param target: Target to train/fit on.
    :param num_boost_round: Number of rounds.
    :param features: List of features to train on.
    :return vaex.ml.xgboost.XGBModel: Fitted XGBoost model.
    '''
    from .xgboost import XGBoostModel
    dataframe = self
    features = features or self.get_column_names()
    features = _ensure_strings_from_expressions(features)
    booster = XGBoostModel(prediction_name=prediction_name,
                           num_boost_round=num_boost_round,
                           features=features,
                           params=params)
    booster.fit(dataframe, target)
    return booster


def lightgbm_model(self, target, num_boost_round, features=None, copy=False, params={},
                   prediction_name='lightgbm_prediction'):
    '''Requires vaex.ml: create a lightgbm model and train/fit it.

    :param target: The target variable to predict.
    :param num_boost_round: Number of boosting iterations.
    :param features: List of features to train on.
    :param bool copy: Copy data or use the modified xgboost library for efficient transfer.
    :return vaex.ml.lightgbm.LightGBMModel: Fitted LightGBM model.
    '''
    from .lightgbm import LightGBMModel
    dataframe = self
    features = features or self.get_column_names(virtual=True)
    features = _ensure_strings_from_expressions(features)

    booster = LightGBMModel(prediction_name=prediction_name,
                            num_boost_round=num_boost_round,
                            features=features,
                            params=params)
    booster.fit(dataframe, target, copy=copy)
    return booster


def catboost_model(self, target, num_boost_round, features=None, params=None, prediction_name='catboost_prediction'):
    '''Requires vaex.ml: create a CatBoostModel model and train/fit it.

    :param target: Target to train/fit on
    :param num_boost_round: Number of rounds
    :param features: List of features to train on
    :return vaex.ml.catboost.CatBoostModel: Fitted CatBoostModel model
    '''
    from .catboost import CatBoostModel
    dataframe = self
    features = features or self.get_column_names()
    features = _ensure_strings_from_expressions(features)
    booster = CatBoostModel(prediction_name=prediction_name,
                            num_boost_round=num_boost_round,
                            features=features,
                            params=params)
    booster.fit(dataframe, target)
    return booster


def pygbm_model(self, label, max_iter, features=None, param={}, classifier=False, prediction_name='pygbm_prediction', **kwargs):
    '''Requires vaex.ml: create a pygbm model and train/fit it.

    :param label: Label to train/fit on
    :param max_iter: Max number of iterations/trees
    :param features: List of features to train on
    :param bool classifier: If true, return a the classifier (will use argmax on the probabilities)
    :return vaex.ml.pygbm.PyGBMModel or vaex.ml.pygbm.PyGBMClassifier: Fitted PyGBM model
    '''
    from .incubator.pygbm import PyGBMModel, PyGBMClassifier
    dataframe = self
    features = features or self.get_column_names()
    features = _ensure_strings_from_expressions(features)
    cls = PyGBMClassifier if classifier else PyGBMModel
    b = cls(prediction_name=prediction_name, max_iter=max_iter, features=features, param=param, **kwargs)
    b.fit(dataframe, label)
    return b


def state_transfer(self):
    from .transformations import StateTransfer
    state = self.state_get()
    state.pop('active_range')  # we are not interested in this..
    return StateTransfer(state=state)


def train_test_split(self, test_size=0.2, strings=True, virtual=True, verbose=True):
    '''Will split the DataFrame in train and test part, assuming it is shuffled.

    :param test_size: The fractional size of the test set.
    :param strings: If True, the output DataFrames will also contain string columns, if any.
    :param virtual: If True, the output DataFrames will also contain virtual contain, if any.
    :param verbose: If True, print warnings to screen.
    '''
    if verbose:
        warnings.warn('Make sure the DataFrame is shuffled')
    initial = None
    try:
        assert self.filtered is False, 'Filtered DataFrames are not yet supported.'
        # full_length = len(self)
        self = self.trim()
        initial = self.get_active_range()
        self.set_active_fraction(test_size)
        test = self.trim()
        __, end = self.get_active_range()
        self.set_active_range(end, self.length_original())
        train = self.trim()
    finally:
        if initial is not None:
            self.set_active_range(*initial)
    return train, test


def add_namespace():
    vaex.dataframe.DataFrame.ml = InnerNamespace({}, vaex.dataframe.DataFrame, prefix='ml_')
    vaex.dataframe.DataFrame.ml._add(train_test_split=train_test_split)

    vaex.dataframe.DataFrame.ml._add(xgboost_model=xgboost_model,
                                     lightgbm_model=lightgbm_model,
                                     catboost_model=catboost_model,
                                     pygbm_model=pygbm_model,
                                     state_transfer=state_transfer,
                                     one_hot_encoder=one_hot_encoder,
                                     label_encoder=label_encoder,
                                     frequency_encoder=frequency_encoder,
                                     pca=pca,
                                     standard_scaler=standard_scaler,
                                     minmax_scaler=minmax_scaler)
add_namespace()

from .transformations import PCA
from .transformations import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from .transformations import LabelEncoder, OneHotEncoder, FrequencyEncoder
