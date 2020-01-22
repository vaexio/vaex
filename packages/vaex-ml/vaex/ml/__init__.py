import warnings

import vaex
import vaex.dataframe
from . import datasets
from .pipeline import Pipeline

from vaex.utils import InnerNamespace
from vaex.utils import _ensure_strings_from_expressions


class DataFrameAccessorML(object):
    def __init__(self, df):
        self.df = df

    def pca(self, n_components=2, features=None, prefix='PCA_', progress=False):
        '''Requires vaex.ml: Create :class:`vaex.ml.transformations.PCA` and fit it.

        :param n_components: Number of components to retain. If None, all the components will be retained.
        :param features: List of features to transform.
        :param prefix: Prefix for the names of the transformed features.
        :param progress: If True, display a progressbar of the PCA fitting process.
        '''
        features = features or self.df.get_column_names()
        features = _ensure_strings_from_expressions(features)
        pca = PCA(n_components=n_components, features=features, progress=progress)
        pca.fit(self.df)
        return pca


    def label_encoder(self, features=None, prefix='label_encoded_', allow_unseen=False):
        '''Requires vaex.ml: Create :class:`vaex.ml.transformations.LabelEncoder` and fit it.

        :param features: List of features to encode.
        :param prefix: Prefix for the names of the encoded features.
        :param allow_unseen: If True, encode unseen value as -1, otherwise an error is raised.
        '''
        features = features or self.df.get_column_names()
        features = _ensure_strings_from_expressions(features)
        label_encoder = LabelEncoder(features=features, prefix=prefix, allow_unseen=allow_unseen)
        label_encoder.fit(self.df)
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
        one_hot_encoder.fit(self.df)
        return one_hot_encoder


    def frequency_encoder(self, features=None, unseen='nan', prefix='frequency_encoded_'):
        '''
        Requires vaex.ml: Create :class:`vaex.ml.transformations.FrequencyEncoder` and fit it.

        :param features: List of features to encode.
        :param unseen: Strategy to deal with unseen values. Accepted arguments are "zero" or "nan".
        :param prefix: Prefix for the names of the encoded features.
        '''
        features = features or self.df.get_column_names()
        features = _ensure_strings_from_expressions(features)
        freq_encoder = FrequencyEncoder(features=features, prefix=prefix)
        freq_encoder.fit(self.df)
        return freq_encoder


    def standard_scaler(self, features=None, with_mean=True, with_std=True, prefix='standard_scaled_'):
        '''Requires vaex.ml: Create :class:`vaex.ml.transformations.StandardScaler` and fit it.

        :param features: List of features to scale.
        :param with_mean: If True, remove the mean from each feature.
        :param with_std: If True, scale each feature to unit variance.
        :param prefix: Prefix for the names of the scaled features.
        '''
        features = features or self.df.get_column_names()
        features = _ensure_strings_from_expressions(features)
        standard_scaler = StandardScaler(features=features, with_mean=with_mean, with_std=with_std, prefix=prefix)
        standard_scaler.fit(self.df)
        return standard_scaler


    def minmax_scaler(self, features=None, feature_range=[0, 1], prefix='minmax_scaled_'):
        '''Requires vaex.ml: Create :class:`vaex.ml.transformations.MinMaxScaler` and fit it.

        :param features: List of features to scale.
        :param feature_range: The range the features are scaled to.
        :param prefix: Prefix for the names of the scaled features.
        '''
        features = features or self.df.get_column_names()
        features = _ensure_strings_from_expressions(features)
        minmax_scaler = MinMaxScaler(features=features, feature_range=feature_range, prefix=prefix)
        minmax_scaler.fit(self.df)
        return minmax_scaler


    def xgboost_model(self, target, features=None, num_boost_round=100, params={}, prediction_name='xgboost_prediction'):
        '''Requires vaex.ml: create a XGBoost model and train/fit it.

        :param target: The name of the target column.
        :param features: List of features to use when training the model. If None, all columns except the target will be used as features.
        :param num_boost_round: Number of boosting rounds.
        :return vaex.ml.xgboost.XGBoostModel: Fitted XGBoost model.
        '''
        from .xgboost import XGBoostModel
        df = self.df
        target = _ensure_strings_from_expressions(target)
        features = features or self.df.get_column_names(virtual=True).remove(target)
        features = _ensure_strings_from_expressions(features)
        booster = XGBoostModel(prediction_name=prediction_name,
                               num_boost_round=num_boost_round,
                               features=features,
                               target=target,
                               params=params)
        booster.fit(df)
        return booster


    def lightgbm_model(self, target, features=None, num_boost_round=100, copy=False, params={}, prediction_name='lightgbm_prediction'):
        '''Requires vaex.ml: create a lightgbm model and train/fit it.

        :param target: The name of the target column.
        :param features: List of features to use when training the model. If None, all columns except the target will be used as features.
        :param num_boost_round: Number of boosting rounds.
        :param bool copy: If True Copy the data, otherwise use a more memory efficient data transfer method.
        :return vaex.ml.lightgbm.LightGBMModel: Fitted LightGBM model.
        '''
        from .lightgbm import LightGBMModel
        dataframe = self.df
        target = _ensure_strings_from_expressions(target)
        features = features or self.df.get_column_names(virtual=True).remove(target)
        features = _ensure_strings_from_expressions(features)

        booster = LightGBMModel(prediction_name=prediction_name,
                                num_boost_round=num_boost_round,
                                features=features,
                                target=target,
                                params=params)
        booster.fit(dataframe, copy=copy)
        return booster


    def catboost_model(self, target, features=None, num_boost_round=100, params=None, prediction_name='catboost_prediction'):
        '''Requires vaex.ml: create a CatBoostModel model and train/fit it.

        :param target: The name of the target column.
        :param features: List of features to use when training the model. If None, all columns except the target will be used as features.
        :param num_boost_round: Number of boosting rounds.
        :return vaex.ml.catboost.CatBoostModel: Fitted CatBoostModel model.
        '''
        from .catboost import CatBoostModel
        dataframe = self.df
        target = _ensure_strings_from_expressions(target)
        features = features or self.df.get_column_names(virtual=True).remove(target)
        features = _ensure_strings_from_expressions(features)
        booster = CatBoostModel(prediction_name=prediction_name,
                                num_boost_round=num_boost_round,
                                features=features,
                                target=target,
                                params=params)
        booster.fit(dataframe)
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
        dataframe = self.df
        features = features or self.df.get_column_names()
        features = _ensure_strings_from_expressions(features)
        cls = PyGBMClassifier if classifier else PyGBMModel
        b = cls(prediction_name=prediction_name, max_iter=max_iter, features=features, param=param, **kwargs)
        b.fit(dataframe, label)
        return b


    def state_transfer(self):
        from .transformations import StateTransfer
        state = self.df.state_get()
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
            assert self.df.filtered is False, 'Filtered DataFrames are not yet supported.'
            # full_length = len(self)
            df = self.df.trim()
            initial = self.df.get_active_range()
            df.set_active_fraction(test_size)
            test = df.trim()
            __, end = df.get_active_range()
            df.set_active_range(end, df.length_original())
            train = df.trim()
        finally:
            if initial is not None:
                df.set_active_range(*initial)
        return train, test


from .transformations import PCA
from .transformations import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from .transformations import LabelEncoder, OneHotEncoder, FrequencyEncoder
from .transformations import CycleTransformer
from .transformations import BayesianTargetEncoder
from .transformations import WeightOfEvidenceEncoder
