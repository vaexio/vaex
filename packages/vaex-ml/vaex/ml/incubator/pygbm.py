import vaex
import pygbm
import numpy as np
import vaex.serialize
from .. import state
import traitlets
import pygbm.binning
import contextlib
import cloudpickle as pickle
import base64

import pygbm.gradient_boosting
from .. import generate

@contextlib.contextmanager
def using_vaex(features):
    """Temporary monkey patches pygbm"""
    class VaexBinMapper(traitlets.HasTraits):
        max_bins = traitlets.CInt(255)
        random_state = traitlets.Any()
        subsample = traitlets.CInt(int(1e5))

        def fit(self, dataframe_wrapper):
            dataframe = dataframe_wrapper.ds
            self.bin_thresholds_ = []
            for feature in features:
                X = dataframe[feature].values.reshape((-1, 1)).astype(np.float32)
                midpoints = pygbm.binning.find_binning_thresholds(
                    X, self.max_bins, subsample=self.subsample,
                    random_state=self.random_state)[0]
                self.bin_thresholds_.append(midpoints)
            self.bin_thresholds_

        def transform(self, dataframe_wrapper):
            dataframe = dataframe_wrapper.ds
            N = len(dataframe)
            M = len(features)
            # fortran order so 1 column is contiguous in memory
            binned = np.zeros((N, M), dtype=np.uint8, order='F')
            for m, feature in enumerate(features):
                X = dataframe[feature].values.reshape((-1, 1)).astype(np.float32)
                binned1 = pygbm.binning.map_to_bins(X, binning_thresholds=self.bin_thresholds_)
                assert binned1.shape[1] == 1
                binned[:,m] = binned1[:,0]
            return binned

        def fit_transform(self, X):
            self.fit(X)
            return self.transform(X)
    try:
        check_X_y = pygbm.gradient_boosting.check_X_y
        BinMapper = pygbm.gradient_boosting.BinMapper

        pygbm.gradient_boosting.BinMapper = VaexBinMapper
        pygbm.gradient_boosting.check_X_y = lambda *x, **kwargs: x
        yield
    finally:
        pygbm.gradient_boosting.BinMapper = BinMapper
        pygbm.gradient_boosting.check_X_y = check_X_y



class DataFrameWrapper:
    def __init__(self, ds):
        self.ds = ds

    @property
    def nbytes(self):
        return self.ds.byte_size()


@vaex.serialize.register
@generate.register
class PyGBMModel(state.HasState):

    features = traitlets.List(traitlets.Unicode())
    num_round = traitlets.CInt()
    param = traitlets.Dict()
    prediction_name = traitlets.Unicode(default_value='pygbm_prediction')
    learning_rate = traitlets.Float(0.1)
    max_iter = traitlets.Int(10)
    max_bins = traitlets.Int(255)
    max_leaf_nodes = traitlets.Int(31)
    random_state = traitlets.Int(0)
    verbose = traitlets.Int(1)
    prediction_name = traitlets.Unicode(default_value='pygbm_prediction')


    def fit(self, dataframe, label):
        self.pygbm_model = pygbm.GradientBoostingMachine(
                            learning_rate=self.learning_rate,
                            max_iter=self.max_iter,
                            max_bins=self.max_bins,
                            max_leaf_nodes=self.max_leaf_nodes,
                            random_state=self.random_state,
                            scoring=None,
                            verbose=self.verbose,
                            validation_split=None)
        if not hasattr(label, 'values'):
            label = dataframe[label]
        y = label.values.astype(np.float32)
        with using_vaex(self.features):
            dsw = DataFrameWrapper(dataframe)
            self.pygbm_model.fit(dsw, y)

    def predict(self, dataframe):
        data = np.vstack([dataframe[k].values for k in self.features]).T
        return self.pygbm_model.predict(data)

    def __call__(self, *args):
        data = np.vstack([arg.astype(np.float32) for arg in args]).T.copy()
        return self.pygbm_model.predict(data)

    def transform(self, dataframe):
        copy = dataframe.copy()
        lazy_function = copy.add_function('pygbm_prediction_function', self)
        expression = lazy_function(*self.features)
        copy.add_virtual_column(self.prediction_name, expression, unique=False)
        return copy

    def state_get(self):
        return dict(tree_state=base64.encodebytes(pickle.dumps(self.pygbm_model)).decode('ascii'),
                    substate=super(PyGBMModel, self).state_get())

    def state_set(self, state, trusted=True):
        super(PyGBMModel, self).state_set(state['substate'])
        if trusted is False:
            raise ValueError("Will not unpickle data when source is not trusted")
        self.pygbm_model = pickle.loads(base64.decodebytes(state['tree_state'].encode('ascii')))

@vaex.serialize.register
class PyGBMClassifier(PyGBMModel):
    def __call__(self, *args):
        return np.argmax(super(PyGBMClassifier, self).__call__(*args), axis=1)
    def predict(self, dataframe, copy=False):
        return np.argmax(super(PyGBMClassifier, self).predict(dataframe, copy=copy), axis=1)
