import warnings
import json
import os

import vaex
from . import datasets
from .pipeline import Pipeline
from vaex.utils import _ensure_strings_from_expressions


class DataFrameAccessorML(object):
    def __init__(self, df):
        self.df = df

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


filename_spec = os.path.join(os.path.dirname(__file__), 'spec.json')

if os.path.exists(filename_spec):
    # add DataFrameAccessorML.<snake_name> wrapper methods
    with open(filename_spec) as f:
        try:
            spec = json.load(f)
        except json.decoder.JSONDecodeError:
            pass  # we are generating the file probably
        else:
            for class_spec in spec:
                def closure(class_spec=class_spec):
                    def wrapper(self, features=None, transform=True, **kwargs):
                        kwargs = kwargs.copy()  # we do modifications, so make a copy
                        features = features or self.df.get_column_names()
                        features = _ensure_strings_from_expressions(features)
                        import importlib
                        module = importlib.import_module(class_spec['module'])
                        cls = getattr(module, class_spec['classname'])
                        if 'target' in kwargs:
                            kwargs['target'] = str(kwargs['target'])

                        object = cls(features=features, **kwargs)
                        object.fit(self.df)
                        if transform:
                            dft = object.transform(self.df)
                            return dft
                        else:
                            return object

                    # Append trait help strings to the docstring
                    doc = '\n'
                    for trait in class_spec['traits']:
                        doc += f':param {trait["name"]}: {trait["help"]} \n'
                    doc += ':param transform: If True, return a shallow copy of the transformed DataFrame, otherwise the return fitted transformer. \n'
                    try:
                        wrapper.__doc__= class_spec['doc'] + doc
                    except TypeError:  # unsupported operand type(s) for +: 'NoneType' and 'str'
                        wrapper.__doc__= doc

                    return wrapper
                accessor = DataFrameAccessorML
                name = class_spec['snake_name']
                setattr(accessor, name, closure())


from .transformations import PCA, PCAIncremental, RandomProjections
from .transformations import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from .transformations import LabelEncoder, OneHotEncoder, FrequencyEncoder
from .transformations import CycleTransformer
from .transformations import BayesianTargetEncoder
from .transformations import WeightOfEvidenceEncoder
from .transformations import GroupByTransformer, KBinsDiscretizer
