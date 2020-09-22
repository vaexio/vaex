import collections
import inspect
import json
import logging
import marshal
import os
import time
import types
from collections import OrderedDict
from copy import copy as _copy

import numpy as np
import pandas as pd
import s3fs
import traitlets
import vaex
from pandas.io.common import is_s3_url

logging.basicConfig()
logger = logging.getLogger(__name__)
from vaex.ml import state

EXAMPLE = 'example'
PIPELINE_FIT = '_PIPELINE_FIT'


@vaex.register_dataframe_accessor('pipeline', override=True)
class DataFrameAccessorPipeline(object):
    def __init__(self, df):
        self.df = df

    def sample_first(self, data):
        try:
            ret = data[:1].to_dict(virtual=True)
        except:
            ret = None
        return ret

    @staticmethod
    def is_vaex_dataset(ds):
        return isinstance(ds, vaex.dataset.DataFrame) or isinstance(ds, vaex.dataset.DataFrameArrays) or isinstance(
            ds, vaex.dataset.DataFrameArrays) or isinstance(ds, vaex.dataset.DataFrameConcatenated)

    @classmethod
    def from_dict(cls, state):
        if 'state' in state:
            ret = Pipeline()
            ret.state_set(state)
            ret.updated = int(time.time())
            return ret
        return Pipeline(state=state, example=None, fit_func=None)

    def from_dataset(self, fit=None, predict_column=None):

        copy = self.df.copy()
        Pipeline.verify_vaex_dataset(self.df)
        example = None if len(self.df) == 0 else self.sample_first(self.df)
        if fit is not None:
            copy.add_function(PIPELINE_FIT, fit)
        return Pipeline(state=copy.state_get(), example=example, predict_column=predict_column)

    def __call__(self, *args, **kwargs):
        return self.from_dataset(**kwargs)


def from_file(path):
    if is_s3_url(path):
        fs = s3fs.S3FileSystem(anon=False)
        with fs.open(path, 'r') as f:
            state = Pipeline.json_load(f.read())
    else:
        with open(path, 'r') as f:
            state = Pipeline.json_load(f.read())
    ret = Pipeline.from_dict(state)
    # ret.reload_fit_func()
    return ret


class Pipeline(state.HasState):
    current_time = int(time.time())
    created = traitlets.Int(default_value=current_time, allow_none=False, help='Created time')
    updated = traitlets.Int(default_value=current_time, allow_none=False, help='Updated time')
    example = traitlets.Dict(default_value=None, allow_none=True, help='An example of the dataset to infer')
    state = traitlets.Dict(default_value=None, allow_none=True, help='The state to apply on inference')
    predict_column = traitlets.Unicode(default_value=None, allow_none=True,
                                       help='This is the column provided when using predict')
    _cache = traitlets.Dict(default_value=None, allow_none=True, help='cached statistics during training')

    def state_set(self, state):
        self.state = state.pop('state', None)
        self.example = state.pop('example', None)
        self.created = state.pop('created', None)
        self.updated = int(time.time())
        # self.fit_func = state.pop('fit_func', None)
        return self

    def not_implemented(self):
        return None

    @property
    def virtual_columns(self):
        return self.state.get('virtual_columns')

    @property
    def functions(self):
        return self.state.get('functions')

    @classmethod
    def verify_vaex_dataset(cls, ds):
        if not cls.is_vaex_dataset(ds):
            raise ValueError('ds should be a vaex.dataset.DatasetArrays or vaex.hdf5.dataset.Hdf5MemoryMapped')
        return True

    @classmethod
    def from_dict(cls, state):
        if 'state' in state:
            ret = Pipeline()
            ret.state_set(state)
            ret.updated = int(time.time())
            return ret
        return Pipeline(state=state, example=None, fit_func=None)

    @property
    def origin_columns(self):
        if self.example is None:
            return None
        ret = []
        test = _copy(self.example)
        virtual_columns = list(self.state.get('virtual_columns', {}).keys())
        column_names = _copy(self.state['column_names'])
        for column in virtual_columns:
            test.pop(column, None)

        for column in column_names:
            if column in virtual_columns:
                continue
            value = test.pop(column, None)
            try:
                self.inference(test).evaluate(virtual_columns)
            except Exception as e:
                ret.append(column)
            test[column] = value
        return ret

    @classmethod
    def from_dataset(cls, df, fit=None, predict_column=None):
        copy = df.copy()
        cls.verify_vaex_dataset(df)
        example = None if len(df) == 0 else cls.sample_first(df)
        if fit is not None:
            copy.add_function(PIPELINE_FIT, fit)
        return Pipeline(state=copy.state_get(), example=example, predict_column=predict_column)

    @classmethod
    def from_file(cls, path):
        if is_s3_url(path):
            fs = s3fs.S3FileSystem(anon=False)
            with fs.open(path, 'r') as f:
                state = cls.json_load(f.read())
        else:
            with open(path, 'r') as f:
                state = cls.json_load(f.read())
        ret = Pipeline.from_dict(state)
        # ret.reload_fit_func()
        return ret

    def reload_fit_func(self):
        if self.fit_func is not None and not callable(self.fit_func):
            self.set_fit(self.code_to_func(self.fit_func))
            return True
        return False

    @classmethod
    def func_to_code(cls, fn):
        if fn is not None:
            return marshal.dumps(fn.__code__)
        return None

    @classmethod
    def code_to_func(cls, code):
        if code is not None:
            return types.FunctionType(marshal.loads(code), globals())
        return None

    def json_dumps(self):
        from vaex.json import VaexJsonEncoder
        return json.dumps(_copy(self.state_get()), indent=2, cls=VaexJsonEncoder)

    @classmethod
    def json_load(cls, state):
        from vaex.json import VaexJsonDecoder
        return json.loads(state, cls=VaexJsonDecoder)

    def save(self, path):
        if self.state is None:
            raise RuntimeError("Pipeline has no state to save")

        state_to_write = self.json_dumps()
        if is_s3_url(path):
            fs = s3fs.S3FileSystem(anon=False)
            with fs.open(path, 'w') as f:
                f.write(state_to_write)
        else:
            try:
                os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
            except AttributeError as e:
                pass
            with open(path, 'w') as outfile:
                outfile.write(state_to_write)
        # self.reload_fit_func()
        return path

    @classmethod
    def is_vaex_dataset(cls, ds):
        return isinstance(ds, vaex.dataset.DataFrame) or isinstance(ds, vaex.dataset.DataFrameArrays) or isinstance(
            ds, vaex.dataset.DataFrameArrays) or isinstance(ds, vaex.dataset.DataFrameConcatenated)

    def load_state(self, state):
        if type(state) == str:
            temp = Pipeline.from_file(state)
            self.set_fit(temp.fit_func)
            self.state = temp.state
        elif isinstance(state, dict):
            self.state = state
        elif Pipeline.is_vaex_dataset(state):
            self.state = state.state_get()
            self.set_fit(state.fit_func)
        else:
            raise ValueError('state type should be a json file path or a dictionary')

    def __getitem__(self, index):
        virtual_columns = list(OrderedDict(self.virtual_columns).items())
        return virtual_columns[index]

    def reload_state(self, state):
        self.state = self.load_state(state)

    @staticmethod
    def remove_selections(state):
        return {**state, **{'selections': {'__filter__': None}}}

    def apply_state_options(self, df, **kwargs):
        selections = kwargs.get('selections')
        state = kwargs.get('state', self.state)
        if not bool(selections):
            state = self.remove_selections(state)
        data_columns = df.get_column_names()
        virtual_columns = list(state.get('virtual_columns', {}).keys())
        all_columns = data_columns + virtual_columns
        state['column_names'] = [column for column in state.get('column_names') if column in all_columns]
        return state

    def inference(self, df, **kwargs):
        df = self.infer(df).copy()
        state = self.apply_state_options(df, **kwargs)
        ret = self.transform(df, state=state)
        columns = kwargs.get('columns')
        virtual = kwargs.get('virtual', True)
        output_type = kwargs.get('output_type')
        if columns is not None:
            common_columns = set(columns).intersection(set(ret.get_column_names(virtual=virtual)))
            if len(common_columns) > 0:
                columns = list(common_columns)
            else:
                columns = None
        if columns:
            ret = ret[columns]
        if output_type is not None:
            if output_type == 'pandas':
                ret = ret.to_pandas_df(virtual=virtual)
            elif output_type == 'numpy':
                columns_count = len(ret.get_column_names())
                ret = ret.to_pandas_df(virtual=virtual).to_numpy()
                if columns_count == 1:
                    ret = ret.reshape(-1)
            elif output_type == 'json':
                ret = ret.to_pandas_df(virtual=virtual).to_json(orient=kwargs.get('orient', 'records'))
            else:
                raise RuntimeError(f"output_type {output_type} is invalid")
        return ret

    def transform(self, df, state=None):
        df = self.infer(df).copy()
        df.state_set(state or self.state)
        return df

    def predict(self, df, predict_column=None, inference=False):
        predict_column = predict_column or self.predict_column
        if predict_column is None:
            raise RuntimeError("predict_column is not provided - please provide a prediction column")
        copy = self.inference(df) if inference else self.transform(df)
        return copy[predict_column].values

    def evaluate(self, df):
        raise NotImplementedError('evaluate not implemented')

    def partial_fit(self, df):
        raise NotImplementedError('partial_fit not implemented')

    @staticmethod
    def infer(input):
        vaex_ds = None
        if type(input) == bytes or type(input) == str:
            input = json.loads(input)
        if type(input) == np.array:
            pass
        if type(input) == list:
            vaex_ds = vaex.from_pandas(pd.DataFrame(input))
        elif type(input) == dict:
            random_key = list(input.keys())[0]
            if type(input[random_key]) == list or type(input[random_key]) == np.ndarray:
                vaex_ds = vaex.from_arrays(**input)
            else:
                vaex_ds = vaex.from_scalars(**input)
        elif isinstance(input, vaex.dataset.DataFrame):
            vaex_ds = input
        elif isinstance(input, pd.DataFrame):
            vaex_ds = vaex.from_pandas(input)
        if vaex_ds is None:
            raise RuntimeError("Could not infer a vaex type")
        return vaex_ds

    def get_variable(self, variable):
        if self.state is None:
            logger.debug('state is None')
            return None
        return self.state['variables'].get(variable)

    def presiste_code(self):
        self.fit_code = marshal.dumps(self.fit_func.__code__)
        return self.fit_func

    def get_example(self):
        return self.example

    def get_columns_names(self, virtual=True, strings=True, hidden=False, regex=None):
        return self.transform(self.example).get_column_names(virtual=virtual, strings=strings, hidden=hidden,
                                                             regex=regex)

    def get_input_columns(self, regex=None):
        """
        :param regex: regex for filtering the results
        :return: List of columns that provided in the original dataset
        """
        return self.transform(self.example).get_column_names(virtual=False, strings=False, hidden=False, regex=regex)

    def set_train_state(self, func):
        if not callable(func):
            raise ValueError('f must be a function')
        self.train_state = func

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    @staticmethod
    def train_state(data):
        return False

    def is_valid_fit(self, f):
        return f is not None and callable(f) and len(inspect.getfullargspec(f)[0]) > 0

    def set_fit(self, f=None):
        if self.is_valid_fit(f):
            self.fit_func = f

    def fit(self, df):
        copy = self.infer(df).copy()
        self.verify_vaex_dataset(copy)
        from vaex.serialize import from_dict
        fit_func = from_dict(self.state.get('functions', {}).get(PIPELINE_FIT)).f
        if fit_func is None:
            raise RuntimeError("'fit()' was not set for this pipeline")
        self.example = None if len(copy) == 0 else self.sample_first(copy)
        trained = fit_func(copy)
        if Pipeline.is_vaex_dataset(trained):
            trained.add_function(PIPELINE_FIT, fit_func)
            self.state = trained.state_get()
        else:
            if isinstance(trained, dict):
                self.state = trained
            else:
                raise ValueError("'fit_func' should return a vaex dataset or a state, got {}".format(type(trained)))
        self.updated = int(time.time())

    @classmethod
    def sample_first(cls, data):
        try:
            ret = data[:1].to_dict(virtual=True)
        except:
            ret = None
        return ret

    def cache_set(self, key, values):
        self._cache[key] = {**self._cache.get(key, {}), **values}

    def cache_get(self, column, key):
        return self._cache.get(column, {}).get((key))

    def cache_clear(self, key=None):
        if key is None:
            self._cache = collections.defaultdict(dict)
            return self._cache
        else:
            return self._cache.pop(key, None)
