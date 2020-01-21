import vaex
import json
from . import executor
import numpy as np
from vaex.encoding import Encoding, serialize, deserialize

from . import client

class DummyColumn(vaex.column.Column):
    def __init__(self, name, length, dtype):
        self.name = name
        self.length = length
        self.dtype = dtype

    def __len__(self):
        return self.length


def create_df(name, info, executor):
    length = info['length_original']
    df = vaex.dataframe.DataFrameLocal(name, '', [])
    dtypes = {name: np.dtype(dtype) for name, dtype in info['dtypes'].items()}
    for name in info['column_names']:
        df.add_column(name, DummyColumn(name, length, dtypes[name]))
    state = info['state']
    df.state_set(state)
    df.executor = executor
    return df


class Server:
    def __init__(self, service):
        self.service = service

    def list(self):
        df_map = self.service.list()
        return df_map

    def execute(self, df_name, spec_data):
        encoding = Encoding()
        specs = deserialize(spec_data, encoding)
        df = self.service.df_map[df_name]
        tasks = encoding.decode_list('task', specs, df=df)
        results = self.service.execute(df, tasks)
        return results

    def _rmi(self, df, methodname, args, kwargs):
        method = getattr(self.df_map[df.name], methodname)
        return method(*args, **kwargs)


class Client(client.Client):
    def __init__(self, server):
        self.server = server
        self.executor = executor.Executor(self)
        self.update()

    def _list(self):
        return self.server.list()

    def get(self, name):
        return self.df_map[name]

    def execute(self, df, tasks):
        encoding = Encoding()
        task_specs = encoding.encode_list("task", tasks)
        return self.server.execute(df.name, serialize(task_specs, encoding))

    def _rmi(self, df, methodname, args, kwargs):
        self.server._rmi(df.name, methodname, args, kwargs)
