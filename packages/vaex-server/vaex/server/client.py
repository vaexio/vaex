import numpy as np
import uuid
import warnings

import vaex
from .dataframe import DataFrameRemote


def create_df(name, info, executor):
    _dtypes = {name: vaex.encoding.dtype_encoding.decode(None, dtype) for name, dtype in info['dtypes'].items()}
    df = DataFrameRemote(name=name,
                         length_original=info['length_original'],
                         column_names=info['column_names'],
                         dtypes=_dtypes)
    df.executor = executor
    df.state_set(info['state'])
    return df


class Client:
    def __init__(self, secure=False):
        self.secure = secure
        self.df_map = {}
        self.executor = None
        self._msg_id_to_tasks = {}

    def _check_version(self):
        versions = self.get_versions()
        import vaex.server._version
        local_version = vaex.server._version.__version_tuple__
        remote_version = tuple(versions['vaex.server'])
        if (local_version[0] != remote_version[0]) or (local_version[1] < remote_version[1]):
            warnings.warn(f'Version mismatch: server {remote_version}, while we have {local_version}')

    def get_versions(self):
        reply, encoding = self._send({'command': 'versions'})
        return reply

    def close(self):
        raise NotImplementedError

    def connect(self):
        raise NotImplementedError

    def _send(self, msg, msg_id=None):
        raise NotImplementedError

    async def _send_async(self, msg, msg_id=None):
        raise NotImplementedError

    def update(self):
        self.df_info = self._list()
        self.df_map = {name: create_df(name, info, self.executor) for name, info in self.df_info.items()}

    def __getitem__(self, name):
        if name not in self.df_map:
            raise KeyError("no such DataFrame '%s' at server, possible names: %s" % (name, ", ".join(self.df_map.keys())))
        return self.df_map[name]

    def _list(self):
        msg = {'command': 'list'}
        reply, encoding = self._send(msg)
        return reply

    def _rmi(self, df, methodname, args, kwargs):
        args = [str(k) if isinstance(k, vaex.expression.Expression) else k for k in args]
        msg = {'command': 'call-dataframe', 'method': methodname, 'df': df.name, 'state': df.state_get(), 'args': args, 'kwargs': kwargs}
        result, encoding = self._send(msg)
        if msg['method'] == "_evaluate_implementation":
            result = encoding.decode('vaex-evaluate-result', result)
        else:
            result = encoding.decode('vaex-rmi-result', result)
        return result

    def execute(self, df, tasks):
        from vaex.encoding import Encoding
        encoder = Encoding()
        msg_id = str(uuid.uuid4())
        self._msg_id_to_tasks[msg_id] = tuple(tasks)
        task_specs = encoder.encode_list("task", tasks)
        msg = {'command': 'execute', 'df': df.name, 'state': df.state_get(), 'tasks': task_specs}
        try:
            results, encoding = self._send(msg, msg_id=msg_id)
            results = encoding.decode_list('vaex-task-result', results)
            return results
        finally:
            del self._msg_id_to_tasks[msg_id]

    async def execute_async(self, df, tasks):
        from vaex.encoding import Encoding
        encoder = Encoding()
        msg_id = str(uuid.uuid4())
        self._msg_id_to_tasks[msg_id] = tuple(tasks)
        task_specs = encoder.encode_list("task", tasks)
        msg = {'command': 'execute', 'df': df.name, 'state': df.state_get(), 'tasks': task_specs}
        try:
            results, encoding = await self._send_async(msg, msg_id=msg_id)
            results = encoding.decode_list('vaex-task-result', results)
            return results
        finally:
            del self._msg_id_to_tasks[msg_id]
            pass

    @property
    def url(self):
        protocol = "wss" if self.secure else "ws"
        return "%s://%s:%d%s" % (protocol, self.hostname, self.port, self.base_path)

    @property
    def _url(self):
        protocol = "wss" if self.secure else "ws"
        return "%s://%s:%d%swebsocket" % (protocol, self.hostname, self.port, self.base_path)

