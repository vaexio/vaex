import base64


import vaex.encoding
import vaex.utils


def load(file, fs_options=None, fs=None, trusted=False):
    data = vaex.utils.read_json_or_yaml(file, fs_options=fs_options, fs=fs, old_style=False)
    encoding = vaex.encoding.Encoding()
    encoding.blobs = {key: base64.b64decode(value.encode('ascii')) for key, value in data['blobs'].items()}
    return encoding.decode('transformer', data['transformer'], trusted=trusted)


class PipeLine:
    def __init__(self, df=None, transformer=None):
        self.df = df
        self.transformer = transformer

    def _encode(self):
        encoding = vaex.encoding.Encoding()
        return encoding, {'transformer': encoding.encode('transformer', self.transformer)}
    
    # @classmethod
    # def _decode(data, trusted=False):
    #     encoding = vaex.encoding.Encoding()
    #     encoding.blobs = {key: base64.b64decode(value.encode('ascii')) for key, value in data['blobs'].items()}
    #     return encoding.decode('transformer', data['transformer'], trusted=trusted)

    def save(self, file, fs_options=None, fs=None):
        encoding, data = self._encode()
        data['blobs'] = {key: base64.b64encode(value).decode('ascii') for key, value in encoding.blobs.items()}
        fs_options = fs_options or {}
        vaex.utils.write_json_or_yaml(file, data, fs_options=fs_options, fs=fs, old_style=False)

    def transform(self, df, trusted=False):
        return self.transformer.apply_deep(df)

    def load_and_transform(self, file, fs_options=None, fs=None, trusted=False):
        transformer = load(file, fs_options=fs_options, fs=fs, trusted=trusted)
        return transformer.apply_deep(self.df)
        
        # data = vaex.utils.read_json_or_yaml(file, fs_options=fs_options, fs=fs, old_style=not self._future_behaviour)
        # self.state_set(state, use_active_range=use_active_range, keep_columns=keep_columns, set_filter=set_filter, trusted=trusted)

