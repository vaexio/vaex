import vaex.utils
import traitlets
import pickle
import base64


def object_pickle(trait_name, value, state_obj):
    binary_data = pickle.dumps(value)
    data = base64.encodebytes(binary_data).decode('ascii')
    return data


def object_unpickle(trait_name, data, state_obj):
    binary_data = base64.decodebytes(data.encode('ascii'))
    value = pickle.loads(binary_data)
    return value


serialize_pickle = {
    'to_json': object_pickle,
    'from_json': object_unpickle
}


def default_to_json(trait_name, value, state_obj):
    return value


def default_from_json(trait_name, data, state_obj):
    return data


class HasState(traitlets.HasTraits):

    @classmethod
    def state_from(cls, state, trusted=True):
        obj = cls()
        obj.state_set(state)
        return obj

    def state_get(self):
        state = {}
        for name in self.trait_names():
            to_json = self.trait_metadata(name, 'to_json', default_to_json)
            value = to_json(name, getattr(self, name), self)
            state[name] = value
        return state

    def state_set(self, state, trusted=True):
        for name in self.trait_names():
            if name in state:
                from_json = self.trait_metadata(name, 'from_json', default_from_json)
                value = from_json(name, state[name], self)
                setattr(self, name, value)

    def state_write(self, f):
        vaex.utils.write_json_or_yaml(f, self.state_get())

    def state_load(self, f):
        state = vaex.utils.read_json_or_yaml(f)
        self.state_set(state)
