import os
import logging
import vaex.utils
import collections
from dataclasses import dataclass

try:
    collections_abc = collections.abc
except AttributeError:
    collections_abc = collections

logger = logging.getLogger("vaex.settings")


class Files(object):
    def __init__(self, open, recent):
        self.open = open
        self.recent = recent


class Settings(object):
    def __init__(self, filename):
        self.filename = filename
        if not os.path.exists(filename):
            with open(filename, "w"):
                pass
        with open(self.filename) as f:
            self.settings = vaex.utils.yaml_load(f)  # yaml.load(f, Loader=Loader)
        if self.settings is None:
            self.settings = {}
        # logger.debug("settings: %r", self.settings)

    def auto_store_dict(self, key, autostore=False):
        # TODO: no nested keys supported yet
        if key not in self.settings:
            self.settings[key] = {}
        return AutoStoreDict(self, self.settings[key], autostore)

    def store(self, key, value):
        parts = key.split(".")
        obj = self.settings
        for part in parts[:-1]:
            # print part,
            if part not in obj:
                obj[part] = {}
            obj = obj[part]
        obj[parts[-1]] = value
        # print self.settings
        self.dump()

    def dump(self):
        with open(self.filename, "w") as f:
            # yaml.dump(self.settings, f)
            vaex.utils.yaml_dump(f, self.settings)

    def get(self, key, default=None):
        logger.debug("get %r", key)
        parts = key.split(".")
        obj = self.settings
        for part in parts:
            if part not in obj:
                logger.debug("return %r (default)", default)
                return default
            obj = obj[part]
        logger.debug("return %r", obj)
        return obj


settings = {}


class AutoStoreDict(collections_abc.MutableMapping):
    def __init__(self, settings, store, autostore):
        self._store = store
        self._settings = settings
        self._autostore = autostore

    def save(self):
        self._settings.dump()

    def __getitem__(self, key):
        return self._store[self.__keytransform__(key)]

    def __setitem__(self, key, value):
        self._store[self.__keytransform__(key)] = value
        if self._autostore:
            self._settings.dump()

    def __delitem__(self, key):
        del self._store[self.__keytransform__(key)]
        if self._autostore:
            self._settings.dump()

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    def __keytransform__(self, key):
        return key

    def __dir__(self):
        return list(self._store.keys())

    def __setattr__(self, name, value):
        if name.startswith("_"):
            self.__dict__[name] = value
        else:
            self[name] = value

    def __getattr__(self, name):
        if name.startswith("_"):
            return self.__dict__[name]
        else:
            return self[name]

    def __repr__(self) -> str:
        s = repr(self._store)
        return f'auto_store_dict({s})'


main = Settings(os.path.join(vaex.utils.get_private_dir(), "main.yml"))
webclient = Settings(os.path.join(vaex.utils.get_private_dir(), "webclient.yml"))
webserver = Settings(os.path.join(vaex.utils.get_private_dir(), "webserver.yml"))
cluster = Settings(os.path.join(vaex.utils.get_private_dir(), "cluster.yml"))
display = main.auto_store_dict("display")
aliases = main.auto_store_dict("aliases")


def save():
    '''Save all settings.'''
    main.dump()
    webclient.dump()
    webserver.dump()
    cluster.dump()

# default values
_display_default = dict(
    max_columns=200,
    max_rows=10,
)
for name, value in _display_default.items():
    if name not in display:
        display[name] = value


if __name__ == "__main__":
    import sys
    print(f"main.yml is at {main.filename}")
    vaex.utils.yaml_dump(sys.stdout, main.settings)
