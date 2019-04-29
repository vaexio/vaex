import os
import logging
import vaex.utils
import collections

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

    def auto_store_dict(self, key):
        # TODO: no nested keys supported yet
        if key not in self.settings:
            self.settings[key] = {}
        return AutoStoreDict(self, self.settings[key])

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
    def __init__(self, settings, store):
        self.store = store
        self.settings = settings

    def __getitem__(self, key):
        return self.store[self.__keytransform__(key)]

    def __setitem__(self, key, value):
        self.store[self.__keytransform__(key)] = value
        self.settings.dump()

    def __delitem__(self, key):
        del self.store[self.__keytransform__(key)]
        self.settings.dump()

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __keytransform__(self, key):
        return key


main = Settings(os.path.join(vaex.utils.get_private_dir(), "main.yml"))
webclient = Settings(os.path.join(vaex.utils.get_private_dir(), "webclient.yml"))
webserver = Settings(os.path.join(vaex.utils.get_private_dir(), "webserver.yml"))
cluster = Settings(os.path.join(vaex.utils.get_private_dir(), "cluster.yml"))

# yaml.load()

if __name__ == "__main__":
    webclient.store("bla.la.l", 1)
