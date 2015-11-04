import logging

logger = logging.getLogger("vaex.settings")

class Files(object):
	def __init__(self, open, recent):
		self.open = open
		self.recent = recent

import yaml
from yaml import Loader, Dumper

class Settings(object):
	def __init__(self, filename):
		self.filename = filename
		if not os.path.exists(filename):
			with open(filename, "w"):
				pass
		with open(self.filename) as f:
			self.settings = yaml.load(f, Loader=Loader)
		if self.settings is None:
			self.settings = {}
		#logger.debug("settings: %r", self.settings)

	def store(self, key, value):
		parts = key.split(".")
		obj = self.settings
		for part in parts[:-1]:
			#print part,
			if part not in obj:
				obj[part] = {}
			obj = obj[part]
		obj[parts[-1]] = value
		#print self.settings
		with open(self.filename, "w") as f:
			yaml.dump(self.settings, f)

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

import vaex.utils
import os


webclient = Settings(os.path.join(vaex.utils.get_private_dir(), "webclient.yml"))
webserver = Settings(os.path.join(vaex.utils.get_private_dir(), "webserver.yml"))

#yaml.load()

if __name__ == "__main__":
	webclient.store("bla.la.l", 1)