import json
import os
import vaex.events

__author__ = 'maartenbreddels'

import logging
logger = logging.getLogger("plugin.favorites")

class Storage(object):
	def __init__(self, name):
		self.dir_path= os.path.expanduser('~/.vaex')
		if not os.path.exists(self.dir_path):
			os.makedirs(self.dir_path)
		self.path = os.path.join(self.dir_path, '%s.json' % name)
		self.all_options = []
		self.load()
		self.changed = vaex.events.Signal("changed")

	def load(self):
		if os.path.exists(self.path):
			try:
				self.all_options = json.load(open(self.path))
			except:
				logger.exception("error parsing settings from:" +self.path)

	def get_all(self, type_name, dataset):
		return [a for a in self.all_options if self._fuzzy_match(a, dataset, type_name)]

	def _fuzzy_match(self, options, dataset, type_name):
		ids = options["identifiers"]
		return ((ids["path"] == dataset.path) or\
		       (ids["name"] == dataset.name) or \
		       (list(sorted(ids["column_names"])) == list(sorted(dataset.get_column_names())))) and\
				type_name == options["type_name"]

	def _make_key(self, name, type_name, dataset):
		#return os.path.join(dataset.path, name)
		return dataset.path +"?type=%s&options=%s" % (type_name, name)

	def exists(self, name, type_name, dataset):
		key = self._make_key(name, type_name, dataset)
		for option in self.all_options:
			if key == option["key"]:
				return True
		return False

	def add(self, name, type_name, dataset, options):
		key = self._make_key(name, type_name, dataset)
		for stored_options in self.all_options:
			if stored_options["identifiers"]["path"] == dataset.path and dict(options) == dict(stored_options["options"]):
				return # duplicate found
		#for stored_options in self.all_options:
		# make sure we overwrite older settings
		self.all_options = [set for set in self.all_options if set["key"] != key]

		identifiers = {}
		identifiers["name"] = dataset.name
		identifiers["path"] = dataset.path
		identifiers["column_names"] = dataset.get_column_names()
		self.all_options.append({"identifiers":identifiers, "options":options, "key":key, "name":name, "type_name": type_name})


		#print((self.all_options))
		logger.debug("writing favorites to: %s" % self.path)
		json.dump(self.all_options, open(self.path, "w"), indent=4)
		self.changed.emit()
