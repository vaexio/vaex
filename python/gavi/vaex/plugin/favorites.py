import sys
import gavi.vaex.plugin
from gavi.vaex.qt import *
from gavi.icons import iconfile
import gavi.events

import gavi.logging
import json

logger = gavi.logging.getLogger("plugin.favorites")

class Storage(object):
	def __init__(self, name):
		self.dir_path= os.path.expanduser('~/.vaex')
		if not os.path.exists(self.dir_path):
			os.makedirs(self.dir_path)
		self.path = os.path.join(self.dir_path, '%s.json' % name)
		self.all_options = []
		self.load()
		self.changed = gavi.events.Signal("changed")
		
	def load(self):
		if os.path.exists(self.path):
			try:
				self.all_options = json.load(file(self.path))
			except:
				logger.exception("error parsing settings from:" +self.path)

	def get_all(self, type_name, dataset):
		return filter(lambda a: self._fuzzy_match(a, dataset, type_name), self.all_options)

	def _fuzzy_match(self, options, dataset, type_name):
		ids = options["identifiers"]
		return ((ids["path"] == dataset.path) or\
		       (ids["name"] == dataset.name) or \
		       (list(sorted(ids["column_names"])) == list(sorted(dataset.get_column_names())))) and\
				type_name == options["type_name"]

	def _make_key(self, name, type_name, dataset):
		#return os.path.join(dataset.path, name)
		return dataset.get_path() +"?type=%s&options=%s" % (type_name, name)

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
		self.all_options = filter(lambda set: set["key"] != key, self.all_options)

		identifiers = {}
		identifiers["name"] = dataset.name
		identifiers["path"] = dataset.path
		identifiers["column_names"] = dataset.get_column_names()
		self.all_options.append({"identifiers":identifiers, "options":options, "key":key, "name":name, "type_name": type_name})

		print self.all_options
		logger.debug("writing favorites to: %s" % self.path)
		json.dump(self.all_options, file(self.path, "w"), indent=4)
		self.changed.emit()
		
storage_plots = Storage("favorite-plots")

class FavStorePlugin(gavi.vaex.plugin.PluginPlot):
	name="favorites"
	def __init__(self, dialog):
		super(FavStorePlugin, self).__init__(dialog)
		dialog.plug_toolbar(self.plug_toolbar, 1.1)

	def plug_toolbar(self):
		logger.info("adding %s plugin" % self.name)
		self.action_store  = QtGui.QAction(QtGui.QIcon(iconfile('star')), 'Bookmark', self.dialog)
		self.dialog.toolbar.addAction(self.action_store)
		self.action_store.triggered.connect(self.on_store)
		self.changed_handle = storage_plots.changed.connect(self.load_options_menu)
		self.load_options_menu()

	def clean_up(self):
		logger.info("cleaning up, disconnecting event handler")
		storage_plots.changed.disconnect(self.changed_handle)

	def load_options(self, name, update=True):
		found = False
		names = []
		for options in storage_plots.get_all(self.dialog.type_name, self.dialog.dataset):
			names.append(options["name"])
			if options["name"] == name:
				self.dialog.apply_options(options["options"], update=update)
				found = True
		if not found:
			list = "".join(["\t'%s'\n" % k for k in names])
			print >>sys.stderr, "options %r not found, possible options:\n%s" % (name, list)
			sys.exit(-2)

	def load_options_menu(self):
		self.fav_menu = QtGui.QMenu()
		for options in storage_plots.get_all(self.dialog.type_name, self.dialog.dataset):
			action = QtGui.QAction(options["name"], self.fav_menu)
			def onLoad(_=None, options=options):
				self.dialog.apply_options(options["options"])
			action.triggered.connect(onLoad)
			self.fav_menu.addAction(action)
		self.action_store.setMenu(self.fav_menu)

	def on_store(self, _=None):
		index = len(storage_plots.get_all(self.dialog.type_name, self.dialog.dataset)) + 1
		default_name = "Settings%d" % index
		new_name = gettext(self.dialog, "Store settings", "Give a name for the stored settings", default_name)
		if new_name:
			if (not storage_plots.exists(new_name, self.dialog.type_name, self.dialog.dataset)) or confirm(self.dialog, "Store settings", "Setting with this name already exists, overwrite"):
				storage_plots.add(new_name, self.dialog.type_name, self.dialog.dataset, self.dialog.get_options())
		

class FavLoadPlugin(gavi.vaex.plugin.PluginDataset):
	name="favorites"
	def __init__(self):
		pass