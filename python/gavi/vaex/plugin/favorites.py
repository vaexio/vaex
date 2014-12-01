import gavi.vaex.plugin
from gavi.vaex.qt import *
from gavi.icons import iconfile

import gavi.logging
import json

logger = gavi.logging.getLogger("plugin.favorites")

class Storage(object):
	def __init__(self, name):
		self.path = os.path.expanduser('~/.vaex/%s.json' % name)
		self.all_options = []
		self.load()
		
	def load(self):
		if os.path.exists(self.path):
			try:
				self.all_options = json.load(file(self.path))
			except:
				logger.exception("error parsing settings from:" +self.path)
			
		
	def add(self, dataset, options):
		for stored_options in self.all_options:
			if stored_options["identifiers"]["path"] == dataset.path and dict(options) == dict(stored_options["options"]):
				return # duplicate found
		identifiers = {}
		identifiers["name"] = dataset.name
		identifiers["path"] = dataset.path
		identifiers["column_names"] = dataset.get_column_names()
		self.all_options.append({"identifiers":identifiers, "options":options})

		print self.all_options
		logger.debug("writing favorites to: %s" % self.path)
		json.dump(self.all_options, file(self.path, "w"), indent=4)
		
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
		#self.action_store.
		
	def on_store(self, _=None):
		storage_plots.add(self.dialog.dataset, self.dialog.get_options())
		

class FavLoadPlugin(gavi.vaex.plugin.PluginDataset):
	name="favorites"
	def __init__(self):
		pass