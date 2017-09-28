from __future__ import print_function
import sys

import vaex.ui.plugin
from vaex.ui.qt import *
from vaex.ui.icons import iconfile
import vaex.events
import logging
import vaex.ui.storage


logger = logging.getLogger("plugin.favorites")


storage_plots = vaex.ui.storage.Storage("favorite-plots")

class FavStorePlugin(vaex.ui.plugin.PluginPlot):
	name="favorites"
	def __init__(self, dialog):
		super(FavStorePlugin, self).__init__(dialog)
		dialog.plug_toolbar(self.plug_toolbar, 1.1)

	def plug_toolbar(self):
		logger.info("adding %s plugin" % self.name)
		self.menu = QtGui.QMenu("F&avorites")
		self.dialog.menu_bar.addMenu(self.menu)

		self.action_store  = QtGui.QAction(QtGui.QIcon(iconfile('star')), 'Store', self.dialog)
		self.action_store.setShortcut("Ctrl+B")
		self.action_store_toolbar  = QtGui.QAction(QtGui.QIcon(iconfile('star')), 'Store', self.dialog)


		self.dialog.toolbar.addAction(self.action_store_toolbar)
		self.action_store.triggered.connect(self.on_store)
		self.action_store_toolbar.triggered.connect(self.on_store)
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
			print("options %r not found, possible options:\n%s" % (name, list), file=sys.stderr)
			sys.exit(-2)

	def load_options_menu(self):
		self.fav_menu = QtGui.QMenu()
		self.menu.clear()
		self.menu.addAction(self.action_store)
		self.menu.addSeparator()

		for options in storage_plots.get_all(self.dialog.type_name, self.dialog.dataset):
			action = QtGui.QAction("Load:"+options["name"], self.fav_menu)
			def onLoad(_=None, options=options):
				self.dialog.apply_options(options["options"])
			action.triggered.connect(onLoad)
			self.fav_menu.addAction(action)
			self.menu.addAction(action)
		self.action_store_toolbar.setMenu(self.fav_menu)

	def on_store(self, _=None):
		layer = self.dialog.current_layer
		if layer is None:
			dialog_error(self, "No active layer", "Can only store settings when a layer is present")
			return
		index = len(storage_plots.get_all(self.dialog.type_name, self.dialog.dataset)) + 1
		default_name = "Settings%d" % index
		new_name = gettext(self.dialog, "Store settings", "Give a name for the stored settings", default_name)
		if new_name:
			if (not storage_plots.exists(new_name, self.dialog.type_name, self.dialog.dataset)) or confirm(self.dialog, "Store settings", "Setting with this name already exists, overwrite"):
				storage_plots.add(new_name, self.dialog.type_name, self.dialog.dataset, self.dialog.get_options())
		

class FavLoadPlugin(vaex.ui.plugin.PluginDataset):
	name="favorites"
	def __init__(self):
		pass