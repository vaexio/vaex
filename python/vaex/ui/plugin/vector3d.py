__author__ = 'maartenbreddels'

import numpy as np

import vaex.ui.plugin
from vaex.ui.qt import *
import vaex.ui.plot_windows
import logging


logger = logging.getLogger("plugin.zoom")


@vaex.ui.plugin.pluginclass
class Vector3dPlugin(vaex.ui.plugin.PluginLayer):
	name = "vector3"
	def __init__(self, parent, layer):
		super(Vector3dPlugin, self).__init__(parent, layer)
		layer.plug_page(self.plug_page, "Vector field", 2., 2.)

	@staticmethod
	def useon(dialog_class):
		return issubclass(dialog_class, vaex.ui.plot_windows.VolumeRenderingPlotDialog)

	def plug_page(self, page):
		existing_layout = page.layout()
		if isinstance(existing_layout, QtGui.QGridLayout):
			layout = existing_layout
		else:
			#raise NotImplementedError("expected different layout")
			self.layout = layout = QtGui.QGridLayout()
			existing_layout.addLayout(self.layout)
		page.setLayout(self.layout)
		layout.setSpacing(0)
		layout.setContentsMargins(0,0,0,0)
		layout.setAlignment(QtCore.Qt.AlignTop)

		row = 0


		if 0:
			def setter(value):
				self.layer.plot_window.widget_volume.draw_vectors = value
				self.layer.plot_window.widget_volume.update()
			self.vector3d_show_checkbox = self.layer.plot_window.create_checkbox(page, "show 3d vectors", lambda : self.layer.plot_window.widget_volume.draw_vectors, setter)
			layout.addWidget(self.vector3d_show_checkbox, row, 1)
			row += 1


		self.vector3d_show_checkbox = Checkbox(page, "show 3d vectors", getter=attrgetter(self.layer.plot_window.widget_volume, "draw_vectors"), setter=attrsetter(self.layer.plot_window.widget_volume, "draw_vectors"), update=self.layer.plot_window.widget_volume.update)
		row = self.vector3d_show_checkbox.add_to_grid_layout(row, layout)



		self.vector3d_auto_scale_checkbox = Checkbox(page, "auto scale 3d vectors", getter=attrgetter(self.layer.plot_window.widget_volume, "vector3d_auto_scale"), setter=attrsetter(self.layer.plot_window.widget_volume, "vector3d_auto_scale"), update=self.layer.plot_window.widget_volume.update)
		row = self.vector3d_auto_scale_checkbox.add_to_grid_layout(row, layout)


		self.vector3d_min_level_label = Slider(page, "min level 3d", 0, 1, 1000, getter=attrgetter(self.layer.plot_window.widget_volume, "min_level_vector3d"),
		                                     setter=attrsetter(self.layer.plot_window.widget_volume, "min_level_vector3d"), update=self.layer.plot_window.widget_volume.update)
		row = self.vector3d_min_level_label.add_to_grid_layout(row, layout)

		self.vector3d_max_level_label = Slider(page, "max level 3d", 0, 1, 1000, getter=attrgetter(self.layer.plot_window.widget_volume, "max_level_vector3d"),
		                                     setter=attrsetter(self.layer.plot_window.widget_volume, "max_level_vector3d"), update=self.layer.plot_window.widget_volume.update)
		row = self.vector3d_max_level_label.add_to_grid_layout(row, layout)
		return

		def setter(value):
			self.layer.plot_window.widget_volume.vector3d_auto_scale = value
			self.layer.plot_window.widget_volume.update()
		self.vector3d_auto_scale_checkbox = self.layer.plot_window.create_checkbox(page, "auto scale 3d vectors", lambda : self.layer.plot_window.widget_volume.vector3d_auto_scale, setter)
		layout.addWidget(self.vector3d_auto_scale_checkbox, row, 1)
		row += 1


		def setter(value):
			self.layer.plot_window.widget_volume.min_level_vector3d = value
			self.layer.plot_window.widget_volume.update()
		self.vector3d_min_level_label, self.vector3d_min_level_slider, self.vector3d_min_level_value_label =\
				self.layer.plot_window.create_slider(page, "min level: ", 0., 1., lambda : self.layer.plot_window.widget_volume.min_level_vector3d, setter)
		layout.addWidget(self.vector3d_min_level_label, row, 0)
		layout.addWidget(self.vector3d_min_level_slider, row, 1)
		layout.addWidget(self.vector3d_min_level_value_label, row, 2)
		row += 1

		def setter(value):
			self.layer.plot_window.widget_volume.max_level_vector3d = value
			self.layer.plot_window.widget_volume.update()
		self.vector3d_max_level_label, self.vector3d_max_level_slider, self.vector3d_max_level_value_label =\
				self.layer.plot_window.create_slider(page, "max level: ", 0., 1., lambda : self.layer.plot_window.widget_volume.max_level_vector3d, setter)
		layout.addWidget(self.vector3d_max_level_label, row, 0)
		layout.addWidget(self.vector3d_max_level_slider, row, 1)
		layout.addWidget(self.vector3d_max_level_value_label, row, 2)
		row += 1

		def setter(value):
			self.layer.plot_window.widget_volume.vector3d_scale = value
			self.layer.plot_window.widget_volume.update()
		self.vector3d_scale_level_label, self.vector3d_scale_level_slider, self.vector3d_scale_level_value_label =\
				self.layer.plot_window.create_slider(page, "scale: ", 1./20, 20., lambda : self.layer.plot_window.widget_volume.vector3d_scale, setter, format=" {0:>05.2f}", transform=lambda x: 10**x, inverse=lambda x: np.log10(x))
		layout.addWidget(self.vector3d_scale_level_label, row, 0)
		layout.addWidget(self.vector3d_scale_level_slider, row, 1)
		layout.addWidget(self.vector3d_scale_level_value_label, row, 2)
		row += 1

		layout.setRowMinimumHeight(row, 8)
		row += 1

