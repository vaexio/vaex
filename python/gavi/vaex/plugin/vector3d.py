__author__ = 'maartenbreddels'


import gavi.vaex.plugin
from gavi.vaex.qt import *
import gavi.vaex.plot_windows
from gavi.icons import iconfile
import matplotlib.widgets
import functools
import gavi.logging
import gavi.vaex.undo as undo

logger = gavi.logging.getLogger("plugin.zoom")


class Vector3dPlugin(gavi.vaex.plugin.PluginPlot):
	name = "vector3"
	def __init__(self, dialog):
		super(Vector3dPlugin, self).__init__(dialog)
		dialog.plug_page(self.plug_page, "Vector3d", 2.5, 1.2)

	@staticmethod
	def useon(dialog_class):
		return issubclass(dialog_class, gavi.vaex.plot_windows.VolumeRenderingPlotDialog)

	def plug_page(self, page):
		layout = self.layout = QtGui.QGridLayout()
		page.setLayout(self.layout)
		layout.setSpacing(0)
		layout.setContentsMargins(0,0,0,0)
		layout.setAlignment(QtCore.Qt.AlignTop)

		row = 0

		self.vector3d_show_checkbox = QtGui.QCheckBox("show 3d vectors", page)
		layout.addWidget(self.vector3d_show_checkbox, row, 1)
		row += 1

		label_vector3d_min_level = QtGui.QLabel("min level: ", page)
		label_vector3d_min_level_value = QtGui.QLabel("", page)
		slider_vector3d_min_level = QtGui.QSlider(page)
		slider_vector3d_min_level.setOrientation(QtCore.Qt.Horizontal)
		slider_vector3d_min_level.setRange(0, 1000)

		layout.addWidget(label_vector3d_min_level, row, 0)
		layout.addWidget(slider_vector3d_min_level, row, 1)
		layout.addWidget(label_vector3d_min_level_value, row, 2)
		row += 1


		def update_text_vector3d_min_level(label_vector3d_min_level_value=label_vector3d_min_level_value):
			#label.setText("mean/sigma: {0:.3g}/{1:.3g} opacity: {2:.3g}".format(self.tool.function_means[i], self.tool.function_sigmas[i], self.tool.function_opacities[i]))
			label_vector3d_min_level_value.setText(" {0:.3g}".format(self.dialog.widget_volume.min_level_vector3d))
		def on_vector3d_min_level_change(index, update_text_vector3d_min_level=update_text_vector3d_min_level):
			value = index / 1000.
			print value
			self.dialog.widget_volume.min_level_vector3d = value
			self.dialog.widget_volume.update()
			update_text_vector3d_min_level()
			#self.dialog.tool.update()
		#slider_vector3d_min_level.setValue(int((np.log10(self.dialog.widget_volume.brightness)/2.+1)/2.*1000))
		slider_vector3d_min_level.setValue(int(self.dialog.widget_volume.min_level_vector3d*1000))
		update_text_vector3d_min_level()
		slider_vector3d_min_level.valueChanged.connect(on_vector3d_min_level_change)

		label_vector3d_max_level = QtGui.QLabel("max level: ", page)
		label_vector3d_max_level_value = QtGui.QLabel("", page)
		slider_vector3d_max_level = QtGui.QSlider(page)
		slider_vector3d_max_level.setOrientation(QtCore.Qt.Horizontal)
		slider_vector3d_max_level.setRange(0, 1000)

		layout.addWidget(label_vector3d_max_level, row, 0)
		layout.addWidget(slider_vector3d_max_level, row, 1)
		layout.addWidget(label_vector3d_max_level_value, row, 2)
		row += 1

		def update_text_vector3d_max_level(label_vector3d_max_level_value=label_vector3d_max_level_value):
			#label.setText("mean/sigma: {0:.3g}/{1:.3g} opacity: {2:.3g}".format(self.tool.function_means[i], self.tool.function_sigmas[i], self.tool.function_opacities[i]))
			label_vector3d_max_level_value.setText(" {0:0.3g}".format(self.dialog.widget_volume.max_level_vector3d))
		def on_vector3d_max_level_change(index, update_text_vector3d_max_level=update_text_vector3d_max_level):
			value = index/1000.
			print value
			self.dialog.widget_volume.max_level_vector3d = value
			self.dialog.widget_volume.update()
			update_text_vector3d_max_level()
			#self.tool.update()
		#slider_vector3d_max_level.setValue(int((np.log10(self.dialog.widget_volume.brightness)/2.+1)/2.*1000))
		slider_vector3d_max_level.setValue(int(self.dialog.widget_volume.max_level_vector3d*1000))
		update_text_vector3d_max_level()
		slider_vector3d_max_level.valueChanged.connect(on_vector3d_max_level_change)

		layout.setRowMinimumHeight(row, 8)
		row += 1

