__author__ = 'maartenbreddels'

import numpy as np
import gavi.vaex.plugin
from gavi.vaex.qt import *
import gavi.vaex.plot_windows
#from gavi.icons import iconfile
#import matplotlib.widgets
#import functools
import gavi.logging
import gavi.vaex.undo as undo
from gavi.vaex import widgets

logger = gavi.logging.getLogger("plugin.transferfunction")


class TransferFunctionPlugin(gavi.vaex.plugin.PluginPlot):
	name = "transferfunction"
	def __init__(self, dialog):
		super(TransferFunctionPlugin, self).__init__(dialog)
		dialog.plug_page(self.plug_page, "Transfer function", 2.5, 1.0)

	@staticmethod
	def useon(dialog_class):
		return issubclass(dialog_class, gavi.vaex.plot_windows.VolumeRenderingPlotDialog)

	def plug_page(self, page):
		layout = self.layout = QtGui.QGridLayout()
		self.widget_volume = self.dialog.widget_volume
		page.setLayout(self.layout)
		layout.setSpacing(0)
		layout.setContentsMargins(0,0,0,0)
		layout.setAlignment(QtCore.Qt.AlignTop)

		self.button_orbit = QtGui.QPushButton("orbit", page)
		self.button_orbit.setCheckable(True)
		self.button_orbit.setAutoDefault(False)
		layout.addWidget(self.button_orbit, 0, 1)
		def on_toggle_orbit(checked, button=self.button_orbit):
			if checked: #button.isChecked
				self.widget_volume.orbit_start()
			else:
				self.widget_volume.orbit_stop()
		self.button_orbit.toggled.connect(on_toggle_orbit)
		layout.setRowMinimumHeight(1, 8)

		self.tool = widgets.HistogramAndTransfer(page, self.dialog.colormap)
		#self.tool.setMinimumHeight(100)
		layout.addWidget(self.tool, 2, 1)


		self.slider_transfer_functions_mean = []
		self.slider_transfer_functions_signa = []
		self.slider_transfer_functions_opacity = []

		row = 3
		for i in range(self.tool.function_count):

			#label = QtGui.QLabel("", page)
			#layout.addWidget(label, row, 0, 1, 2)
			#row += 1

			label_mean = QtGui.QLabel("mean: ", page)
			label_mean_value = QtGui.QLabel("", page)
			label_sigma = QtGui.QLabel("sigma: ", page)
			label_sigma_value = QtGui.QLabel("", page)
			label_opacity = QtGui.QLabel("opacity: ", page)
			label_opacity_value = QtGui.QLabel("", page)
			#label2 = QtGui.QLabel("sigma", page)

			slider_mean = QtGui.QSlider(page)
			slider_mean.setOrientation(QtCore.Qt.Horizontal)
			slider_sigma = QtGui.QSlider(page)
			slider_sigma.setOrientation(QtCore.Qt.Horizontal)
			slider_opacity = QtGui.QSlider(page)
			slider_opacity.setOrientation(QtCore.Qt.Horizontal)

			layout.addWidget(label_mean, row, 0)
			layout.addWidget(slider_mean, row, 1)
			layout.addWidget(label_mean_value, row, 2)
			row += 1

			layout.addWidget(label_sigma, row, 0)
			layout.addWidget(slider_sigma, row, 1)
			layout.addWidget(label_sigma_value, row, 2)
			row += 1

			layout.addWidget(label_opacity, row, 0)
			layout.addWidget(slider_opacity, row, 1)
			layout.addWidget(label_opacity_value, row, 2)
			row += 1

			self.slider_transfer_functions_mean.append(slider_mean)
			slider_mean.setRange(0, 1000)
			slider_sigma.setRange(0, 1000)
			slider_opacity.setRange(0, 1000)
			def update_text(i=i, slider_mean=slider_mean, slider_sigma=slider_sigma, slider_opacity=slider_opacity, label_mean_value=label_mean_value, label_sigma_value=label_sigma_value, label_opacity_value=label_opacity_value):
				#label.setText("mean/sigma: {0:<0.3f}/{1:.3g} opacity: {2:.3g}".format(self.tool.function_means[i], self.tool.function_sigmas[i], self.tool.function_opacities[i]))
				label_mean_value.setText(" {0:<0.3f}".format(self.tool.function_means[i]))
				label_sigma_value.setText(" {0:<0.3f}".format(self.tool.function_sigmas[i]))
				label_opacity_value.setText(" {0:<0.3f}".format(self.tool.function_opacities[i]))
			def on_mean_change(index, i=i, update_text=update_text):
				value = index/1000.
				self.tool.function_means[i] = value
				self.widget_volume.function_means[i] = value
				self.widget_volume.update()
				update_text()
				self.tool.update()
			def on_sigma_change(index, i=i, update_text=update_text):
				value = index/1000.
				value = 10**((value-1)*3)
				self.tool.function_sigmas[i] = value
				self.widget_volume.function_sigmas[i] = value
				self.widget_volume.update()
				update_text()
				self.tool.update()
			def on_opacity_change(index, i=i, update_text=update_text):
				value = index/1000.
				value = 10**((value-1)*3)
				self.tool.function_opacities[i] = value
				self.widget_volume.function_opacities[i] = value
				self.widget_volume.update()
				update_text()
				self.tool.update()
			self.widget_volume.function_opacities[i] = self.tool.function_opacities[i]
			self.widget_volume.function_sigmas[i] = self.tool.function_sigmas[i]
			self.widget_volume.function_means[i] = self.tool.function_means[i]
			slider_mean.valueChanged.connect(on_mean_change)
			slider_sigma.valueChanged.connect(on_sigma_change)
			slider_opacity.valueChanged.connect(on_opacity_change)
			update_text()
			slider_mean.setValue(int(self.tool.function_means[i] * 1000))
			#slider_sigma.setValue(int(self.tool.function_sigmas[i] * 2000))
			slider_opacity.setValue(int((np.log10(self.tool.function_sigmas[i])/3+1) * 1000))
			slider_opacity.setValue(int((np.log10(self.tool.function_opacities[i])/3+1) * 1000))

			layout.setRowMinimumHeight(row, 8)
			row += 1

		label_brightness = QtGui.QLabel("brightness: ", page)
		label_brightness_value = QtGui.QLabel("", page)
		slider_brightness = QtGui.QSlider(page)
		slider_brightness.setOrientation(QtCore.Qt.Horizontal)
		slider_brightness.setRange(0, 1000)

		layout.addWidget(label_brightness, row, 0)
		layout.addWidget(slider_brightness, row, 1)
		layout.addWidget(label_brightness_value, row, 2)
		row += 1

		def update_text_brightness(i=i, label_brightness_value=label_brightness_value):
			#label.setText("mean/sigma: {0:<0.3f}/{1:.3g} opacity: {2:.3g}".format(self.tool.function_means[i], self.tool.function_sigmas[i], self.tool.function_opacities[i]))
			label_brightness_value.setText(" {0:<0.3f}".format(self.widget_volume.brightness))
		def on_brightness_change(index, update_text_brightness=update_text_brightness):
			value = 10**(2*(index/1000.*2-1.))
			print value
			self.widget_volume.brightness = value
			self.widget_volume.update()
			update_text_brightness()
			self.tool.update()
		slider_brightness.setValue(int((np.log10(self.widget_volume.brightness)/2.+1)/2.*1000))
		update_text_brightness()
		slider_brightness.valueChanged.connect(on_brightness_change)

		layout.setRowMinimumHeight(row, 8)
		row += 1

		label_min_level = QtGui.QLabel("min_level: ", page)
		label_min_level_value = QtGui.QLabel("", page)
		slider_min_level = QtGui.QSlider(page)
		slider_min_level.setOrientation(QtCore.Qt.Horizontal)
		slider_min_level.setRange(0, 1000)

		layout.addWidget(label_min_level, row, 0)
		layout.addWidget(slider_min_level, row, 1)
		layout.addWidget(label_min_level_value, row, 2)
		row += 1

		def update_text_min_level(i=i, label_min_level_value=label_min_level_value):
			#label.setText("mean/sigma: {0:<0.3f}/{1:.3g} opacity: {2:.3g}".format(self.tool.function_means[i], self.tool.function_sigmas[i], self.tool.function_opacities[i]))
			label_min_level_value.setText(" {0:<0.3f}".format(self.widget_volume.min_level))
		self.handling_nested_min_max_level = False
		def on_min_level_change(index, update_text_min_level=update_text_min_level):
			value = index/1000.
			print value
			self.widget_volume.min_level = value
			if (self.handling_nested_min_max_level is False) and (QtGui.QApplication.keyboardModifiers() == QtCore.Qt.AltModifier) or (QtGui.QApplication.keyboardModifiers() == QtCore.Qt.ControlModifier):
				self.handling_nested_min_max_level = True
				try:
					delta = value - self.previous_volume_rendering_min_level
					max_level = self.previous_volume_rendering_max_level + delta
					slider_max_level.setValue(int(max_level*1000))
					self.previous_volume_rendering_max_level = max_level
				finally:
					self.handling_nested_min_max_level = False

			self.previous_volume_rendering_min_level = value
			self.widget_volume.update()
			update_text_min_level()
			self.tool.update()

		slider_min_level.setValue(int(self.widget_volume.min_level*1000))
		update_text_min_level()
		slider_min_level.valueChanged.connect(on_min_level_change)



		label_max_level = QtGui.QLabel("max_level: ", page)
		label_max_level_value = QtGui.QLabel("", page)
		slider_max_level = QtGui.QSlider(page)
		slider_max_level.setOrientation(QtCore.Qt.Horizontal)
		slider_max_level.setRange(0, 1000)

		layout.addWidget(label_max_level, row, 0)
		layout.addWidget(slider_max_level, row, 1)
		layout.addWidget(label_max_level_value, row, 2)
		row += 1

		def update_text_max_level(i=i, label_max_level_value=label_max_level_value):
			#label.setText("mean/sigma: {0:<0.3f}/{1:.3g} opacity: {2:.3g}".format(self.tool.function_means[i], self.tool.function_sigmas[i], self.tool.function_opacities[i]))
			label_max_level_value.setText(" {0:<0.3f}".format(self.widget_volume.max_level))
		def on_max_level_change(index, update_text_max_level=update_text_max_level):
			value = index/1000.
			print value
			self.widget_volume.max_level = value
			if (self.handling_nested_min_max_level is False) and (QtGui.QApplication.keyboardModifiers() == QtCore.Qt.AltModifier) or (QtGui.QApplication.keyboardModifiers() == QtCore.Qt.ControlModifier):
				self.handling_nested_min_max_level = True
				try:
					delta = value - self.previous_volume_rendering_max_level
					min_level = self.previous_volume_rendering_min_level + delta
					slider_min_level.setValue(int(min_level*1000))
					self.previous_volume_rendering_min_level = min_level
				finally:
					self.handling_nested_min_max_level = False

			self.widget_volume.update()
			update_text_max_level()
			self.previous_volume_rendering_max_level = value
			self.tool.update()
		slider_max_level.setValue(int(self.widget_volume.max_level*1000))
		update_text_max_level()
		slider_max_level.valueChanged.connect(on_max_level_change)

		self.previous_volume_rendering_min_level = self.widget_volume.min_level
		self.previous_volume_rendering_max_level = self.widget_volume.max_level


