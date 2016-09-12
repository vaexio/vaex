__author__ = 'maartenbreddels'

import numpy as np

import vaex.ui.plugin
from vaex.ui.qt import *
import vaex.ui.plot_windows



#from vaex.ui.icons import iconfile
#import matplotlib.widgets
#import functools
import logging
from vaex.ui import widgets

logger = logging.getLogger("plugin.transferfunction")


@vaex.ui.plugin.pluginclass
class TransferFunctionPlugin(vaex.ui.plugin.PluginLayer):
	name = "transferfunction"
	def __init__(self, parent, layer):
		super(TransferFunctionPlugin, self).__init__(parent, layer)
		layer.plug_page(self.plug_page, "Transfer function", 2.5, 1.0)
		self.properties = [] # list of names with set_<name> and get_<name>

	@staticmethod
	def useon(dialog_class):
		return issubclass(dialog_class, vaex.ui.plot_windows.VolumeRenderingPlotDialog)

	def plug_page(self, page):
		layout = self.layout = QtGui.QGridLayout()
		self.widget_volume = self.layer.plot_window.widget_volume
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

		self.tool = widgets.HistogramAndTransfer(page, self.layer.state.colormap)
		#self.tool.setMinimumHeight(100)
		layout.addWidget(self.tool, 2, 1)


		self.slider_transfer_functions_mean = []
		self.slider_transfer_functions_signa = []
		self.slider_transfer_functions_opacity = []

		row = 3

		self.tool.function_means[:] = eval(self.layer.options.get("tf_means", str(self.tool.function_means)))
		self.tool.function_opacities[:] = eval(self.layer.options.get("tf_opacities", str(self.tool.function_opacities)))
		self.tool.function_sigmas[:] = eval(self.layer.options.get("tf_sigmas", str(self.tool.function_sigmas)))
		print(("Set opacities", self.tool.function_opacities[:]))
		#dsa
		#self.widget_volume.function_opacities[i] = self.tool.function_opacities[i]
		#self.widget_volume.function_sigmas[i] = self.tool.function_sigmas[i]
		#self.widget_volume.function_means[i] = self.tool.function_means[i]

		for i in range(self.tool.function_count):

			#label = QtGui.QLabel("", page)
			#layout.addWidget(label, row, 0, 1, 2)
			#row += 1
			def setter(value, update=True, i=i):
				self.tool.function_means[i] = value
				self.widget_volume.function_means[i] = value
				if update:
					self.widget_volume.update()
				self.tool.update() # always execute
			def getter(i=i):
				return self.tool.function_means[i]
			#self.widget_volume.ambient_coefficient = eval(self.dialog.options.get("ambient", str(getter())))
			label, slider, label_value = self.make_slider(page, "mean_%d" % i, 0., 1., 1000, "{0:<0.3f}", getter, setter)
			layout.addWidget(label, row, 0)
			layout.addWidget(slider, row, 1)
			layout.addWidget(label_value, row, 2)
			row += 1

		layout.setRowMinimumHeight(row, 8)
		row += 1


		for i in range(self.tool.function_count):

			#label = QtGui.QLabel("", page)
			#layout.addWidget(label, row, 0, 1, 2)
			#row += 1
			def setter(value, update=True, i=i):
				self.tool.function_sigmas[i] = value
				self.widget_volume.function_sigmas[i] = value
				if update:
					self.widget_volume.update()
				self.tool.update() # always execute
			def getter(i=i):
				return self.tool.function_sigmas[i]
			#self.widget_volume.ambient_coefficient = eval(self.dialog.options.get("ambient", str(getter())))
			label, slider, label_value = self.make_slider(page, "sigma_%d" % i, 0.0001, 1., 1000, "{0:<0.3f}", getter, setter, transform=lambda x: 10**x, inverse=lambda x: np.log10(x))
			layout.addWidget(label, row, 0)
			layout.addWidget(slider, row, 1)
			layout.addWidget(label_value, row, 2)
			row += 1

		layout.setRowMinimumHeight(row, 8)
		row += 1

		for i in range(self.tool.function_count):

			#label = QtGui.QLabel("", page)
			#layout.addWidget(label, row, 0, 1, 2)
			#row += 1
			def setter(value, update=True, i=i):
				self.tool.function_opacities[i] = value
				self.widget_volume.function_opacities[i] = value
				if update:
					self.widget_volume.update()
				self.tool.update() # always execute
			def getter(i=i):
				return self.tool.function_opacities[i]
			#self.widget_volume.ambient_coefficient = eval(self.dialog.options.get("ambient", str(getter())))
			label, slider, label_value = self.make_slider(page, "opacity_%d" % i, 0.0001, 1., 1000, "{0:<0.3f}", getter, setter, transform=lambda x: 10**x, inverse=lambda x: np.log10(x))
			layout.addWidget(label, row, 0)
			layout.addWidget(slider, row, 1)
			layout.addWidget(label_value, row, 2)
			row += 1


		def setter(value):
			self.widget_volume.brightness = value
			self.widget_volume.update()
			#self.tool.update()
		def getter():
			return self.widget_volume.brightness
		self.widget_volume.brightness = eval(self.layer.options.get("brightness", str(getter())))
		label, slider, label_value = self.make_slider(page, "brightness", 0.1, 5., 1000, "{0:<0.3f}", getter, setter)
		layout.addWidget(label, row, 0)
		layout.addWidget(slider, row, 1)
		layout.addWidget(label_value, row, 2)
		row += 1


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

		self.properties.append("min_level")
		def set_min_level(value, update=True):
			self.widget_volume.min_level = value
			slider_min_level.setValue(int(value*1000))
			if update:
				self.widget_volume.update()
				update_text_min_level()
				self.tool.update()
		self.set_min_level = set_min_level
		self.get_min_level = lambda: self.widget_volume.min_level

		def update_text_min_level(i=i, label_min_level_value=label_min_level_value):
			#label.setText("mean/sigma: {0:<0.3f}/{1:.3g} opacity: {2:.3g}".format(self.tool.function_means[i], self.tool.function_sigmas[i], self.tool.function_opacities[i]))
			label_min_level_value.setText(" {0:<0.3f}".format(self.widget_volume.min_level))
		self.handling_nested_min_max_level = False
		def on_min_level_change(index, update_text_min_level=update_text_min_level):
			value = index/1000.
			print(value)
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




		self.properties.append("max_level")
		def set_max_level(value, update=True):
			self.widget_volume.max_level = value
			slider_max_level.setValue(int(value*1000))
			if update:
				self.widget_volume.update()
				update_text_max_level()
				self.tool.update()
		self.set_max_level = set_max_level
		self.get_max_level = lambda: self.widget_volume.max_level

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
			print(value)
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

		def setter(value):
			self.widget_volume.depth_peel = value
			self.widget_volume.update()
			#self.tool.update()
		def getter():
			return self.widget_volume.depth_peel
		self.widget_volume.depth_peel = eval(self.layer.options.get("depth_peel", str(getter())))
		label, slider, label_value = self.make_slider(page, "depth_peel", 0., 1., 1000, "{0:<0.3f}", getter, setter)
		layout.addWidget(label, row, 0)
		layout.addWidget(slider, row, 1)
		layout.addWidget(label_value, row, 2)
		row += 1


		layout.setRowMinimumHeight(row, 8)
		row += 1


		def setter(value):
			self.widget_volume.ambient_coefficient = value
			self.widget_volume.update()
			#self.tool.update()
		def getter():
			return self.widget_volume.ambient_coefficient
		self.widget_volume.ambient_coefficient = eval(self.layer.options.get("ambient", str(getter())))
		label, slider, label_value = self.make_slider(page, "ambient", 0., 1., 1000, "{0:<0.3f}", getter, setter)
		layout.addWidget(label, row, 0)
		layout.addWidget(slider, row, 1)
		layout.addWidget(label_value, row, 2)
		row += 1

		def setter(value):
			self.widget_volume.diffuse_coefficient = value
			self.widget_volume.update()
			#self.tool.update()
		def getter():
			return self.widget_volume.diffuse_coefficient
		self.widget_volume.diffuse_coefficient = eval(self.layer.options.get("diffuse", str(getter())))
		label, slider, label_value = self.make_slider(page, "diffuse", 0., 1., 1000, "{0:<0.3f}", getter, setter)
		layout.addWidget(label, row, 0)
		layout.addWidget(slider, row, 1)
		layout.addWidget(label_value, row, 2)
		row += 1

		def setter(value):
			self.widget_volume.specular_coefficient = value
			self.widget_volume.update()
			#self.tool.update()
		def getter():
			return self.widget_volume.specular_coefficient
		self.widget_volume.specular_coefficient = eval(self.layer.options.get("specular", str(getter())))
		label, slider, label_value = self.make_slider(page, "specular",  0., 1., 1000, "{0:<0.3f}", getter, setter)
		layout.addWidget(label, row, 0)
		layout.addWidget(slider, row, 1)
		layout.addWidget(label_value, row, 2)
		row += 1

		def setter(value):
			self.widget_volume.specular_exponent = value
			self.widget_volume.update()
			#self.tool.update()
		def getter():
			return self.widget_volume.specular_exponent
		self.widget_volume.specular_exponent = eval(self.layer.options.get("specular_n", str(getter())))
		label, slider, label_value = self.make_slider(page, "specular_n", 0.1, 10., 1000, "{0:<0.3f}", getter, setter)
		layout.addWidget(label, row, 0)
		layout.addWidget(slider, row, 1)
		layout.addWidget(label_value, row, 2)
		row += 1


		layout.setRowMinimumHeight(row, 8)
		row += 1

		def setter(value, update=True, i=i):
			self.widget_volume.foreground_opacity = value
			if update:
				self.widget_volume.update()
		def getter(i=i):
			return self.widget_volume.foreground_opacity
		#self.widget_volume.ambient_coefficient = eval(self.dialog.options.get("ambient", str(getter())))
		label, slider, label_value = self.make_slider(page, "opacity_fg", 0.0001, 10., 1000, "{0:<0.3f}", getter, setter, transform=lambda x: 10**x, inverse=lambda x: np.log10(x))
		layout.addWidget(label, row, 0)
		layout.addWidget(slider, row, 1)
		layout.addWidget(label_value, row, 2)
		row += 1


		def setter(value, update=True, i=i):
			self.widget_volume.background_opacity = value
			if update:
				self.widget_volume.update()
		def getter(i=i):
			return self.widget_volume.background_opacity
		#self.widget_volume.ambient_coefficient = eval(self.dialog.options.get("ambient", str(getter())))
		label, slider, label_value = self.make_slider(page, "opacity_bg", 0.0001, 10., 1000, "{0:<0.3f}", getter, setter, transform=lambda x: 10**x, inverse=lambda x: np.log10(x))
		layout.addWidget(label, row, 0)
		layout.addWidget(slider, row, 1)
		layout.addWidget(label_value, row, 2)
		row += 1



	def make_slider(self, parent, label_text, value_min, value_max, value_steps, format, getter, setter, name=None, transform=lambda x: x, inverse=lambda x: x):
		if name is None:
			name = label_text
		self.properties.append(name)
		label = QtGui.QLabel(label_text, parent)
		label_value = QtGui.QLabel(label_text, parent)
		slider = QtGui.QSlider(parent)
		slider.setOrientation(QtCore.Qt.Horizontal)
		slider.setRange(0, value_steps)

		def wrap_setter(value, update=True):
			slider.setValue((inverse(value) - inverse(value_min))/(inverse(value_max) - inverse(value_min)) * value_steps)
			setter(value)
		# auto getter and setter
		setattr(self, "get_" + label_text, getter)
		setattr(self, "set_" + label_text, wrap_setter)

		def update_text():
			#label.setText("mean/sigma: {0:<0.3f}/{1:.3g} opacity: {2:.3g}".format(self.tool.function_means[i], self.tool.function_sigmas[i], self.tool.function_opacities[i]))
			label_value.setText(format.format(getter()))
		def on_change(index, slider=slider):
			value = index/float(value_steps) * (inverse(value_max) - inverse(value_min)) + inverse(value_min)
			print((label_text, "set to", value))
			setter(transform(value))
			update_text()
		slider.setValue((inverse(getter()) - inverse(value_min))/(inverse(value_max) - inverse(value_min)) * value_steps)
		update_text()
		slider.valueChanged.connect(on_change)
		return label, slider, label_value






	def get_options(self):
		options = {
		}
		for name in self.properties:
			getter = getattr(self, "get_" + name)
			options[name] = getter()
		return options

	def apply_options(self, options):
		for name in self.properties:
			if name in options:
				setter = getattr(self, "set_" + name)
				setter(options[name], update=False)
		pass
