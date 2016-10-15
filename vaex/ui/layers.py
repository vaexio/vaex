import collections
import operator
import vaex.vaexfast
import threading
import matplotlib
import numpy as np
import scipy.ndimage
import matplotlib.colors
import traceback

import vaex
import vaex.delayed
import vaex.ui.storage
import vaex.ui.undo
import vaex.ui.colormaps
import vaex.grids
from vaex.ui.icons import iconfile
import vaex.utils
import vaex.promise
import vaex.ui.qt as dialogs

__author__ = 'maartenbreddels'

import copy
import functools
import time
from vaex.ui.qt import *
import logging
import astropy.units
try:
	import healpy
except:
	healpy = None
	
#from attrdict import AttrDict
from .plot_windows import AttrDict


logger = logging.getLogger("vaex.ui.layer")

storage_expressions = vaex.ui.storage.Storage("expressions")


class multilayer_attrsetter(object):
	def __init__(self, layer, name):
		self.layer = layer
		self.name = name

	def __call__(self, value):
		if QtGui.QApplication.keyboardModifiers()  == QtCore.Qt.ShiftModifier:
			for layer in self.layer.plot_window.layers:
				setattr(layer, self.name, value)
		else:
			setattr(self.layer, self.name, value)


#options.define_options("grid_size", int, validator=options.is_power_of_two)
class LinkButton(QtGui.QToolButton):
	def __init__(self, title, dataset, axis_index, parent):
		super(LinkButton, self).__init__(parent)
		self.setToolTip("link this axes with others (experimental and unstable)")
		self.plot = parent
		self.dataset = dataset
		self.axis_index = axis_index
		self.setText(title)
		#self.setAcceptDrops(True)
		#self.disconnect_icon = QtGui.QIcon(iconfile('network-disconnect-2'))
		#self.connect_icon = QtGui.QIcon(iconfile('network-connect-3'))
		self.disconnect_icon = QtGui.QIcon(iconfile('link_break'))
		self.connect_icon = QtGui.QIcon(iconfile('link'))
		#self.setIcon(self.disconnect_icon)

		#self.action_link_global = QtGui.QAction(self.connect_icon, '&Global link', self)
		#self.action_unlink = QtGui.QAction(self.connect_icon, '&Unlink', self)
		#self.menu = QtGui.QMenu()
		#self.menu.addAction(self.action_link_global)
		#self.menu.addAction(self.action_unlink)
		#self.action_link_global.triggered.connect(self.onLinkGlobal)
		self.setToolTip("Link or unlink axis. When an axis is linked, changing an axis (like zooming) will update all axis of plots that have the same (and linked) axis.")
		self.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
		self.setIcon(self.disconnect_icon)
		#self.setDefaultAction(self.action_link_global)
		self.setCheckable(True)
		self.setChecked(False)
		self.clicked.connect(self.onToggleLink)
		#self.setMenu(self.menu)
		self.link = None

	def onToggleLink(self):
		if self.isChecked():
			logger.debug("connected link")
			self.link = self.dataset.link(self.plot.expressions[self.axis_index], self)
			self.setIcon(self.connect_icon)
		else:
			logger.debug("disconnecting link")
			self.dataset.unlink(self.link, self)
			self.link = None
			self.setIcon(self.disconnect_icon)

	def onLinkGlobal(self):
		self.link = self.dataset.link(self.plot.expressions[self.axis_index], self)
		logger.debug("made global link: %r" % self.link)
		#self.parent.links[self.axis_index] = self.linkHandle

	def onChangeRangeShow(self, range_):
		logger.debug("received range show change for plot=%r, axis_index %r, range=%r" % (self.plot, self.axis_index, range_))
		self.plot.ranges_show[self.axis_index] = range_

	def onChangeRange(self, range_):
		logger.debug("received range change for plot=%r, axis_index %r, range=%r" % (self.plot, self.axis_index, range_))
		self.plot.ranges[self.axis_index] = range_

	def onCompute(self):
		logger.debug("received compute for plot=%r, axis_index %r" % (self.plot, self.axis_index))
		self.plot.compute()

	def onPlot(self):
		logger.debug("received plot command for plot=%r, axis_index %r" % (self.plot, self.axis_index))
		self.plot.plot()

	def onLinkLimits(self, min, max):
		self.plot.expressions[self.axis_index] = expression

	def onChangeExpression(self, expression):
		logger.debug("received change expression for plot=%r, axis_index %r, expression=%r" % (self.plot, self.axis_index, expression))
		self.plot.expressions[self.axis_index] = expression
		self.plot.axisboxes[self.axis_index].lineEdit().setText(expression)



	def _dragEnterEvent(self, e):
		print(e.mimeData())
		print(e.mimeData().text())
		if e.mimeData().hasFormat('text/plain'):
			e.accept()

		else:
			e.ignore()

	def dropEvent(self, e):
		position = e.pos()
		#self.button.move(position)
		print("do", e.mimeData().text())
		e.setDropAction(QtCore.Qt.MoveAction)
		e.accept()

	def _mousePressEvent(self, e):

			super(LinkButton, self).mousePressEvent(e)

			if e.button() == QtCore.Qt.LeftButton:
				print('press')

	def _mouseMoveEvent(self, e):
		if e.buttons() != QtCore.Qt.LeftButton:
			return

		mimeData = QtCore.QMimeData()

		drag = QtGui.QDrag(self)
		drag.setMimeData(mimeData)
		drag.setHotSpot(e.pos() - self.rect().topLeft())
		mimeData.setText("blaat")

		dropAction = drag.start(QtCore.Qt.MoveAction)


import vaex.dataset



class LayerTable(object):
	def __init__(self, plot_window, name, dataset, expressions, axis_names, options, figure, canvas, ranges_grid=None):
		"""
		:type tasks: list[Task]
		:type dataset: Dataset
		:type plot_window: PlotDialog
		"""
		self.plot_window = plot_window
		self.name = name
		self.dataset = dataset
		self.axis_names = axis_names

		self.state = AttrDict()

		self.state.ranges_grid = ranges_grid
		self.state.title = options.get("title")

		self.range_level = None
		self.options = options
		self.state.expressions = expressions
		self.dimensions = len(self.state.expressions)
		self.state.vector_expressions = [None,] * (1 if self.dimensions == 1 else 3)
		self.figure = figure
		self.canvas = canvas
		self.widget_build = False
		self.grid_vector = None

		self._can_plot = False # when every process went though ok, this is True
		self._needs_update = True


		self.widget = None # each layer has a widget, atm only a qt widget is implemented

		self.state.weight_expression = None
		self.state.show_disjoined = False
		self.state.dataset_path = self.dataset.path
		self.state.name = self.dataset.name

		self.compute_counter = 0
		self.sequence_index = 0
		self.state.alpha = float(self.options.get("alpha", "1."))
		self.state.style = options.get("style", "histogram")
		#self.color = self.options.get("color")
		self.level_min = 0.
		self.level_max = 1.
		#self.use_intensity = bool(self.options.get("use_intensity", True))

		self.coordinates_picked_row = None

		self.layer_slice_source = None # the layer we link to for slicing
		self.slice_axis = [] # list of booleans, which axis we listen to

		# we keep a list of vaex.dataset.Task, so that we can cancel, listen
		# to progress etc
		self.tasks = []
		self._task_signals = []


		self._histogram_counter = 0 # TODO: until we can cancel the server, we have to fix it with a counter

		self.state.colormap = "PaulT_plusmin" #"binary"
		self.state.colormap_vector = "binary"
		if "lim" in self.options:
			for i in range(self.dimensions):
				self.state.ranges_grid[i] = eval(self.options["lim"])
		if "ranges" in self.options:
			ranges = self.options["ranges"]
			if isinstance(self.options["ranges"], str):
				ranges = eval(ranges)
			for i in range(self.dimensions):
				self.state.ranges_grid[i] = ranges[i]
		if "xlim" in self.options:
			self.state.ranges_grid[0] = eval(self.options["xlim"])
		if "ylim" in self.options:
			self.state.ranges_grid[1] = eval(self.options["ylim"])
		if "zlim" in self.options:
			self.state.ranges_grid[2] = eval(self.options["zlim"])
		if "aspect" in self.options:
			self.aspect = eval(self.options["aspect"])
			self.action_aspect_lock_one.setChecked(True)
		if "compact" in self.options:
			value = self.options["compact"]
			if value in ["ultra", "+"]:
				self.action_mini_mode_ultra.trigger()
			else:
				self.action_mini_mode_normal.trigger()

		self.first_time = True

		self.state.show_disjoined = False # show p(x,y) as p(x)p(y)




		if self.state.ranges_grid is None:
			self.submit_job_minmax()

		#self.dataset.mask_listeners.append(self.onSelectMask)
		self.dataset.signal_selection_changed.connect(self.on_selection_changed)
		self.dataset.signal_column_changed.connect(self.on_column_changed)
		self.dataset.signal_variable_changed.connect(self.on_variable_changed)
		#self.dataset.signal_selection_changed.
		#self.dataset.row_selection_listeners.append(self.onSelectRow)
		self.dataset.signal_pick.connect(self.on_pick)
		self.dataset.signal_sequence_index_change.connect(self.on_sequence_changed)
		#self.dataset.serie_index_selection_listeners.append(self.onSerieIndexSelect)
		self.plot_density = self.plot_density_imshow
		self.signal_expression_change = vaex.events.Signal("expression_change")
		self.signal_plot_dirty = vaex.events.Signal("plot_dirty")
		self.signal_plot_update = vaex.events.Signal("plot_update")
		self.signal_needs_update = vaex.events.Signal("needs update")
		#self.dataset.signal_pick.connect(self.on)

	def __repr__(self):
		classname = self.__class__.__module__ + "." +self.__class__.__name__
		return "<%s(name=%r, expressions=%r)> instance at 0x%x" % (classname, self.name, self.state.expressions, id(self))

	def restore_state(self, state):
		logger.debug("restoring layer %r to state %r ", self, state)
		self.state = AttrDict(state)
		for dim in range(self.dimensions):
			logger.debug("set expression[%i] to %s", dim, self.state.expressions[dim])
			self.set_expression(self.state.expressions[dim], dim)
		for dim in range(self.vector_dimensions):
			logger.debug("set vector expression[%i] to %s", dim, self.state.vector_expressions[dim])
			self.set_vector_expression(self.state.vector_expressions[dim], dim)
		logger.debug("set weight expression to %s", dim, self.state.weight_expression)
		self.set_weight_expression(self.state.weight_expression)


		self.colorbar_checkbox.set_value(self.state.colorbar)
		for dim in range(self.dimensions):
			self.option_output_unit[dim].set_value(self.state.output_units[dim])
		self.option_label_x.set_value(self.state.labels[0])
		self.option_label_y.set_value(self.state.labels[1])

		logger.debug("remove history change")
		self.plot_window.queue_history_change(None)





	def flag_needs_update(self):
		self._needs_update = True
		self.signal_needs_update.emit()

	def get_needs_update(self):
		return self._needs_update

	@property
	def weight(self):
		"""Expression that is used for the weight"""
		return self.state.weight_expression

	@weight.setter
	def weight(self, value):
		logger.debug("setting self.state.weight_expression to %s" % value)
		self.set_weight_expression(value)
		#self.state.weight_expression = value
		#self.weight_box.lineEdit().setText(value)
		#self.plot_window.queue_update()
		#self.update()

	@weight.deleter
	def weight(self):
		self.state.weight_expression = None
		self.weight_box.lineEdit().setText("")
		#self.plot_window.queue_update()
		self.update()

	@property
	def x(self):
		"""x expression"""
		return self.state.expressions[0]

	@x.setter
	def x(self, value):
		logger.debug("setting self.state.expressions[0] to %s" % value)
		self.set_expression(value, 0)

	@property
	def y(self):
		"""y expression"""
		return self.state.expressions[1]

	@y.setter
	def y(self, value):
		logger.debug("setting self.state.expressions[1] to %s" % value)
		self.set_expression(value, 1)

	@property
	def z(self):
		"""y expression"""
		return self.state.expressions[2]

	@y.setter
	def z(self, value):
		logger.debug("setting self.state.expressions[2] to %s" % value)
		self.set_expression(value, 2)

	@property
	def vx(self):
		"""vector x expression"""
		return self.state.vector_expressions[0]

	@vx.setter
	def vx(self, value):
		logger.debug("setting self.state.vector_expressions[0] to %s" % value)
		self.set_vector_expression(value, 0)


	@property
	def vy(self):
		"""vector y expression"""
		return self.state.vector_expressions[1]

	@vy.setter
	def vy(self, value):
		logger.debug("setting self.state.vector_expressions[1] to %s" % value)
		self.set_vector_expression(value, 1)


	@property
	def vz(self):
		"""vector z expression"""
		return self.state.vector_expressions[2]

	@vz.setter
	def vz(self, value):
		logger.debug("setting self.state.vector_expressions[2] to %s" % value)
		self.set_vector_expression(value, 2)

	@property
	def amplitude(self):
		"""amplitude expression"""
		return self.amplitude_expression

	@amplitude.setter
	def amplitude(self, value):
		logger.debug("setting self.amplitude_expression to %s" % value)
		self.amplitude_expression = value
		self.amplitude_box.lineEdit().setText(value)
		#self.plot_window.queue_update()
		self.update()

	def set_range(self, min, max, dimension=0):
		#was_equal = list(self.plot_window.state.ranges_viewport[dimension]) == [min, max]
		if min is None or max is None:
			self.state.ranges_grid[dimension] = None
		else:
			self.state.ranges_grid[dimension] = [min, max]
		#self.plot_window.state.ranges_viewport[dimension] = list(self.state.ranges_grid[dimension])
		#self.plot_window.set_range(min, max, dimension=dimension)
		if dimension == 0:
			self.option_xrange.set_value((min, max), update=False)
		if dimension == 1:
			self.option_yrange.set_value((min, max), update=False)
		if dimension == 2:
			self.option_zrange.set_value((min, max), update=False)
		#return not was_equal

	def get_range(self, dimension=0):
		return list(self.state.ranges_grid[dimension]) if self.state.ranges_grid[dimension] is not None else None

	@property
	def xlim(self):
		"""vector z expression"""
		return self.get_range(0)

	@xlim.setter
	def xlim(self, value):
		vmin, vmax = value
		self.plot_window.set_range(vmin, vmax, 0)
		self.update()

	@property
	def ylim(self):
		"""vector z expression"""
		return self.get_range(1)

	@ylim.setter
	def ylim(self, value):
		vmin, vmax = value
		self.plot_window.set_range(vmin, vmax, 1)
		self.update()

	@property
	def zlim(self):
		"""vector z expression"""
		return self.get_range(2)

	@xlim.setter
	def zlim(self, value):
		vmin, vmax = value
		self.plot_window.set_range(vmin, vmax, 2)
		self.update()



	def removed(self):
		#self.dataset.mask_listeners.remove(self.onSelectMask)
		self.dataset.signal_selection_changed.disconnect(self.on_selection_changed)
		self.dataset.signal_pick.disconnect(self.on_pick)
		self.dataset.signal_sequence_index_change.disconnect(self.on_sequence_changed)
		#self.dataset.row_selection_listeners.remove(self.onSelectRow)
		#self.dataset.serie_index_selection_listeners.remove(self.onSerieIndexSelect)
		for plugin in self.plugins:
			plugin.clean_up()



	def create_grid_map(self, gridsize, use_selection):
		return {"counts":self.temp_grid, "weighted":None, "weightx":None, "weighty":None, "weightz":None}

	def create_grid_map_(self, gridsize, use_selection):
		locals = {}
		for name in list(self.grids.grids.keys()):
			grid = self.grids.grids[name]
			if name == "counts" or (grid.weight_expression is not None and len(grid.weight_expression) > 0):
				if grid.max_size >= gridsize:
					locals[name] = grid.get_data(gridsize, use_selection=use_selection, disjoined=self.plot_window.show_disjoined)
					#import vaex.kld
					#print("Mutual information", name,  gridsize, self.state.expressions, vaex.kld.mutual_information(locals[name]))
			else:
				locals[name] = None
		for d, name in zip(list(range(self.dimensions)), "xyzw"):
			width = self.state.ranges_grid[d][1] - self.state.ranges_grid[d][0]
			offset = self.state.ranges_grid[d][0]
			x = (np.arange(0, gridsize)+0.5)/float(gridsize) * width + offset
			locals[name] = x
		return locals

	def eval_amplitude(self, expression, locals):
		amplitude = None
		locals = dict(locals)
		if "gf" not in locals:
			locals["gf"] = scipy.ndimage.gaussian_filter
		counts = locals["counts"]
		if self.dimensions == 2:
			peak_columns = np.apply_along_axis(np.nanmax, 1, counts)
			peak_columns[peak_columns==0] = 1.
			peak_columns = peak_columns.reshape((1, -1))#.T
			locals["peak_columns"] = peak_columns


			sum_columns = np.apply_along_axis(np.nansum, 1, counts)
			sum_columns[sum_columns==0] = 1.
			sum_columns = sum_columns.reshape((1, -1))#.T
			locals["sum_columns"] = sum_columns

			peak_rows = np.apply_along_axis(np.nanmax, 0, counts)
			peak_rows[peak_rows==0] = 1.
			peak_rows = peak_rows.reshape((-1, 1))#.T
			locals["peak_rows"] = peak_rows

			sum_rows = np.apply_along_axis(np.nansum, 0, counts)
			sum_rows[sum_rows==0] = 1.
			sum_rows = sum_rows.reshape((-1, 1))#.T
			locals["sum_rows"] = sum_rows

		weighted = locals["weighted"]
		if weighted is None:
			locals["average"] = None
		else:
			average = weighted/counts
			average[counts==0] = np.nan
			locals["average"] = average
		globals = np.__dict__
		amplitude = eval(expression, globals, locals)
		return amplitude

	def error_dialog(self, widget, name, exception):
		dialogs.dialog_error(widget, "Error", "%s: %r" % (name, exception))
	def error_in_field(self, widget, name, exception):
		dialogs.dialog_error(widget, "Error in expression", "Invalid expression for field %s: %r" % (name, exception))
		#self.current_tooltip = QtGui.QToolTip.showText(widget.mapToGlobal(QtCore.QPoint(0, 0)), "Error: " + str(exception), widget)
		#self.current_tooltip = QtGui.QToolTip.showText(widget.mapToGlobal(QtCore.QPoint(0, 0)), "Error: " + str(exception), widget)

	def plot_scatter(self, axes_list):
		for ax in axes_list:
			# TODO: support multiple axes with the axis index
			x = self.dataset.evaluate(self.x)
			y = self.dataset.evaluate(self.y)
			ax.scatter(x, y, alpha=self.state.alpha, color=self.color)
			row = self.dataset.get_current_row()
			if row is not None:
				ax.scatter([x[row]], [y[row]], alpha=self.state.alpha, color=self.color_alt)

	def plot_schlegel(self, axes_list, stack_image):
		if not hasattr(self, "schlegel_map"):
			self.schlegel_map = healpy.read_map('data/lambda_sfd_ebv.fits', nest=False)
		xlim, ylim = self.plot_window.state.ranges_viewport
		phis = np.linspace(np.deg2rad(xlim[0]), np.deg2rad(xlim[1]), self.plot_window.state.grid_size)# + np.pi/2.
		thetas = np.pi-np.linspace(np.deg2rad(ylim[1]) + np.pi/2., np.deg2rad(ylim[0]) + np.pi/2., self.plot_window.state.grid_size)
		#phis = (np.linspace(0, 2*np.pi, 256) - np.pi) % (2*np.pi)
		thetas, phis = np.meshgrid(thetas, phis)

		pix = healpy.ang2pix(512, thetas, phis)
		I = self.schlegel_map[pix].T[::-1,:]
		I = self._normalize_values(np.log(I))
		self.schlegel_projected = I
		rgb = self._to_rgb(I, color=self.color)
		axes_list[0].rgb_images.append(rgb)
		#print "SCHL" * 1000
		#pylab.imshow(np.log(schlegel_map[pix].T))

	def plot(self, axes_list, stack_image):
		if self._can_plot:
			logger.debug("begin plot: %r, style: %r", self, self.state.style)
		else:
			logger.debug("cannot plot layer: %r" % self)
			return

		if not self.visible:
			return
		if self.state.style == "scatter":
			self.plot_scatter(axes_list)
			return
			#return
		logger.debug("total sum of amplitude grid: %s", np.nansum(self.amplitude_grid_view))

		if self.dimensions == 1:
			mask = ~(np.isnan(self.amplitude_grid_view) | np.isinf(self.amplitude_grid_view))
			if np.sum(mask) == 0:
				self.range_level = None
			else:
				values = self.amplitude_grid_view * 1.
				#def nancumsum()
				if self._cumulative:
					values[~mask] = 0
					values = np.cumsum(values)
				if self._normalize:
					if self._cumulative:
						values /= values[-1]
					else:
						values /= np.sum(values[mask]) # TODO: take dx into account?

				if self.dataset.has_selection():
					mask_selected = ~(np.isnan(self.amplitude_grid_selection_view) | np.isinf(self.amplitude_grid_selection_view))
					values_selected = self.amplitude_grid_selection_view * 1.
					if self._cumulative:
						values_selected[~mask_selected] = 0
						values_selected = np.cumsum(values_selected)
					if self._normalize:
						if self._cumulative:
							values_selected /= values_selected[-1]
						else:
							values_selected /= np.sum(values_selected[mask_selected]) # TODO: take dx into account?
				width = self.state.ranges_grid[0][1] - self.state.ranges_grid[0][0]
				x = np.arange(0, self.plot_window.state.grid_size)/float(self.plot_window.state.grid_size) * width + self.state.ranges_grid[0][0]# + width/(Nvector/2.)
				delta = x[1] - x[0]
				for axes in axes_list:
					if self.show in ["total+selection", "total"]:
						if self.display_type == "bar":
							axes.bar(x, values, width=delta, align='center', alpha=self.state.alpha, color=self.color)
						else:
							dx = x[1] - x[0]
							x2 = list(np.ravel(list(zip(x,x+dx))))
							x2p = [x[0]] + x2 + [x[-1]+dx]
							y = values
							y2 = list(np.ravel(list(zip(y,y))))
							y2p = [0] + y2 + [0]
							axes.plot(x2p, y2p, alpha=self.state.alpha, color=self.color)
					if self.show in ["total+selection", "selection"]:
						if self.dataset.has_selection():
							if self.display_type == "bar":
								axes.bar(x, values_selected, width=delta, align='center', color=self.color_alt, alpha=0.6*self.state.alpha)
							else:
								dx = x[1] - x[0]
								x2 = list(np.ravel(list(zip(x,x+dx))))
								x2p = [x[0]] + x2 + [x[-1]+dx]
								y = values_selected
								y2 = list(np.ravel(list(zip(y,y))))
								y2p = [0] + y2 + [0]
								axes.plot(x2p, y2p, drawstyle="steps-mid", alpha=self.state.alpha, color=self.color_alt)

					#3if self.coordinates_picked_row is not None:
					index = self.dataset.get_current_row()
					logger.debug("current row: %r" % index)
					if index is not None:
						x = self.subspace.row(index)
						axes.axvline(x[axes.xaxis_index], color="red")


		#if self.dimensions == 2:
		#	#for axes in axes_list:
		#	assert len(axes_list) == 1
		#	self.plot_density(axes_list[0], self.amplitude_grid, self.amplitude_grid_selection, stack_image)
		if self.dimensions >= 2:
			# for vector we only use the selected map, maybe later also show the full dataset
			#grid_map_vector = self.create_grid_map(self.plot_window.state.vector_grid_size, use_selection)

			self.vector_grid = None
			if 1: #any(self.state.vector_expressions):
				grid_vector = self.grid_vector
				if self.layer_slice_source:
					grid_vector = grid_vector.slice(self.slice_selection_grid)
				vector_grids = None
				if any(self.state.vector_expressions):
					vector_counts = grid_vector.evaluate("counts")
					vector_mask = vector_counts > 0
					if grid_vector.evaluate("weightx") is not None:
						vector_x = grid_vector.evaluate("x")
						vx = grid_vector.evaluate("weightx/counts")
						if self.vectors_subtract_mean:
							vx -= vx[vector_mask].mean()
					else:
						vector_x = None
						vx = None
					if grid_vector.evaluate("weighty") is not None:
						vector_y = grid_vector.evaluate("y")
						vy = grid_vector.evaluate("weighty/counts")
						if self.vectors_subtract_mean:
							vy -= vy[vector_mask].mean()
					else:
						vector_y = None
						vy = None
					if grid_vector.evaluate("weightz") is not None:
						if self.dimensions >= 3:
							vector_z = grid_vector.evaluate("z")
						else:
							vector_z = None
						vz = grid_vector.evaluate("weightz/counts")
						if self.vectors_subtract_mean:
							vz -= vz[vector_mask].mean()
					else:
						vector_z = None
						vz = None
					logger.debug("vx=%s vy=%s vz=%s", vx, vy, vz)
					if vx is not None and vy is not None and vz is not None:
						self.vector_grid = np.zeros((4, ) + ((vx.shape[0],) * 3), dtype=np.float32)
						self.vector_grid[0] = vx
						self.vector_grid[1] = vy
						self.vector_grid[2] = vz
						self.vector_grid[3] = vector_counts
						self.vector_grid = np.swapaxes(self.vector_grid, 0, 3)
						self.vector_grid = self.vector_grid * 1.

					self.vector_grids = vector_grids = [vx, vy, vz]
					vector_positions = [vector_x, vector_y, vector_z]

				for axes in axes_list:
				#if 0:
					# create marginalized grid
					all_axes = list(range(self.dimensions))
					all_axes.remove(self.dimensions-1-axes.xaxis_index)
					all_axes.remove(self.dimensions-1-axes.yaxis_index)

					if 1:
						#grid_map_2d = {key:None if grid is None else (grid if grid.ndim != 3 else vaex.utils.multisum(grid, all_axes)) for key, grid in list(grid_map.items())}
						#grid_context = self.grid_vector
						#amplitude = grid_context(self.amplitude_expression, locals=grid_map_2d)

						grid = self.grid_main.marginal2d(self.dimensions-1-axes.xaxis_index, self.dimensions-1-axes.yaxis_index)
						#grid = self.grid_main.marginal2d(axes.xaxis_index, axes.yaxis_index)
						if self.state.show_disjoined:
							grid = grid.disjoined()
						try:
							amplitude = grid.evaluate(self.amplitude_expression)
						except Exception as e:
							self.error_in_field(self.amplitude_box, "amplitude of layer %s" % self.name, e)
							return
						if self.dataset.has_selection():
							#grid_map_selection_2d = {key:None if grid is None else (grid if grid.ndim != 3 else vaex.utils.multisum(grid, all_axes)) for key, grid in list(grid_map_selection.items())}
							grid_selection = self.grid_main_selection.marginal2d(self.dimensions-1-axes.xaxis_index, self.dimensions-1-axes.yaxis_index)
							if self.state.show_disjoined:
								grid_selection = grid_selection.disjoined()
							amplitude_selection = grid_selection.evaluate(self.amplitude_expression)
						else:
							amplitude_selection = None
						#print("total amplit")
						self.plot_density(axes, amplitude, amplitude_selection, stack_image)

					if len(all_axes) > 2:
						other_axis = all_axes[0]
						assert len(all_axes) == 1, ">3d not supported"
					else:
						other_axis = 2

					if vector_grids:
						#vector_grids[vector_grids==np.inf] = np.nan
						U = vector_grids[axes.xaxis_index]
						V = vector_grids[axes.yaxis_index]
						W = vector_grids[self.dimensions-1-other_axis]
						vx = None if U is None else vaex.utils.multisum(U, all_axes)
						vy = None if V is None else vaex.utils.multisum(V, all_axes)
						vz = None if W is None else vaex.utils.multisum(W, all_axes)
						vector_counts_2d = vaex.utils.multisum(vector_counts, all_axes)
						if vx is not None and vy is not None:
							count_max = vector_counts_2d.max()
							mask = (vector_counts_2d > (self.vector_level_min * count_max)) & \
									(vector_counts_2d <= (self.vector_level_max * count_max))
							x = vector_positions[axes.xaxis_index]
							y = vector_positions[axes.yaxis_index]
							x2d, y2d = np.meshgrid(x, y)
							colors, colormap = None, None
							if True:
								if self.vector_auto_scale:
									length = np.nanmean(np.sqrt(vx[mask]**2 + vy[mask]**2))#  / 1.5
									logger.debug("auto scaling using length: %r", length)
									vx[mask] /= length
									vy[mask] /= length

								scale = self.plot_window.state.vector_grid_size / self.vector_scale
								width = self.vector_head_width * 0.1/self.plot_window.state.vector_grid_size
								xsign = 1 if self.state.ranges_grid[0][0] <= self.state.ranges_grid[0][1] else -1
								ysign = 1 if self.state.ranges_grid[1][0] <= self.state.ranges_grid[1][1] else -1
								if vz is not None and self.vectors_color_code_3rd:
									colors = vz
									colormap = self.state.colormap_vector
									axes.quiver(x2d[mask], y2d[mask], vx[mask] * xsign, vy[mask] * ysign, colors[mask], cmap=colormap, scale_units="width", scale=scale, width=width)
								else:
									axes.quiver(x2d[mask], y2d[mask], vx[mask] * xsign, vy[mask] * ysign, color=self.color, scale_units="width", scale=scale, width=width)
								logger.debug("quiver: %s", self.vector_scale)
								colors = None
		if 0: #if self.coordinates_picked_row is not None:
			if self.dimensions >= 2:
				for axes in axes_list:
					axes.scatter([self.coordinates_picked_row[axes.xaxis_index]], [self.coordinates_picked_row[axes.yaxis_index]], color='red')


		if self.dimensions >= 2:
			for axes in axes_list:
				index = self.dataset.get_current_row()
				logger.debug("current row: %r" % index)
				if index is not None:
					x = self.subspace.row(index)
					axes.scatter([x[axes.xaxis_index]], [x[axes.yaxis_index]], color='red')

	def getVariableDict(self):
		return {} # TODO: remove this? of replace

	def _normalize_values(self, amplitude):
		I = amplitude*1.#self.contrast(amplitude)
		# scale to [0,1]
		mask = ~(np.isnan(I) | np.isinf(I))
		if np.sum(mask) == 0:
			return np.zeros(I.shape, dtype=np.float64)
		I -= I[mask].min()
		I /= I[mask].max()
		return I

	def _to_rgb(self, intensity, color, pre_alpha=1.):
		I = intensity
		mask = ~(np.isnan(I) | np.isinf(I))
		if np.sum(mask) == 0:
			return np.zeros(I.shape + (4,), dtype=np.float64)
		minvalue = I[mask].min()
		maxvalue = I[mask].max()
		if minvalue == maxvalue:
			return np.zeros(I.shape + (4,), dtype=np.float64)
		I -= minvalue
		I /= maxvalue

		# scale [min, max] to [0, 1]
		I -= self.level_min
		I /= (self.level_max - self.level_min)

		#if self.color is not None:

		alpha_mask = (mask) & (I > 0)
		if self.display_type == "solid":
			color_tuple = matplotlib.colors.colorConverter.to_rgb(color)
			rgba = np.zeros(I.shape + (4,), dtype=np.float64)
			rgba[alpha_mask,0:3] = np.array(color_tuple)
		else:
			cmap = matplotlib.cm.cmap_d[self.state.colormap]
			rgba = cmap(I * 1.00)
			rgba[...,3] = (np.clip((I**1.0) * self.state.alpha, 0, 1))
		if self.transparancy == "intensity":
			rgba[...,3] = (np.clip((I**1.0) * self.state.alpha, 0, 1)) * self.state.alpha * pre_alpha
		elif self.transparancy == "constant":
			rgba[alpha_mask,3] = 1. * self.state.alpha * pre_alpha
			rgba[~alpha_mask,3] = 0
		elif self.transparancy == "none":
			rgba[...,3] = pre_alpha
		else:
			raise NotImplemented
		return rgba

	def plot_density_imshow(self, axes, amplitude, amplitude_selection, stack_image):
		if not self.visible:
			return
		ranges = []
		for minimum, maximum in self.state.ranges_grid:
			ranges.append(minimum)
			ranges.append(maximum)
		use_selection = amplitude_selection is not None
		#if isinstance(self.state.colormap, basestring):


		levels = (np.arange(self.contour_count) + 1. ) / (self.contour_count + 1)
		levels = np.linspace(self.level_min, self.level_max, self.contour_count)
		ranges = list(self.state.ranges_grid[0]) + list(self.state.ranges_grid[1])

		amplitude_marginalized = amplitude
		amplitude_marginalized_selected = amplitude_selection
		mask = ~(np.isnan(amplitude_marginalized) | np.isinf(amplitude_marginalized))
		if np.sum(mask) == 0: # if nothing to show
			vmin, vmax = 0, 1
		else:
			vmin, vmax = amplitude_marginalized[mask].min(), amplitude_marginalized[mask].max()
		self.level_ranges = [vmin + self.level_min * (vmax - vmin), vmin + self.level_max * (vmax - vmin)]
		logger.debug("level ranges: %r" % self.level_ranges)

		if self.display_type == "contour":
			if self.contour_count > 0:
				if self.show == "total+selection":
					if use_selection and self.show:
						axes.contour(self._normalize_values(amplitude_marginalized), origin="lower", extent=ranges, levels=levels, linewidths=1, colors=self.color, alpha=0.4*self.state.alpha)
						axes.contour(self._normalize_values(amplitude_marginalized_selected), origin="lower", extent=ranges, levels=levels, linewidths=1, colors=self.color_alt, alpha=self.state.alpha)
					else:
						axes.contour(self._normalize_values(amplitude_marginalized), origin="lower", extent=ranges, levels=levels, linewidths=1, colors=self.color, alpha=self.state.alpha)
				elif self.show == "total":
					axes.contour(self._normalize_values(amplitude_marginalized), origin="lower", extent=ranges, levels=levels, linewidths=1, colors=self.color, alpha=self.state.alpha)
				elif self.show == "selection":
					axes.contour(self._normalize_values(amplitude_marginalized_selected), origin="lower", extent=ranges, levels=levels, linewidths=1, colors=self.color_alt, alpha=self.state.alpha)
		else:
			if self.show == "total+selection":
				I = self._normalize_values(amplitude_marginalized)
				axes.rgb_images.append(self._to_rgb(I, color=self.color, pre_alpha=0.4 if use_selection else 1.0))
				if use_selection:
					I = self._normalize_values(amplitude_marginalized_selected)
					axes.rgb_images.append(self._to_rgb(I, color=self.color_alt))
			elif self.show == "total":
				I = self._normalize_values(amplitude_marginalized)
				axes.rgb_images.append(self._to_rgb(I, color=self.color))
			elif self.show == "selection" and amplitude_marginalized_selected is not None:
				I = self._normalize_values(amplitude_marginalized_selected)
				axes.rgb_images.append(self._to_rgb(I, color=self.color_alt))


	def on_selection_changed(self, dataset):
		self.check_selection_undo_redo()
		#self.plot_window.queue_update(layer=self)
		self.update()
		#self.add_jobs()
		#self.label_selection_info_update()
		#self.plot()

	def on_column_changed(self, dataset, column, type):
		self.update()

	def on_variable_changed(self, dataset, column, type):
		self.update()

	def on_pick(self, dataset, row):
		self.coordinates_picked_row = None
		#self.plot()
		self.signal_plot_dirty.emit(self)

	def on_sequence_changed(self, sequence_index):
		if sequence_index != self.sequence_index: # avoid unneeded event
			self.sequence_index = sequence_index
			#self.seriesbox.setCurrentIndex(self.sequence_index)
		else:
			self.sequence_index = sequence_index
		#self.compute()
		#self.signal_plot_update.emit(delay=0)
		#self.add_jobs()
		#self.plot_window.queue_update(layer=self)
		self.update()

	def get_options(self):
		options = collections.OrderedDict()
		#options["type-names"] = map(str.strip, self.names.split(","))
		options["expressions"] = self.state.expressions
		options["weight"] = self.state.weight_expression
		options["amplitude_expression"] = self.amplitude_expression
		options["ranges_grid"] = self.state.ranges_grid
		options["vx"] = self.vx
		if self.dimensions > 1:
			options["vy"] = self.vy
			options["vz"] = self.vz
		for plugin in self.plugins:
			options.update(plugin.get_options())
		# since options contains reference (like a list of expressions)
		# changes in the gui might reflect previously stored options
		options = copy.deepcopy(options)
		return dict(options)

	def apply_options(self, options, update=True):
		#map = {"expressions",}
		recognize = "expressions weight amplitude_expression ranges_grid aspect vx vy vz".split()
		for key in recognize:
			if key in list(options.keys()):
				value = options[key]
				setattr(self, key, copy.copy(value))
				if key == "amplitude_expression":
					self.amplitude_box.lineEdit().setText(value)
				if key == "weight":
					self.weight = value
				if key == "vx":
					self.weight_x_box.lineEdit().setText(value or "")
				if key == "vy":
					self.weight_y_box.lineEdit().setText(value or "")
				if key == "vz":
					self.weight_z_box.lineEdit().setText(value or "")
				if key == "expressions":
					for expr, box in zip(value, self.axisboxes):
						box.lineEdit().setText(expr)
		for plugin in self.plugins:
			plugin.apply_options(options)
		for key in list(options.keys()):
			if key not in recognize:
				logger.error("option %s not recognized, ignored" % key)
		if update:
			#self.plot_window.queue_update()
			self.update()



	def plug_toolbar(self, callback, order):
		self.plugin_queue_toolbar.append((callback, order))

	def plug_page(self, callback, pagename, pageorder, order):
		self.plugin_queue_page.append((callback, pagename, pageorder, order))

	def plug_grids(self, callback_define, callback_draw):
		self.plugin_grids_defines.append(callback_define)
		self.plugin_grids_draw.append(callback_draw)

	def apply_mask(self, mask):
		# TODO: how to treat this when there is a server
		if self.dataset.is_local():
			self.dataset._set_mask(mask)
			self.execute()
		self.check_selection_undo_redo()
		#self.label_selection_info_update()

	def execute(self):
		error_text = self.dataset.executor.execute()
		if error_text is not None:
			logger.error("error while executing: %r" % error_text)
			dialogs.dialog_error(self.plot_window, "Error when executing", error_text)


	def message(self, *args, **kwargs):
		pass

	def on_error(self, exception):
		logger.exception("unhandled error occured")
		self.finished_tasks()
		import traceback
		traceback.print_exc()
		raise exception

	def add_task(self, task):
		self._task_signals.append(task.signal_progress.connect(self._layer_progress))
		self.tasks.append(task)
		return task

	def _layer_progress(self, fraction):
		total_fraction = 0
		for task in self.tasks:
			total_fraction += task.progress_fraction
		fraction = total_fraction / len(self.tasks)
		self.plot_window.set_layer_progress(self, fraction)
		return True

	def get_progress_fraction(self):
		total_fraction = 0
		for task in self.tasks:
			total_fraction += task.progress_fraction
		fraction = total_fraction / len(self.tasks)
		return fraction

	def finished_tasks(self):
		for task, signal in zip(self.tasks, self._task_signals):
			task.signal_progress.disconnect(signal)
		self.tasks = []
		self._task_signals = []

	def cancel_tasks(self):
		logger.info("cancelling tasks for layer %r", self)
		for task in self.tasks:
			task.cancel()
		self.finished_tasks()

	def add_tasks_ranges(self):
		logger.info("adding ranges jobs for layer: %r, previous ranges_grid = %r", self, self.state.ranges_grid)
		assert not self.tasks, "still were tasks in queue: %r for %r" % (self.tasks, self)
		missing = False
		# TODO, optimize for the case when some dimensions are already known
		for range in self.state.ranges_grid:
			if range is None:
				missing = True
			else:
				vmin, vmax = range
				if vmin is None or vmax is None:
					missing = True
		self.subspace = self.dataset(*self.state.expressions, async=True)
		subspace_ranges = self.subspace
		if self.layer_slice_source:
			all_expressions = self.state.expressions + self.layer_slice_source.expressions
			self.subspace = self.dataset(*all_expressions, async=True)

		if missing:
			logger.debug("first we calculate min max for this layer")
			return self.add_task(subspace_ranges.minmax()).then(self.got_limits, self.on_error).then(None, self.on_error)
		else:
			#self.got_limits(self.state.ranges_grid)
			return vaex.promise.Promise.fulfilled(self)

	def got_limits(self, limits):
		logger.debug("got limits %r for layer %r" % (limits, self))
		self.state.ranges_grid = np.array(limits).tolist() # for this class we need it to be a list
		self.finished_tasks()
		return self

	def add_tasks_histograms(self):
		self._histogram_counter += 1
		assert not self.tasks

		histogram_counter = self._histogram_counter
		self._can_plot = False
		promises = []
		self.grid_main = vaex.grids.GridScope(globals=np.__dict__)
		self.grid_main_selection = vaex.grids.GridScope(globals=np.__dict__)
		self.grid_vector = vaex.grids.GridScope(globals=np.__dict__)


		ranges = np.array(self.state.ranges_grid)
		if self.layer_slice_source:
			ranges = np.array(self.state.ranges_grid + self.layer_slice_source.ranges_grid)
		ranges = np.array(ranges)
		# add the main grid
		histogram_promise = self.add_task(self.subspace.histogram(limits=ranges, size=self.plot_window.state.grid_size))\
			.then(self.grid_main.setter("counts"))\
			.then(None, self.on_error)
		promises.append(histogram_promise)

		if self.dataset.has_selection():
			histogram_promise = self.add_task(self.subspace.selected().histogram(limits=ranges, size=self.plot_window.state.grid_size))\
				.then(self.grid_main_selection.setter("counts"))\
				.then(None, self.on_error)
			promises.append(histogram_promise)

			@vaex.delayed.delayed
			def update_count(count):
				self.label_selection_info_update(count)
			update_count(self.dataset.count(selection=True, async=True))
		else:
			self.label_selection_info_update(None)

		# the weighted ones
		if self.state.weight_expression:
			histogram_weighted_promise = self.add_task(self.subspace.histogram(limits=ranges
					, weight=self.state.weight_expression, size=self.plot_window.state.grid_size))\
				.then(self.grid_main.setter("weighted"))\
				.then(None, self.on_error)
			promises.append(histogram_weighted_promise)

			if self.dataset.has_selection():
				histogram_weighted_promise = self.add_task(self.subspace.selected().histogram(limits=ranges
						, weight=self.state.weight_expression, size=self.plot_window.state.grid_size))\
					.then(self.grid_main_selection.setter("weighted"))\
					.then(None, self.on_error)
				promises.append(histogram_weighted_promise)

		else:
			self.grid_main["weighted"] = None
			self.grid_main_selection["weighted"] = None

		# the vector fields only use the selection if there is one, otherwise the whole dataset
		subspace = self.subspace
		if self.dataset.has_selection():
			subspace = subspace.selected()

		for i, expression in enumerate(self.state.vector_expressions):
			name = "xyzw"[i]

			# add arrays x y z which container the centers of the bins
			if i < self.dimensions:
				gridsize = self.plot_window.state.vector_grid_size
				width = self.state.ranges_grid[i][1] - self.state.ranges_grid[i][0]
				offset = self.state.ranges_grid[i][0]
				x = (np.arange(0, gridsize)+0.5)/float(gridsize) * width + offset
				self.grid_vector[name] = x

			if self.state.vector_expressions[i]:
				histogram_vector_promise = self.add_task(subspace.histogram(limits=ranges
						, weight=self.state.vector_expressions[i], size=self.plot_window.state.vector_grid_size))\
					.then(self.grid_vector.setter("weight"+name))\
					.then(None, self.on_error)
				promises.append(histogram_vector_promise)
			else:
				self.grid_vector["weight" +name] = None
		if any(self.state.vector_expressions):
			histogram_vector_promise = self.add_task(subspace.histogram(limits=ranges
					,size=self.plot_window.state.vector_grid_size))\
				.then(self.grid_vector.setter("counts"))\
				.then(None, self.on_error)
			promises.append(histogram_vector_promise)

		#else:
		#	for name in "xyz":
		#		self.grid_vector["weight" +name] = None


		def check_counter(arg):
			if histogram_counter == self._histogram_counter:
				self.got_grids(arg)
			else:
				logger.debug
				#raise ValueError, "histogram counter got update, cancel redraw etc"
			return arg

		return vaex.promise.listPromise(promises).then(check_counter)#.then(None, error_counter)\
			#.then(self.got_grids)\
			#.then(None, self.on_error)



	def got_grids(self, *args):
		logger.debug("got grids for layer %r" % (self, ))
		self.finished_tasks()
		self.calculate_amplitudes()
		if 0: # TODO: enable again with postfixes like M, insteads of long numbers
			counts = self.grid_main.evaluate("counts")
			visible = np.sum(counts)
			total = len(self.dataset)
			fraction = float(visible)/total
			self.label_visible.setText("{:,} of {:,} ({}) visible".format(visible, total, fraction*100))
		self._needs_update = False
		return self

	def slice_amplitudes(self):
		slice = self.layer_slice_source is not None
		if False: #slice:
			extra_axes = tuple(range(self.subspace.dimension)[len(self.state.expressions):])
			logger.debug("sum over axes: %r", extra_axes)
			msum = vaex.utils.multisum
			logger.debug("shape of grid: %r", self.amplitude_grid[...,self.slice_selection_grid].shape)
			self.amplitude_grid_view = np.sum(self.amplitude_grid[...,self.slice_selection_grid], axis=-1)
			if self.amplitude_grid_selection is not None:
				self.amplitude_grid_selection_view = np.sum(self.amplitude_grid_selection[...,self.slice_selection_grid], axis=-1)
			else:
				self.amplitude_grid_selection_view = None
		#else:
		self.amplitude_grid_view = self.amplitude_grid
		self.amplitude_grid_selection_view = self.amplitude_grid_selection

	def calculate_amplitudes(self):
		logger.debug("calculating amplitudes (in thread %r)" % threading.currentThread())

		slice = self.layer_slice_source is not None

		try:
			grid = self.grid_main
			if self.state.style == "schlegel":
				#self.plot_schlegel(axes_list, stack_image)
				if not hasattr(self, "schlegel_map"):
					self.schlegel_map = healpy.read_map('data/lambda_sfd_ebv.fits', nest=False)
				xlim, ylim = self.plot_window.state.ranges_viewport
				phis = np.linspace(np.deg2rad(xlim[0]), np.deg2rad(xlim[1]), self.plot_window.state.grid_size)# + np.pi/2.
				thetas = np.pi-np.linspace(np.deg2rad(ylim[1]) + np.pi/2., np.deg2rad(ylim[0]) + np.pi/2., self.plot_window.state.grid_size)
				#phis = (np.linspace(0, 2*np.pi, 256) - np.pi) % (2*np.pi)
				thetas, phis = np.meshgrid(thetas, phis)

				pix = healpy.ang2pix(512, thetas, phis)
				grid["counts"] = self.schlegel_map[pix].T[::-1,:]
				#self.amplitude_grid_selection = None

			if slice:
				grid = grid.slice(self.slice_selection_grid)
			if self.state.show_disjoined:
				grid = grid.disjoined()
			self.amplitude_grid = grid.evaluate(self.amplitude_expression)
		except Exception as e:
			logger.exception("amplitude field")
			#traceback.print_exc()
			#print self.error_in_field
			self.error_in_field(self.amplitude_box, "amplitude of layer %s" % self.name, e)
			return
		self.amplitude_grid_selection = None
		logger.debug("shape of amplitude grid: %r" % (self.amplitude_grid.shape, ))
		if self.dataset.has_selection():
			grid = self.grid_main_selection
			if slice:
				grid = grid.slice(self.slice_selection_grid)
			if self.state.show_disjoined:
				grid = grid.disjoined()
			self.amplitude_grid_selection = grid.evaluate(self.amplitude_expression)

		vmin = None
		vmax = None
		def getminmax(grid, vmin, vmax):
			mask = ~(np.isnan(grid) | np.isinf(grid))
			values = grid
			if self.dimensions == 1:
				# DRY, same code in plot(..)
				values = grid * 1. # copy grid
				if self._cumulative:
					values[~mask] = 0
					values = np.cumsum(values)
				if self._normalize:
					if self._cumulative:
						values /= values[-1]
					else:
						values /= np.sum(values[mask]) # TODO: take dx into account?
			if mask.sum() > 0:
				newvmin = values[mask].min()
				newvmax = values[mask].max()
				vmin = min(newvmin, vmin) if vmin is not None else newvmin
				vmax = max(newvmax, vmax) if vmax is not None else newvmax
			return vmin, vmax
		self.range_level = vmin, vmax = getminmax(self.amplitude_grid, vmin, vmax)
		logger.debug("range_level: %r" % [vmin, vmax])
		if self.dataset.has_selection():
			self.range_level = getminmax(self.amplitude_grid_selection, vmin, vmax)
			logger.debug("range_level update: %r" % ([vmin, vmax], ))
		self.slice_amplitudes()
		self._can_plot = True

	def build_widget_qt(self, parent):
		self.widget_build = True

		# create plugins
		self.plugin_grids_defines = []
		self.plugin_grids_draw = []
		self.plugin_queue_toolbar = [] # list of tuples (callback, order)
		self.plugin_queue_page = []
		self.plugins = [cls(parent, self) for cls in vaex.ui.plugin.PluginLayer.registry if cls.useon(self.plot_window.__class__)]
		logger.debug("PLUGINS: %r " % self.plugins)
		self.plugins_map = {plugin.name:plugin for plugin in self.plugins}
		#self.plugin_zoom = plugin.zoom.ZoomPlugin(self)


		self.toolbox = QtGui.QToolBox(parent)
		self.toolbox.setMinimumWidth(250)



		self.plug_page(self.page_main, "Main", 1., 1.)
		self.plug_page(self.page_visual, "Visual", 1.5, 1.)
		self.plug_page(self.page_annotate, "Annotate", 1.75, 1.)
		if self.dimensions >= 2:
			self.plug_page(self.page_vector, "Vector field", 2., 1.)
		else:
			self.vector_dimensions = 0
			self.vector_names = "vx vy vz".split()[:self.vector_dimensions]

		#self.plug_page(self.page_display, "Display", 3., 1.)
		self.plug_page(self.page_selection, "Selection", 3.5, 1.)
		if self.plot_window.enable_slicing:
			self.plug_page(self.page_slice, "Slicing", 3.75, 1.)

		# first get unique page orders
		logger.debug("setting up layer plugins")
		pageorders = {}
		for callback, pagename, pageorder, order in self.plugin_queue_page:
			pageorders[pagename] = pageorder
		self.pages = {}
		for pagename, order in sorted(list(pageorders.items()), key=operator.itemgetter(1)):
			page_frame = QtGui.QFrame(self.toolbox)
			self.pages[pagename] = page_frame
			self.toolbox.addItem(page_frame, pagename)
			logger.debug("created page: "+pagename)
		for pagename, order in sorted(list(pageorders.items()), key=operator.itemgetter(1)):
			logger.debug("filling page: %sr %r" % (pagename, [x for x in self.plugin_queue_page if x[1] == pagename]))
			for callback, pagename_, pageorder, order in sorted([x for x in self.plugin_queue_page if x[1] == pagename], key=operator.itemgetter(3)):
				logger.debug("filling page: "+pagename +" order=" +str(order) + " callback=" +str(callback))
				callback(self.pages[pagename])
		page_name = self.options.get("page", "Main")
		page_frame = self.pages.get(page_name, None)
		if page_frame:
			self.toolbox.setCurrentWidget(page_frame)
		logger.debug("done setting up layer plugins")

		self.widget = self.toolbox

		if "selection" in self.options:
			raise NotImplementedError("selection meaning changed")
			#filename = self.options["selection"]
			#mask = np.load(filename)
			#action = vaex.ui.undo.ActionMask(self.dataset.undo_manager, "selection from %s" % filename, mask, self.apply_mask)
			#action.do()
			#self.apply_mask(mask)
			#self.dataset.selectMask(mask)

		if "slice_link" in self.options:
			window_name, layer_name = self.options["slice_link"].split(".")
			matches = [window for window in self.plot_window.app.windows if window.name == window_name]
			logger.debug("matching windows for slice_link: %r", matches)
			if matches:
				window = matches[0]
				matches = [layer for layer in window.layers if layer.name == layer_name]
				logger.debug("matching layers for slice_link: %r", matches)
				if matches:
					layer = matches[0]
					self.slice_link(layer)


		return self.toolbox

	def grab_layer_control(self, new_parent):
		# no need to take action
		#self.widget_layer_control = page_widget = QtGui.QGroupBox(self.name, parent)
		self.page_visual_groupbox_layout.addWidget(self.page_widget_visual)
		return self.page_visual_groupbox

	def release_layer_control(self, current_parent):
		self.page_visual_groupbox.setParent(None)
		self.page_visual_layout.addWidget(self.page_widget_visual)

	def _build_widget_qt_layer_control(self, parent):
		self.widget_layer_control = page_widget = QtGui.QGroupBox(self.name, parent)
		#self.widget_layer_control.setFlat(True)

		self.layout_layer_control = QtGui.QGridLayout()
		self.widget_layer_control.setLayout(self.layout_layer_control)
		self.layout_layer_control.setSpacing(0)
		self.layout_layer_control.setContentsMargins(0,0,0,0)

		row = 0


	def get_expression_list(self):
		return self.dataset.get_column_names(virtual=True)

	def onExpressionChanged(self, axis_index):
		text = str(self.axisboxes[axis_index].lineEdit().text())
		if text == self.state.expressions[axis_index]:
			logger.debug("same expression, will not update")
		else:
			self.set_expression(text, axis_index)

	def set_expression(self, expression, index):
		self.state.expressions[index] = expression
		try:
			self.dataset.validate_expression(expression)
		except Exception as e:
			logger.exception("error in expression")
			self.error_in_field(self.axisboxes[index], self.axis_names[index], e)
			return
		self.axisboxes[index].lineEdit().setText(expression)
		self.plot_window.queue_history_change("changed expression %s to %s" % (self.axis_names[index], expression))
		# TODO: range reset as option?
		self.state.ranges_grid[index] = None
		self.plot_window.state.ranges_viewport[index] = None
		# TODO: how to handle axis lock.. ?
		if not self.plot_window.state.axis_lock:
			self.state.ranges_grid[index] = None
		linkButton = self.linkButtons[index]
		link = linkButton.link
		if link:
			logger.debug("sending link messages")
			link.sendRanges(self.ranges[index], linkButton)
			link.sendRangesShow(self.ranges_show[index], linkButton)
			link.sendExpression(self.state.expressions[index], linkButton)
			vaex.dataset.Link.sendCompute([link], [linkButton])
		else:
			logger.debug("not linked")
		# let any event handler deal with redraw etc
		self.coordinates_picked_row = None
		#self.add_jobs()
		#self.plot_window.queue_update()
		self.update()
		#self.execute()
		#self.signal_expression_change.emit(self, axis_index, text)
		#self.compute()
		#error_text = self.dataset.executor.execute()
		#if error_text:
		#	dialog_error(self, "Error in expression", "Error: " +error_text)

	def onWeightExpr(self):
		text = str(self.weight_box.lineEdit().text())
		if (text == self.state.weight_expression) or (text == "" and self.state.weight_expression == None):
			logger.debug("same weight expression, will not update")
			return
		else:
			self.set_weight_expression(text)

	def set_weight_expression(self, expression):
		expression = expression or ""
		if expression.strip() == "":
			self.state.weight_expression = None
		else:
			self.state.weight_expression = expression
		if expression:
			try:
				self.dataset.validate_expression(expression)
			except Exception as e:
				self.error_in_field(self.weight_box, "weight", e)
				return
		self.weight_box.lineEdit().setText(expression)
		self.plot_window.queue_history_change("changed weight expression to %s" % (expression))
		self.range_level = None
		self.plot_window.range_level_show = None
		#self.plot_window.queue_update(layer=self)
		self.update()
		#self.add_jobs()
		#self.execute()
		#self.plot()

	def onTitleExpr(self):
		self.title_expression = str(self.title_box.lineEdit().text())
		self.plot()

	def onWeightXExpr(self):
		text = str(self.weight_x_box.lineEdit().text())
		self.set_vector_expression(text, 0)

	def onWeightYExpr(self):
		text = str(self.weight_y_box.lineEdit().text())
		self.set_vector_expression(text, 1)

	def onWeightZExpr(self):
		text = str(self.weight_z_box.lineEdit().text())
		self.set_vector_expression(text, 2)

	def set_vector_expression(self, expression, axis_index):
		# is we set the text to "", check if some of the grids are existing, and simply 'disable' the and replot
		# otherwise check if it changed, if it did, see if we should do the grid computation, since
		# if only 1 grid is defined, we don't need it
		name = "xyz"[axis_index]
		weight_name = ("weight" + name)
		if (not expression) or expression.strip() == "":
			expression = ""
		self.vector_boxes[axis_index].lineEdit().setText(expression)
		if expression == self.state.vector_expressions[axis_index]:
			logger.debug("same vector_expression[%d], will not update", axis_index)
			return

		self.state.vector_expressions[axis_index] = expression
		if expression:
			try:
				self.dataset.validate_expression(expression)
			except Exception as e:
				self.error_in_field(self.weight_x_box, "v" +name, e)
				return

		self.plot_window.queue_history_change("changed vector expression %s to %s" % (name, expression))
		if expression is None:
			if self.grid_vector and weight_name in self.grid_vector and self.grid_vector[weight_name] is not None:
				logger.debug("avoided update due to change in vector_expression[%d]", axis_index)
				self.grid_vector[weight_name] = None
				self.plot_window.queue_replot()
				self.plot_window.queue_push_full_state()
				return

		self.range_level = None
		self.plot_window.range_level_show = None

		logger.debug("current vector expressions: %r" % self.state.vector_expressions)
		non_none_expressions = [k for k in self.state.vector_expressions if k is not None and len(k) > 0]
		if len(non_none_expressions) >= 2:
			logger.debug("do an update due to change in vector_expression[%d]" % axis_index)
			#self.add_jobs()
			#self.execute()
			#self.plot_window.queue_update(layer=self)
			self.update()
		else:
			self.plot_window.queue_push_full_state()

	def onAmplitudeExpr(self):
		text = str(self.amplitude_box.lineEdit().text())
		if len(text) == 0 or text == self.amplitude_expression:
			logger.debug("same expression, skip")
			return
		self.amplitude_expression = text
		self.calculate_amplitudes()
		self.plot_window.calculate_range_level_show()
		self.plot_window.plot()
		#self.plot()

	def page_main(self, page):
		self.frame_options_main = page #QtGui.QFrame(self)
		self.layout_frame_options_main =  QtGui.QVBoxLayout()
		self.frame_options_main.setLayout(self.layout_frame_options_main)
		self.layout_frame_options_main.setSpacing(0)
		self.layout_frame_options_main.setContentsMargins(0,0,0,0)
		self.layout_frame_options_main.setAlignment(QtCore.Qt.AlignTop)

		self.button_layout = QtGui.QVBoxLayout()
		if self.dimensions > 1:
			self.buttonFlipXY = QtGui.QPushButton("exchange x and y")
			def flipXY():
				self.state.expressions.reverse()
				self.state.ranges_grid.reverse()
				# TODO: how to handle layers?
				self.plot_window.state.ranges_viewport.reverse()
				for box, expr in zip(self.axisboxes, self.state.expressions):
					box.lineEdit().setText(expr)
				#self.plot_window.queue_update() # only update thislayer??
				self.update()
				self.execute()
			self.buttonFlipXY.clicked.connect(flipXY)
			self.button_layout.addWidget(self.buttonFlipXY, 0.)
			self.buttonFlipXY.setAutoDefault(False)
			self.button_flip_colormap = QtGui.QPushButton("exchange colormaps")
			def flip_colormap():
				index1 = self.colormap_box.currentIndex()
				index2 = self.colormap_vector_box.currentIndex()
				self.colormap_box.setCurrentIndex(index2)
				self.colormap_vector_box.setCurrentIndex(index1)
			self.button_flip_colormap.clicked.connect(flip_colormap)
			self.button_layout.addWidget(self.button_flip_colormap)
			self.button_flip_colormap.setAutoDefault(False)
		self.layout_frame_options_main.addLayout(self.button_layout, 0)

		self.axisboxes = []
		self.onExpressionChangedPartials = []
		axis_index = 0

		self.grid_layout = QtGui.QGridLayout()
		#self.grid_layout.setColumnStretch(2, 1)
		self.grid_layout.setColumnStretch(1, 1)
		self.grid_layout.setSpacing(0)
		self.grid_layout.setContentsMargins(2,1,2,1)
		self.grid_layout.setAlignment(QtCore.Qt.AlignTop)
		#row = 0
		self.linkButtons = []
		for axis_name in self.axis_names:
			row = axis_index
			import vaex.ui.completer
			axisbox = vaex.ui.completer.ExpressionCombobox(page, self.dataset, variables=True) #$QtGui.QComboBox(page)
			#$axisbox.setEditable(True)
			#axisbox.setMinimumContentsLength(10)
			#self.form_layout.addRow(axis_name + '-axis:', axisbox)
			self.grid_layout.addWidget(QtGui.QLabel(axis_name + '-axis:', page), row, 0)
			self.grid_layout.addWidget(axisbox, row, 1)
			linkButton = LinkButton("link", self.dataset, axis_index, page)
			self.linkButtons.append(linkButton)
			linkButton.setChecked(True)
			linkButton.setVisible(False)
			# obove doesn't fire event, do manually
			#linkButton.onToggleLink()
			if 1:
				functionButton = QtGui.QToolButton(page)
				functionButton.setIcon(QtGui.QIcon(iconfile('edit-mathematics')))
				menu = QtGui.QMenu()
				functionButton.setMenu(menu)
				functionButton.setPopupMode(QtGui.QToolButton.InstantPopup)
				#link_action = QtGui.QAction(QtGui.QIcon(iconfile('network-connect-3')), '&Link axis', self)
				#unlink_action = QtGui.QAction(QtGui.QIcon(iconfile('network-disconnect-2')), '&Unlink axis', self)
				templates = ["log10(%s)", "sqrt(%s)", "1/(%s)", "abs(%s)"]

				for template in templates:
					action = QtGui.QAction(template % "...", page)
					def add(checked=None, axis_index=axis_index, template=template):
						logger.debug("adding template %r to axis %r" % (template, axis_index))
						expression = self.state.expressions[axis_index].strip()
						if "#" in expression:
							expression = expression[:expression.index("#")].strip()
						self.state.expressions[axis_index] = template % expression
						# this doesn't cause an event causing jobs to be added?
						self.axisboxes[axis_index].lineEdit().setText(self.state.expressions[axis_index])
						self.state.ranges_grid[axis_index] = None
						self.coordinates_picked_row = None
						if not self.plot_window.state.axis_lock:
							self.plot_window.state.ranges_viewport[axis_index] = None
						# to add them
						#self.add_jobs()
						#self.execute()
						self.update()
					action.triggered.connect(add)
					menu.addAction(action)
				self.grid_layout.addWidget(functionButton, row, 2, QtCore.Qt.AlignLeft)
				#menu.addAction(unlink_action)
				#self.grid_layout.addWidget(functionButton, row, 2)
			#self.grid_layout.addWidget(linkButton, row, 0)
			#if axis_index == 0:
			extra_expressions = []
			expressionList = self.get_expression_list()
			for prefix in ["", "v", "v_"]:
				names = "x y z".split()
				allin = True
				for name in names:
					if prefix + name not in expressionList:
						allin = False
				# if all items found, add it
				#if allin:
				#	expression = "l2(%s) # l2 norm" % (",".join([prefix+name for name in names]))
				#	extra_expressions.append(expression)

				if 0: # this gives too much clutter
					for name1 in names:
						for name2 in names:
							if name1 != name2:
								if name1 in expressionList and name2 in expressionList:
									expression = "d(%s)" % (",".join([prefix+name for name in [name1, name2]]))
									extra_expressions.append(expression)


			axisbox.addItems(extra_expressions + self.get_expression_list())
			#axisbox.setCurrentIndex(self.state.expressions[axis_index])
			#axisbox.currentIndexChanged.connect(functools.partial(self.onAxis, axis_index=axis_index))
			axisbox.lineEdit().setText(self.state.expressions[axis_index])
			# keep a list to be able to disconnect
			self.onExpressionChangedPartials.append(functools.partial(self.onExpressionChanged, axis_index=axis_index))
			axisbox.lineEdit().editingFinished.connect(self.onExpressionChangedPartials[axis_index])
			# if the combox pulldown is clicked, execute the same command
			axisbox.currentIndexChanged.connect(lambda _, axis_index=axis_index: self.onExpressionChangedPartials[axis_index]())
			axis_index += 1
			self.axisboxes.append(axisbox)
		row += 1
		self.layout_frame_options_main.addLayout(self.grid_layout, 0)
		#self.layout_frame_options_main.addLayout(self.form_layout, 0) # TODO: form layout can be removed?

		self.amplitude_box = QtGui.QComboBox(page)
		self.amplitude_box.setEditable(True)
		if "amplitude" in self.options:
			self.amplitude_box.addItems([self.options["amplitude"]])
		self.amplitude_box.addItems(["log(counts) if weighted is None else average", "counts", "counts**2", "average", "sqrt(counts)"])
		self.amplitude_box.addItems(["log(counts+1)"])
		self.amplitude_box.addItems(["gf(log(counts+1),1)"])
		self.amplitude_box.addItems(["gf(log(counts+1),2)"])
		self.amplitude_box.addItems(["dog(counts, 2, 2.1)"])
		if 0:
			self.amplitude_box.addItems(["counts/peak_columns # divide by peak value in every row"])
			self.amplitude_box.addItems(["counts/sum_columns # normalize columns"])
			self.amplitude_box.addItems(["counts/peak_rows # divide by peak value in every row"])
			self.amplitude_box.addItems(["counts/sum_rows # normalize rows"])
			self.amplitude_box.addItems(["log(counts/peak_columns)"])
			self.amplitude_box.addItems(["log(counts/sum_columns)"])
			self.amplitude_box.addItems(["log(counts/peak_rows)"])
			self.amplitude_box.addItems(["log(counts/sum_rows)"])
			self.amplitude_box.addItems(["abs(fft.fftshift(fft.fft2(counts))) # 2d fft"])
			self.amplitude_box.addItems(["abs(fft.fft(counts, axis=1)) # ffts along y axis"])
			self.amplitude_box.addItems(["abs(fft.fft(counts, axis=0)) # ffts along x axis"])
		self.amplitude_box.setMinimumContentsLength(10)
		self.grid_layout.addWidget(QtGui.QLabel("amplitude="), row, 0)
		self.grid_layout.addWidget(self.amplitude_box, row, 1, QtCore.Qt.AlignLeft)
		#self.amplitude_box.lineEdit().editingFinished.connect(self.onAmplitudeExpr)
		#self.amplitude_box.currentIndexChanged.connect(lambda _: self.onAmplitudeExpr())
		def onchange(*args, **kwargs):
			self.onAmplitudeExpr()
		def onchange_line(*args, **kwargs):
			if len(str(self.amplitude_box.lineEdit().text())) == 0:
				self.onAmplitudeExpr()
		#self.amplitude_box.currentIndexChanged.connect(functools.partial(onchange, event="currentIndexChanged"))
		#self.amplitude_box.editTextChanged.connect(functools.partial(onchange, event="editTextChanged"))
		#self.amplitude_box.lineEdit().editingFinished.connect(functools.partial(onchange, event="editingFinished"))

		# this event is also fired when the line edit is finished, except when an empty entry is given
		self.amplitude_box.currentIndexChanged.connect(onchange)
		self.amplitude_box.lineEdit().editingFinished.connect(functools.partial(onchange_line, event="editingFinished"))


		self.amplitude_expression = str(self.amplitude_box.lineEdit().text())

		row += 1


		if 0: # TODO: this should go out of layer...
			self.title_box = QtGui.QComboBox(page)
			self.title_box.setEditable(True)
			self.title_box.addItems([""] + self.getTitleExpressionList())
			self.title_box.setMinimumContentsLength(10)
			self.grid_layout.addWidget(QtGui.QLabel("title="), row, 0)
			self.grid_layout.addWidget(self.title_box, row, 1)
			self.title_box.lineEdit().editingFinished.connect(self.onTitleExpr)
			self.title_box.currentIndexChanged.connect(lambda _: self.onTitleExpr())
			self.title_expression = str(self.title_box.lineEdit().text())
			row += 1

		self.weight_box = vaex.ui.completer.ExpressionCombobox(page, self.dataset, variables=True)#QtGui.QComboBox(page)
		self.weight_box.setEditable(True)
		self.weight_box.addItems([self.options.get("weight", "")] + self.get_expression_list())
		self.weight_box.setMinimumContentsLength(10)
		self.grid_layout.addWidget(QtGui.QLabel("weight="), row, 0)
		self.grid_layout.addWidget(self.weight_box, row, 1)
		self.weight_box.lineEdit().editingFinished.connect(self.onWeightExpr)
		self.weight_box.currentIndexChanged.connect(lambda _: self.onWeightExpr())
		self.state.weight_expression = str(self.weight_box.lineEdit().text())
		if len(self.state.weight_expression.strip()) == 0:
			self.state.weight_expression = None
		row += 1


		if 0:
			self.flip_x = False
			ucd_x = self.dataset.ucds.get(self.x)
			if ucd_x and ("pos.galactic.lon" in ucd_x or "pos.eq.ra" in ucd_x):
				self.flip_x = True

			self.checkbox_flip_x = Checkbox(page, "flip_x", getter=attrgetter(self, "flip_x"), setter=attrsetter(self, "flip_x"), update=self.signal_plot_dirty.emit)
			row = self.checkbox_flip_x.add_to_grid_layout(row, self.grid_layout, 1)

			self.flip_y = False
			ucd_x = self.dataset.ucds.get(self.y)
			#if ucd_x and ("pos.galactic.lon" in ucd_x or "pos.eq.ra" in ucd_x):
			#	self.flip_x = True

			self.checkbox_flip_y = Checkbox(page, "flip_y", getter=attrgetter(self, "flip_y"), setter=attrsetter(self, "flip_y"), update=self.signal_plot_dirty.emit)
			row = self.checkbox_flip_y.add_to_grid_layout(row, self.grid_layout, 1)


		self.option_xrange = RangeOption(page, "x-range", [0], lambda: self.get_range(0), lambda value: self.plot_window.set_range(value[0], value[1], 0), update=self.update)
		row = self.option_xrange.add_to_grid_layout(row, self.grid_layout)

		if self.dimensions >= 2:
			self.option_yrange = RangeOption(page, "y-range", [0], lambda: self.get_range(1), lambda value: self.plot_window.set_range(value[0], value[1], 1), update=self.update)
			row = self.option_yrange.add_to_grid_layout(row, self.grid_layout)
		if self.dimensions >= 3:
			self.option_zrange = RangeOption(page, "z-range", [0], lambda: self.get_range(2), lambda value: self.plot_window.set_range(value[0], value[1], 2), update=self.update)
			row = self.option_zrange.add_to_grid_layout(row, self.grid_layout)

		self.state.output_units = []
		self.option_output_unit = []
		for dim in range(self.dimensions):
			name = "unit"+self.axis_names[dim]
			self.state.output_units.append(self.options.get(name, ""))
			def get(dim=dim):
				return self.state.output_units[dim]
			def set(value, dim=dim, name=name):
				if not value:
					self.state.output_units[dim] = ""
					self.plot_window.queue_history_change("changed %s to default" % name)
					self.plot_window.queue_push_full_state()
				else:
					try:
						unit_output = astropy.units.Unit(value)
						unit_input = self.dataset.unit(self.state.expressions[dim])
						unit_input.to(unit_output)
						self.plot_window.queue_history_change("changed %s to %s" % (name, value))
						self.plot_window.queue_push_full_state()
					except Exception as e:
						self.error_dialog(self.option_output_unit[dim].textfield, "Error converting units", e)
					else:
						self.state.output_units[dim] = value
			self.option_output_unit.append(
				dialogs.TextOption(page, name, get(), placeholder="output units", getter=get, setter=set, update=self.signal_plot_dirty.emit)
			)
			self.option_output_unit[-1].set_unit_completer()
			row = self.option_output_unit[-1].add_to_grid_layout(row, self.grid_layout)

		if 0:
			self.grid_layout.addWidget(QtGui.QLabel("visible:"), row, 0)
			self.label_visible = QtGui.QLabel("", page)
			self.grid_layout.addWidget(self.label_visible, row, 1)

		row += 1

	def page_visual(self, page):

		# this widget is used for the layer control, it is wrapped around the page_widget
		self.page_visual_groupbox = QtGui.QGroupBox(self.name)
		self.page_visual_groupbox_layout = QtGui.QVBoxLayout(page)
		self.page_visual_groupbox_layout.setAlignment(QtCore.Qt.AlignTop)
		self.page_visual_groupbox_layout.setSpacing(0)
		self.page_visual_groupbox_layout.setContentsMargins(0,0,0,0)
		self.page_visual_groupbox.setLayout(self.page_visual_groupbox_layout)

		self.page_visual_widget = page # refactor, change -> page_X to fill_page_X and use page_X for the wiget

		self.page_visual_layout = layout = QtGui.QVBoxLayout(page)
		layout.setAlignment(QtCore.Qt.AlignTop)
		layout.setSpacing(0)
		layout.setContentsMargins(0,0,0,0)
		page.setLayout(layout)


		# put all children in one parent widget to easily move them (for layer control)
		self.page_widget_visual = page_widget = QtGui.QWidget(page)
		layout.addWidget(page_widget)

		grid_layout = QtGui.QGridLayout()
		grid_layout.setColumnStretch(2, 1)
		page_widget.setLayout(grid_layout)
		grid_layout.setAlignment(QtCore.Qt.AlignTop)
		grid_layout.setSpacing(0)
		grid_layout.setContentsMargins(0,0,0,0)

		row = 1
		
		
		self.visible = True
		self.checkbox_visible = Checkbox(page_widget, "visible", getter=attrgetter(self, "visible"), setter=attrsetter(self, "visible"), update=self.signal_plot_dirty.emit)
		row = self.checkbox_visible.add_to_grid_layout(row, grid_layout)

		#self.checkbox_intensity_as_opacity = Checkbox(page_widget, "use_intensity", getter=attrgetter(self, "use_intensity"), setter=attrsetter(self, "use_intensity"), update=self.signal_plot_dirty.emit)
		#row = self.checkbox_intensity_as_opacity.add_to_grid_layout(row, grid_layout)

		if self.dimensions <= 3:
			show_options = ["total+selection", "total", "selection"]
			self.show = self.options.get("show", "total+selection")
			self.option_show = Option(page_widget, "show", show_options, getter=attrgetter(self, "show"), setter=attrsetter(self, "show"), update=self.signal_plot_dirty.emit)
			row = self.option_show.add_to_grid_layout(row, grid_layout)

		if self.dimensions >= 2:
			transparancies = ["intensity", "constant", "none"]
			self.transparancy = self.options.get("transparancy", "constant")
			self.option_transparancy = Option(page_widget, "transparancy", transparancies, getter=attrgetter(self, "transparancy"), setter=attrsetter(self, "transparancy"), update=self.signal_plot_dirty.emit)
			row = self.option_transparancy.add_to_grid_layout(row, grid_layout)

		self.slider_layer_alpha = Slider(page_widget, "opacity", 0, 1, 1000, getter=attrgetter(self.state, "alpha"), setter=attrsetter(self, "alpha"), update=self.signal_plot_dirty.emit)
		row = self.slider_layer_alpha.add_to_grid_layout(row, grid_layout)

		if self.dimensions >= 2:
			self.slider_layer_level_min = Slider(page_widget, "level_min", 0, 1, 1000, getter=attrgetter(self, "level_min"), setter=attrsetter(self, "level_min"), update=self.signal_plot_dirty.emit)
			row = self.slider_layer_level_min.add_to_grid_layout(row, grid_layout)

			self.slider_layer_level_max = Slider(page_widget, "level_max", 0, 1, 1000, getter=attrgetter(self, "level_max"), setter=attrsetter(self, "level_max"), update=self.signal_plot_dirty.emit)
			row = self.slider_layer_level_max.add_to_grid_layout(row, grid_layout)

			self.display_type = self.options.get("display_type", "colormap")
			self.option_display_type = Option(page_widget, "display", ["colormap", "solid", "contour"], getter=attrgetter(self, "display_type"), setter=attrsetter(self, "display_type"), update=self.signal_plot_dirty.emit)
			row = self.option_display_type.add_to_grid_layout(row, grid_layout)


		colors =["red", "green", "blue", "orange", "cyan", "magenta", "black", "gold", "purple"]
		default_color = colors[self.plot_window.layers.index(self)]
		self.color = self.options.get("color", default_color)
		self.option_solid_color = Option(page_widget, "color", colors, getter=attrgetter(self, "color"), setter=attrsetter(self, "color"), update=self.signal_plot_dirty.emit)
		row = self.option_solid_color.add_to_grid_layout(row, grid_layout)

		colors =["red", "green", "blue", "orange", "cyan", "magenta", "black", "gold", "purple"]
		default_color = colors[-1-self.plot_window.layers.index(self)]
		self.color_alt = self.options.get("color_alt", default_color)
		self.option_solid_color_alt = Option(page_widget, "color_alt", colors, getter=attrgetter(self, "color_alt"), setter=attrsetter(self, "color_alt"), update=self.signal_plot_dirty.emit)
		row = self.option_solid_color_alt.add_to_grid_layout(row, grid_layout)

		if self.dimensions == 1:

			self.display_type = self.options.get("display_type", "bar")
			self.option_display_type = Option(page_widget, "display", ["bar", "line"], getter=attrgetter(self, "display_type"), setter=attrsetter(self, "display_type"), update=self.signal_plot_dirty.emit)
			row = self.option_display_type.add_to_grid_layout(row, grid_layout)


			self._normalize = eval(self.options.get("normalize", "False"))
			self.checkbox_normalize = Checkbox(page_widget, "normalize", getter=attrgetter(self, "_normalize"), setter=attrsetter(self, "_normalize"), update=self.signal_plot_dirty.emit)
			row = self.checkbox_normalize.add_to_grid_layout(row, grid_layout)

			self._cumulative = eval(self.options.get("cumulative", "False"))
			self.checkbox_cumulative = Checkbox(page_widget, "cumulative", getter=attrgetter(self, "_cumulative"), setter=attrsetter(self, "_cumulative"), update=self.signal_plot_dirty.emit)
			row = self.checkbox_cumulative.add_to_grid_layout(row, grid_layout)


		if self.dimensions > 1:
			vaex.ui.colormaps.process_colormaps()
			self.colormap_box = QtGui.QComboBox(page_widget)
			self.colormap_box.setIconSize(QtCore.QSize(16, 16))
			model = QtGui.QStandardItemModel(self.colormap_box)
			for colormap_name in vaex.ui.colormaps.colormaps:
				colormap = matplotlib.cm.get_cmap(colormap_name)
				pixmap = vaex.ui.colormaps.colormap_pixmap[colormap_name]
				icon = QtGui.QIcon(pixmap)
				item = QtGui.QStandardItem(icon, colormap_name)
				model.appendRow(item)
			self.colormap_box.setModel(model);
			#self.form_layout.addRow("colormap=", self.colormap_box)
			self.label_colormap = QtGui.QLabel("colormap=")
			grid_layout.addWidget(self.label_colormap, row, 0)
			grid_layout.addWidget(self.colormap_box, row, 1, QtCore.Qt.AlignLeft)
			def onColorMap(index):
				colormap_name = str(self.colormap_box.itemText(index))
				logger.debug("selected colormap: %r" % colormap_name)
				self.state.colormap = colormap_name
				if hasattr(self, "widget_volume"):
					self.plugins_map["transferfunction"].tool.colormap = self.state.colormap
					self.plugins_map["transferfunction"].tool.update()
					self.widget_volume.colormap_index = index
					self.widget_volume.update()
				#self.plot()
				self.signal_plot_dirty.emit(self)
			cmapnames = "cmap colormap colourmap".split()
			if not set(cmapnames).isdisjoint(self.options):
				for name in cmapnames:
					if name in self.options:
						break
				cmap = self.options[name]
				if cmap not in vaex.ui.colormaps.colormaps:
					colormaps_sorted = sorted(vaex.ui.colormaps.colormaps)
					colormaps_string = " ".join(colormaps_sorted)
					dialogs.dialog_error(self, "Wrong colormap name", "colormap {cmap} does not exist, choose between: {colormaps_string}".format(**locals()))
					index = 0
				else:
					index = vaex.ui.colormaps.colormaps.index(cmap)
				self.colormap_box.setCurrentIndex(index)
				self.state.colormap = vaex.ui.colormaps.colormaps[index]
			self.colormap_box.currentIndexChanged.connect(onColorMap)

		row += 1

		self.contour_count = int(self.options.get("contour_count", 4))
		self.slider_contour_count = Slider(page_widget, "contour_count", 0, 20, 20, getter=attrgetter(self, "contour_count"), setter=attrsetter(self, "contour_count"), update=self.signal_plot_dirty.emit, format="{0:<3d}", numeric_type=int)
		row = self.slider_contour_count.add_to_grid_layout(row, grid_layout)

	def page_annotate(self, page):
		#self.frame_options_vector2d = page #QtGui.QFrame(self)
		#self.layout_frame_options_vector2d =  QtGui.QVBoxLayout()
		#self.frame_options_vector2d.setLayout(self.layout_frame_options_vector2d)
		#self.layout_frame_options_vector2d.setSpacing(0)
		#self.layout_frame_options_vector2d.setContentsMargins(0,0,0,0)
		#self.layout_frame_options_vector2d.setAlignment(QtCore.Qt.AlignTop)

		self.grid_layout_annotate = QtGui.QGridLayout()
		self.grid_layout_annotate.setColumnStretch(1, 1)
		self.grid_layout_annotate.setSpacing(0)
		self.grid_layout_annotate.setContentsMargins(2,1,2,1)
		self.grid_layout_annotate.setAlignment(QtCore.Qt.AlignTop)
		page.setLayout(self.grid_layout_annotate)
		row = 0

		def get():
			return self.state.title
		def set(value):
			self.state.title = value
			self.plot_window.queue_history_change("changed title to %s" % (value))
			self.plot_window.queue_push_full_state()
		#def default():
		#	#return "default label"
		#	return self.plot_window.get_default_label(0)
		self.option_title = TextOption(page, "title", self.state.title, None, get, set, self.signal_plot_dirty.emit)
		row = self.option_title.add_to_grid_layout(row, self.grid_layout_annotate)



		self.state.labels = []

		self.state.labels.append(self.options.get("label_x"))
		def get():
			return self.state.labels[0]
		def set(value):
			self.state.labels[0] = value
			self.plot_window.queue_history_change("changed label_x to %s" % (value))
			self.plot_window.queue_push_full_state()
		def default():
			#return "default label"
			return self.plot_window.get_default_label(0)
		self.option_label_x = TextOption(page, "label_x", self.state.labels[0], default, get, set, self.signal_plot_dirty.emit)
		row = self.option_label_x.add_to_grid_layout(row, self.grid_layout_annotate)

		self.state.labels.append(self.options.get("label_y"))
		def get():
			return self.state.labels[1]
		def set(value):
			self.state.labels[1] = value
			self.plot_window.queue_history_change("changed label_y to %s" % (value))
			self.plot_window.queue_push_full_state()
		def default():
			#return "default label"
			return self.plot_window.get_default_label(1)
		self.option_label_y = TextOption(page, "label_y", self.state.labels[1], default, get, set, self.signal_plot_dirty.emit)
		row = self.option_label_y.add_to_grid_layout(row, self.grid_layout_annotate)

		if self.dimensions > 2:
			self.state.labels.append(self.options.get("label_z"))
			def get():
				return self.state.labels[2]
			def set(value):
				self.state.labels[2] = value
				self.plot_window.queue_history_change("changed label_z to %s" % (value))
				self.plot_window.queue_push_full_state()
			def default():
				#return "default label"
				return self.plot_window.get_default_label(2)
			self.option_label_z = TextOption(page, "label_z", self.state.labels[2], default, get, set, self.signal_plot_dirty.emit)
			row = self.option_label_z.add_to_grid_layout(row, self.grid_layout_annotate)


		def get():
			return self.state.colorbar
		def set(value):
			self.state.colorbar = value
			self.plot_window.queue_history_change("enabled colorbar" if value else "disabled colorbar")
			self.plot_window.queue_push_full_state()
		self.state.colorbar = eval(self.options.get("colorbar", "True"))
		self.colorbar_checkbox = Checkbox(page, "colorbar", getter=get, setter=set, update=self.signal_plot_dirty.emit)
		row = self.colorbar_checkbox.add_to_grid_layout(row, self.grid_layout_annotate)


	def page_slice(self, page):
		class PageWrapper(object):
			def __init__(self, layer, page_widget):
				self.layer = layer
				self.page_widget = page_widget
				self.layout_page =  QtGui.QVBoxLayout()
				self.page_widget.setLayout(self.layout_page)
				self.layout_page.setSpacing(0)
				self.layout_page.setContentsMargins(0,0,0,0)
				self.layout_page.setAlignment(QtCore.Qt.AlignTop)
				self.row = 0

				self.grid_layout = QtGui.QGridLayout()
				self.grid_layout.setColumnStretch(2, 1)
				self.layout_page.addLayout(self.grid_layout)


			def add(self, name, widget):
				self.grid_layout.addWidget(QtGui.QLabel(name), self.row, 0)
				self.grid_layout.addWidget(widget, self.row, 1)
				self.row += 1

			def add_slider_linear(self, name, value, min, max, steps=1000):
				def getter():
					return getattr(self.layer, name)
				def setter(value):
					setattr(self.layer, name, value)
				slider = Slider(self.page_widget, name, min, max, steps, getter, setter)
				self.row = slider.add_to_grid_layout(self.row, self.grid_layout)

		page = PageWrapper(self, page)
		self.menu_button_slice_link = QtGui.QPushButton("No link", page.page_widget)
		self.menu_slice_link = QtGui.QMenu()
		self.menu_button_slice_link.setMenu(self.menu_slice_link)

		action = QtGui.QAction("unlink", self.menu_slice_link)
		action.triggered.connect(lambda *x: self.slice_unlink())
		self.menu_slice_link.addAction(action)


		for window in self.plot_window.app.windows:
			layers = [layer for layer in window.layers if layer.dataset == self.dataset]
			if layers:
				menu_window = QtGui.QMenu(window.name, self.menu_slice_link)
				self.menu_slice_link.addMenu(menu_window)
				#self.menu_slice_link.add
				for layer in window.layers:
					#menu_layer = QtGui.QMenu(layer.name)
					action = QtGui.QAction(layer.name, self.menu_slice_link)
					def on_link(ignore=None, layer=layer):
						self.slice_link(layer)
					action.triggered.connect(on_link)
					menu_window.addAction(action)
				#self.menu_slice_link.

		page.add("slice_link", self.menu_button_slice_link)
		self.signal_slice_change = vaex.events.Signal("slice changed")

		self._slice_radius = 0.1
		page.add_slider_linear("slice_radius", self.slice_radius, 0.0, 1.0)

	@property
	def slice_radius(self): return self._slice_radius
	@slice_radius.setter
	def slice_radius(self, value):
		self._slice_radius = value
		self.plot_window.slice_radius = value
		self.plot_window.setMode(self.plot_window.lastAction)

	def slice_link(self, layer):
		if self.plot_window.state.grid_size != layer.plot_window.state.grid_size:
			msg = "Source layer has a gridsize of %d, while the linked layer has a gridsize of %d, only linking with equal gridsize is supported" % (self.plot_window.state.grid_size, layer.plot_window.state.grid_size)
			dialogs.dialog_error(self.plot_window, "Unequal gridsize", msg)
			return
		dim = self.plot_window.dimensions * layer.plot_window.dimensions
		bytes_required = (layer.plot_window.state.grid_size ** dim) * 8
		if memory_check_ok(self.plot_window, bytes_required):
			name = layer.plot_window.name + "." + layer.name
			self.menu_button_slice_link.setText(name)
			self.slice_unlink()

			self.layer_slice_source = layer
			self.slice_axis = [True] * layer.dimensions
			shape = (layer.plot_window.state.grid_size, ) * layer.dimensions
			self.slice_selection_grid = np.ones(shape, dtype=np.bool)
			self.layer_slice_source.signal_slice_change.connect(self.on_slice_change)
			self.layer_slice_source.signal_needs_update.connect(self.on_slice_source_needs_update)
			self.update()

	def slice_unlink(self):
		if self.layer_slice_source is not None:
			self.layer_slice_source.signal_slice_change.disconnect(self.on_slice_change)
			self.layer_slice_source.signal_needs_update.disconnect(self.on_slice_source_needs_update)
			self.layer_slice_source = None
			self.update()

	def on_slice_source_needs_update(self):
		self.update()

	def on_slice_change(self, selection_grid, clicked):
		self.slice_selection_grid = selection_grid
		self.calculate_amplitudes()
		self.signal_plot_dirty.emit()


	def page_selection(self, page):
		self.layout_page_selection =  QtGui.QVBoxLayout()
		page.setLayout(self.layout_page_selection)
		self.layout_page_selection.setSpacing(0)
		self.layout_page_selection.setContentsMargins(0,0,0,0)
		self.layout_page_selection.setAlignment(QtCore.Qt.AlignTop)

		#button_layout = QtGui.QVBoxLayout()

		self.button_selection_undo = QtGui.QPushButton(QtGui.QIcon(iconfile('undo')), "Undo", page )
		self.button_selection_redo = QtGui.QPushButton(QtGui.QIcon(iconfile('redo')), "Redo", page)
		self.layout_page_selection.addWidget(self.button_selection_undo)
		self.layout_page_selection.addWidget(self.button_selection_redo)
		def on_undo(checked=False):
			self.dataset.selection_undo()
			self.check_selection_undo_redo()
		def on_redo(checked=False):
			self.dataset.selection_redo()
			self.check_selection_undo_redo()
		self.button_selection_undo.clicked.connect(on_undo)
		self.button_selection_redo.clicked.connect(on_redo)
		self.check_selection_undo_redo()

		self.label_selection_info = QtGui.QLabel("should not see me", page)
		self.layout_page_selection.addWidget(self.label_selection_info)
		#self.label_selection_info_update()

		def on_select_expression():
			logger.debug("making selection by expression")
			all = storage_expressions.get_all("selection", self.dataset)
			expressions = []
			for stored in all:
				for ex in stored["options"]["expressions"]:
					if ex not in expressions:
						expressions.append(ex)
			for column in self.dataset.get_column_names():
				ex = "%s < 0" % column
				if ex not in expressions:
					expressions.append(ex)
			cancelled = False
			while not cancelled:
				expression = dialogs.choose(self.plot_window, "Give expression", "Expression for selection: ", expressions, 0, True)
				if not expression:
					cancelled = True
				else:
					expression = str(expression)
					try:
						self.dataset.validate_expression(expression)
					except Exception as e:
						expressions[0] = expression
						self.error_in_field(self.button_selection_expression, "selection", e)
						continue

					if expression not in expressions:
						expressions.insert(0, expression)
					#dialog_info(self.plot_window, "expr", expression)
					storage_expressions.add("", "selection", self.dataset, {"expressions": expressions} )

					mode = self.plot_window.select_mode
					self.dataset.select(expression, mode)
					self.update()
					#mask = self.dataset.mask
					#action = vaex.ui.undo.ActionMask(self.dataset.undo_manager, "expression: " + expression, mask, self.apply_mask)
					#action.do()

					self.check_selection_undo_redo()
					return

					mask = np.zeros(self.dataset._fraction_length, dtype=np.bool)
					t0 = time.time()
					def select(info, blockmask):
						self.message("selection at %.1f%% (%.2fs)" % (info.percentage, time.time() - t0), index=40)
						QtCore.QCoreApplication.instance().processEvents()
						mask[info.i1:info.i2] = self.plot_window.select_mode(None if self.dataset.mask is None else self.dataset.mask[info.i1:info.i2], blockmask == 1)
						#if info.last:
						#	self.message("selection %.2fs" % (time.time() - t0), index=40)

					#layer = self.current_layer
					#if layer is not None:
					if 1:
						self.dataset.evaluate(select, expression, **self.getVariableDict())

						#self.plot_window.checkUndoRedo()
						#self.setMode(self.lastAction)


		self.button_selection_expression = QtGui.QPushButton(QtGui.QIcon(iconfile('undo')), "Add expression", page )
		self.button_selection_expression.clicked.connect(on_select_expression)
		self.layout_page_selection.addWidget(self.button_selection_expression)




	def label_selection_info_update(self, count):
		# TODO: support this again
		#return
		if count is None:
			self.label_selection_info.setText("no selection")
		else:
			N_sel = int(count)
			N_total = len(self.dataset)
			self.label_selection_info.setText("selected {:,} ({:.2f}%)".format(N_sel, N_sel*100./float(N_total)))

	def check_selection_undo_redo(self):
		#if self.widget_build:
		#self.button_selection_undo.setEnabled(self.dataset.undo_manager.can_undo())
		#self.button_selection_redo.setEnabled(self.dataset.undo_manager.can_redo())
		self.button_selection_undo.setEnabled(self.dataset.selection_can_undo())
		self.button_selection_redo.setEnabled(self.dataset.selection_can_redo())



	def page_display(self, page):

		self.frame_options_visuals = page#QtGui.QFrame(self)
		self.layout_frame_options_visuals =  QtGui.QVBoxLayout()
		self.frame_options_visuals.setLayout(self.layout_frame_options_visuals)
		self.layout_frame_options_visuals.setAlignment(QtCore.Qt.AlignTop)

		if self.dimensions > 1:
			if 0: # TODO: reimplement contrast
				self.action_group_constrast = QtGui.QActionGroup(self)
				self.action_image_contrast = QtGui.QAction(QtGui.QIcon(iconfile('contrast')), '&Contrast', self)
				self.action_image_contrast_auto = QtGui.QAction(QtGui.QIcon(iconfile('contrast')), '&Contrast', self)
				self.toolbar2.addAction(self.action_image_contrast)

				self.action_image_contrast.triggered.connect(self.onActionContrast)
				self.contrast_list = [self.contrast_none, functools.partial(self.contrast_none_auto, percentage=0.1) , functools.partial(self.contrast_none_auto, percentage=1), functools.partial(self.contrast_none_auto, percentage=5)]
			self.contrast = self.contrast_none

			if 1:
				self.slider_gamma = QtGui.QSlider(page)
				self.label_gamma = QtGui.QLabel("...", self.frame_options_visuals)
				self.layout_frame_options_visuals.addWidget(self.label_gamma)
				self.layout_frame_options_visuals.addWidget(self.slider_gamma)
				self.slider_gamma.setRange(-100, 100)
				self.slider_gamma.valueChanged.connect(self.onGammaChange)
				self.slider_gamma.setValue(0)
				self.slider_gamma.setOrientation(QtCore.Qt.Horizontal)
				#self.slider_gamma.setMaximumWidth(100)
			self.image_gamma = 1.
			self.update_gamma_label()

			self.image_invert = False
			#self.action_image_invert = QtGui.QAction(QtGui.QIcon(iconfile('direction')), 'Invert image', self)
			#self.action_image_invert.setCheckable(True)
			#self.action_image_invert.triggered.connect(self.onActionImageInvert)
			#self.toolbar2.addAction(self.action_image_invert)
			self.button_image_invert = QtGui.QPushButton(QtGui.QIcon(iconfile('direction')), 'Invert image', self.frame_options_visuals)
			self.button_image_invert.setCheckable(True)
			self.button_image_invert.setAutoDefault(False)
			self.button_image_invert.clicked.connect(self.onActionImageInvert)
			self.layout_frame_options_visuals.addWidget(self.button_image_invert)


	def create_slider(self, parent, label_text, value_min, value_max, getter, setter, value_steps=1000, format=" {0:<0.3f}", transform=lambda x: x, inverse=lambda x: x):
		label = QtGui.QLabel(label_text, parent)
		label_value = QtGui.QLabel(label_text, parent)
		slider = QtGui.QSlider(parent)
		slider.setOrientation(QtCore.Qt.Horizontal)
		slider.setRange(0, value_steps)

		def update_text():
			#label.setText("mean/sigma: {0:<0.3f}/{1:.3g} opacity: {2:.3g}".format(self.tool.function_means[i], self.tool.function_sigmas[i], self.tool.function_opacities[i]))
			label_value.setText(format.format(getter()))
		def on_change(index, slider=slider):
			value = index/float(value_steps) * (inverse(value_max) - inverse(value_min)) + inverse(value_min)
			setter(transform(value))
			update_text()
		slider.setValue((inverse(getter()) - inverse(value_min))/(inverse(value_max) - inverse(value_min)	) * value_steps)
		update_text()
		slider.valueChanged.connect(on_change)
		return label, slider, label_value

	def create_checkbox(self, parent, label, getter, setter):
		checkbox = QtGui.QCheckBox(label, parent)
		checkbox.setChecked(getter())
		def stateChanged(state):
			value = state == QtCore.Qt.Checked
			setter(value)

		checkbox.stateChanged.connect(stateChanged)
		return checkbox

	def page_vector(self, page):
		self.frame_options_vector2d = page #QtGui.QFrame(self)
		self.layout_frame_options_vector2d =  QtGui.QVBoxLayout()
		self.frame_options_vector2d.setLayout(self.layout_frame_options_vector2d)
		self.layout_frame_options_vector2d.setSpacing(0)
		self.layout_frame_options_vector2d.setContentsMargins(0,0,0,0)
		self.layout_frame_options_vector2d.setAlignment(QtCore.Qt.AlignTop)

		self.grid_layout_vector = QtGui.QGridLayout()
		self.grid_layout_vector.setColumnStretch(2, 1)
		self.layout_frame_options_vector2d.addLayout(self.grid_layout_vector)

		row = 0

		self.vectors_subtract_mean = bool(eval(self.options.get("vsub_mean", "False")))
		def setter(value):
			self.vectors_subtract_mean = value
			#self.plot()
			self.signal_plot_dirty.emit()
		self.vector_subtract_mean_checkbox = self.create_checkbox(page, "subtract mean", lambda : self.vectors_subtract_mean, setter)
		self.grid_layout_vector.addWidget(self.vector_subtract_mean_checkbox, row, 1)
		row += 1

		self.vectors_color_code_3rd = bool(eval(self.options.get("vcolor_3rd", "True" if self.dimensions <=2 else "False")))
		def setter(value):
			self.vectors_color_code_3rd = value
			#self.plot()
			self.signal_plot_dirty.emit()
		self.vectors_color_code_3rd_checkbox = self.create_checkbox(page, "color code 3rd axis", lambda : self.vectors_color_code_3rd, setter)
		self.grid_layout_vector.addWidget(self.vectors_color_code_3rd_checkbox, row, 1)
		row += 1

		self.vector_auto_scale = (eval(self.options.get("vector_auto_scale", "True")))
		def setter(value):
			self.vector_auto_scale = value
			#self.plot()
			self.signal_plot_dirty.emit()
		self.vector_auto_scale_checkbox = self.create_checkbox(page, "vector_auto_scale", lambda : self.vector_auto_scale, setter)
		self.grid_layout_vector.addWidget(self.vector_auto_scale_checkbox, row, 1)
		row += 1

		self.vector_level_min = float(eval(self.options.get("vector_level_min", "0")))
		self.slider_layer_level_min = Slider(page, "vector_level_min", 0, 1, 1000, getter=attrgetter(self, "vector_level_min"),
		                                     setter=attrsetter(self, "vector_level_min"), update=self.signal_plot_dirty.emit,
		                                     )#inverse=lambda x: math.log10(x), transform=lambda x: 10**x)
		row = self.slider_layer_level_min.add_to_grid_layout(row, self.grid_layout_vector)

		self.vector_level_max = float(eval(self.options.get("vector_level_max", "1.0")))
		self.slider_layer_level_max = Slider(page, "vector_level_max", 0, 1, 1000, getter=attrgetter(self, "vector_level_max"),
		                                     setter=attrsetter(self, "vector_level_max"), update=self.signal_plot_dirty.emit,
		                                     )#inverse=lambda x: math.log10(x), transform=lambda x: 10**x)
		row = self.slider_layer_level_max.add_to_grid_layout(row, self.grid_layout_vector)


		self.vector_scale = 1.
		self.slider_vector_scale = Slider(page, "vector_scale", 0.01, 100, 100, getter=attrgetter(self, "vector_scale"),
		                                     setter=multilayer_attrsetter(self, "vector_scale"), update=self.signal_plot_dirty.emit, uselog=True)
		                                     #format=" {0:>05.2f}", transform=lambda x: 10**x, inverse=lambda x: float(np.log10(x))
		#)#inverse=lambda x: math.log10(x), transform=lambda x: 10**x)
		row = self.slider_vector_scale.add_to_grid_layout(row, self.grid_layout_vector)

		self.vector_head_width = 1
		self.slider_vector_head_width = Slider(page, "vector_head_width", 0.01, 100, 100, getter=attrgetter(self, "vector_head_width"),
		                                     setter=multilayer_attrsetter(self, "vector_head_width"), update=self.signal_plot_dirty.emit, uselog=True)
		                                     #format=" {0:>05.2f}", transform=lambda x: 10**x, inverse=lambda x: float(np.log10(x))
		#)#inverse=lambda x: math.log10(x), transform=lambda x: 10**x)
		row = self.slider_vector_head_width.add_to_grid_layout(row, self.grid_layout_vector)


		self.vector_boxes = []
		if self.dimensions > -1:
			self.weight_x_box = vaex.ui.completer.ExpressionCombobox(page, self.dataset, variables=True)#QtGui.QComboBox(page)
			self.weight_x_box.setMinimumContentsLength(10)
			self.weight_x_box.setEditable(True)
			#self.weight_x_box.addItems([self.options.get("vx", "")] + self.get_expression_list())
			self.weight_x_box.lineEdit().setText(self.options.get("vx", ""))
			self.weight_x_box.setMinimumContentsLength(10)
			self.grid_layout_vector.addWidget(QtGui.QLabel("vx="), row, 0)
			self.grid_layout_vector.addWidget(self.weight_x_box, row, 1)
			#def onWeightXExprLine(*args, **kwargs):
			#	if len(str(self.weight_x_box.lineEdit().text())) == 0:
			#		self.onWeightXExpr()
			self.weight_x_box.lineEdit().editingFinished.connect(lambda _=None: self.onWeightXExpr())
			self.weight_x_box.currentIndexChanged.connect(lambda _=None: self.onWeightXExpr())
			self.state.vector_expressions[0] = str(self.weight_x_box.lineEdit().text())
			self.vector_boxes.append(self.weight_x_box)
			if 0:
				for name in "x y z".split():
					if name in self.state.expressions[0]:
						for prefix in "v v_".split():
							expression = (prefix+name)
							if expression in self.get_expression_list():
								self.weight_x_box.lineEdit().setText(expression)
								self.state.vector_expressions[0] = expression

			row += 1

		if self.dimensions > -1:
			self.weight_y_box = vaex.ui.completer.ExpressionCombobox(page, self.dataset, variables=True)#QtGui.QComboBox(page)
			self.weight_y_box.setEditable(True)
			#self.weight_y_box.addItems([self.options.get("vy", "")] + self.get_expression_list())
			self.weight_y_box.lineEdit().setText(self.options.get("vy", ""))
			self.weight_y_box.setMinimumContentsLength(10)
			self.grid_layout_vector.addWidget(QtGui.QLabel("vy="), row, 0)
			self.grid_layout_vector.addWidget(self.weight_y_box, row, 1)
			#def onWeightYExprLine(*args, **kwargs):
			#	if len(str(self.weight_y_box.lineEdit().text())) == 0:
			#		self.onWeightYExpr()
			self.weight_y_box.lineEdit().editingFinished.connect(lambda _=None: self.onWeightYExpr())
			self.weight_y_box.currentIndexChanged.connect(lambda _=None: self.onWeightYExpr())
			self.state.vector_expressions[1] = str(self.weight_y_box.lineEdit().text())
			self.vector_boxes.append(self.weight_y_box)
			if 0:
				for name in "x y z".split():
					if self.dimensions > 1:
						if name in self.state.expressions[1]:
							for prefix in "v v_".split():
								expression = (prefix+name)
								if expression in self.get_expression_list():
									self.weight_y_box.lineEdit().setText(expression)
									self.state.vector_expressions[0] = expression

			row += 1

			self.weight_z_box = vaex.ui.completer.ExpressionCombobox(page, self.dataset, variables=True) #QtGui.QComboBox(page)
			self.weight_z_box.setEditable(True)
			#self.weight_z_box.addItems([self.options.get("vz", "")] + self.get_expression_list())
			self.weight_z_box.lineEdit().setText(self.options.get("vz", ""))
			self.weight_z_box.setMinimumContentsLength(10)
			self.grid_layout_vector.addWidget(QtGui.QLabel("vz="), row, 0)
			self.grid_layout_vector.addWidget(self.weight_z_box, row, 1)
			#def onWeightZExprLine(*args, **kwargs):
			#	if len(str(self.weight_z_box.lineEdit().text())) == 0:
			#		self.onWeightZExpr()
			self.weight_z_box.lineEdit().editingFinished.connect(lambda _=None: self.onWeightZExpr())
			self.weight_z_box.currentIndexChanged.connect(lambda _=None: self.onWeightZExpr())
			self.state.vector_expressions[2] = str(self.weight_z_box.lineEdit().text())
			self.vector_boxes.append(self.weight_z_box)

			row += 1

		self.vector_dimensions = len(self.vector_boxes)
		self.vector_names = "vx vy vz".split()[:self.vector_dimensions]

		if self.dimensions > -1:
			vaex.ui.colormaps.process_colormaps()
			self.colormap_vector_box = QtGui.QComboBox(page)
			self.colormap_vector_box.setIconSize(QtCore.QSize(16, 16))
			model = QtGui.QStandardItemModel(self.colormap_vector_box)
			for colormap_name in vaex.ui.colormaps.colormaps:
				colormap = matplotlib.cm.get_cmap(colormap_name)
				pixmap = vaex.ui.colormaps.colormap_pixmap[colormap_name]
				icon = QtGui.QIcon(pixmap)
				item = QtGui.QStandardItem(icon, colormap_name)
				model.appendRow(item)
			self.colormap_vector_box.setModel(model);
			#self.form_layout.addRow("colormap=", self.colormap_vector_box)
			self.grid_layout_vector.addWidget(QtGui.QLabel("vz_cmap="), row, 0)
			self.grid_layout_vector.addWidget(self.colormap_vector_box, row, 1, QtCore.Qt.AlignLeft)
			def onColorMap(index):
				colormap_name = str(self.colormap_vector_box.itemText(index))
				logger.debug("selected colormap for vector: %r" % colormap_name)
				self.state.colormap_vector = colormap_name
				#self.plot()
				self.signal_plot_dirty.emit()

			cmapnames = "vz_cmap vz_colormap vz_colourmap".split()
			if not set(cmapnames).isdisjoint(self.options):
				for name in cmapnames:
					if name in self.options:
						break
				cmap = self.options[name]
				if cmap not in vaex.ui.colormaps.colormaps:
					colormaps_sorted = sorted(vaex.ui.colormaps.colormaps)
					colormaps_string = " ".join(colormaps_sorted)
					dialog_error(self, "Wrong colormap name", "colormap {cmap} does not exist, choose between: {colormaps_string}".format(**locals()))
					index = 0
				else:
					index = vaex.ui.colormaps.colormaps.index(cmap)
				self.colormap_vector_box.setCurrentIndex(index)
				self.state.colormap_vector = vaex.ui.colormaps.colormaps[index]
			else:
				index = vaex.ui.colormaps.colormaps.index(self.state.colormap_vector)
				self.colormap_vector_box.setCurrentIndex(index)
			self.colormap_vector_box.currentIndexChanged.connect(onColorMap)

			row += 1

		#self.toolbox.addItem(self.frame_options_main, " Main")
		#self.toolbox.addItem(self.frame_options_vector2d, "Vector 2d")
		#self.toolbox.addItem(self.frame_options_visuals, "Display")
		#self.add_pages(self.toolbox)



		#self.form_layout = QtGui.QFormLayout()


		#self.setStatusBar(self.status_bar)
		#layout.setMargin(0)
		#self.grid_layout.setMargin(0)
		self.grid_layout.setHorizontalSpacing(0)
		self.grid_layout.setVerticalSpacing(0)
		self.grid_layout.setContentsMargins(0, 0, 0, 0)

		self.button_layout.setContentsMargins(0, 0, 0, 0)
		self.button_layout.setSpacing(0)
		#self.form_layout.setContentsMargins(0, 0, 0, 0)
		#self.form_layout.setSpacing(0)
		self.grid_layout.setContentsMargins(0, 0, 0, 0)
		self.messages = {}
		#super(self.__class__, self).afterLayout()



		#self.add_shortcut(self.action_fullscreen, "F")
		#self.add_shortcut(self.action_undo, "Ctrl+Z")
		#self.add_shortcut(self.action_redo, "Alt+Y")

		#self.add_shortcut(self.action_display_mode_both, "1")
		#self.add_shortcut(self.action_display_mode_full, "2")
		#self.add_shortcut(self.action_display_mode_selection, "3")
		#self.add_shortcut(self.action_display_mode_both_contour, "4")

		#if "zoom" in self.options:
		#	factor = eval(self.options["zoom"])
		#	self.zoom(factor)
		#self.checkUndoRedo()

	def onActionImageInvert(self, ignore=None):
		self.image_invert = self.button_image_invert.isChecked()
		self.plot()

	def update_gamma_label(self):
		text = "gamma=%.3f" % self.image_gamma
		self.label_gamma.setText(text)

	def onGammaChange(self, gamma_index):
		self.image_gamma = 10**(gamma_index / 100./2)
		self.update_gamma_label()
		self.queue_replot()

	def normalize(self, array):
		#return (array - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array))
		return array

	def image_post(self, array):
		return -array if self.image_invert else array

	def contrast_none(self, array):
		return self.image_post(self.normalize(array)**(self.image_gamma))

	def contrast_none_auto(self, array, percentage=1.):
		values = array.reshape(-1)
		mask = np.isinf(values)
		values = values[~mask]
		indices = np.argsort(values)
		min, max = np.nanmin(values), np.nanmax(values)
		N = len(values)
		i1, i2 = int(N * percentage / 100), int(N-N * percentage / 100)
		v1, v2 = values[indices[i1]], values[indices[i2]]
		return self.image_post(self.normalize(np.clip(array, v1, v2))**self.image_gamma)

	def onActionContrast(self):
		index = self.contrast_list.index(self.contrast)
		next_index = (index + 1) % len(self.contrast_list)
		self.contrast = self.contrast_list[next_index]
		self.plot()

	def validate_all_fields(self):
		for i in range(self.dimensions):
			logger.debug("validating %r", self.state.expressions[i])
			try:
				self.dataset.validate_expression(self.state.expressions[i])
			except Exception as e:
				self.error_in_field(self.axisboxes[i], self.axis_names[i], e)
				return False
		for i in range(self.vector_dimensions):
			logger.debug("validating %r", self.state.vector_expressions[i])
			try:
				if self.state.vector_expressions[i]:
					self.dataset.validate_expression(self.state.vector_expressions[i])
			except Exception as e:
				self.error_in_field(self.vector_boxes[i], self.vector_names[i], e)
				return False

		try:
			if self.state.weight_expression:
				self.dataset.validate_expression(self.state.weight_expression)
		except Exception as e:
			self.error_in_field(self.weight_box, "weight", e)
			return False
		return True

	def update(self):
		if self.validate_all_fields():
			self.flag_needs_update()
			self.plot_window.queue_update()

from vaex.dataset import Dataset, Task
from vaex.ui.plot_windows import PlotDialog
