import collections
import operator
import math
import gavifast
import matplotlib
import gavifast
import gavi
import gavi.vaex.storage
import gavi.vaex.undo
import gavi.vaex.colormaps
import gavi.vaex.grids
from gavi.icons import iconfile
import numpy as np
import scipy.ndimage
import matplotlib.colors
import gavi.utils
__author__ = 'maartenbreddels'

import copy
import functools
import time
from qt import *

logger = gavi.logging.getLogger("gavi.vaex")

storage_expressions = gavi.vaex.storage.Storage("expressions")


#options.define_options("grid_size", int, validator=options.is_power_of_two)
class LinkButton(QtGui.QToolButton):
	def __init__(self, title, dataset, axisIndex, parent):
		super(LinkButton, self).__init__(parent)
		self.setToolTip("link this axes with others (experimental and unstable)")
		self.plot = parent
		self.dataset = dataset
		self.axisIndex = axisIndex
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
			self.link = self.dataset.link(self.plot.expressions[self.axisIndex], self)
			self.setIcon(self.connect_icon)
		else:
			logger.debug("disconnecting link")
			self.dataset.unlink(self.link, self)
			self.link = None
			self.setIcon(self.disconnect_icon)

	def onLinkGlobal(self):
		self.link = self.dataset.link(self.plot.expressions[self.axisIndex], self)
		logger.debug("made global link: %r" % self.link)
		#self.parent.links[self.axisIndex] = self.linkHandle

	def onChangeRangeShow(self, range_):
		logger.debug("received range show change for plot=%r, axisIndex %r, range=%r" % (self.plot, self.axisIndex, range_))
		self.plot.ranges_show[self.axisIndex] = range_

	def onChangeRange(self, range_):
		logger.debug("received range change for plot=%r, axisIndex %r, range=%r" % (self.plot, self.axisIndex, range_))
		self.plot.ranges[self.axisIndex] = range_

	def onCompute(self):
		logger.debug("received compute for plot=%r, axisIndex %r" % (self.plot, self.axisIndex))
		self.plot.compute()

	def onPlot(self):
		logger.debug("received plot command for plot=%r, axisIndex %r" % (self.plot, self.axisIndex))
		self.plot.plot()

	def onLinkLimits(self, min, max):
		self.plot.expressions[self.axisIndex] = expression

	def onChangeExpression(self, expression):
		logger.debug("received change expression for plot=%r, axisIndex %r, expression=%r" % (self.plot, self.axisIndex, expression))
		self.plot.expressions[self.axisIndex] = expression
		self.plot.axisboxes[self.axisIndex].lineEdit().setText(expression)



	def _dragEnterEvent(self, e):
		print e.mimeData()
		print e.mimeData().text()
		if e.mimeData().hasFormat('text/plain'):
			e.accept()

		else:
			e.ignore()

	def dropEvent(self, e):
		position = e.pos()
		#self.button.move(position)
		print "do", e.mimeData().text()
		e.setDropAction(QtCore.Qt.MoveAction)
		e.accept()

	def _mousePressEvent(self, e):

			super(LinkButton, self).mousePressEvent(e)

			if e.button() == QtCore.Qt.LeftButton:
				print 'press'

	def _mouseMoveEvent(self, e):
		if e.buttons() != QtCore.Qt.LeftButton:
			return

		mimeData = QtCore.QMimeData()

		drag = QtGui.QDrag(self)
		drag.setMimeData(mimeData)
		drag.setHotSpot(e.pos() - self.rect().topLeft())
		mimeData.setText("blaat")

		dropAction = drag.start(QtCore.Qt.MoveAction)



class LayerTable(object):
	def __init__(self, plot_window, name, dataset, expressions, axis_names, options, jobs_manager, thread_pool, figure, canvas, ranges_grid=None):
		self.plot_window = plot_window
		self.name = name
		self.dataset = dataset
		self.expressions = expressions
		self.axis_names = axis_names
		self.ranges_grid = ranges_grid
		self.range_level = None
		self.jobs_manager = jobs_manager
		self.thread_pool = thread_pool
		self.dimensions = len(self.expressions)
		self.options = options
		self.grids = gavi.vaex.grids.Grids(self.dataset, self.thread_pool, *expressions)
		self.grids.ranges = self.ranges_grid
		self.expressions_vector = [None,] * 3
		self.figure = figure
		self.canvas = canvas
		self.widget_build = False

		self.widget = None # each layer has a widget, atm only a qt widget is implemented

		self.weight_expression = None

		self.compute_counter = 0
		self.sequence_index = 0
		self.alpha = float(self.options.get("alpha", "1."))
		#self.color = self.options.get("color")
		self.level_min = 0.
		self.level_max = 1.
		#self.use_intensity = bool(self.options.get("use_intensity", True))

		self.coordinates_picked_row = None



		self.colormap = "PaulT_plusmin" #"binary"
		self.colormap_vector = "binary"
		if "lim" in self.options:
			for i in range(self.dimensions):
				self.ranges_grid[i] = eval(self.options["lim"])
		if "xlim" in self.options:
			self.ranges_grid[0] = eval(self.options["xlim"])
		if "ylim" in self.options:
			self.ranges_grid[1] = eval(self.options["ylim"])
		if "zlim" in self.options:
			self.ranges_grid[2] = eval(self.options["zlim"])
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

		self.show_disjoined = False # show p(x,y) as p(x)p(y)




		if self.ranges_grid is None:
			self.submit_job_minmax()

		self.dataset.mask_listeners.append(self.onSelectMask)
		self.dataset.row_selection_listeners.append(self.onSelectRow)
		self.dataset.serie_index_selection_listeners.append(self.onSerieIndexSelect)
		self.plot_density = self.plot_density_imshow
		self.signal_expression_change = gavi.events.Signal("expression_change")
		self.signal_plot_dirty = gavi.events.Signal("plot_dirty")
		self.signal_plot_update = gavi.events.Signal("plot_update")
		#self.dataset.signal_pick.connect(self.on)

	def removed(self):
		self.dataset.mask_listeners.remove(self.onSelectMask)
		self.dataset.row_selection_listeners.remove(self.onSelectRow)
		self.dataset.serie_index_selection_listeners.remove(self.onSerieIndexSelect)
		for plugin in self.plugins:
			plugin.clean_up()



	def create_grid_map(self, gridsize, use_selection):
		locals = {}
		for name in self.grids.grids.keys():
			grid = self.grids.grids[name]
			if name == "counts" or (grid.weight_expression is not None and len(grid.weight_expression) > 0):
				if grid.max_size >= gridsize:
					locals[name] = grid.get_data(gridsize, use_selection=use_selection)
			else:
				locals[name] = None
		for d, name in zip(range(self.dimensions), "xyzw"):
			width = self.ranges_grid[d][1] - self.ranges_grid[d][0]
			offset = self.ranges_grid[d][0]
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

	def error_in_field(self, widget, name, exception):
		dialog_error(widget, "Error in expression", "Invalid expression for field %s: %r" % (name, exception))
		#self.current_tooltip = QtGui.QToolTip.showText(widget.mapToGlobal(QtCore.QPoint(0, 0)), "Error: " + str(exception), widget)
		#self.current_tooltip = QtGui.QToolTip.showText(widget.mapToGlobal(QtCore.QPoint(0, 0)), "Error: " + str(exception), widget)

	def plot(self, axes_list, stack_image):
		import traceback
		#print "stack trace", ''.join(traceback.format_stack())
		grid_map = self.create_grid_map(self.plot_window.grid_size, False)
		try:
			self.amplitude = amplitude = self.eval_amplitude(self.amplitude_expression, locals=grid_map)
		except Exception, e:
			logger.exception("amplitude field")
			self.error_in_field(self.amplitude_box, "amplitude", e)
			return
		self.amplitude_selection = amplitude_selection = None
		use_selection = self.dataset.mask is not None
		if use_selection:
			grid_map_selection = self.create_grid_map(self.plot_window.grid_size, True)
			self.amplitude_selection = amplitude_selection = self.eval_amplitude(self.amplitude_expression, locals=grid_map_selection)

		grid_map_vector = self.create_grid_map(self.plot_window.vector_grid_size, use_selection)
		for callback in self.plugin_grids_draw:
			callback(axes, grid_map, grid_map_vector)

		#return

		locals = {}
		for name in self.grids.grids.keys():
			grid = self.grids.grids[name]
			if name == "counts" or (grid.weight_expression is not None and len(grid.weight_expression) > 0):
				if grid.max_size >= self.plot_window.vector_grid_size:
					locals[name] = grid.get_data(self.plot_window.vector_grid_size, use_selection)
			else:
				locals[name] = None

		index = self.dataset.selected_row_index
		if index is not None and self.coordinates_picked_row is None:
			logger.debug("point selected but after computation")
			# TODO: optimize
			def find_selected_point(info, *blocks):
				if index >= info.i1 and index < info.i2: # selected point is in this block
					self.coordinates_picked_row = [block[index-info.i1] for block in blocks]
			self.dataset.evaluate(find_selected_point, *self.expressions, **self.getVariableDict())


		if self.dimensions == 1:
			mask = ~(np.isnan(amplitude) | np.isinf(amplitude))
			if np.sum(mask) == 0:
				self.range_level = None
			else:
				if self.range_level is None:
					#self.range_level = [np.min(amplitude[mask]), np.max(amplitude[mask])]
					self.range_level = [min(0, np.min(amplitude[mask])), np.max(amplitude[mask])]
				width = self.ranges_grid[0][1] - self.ranges_grid[0][0]
				x = np.arange(0, self.plot_window.grid_size)/float(self.plot_window.grid_size) * width + self.ranges_grid[0][0]# + width/(Nvector/2.)
				delta = x[1] - x[0]
				for axes in axes_list:
					if self.display_type == "bar":
						axes.bar(x, amplitude, width=delta, align='center', alpha=self.alpha, color=self.color)
					else:
						print len(x), len(amplitude)
						dx = x[1] - x[0]
						x2 = list(np.ravel(zip(x,x+dx)))
						x2p = [x[0]] + x2 + [x[-1]+dx]
						y = amplitude
						y2 = list(np.ravel(zip(y,y)))
						y2p = [0] + y2 + [0]
						axes.plot(x2p, y2p, alpha=self.alpha, color=self.color)
					if use_selection:
						if self.display_type == "bar":
							axes.bar(x, amplitude_selection, width=delta, align='center', color=self.color_alt, alpha=0.6*self.alpha)
						else:
							dx = x[1] - x[0]
							x2 = list(np.ravel(zip(x,x+dx)))
							x2p = [x[0]] + x2 + [x[-1]+dx]
							y = amplitude_selection
							y2 = list(np.ravel(zip(y,y)))
							y2p = [0] + y2 + [0]
							axes.plot(x2p, y2p, drawstyle="steps-mid", alpha=self.alpha, color=self.color_alt)
					if self.coordinates_picked_row is not None:
						axes.axvline(self.coordinates_picked_row[axes.xaxis_index], color="red")


		if self.dimensions == 2:
			#for axes in axes_list:
			assert len(axes_list) == 1
			self.plot_density(axes_list[0], amplitude, amplitude_selection, stack_image)
		if self.dimensions >= 2:
			# for vector we only use the selected map, maybe later also show the full dataset
			grid_map_vector = self.create_grid_map(self.plot_window.vector_grid_size, use_selection)

			self.vector_grid = None
			vector_counts = grid_map_vector["counts"]
			vector_mask = vector_counts > 0
			if grid_map_vector["weightx"] is not None:
				vector_x = grid_map_vector["x"] if "x" in grid_map_vector else None
				vx = self.eval_amplitude("weightx/counts", locals=grid_map_vector)
				if self.vectors_subtract_mean:
					vx -= vx[vector_mask].mean()
			else:
				vector_x = None
				vx = None
			if grid_map_vector["weighty"] is not None:
				vector_y = grid_map_vector["y"] if "y" in grid_map_vector else None
				vy = self.eval_amplitude("weighty/counts", locals=grid_map_vector)
				if self.vectors_subtract_mean:
					vy -= vy[vector_mask].mean()
			else:
				vector_y = None
				vy = None
			if grid_map_vector["weightz"] is not None:
				vector_z = grid_map_vector["z"] if "z" in grid_map_vector else None
				vz = self.eval_amplitude("weightz/counts", locals=grid_map_vector)
				if self.vectors_subtract_mean:
					vz -= vz[vector_mask].mean()
			else:
				vector_z = None
				vz = None
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
				# create marginalized grid
				all_axes = range(self.dimensions)
				all_axes.remove(self.dimensions-1-axes.xaxis_index)
				all_axes.remove(self.dimensions-1-axes.yaxis_index)


				grid_map_2d = {key:None if grid is None else (grid if grid.ndim != 3 else gavi.utils.multisum(grid, all_axes)) for key, grid in grid_map.items()}
				amplitude = self.eval_amplitude(self.amplitude_expression, locals=grid_map_2d)
				if use_selection:
					grid_map_selection_2d = {key:None if grid is None else (grid if grid.ndim != 3 else gavi.utils.multisum(grid, all_axes)) for key, grid in grid_map_selection.items()}
					amplitude_selection = self.eval_amplitude(self.amplitude_expression, locals=grid_map_selection_2d)
				else:
					amplitude_selection = None
				self.plot_density(axes, amplitude, amplitude_selection, stack_image)

				if len(all_axes) > 2:
					other_axis = all_axes[0]
					assert len(all_axes) == 1, ">3d not supported"
				else:
					other_axis = 2

				U = vector_grids[axes.xaxis_index]
				V = vector_grids[axes.yaxis_index]
				W = vector_grids[self.dimensions-1-other_axis]
				vx = None if U is None else gavi.utils.multisum(U, all_axes)
				vy = None if V is None else gavi.utils.multisum(V, all_axes)
				vz = None if W is None else gavi.utils.multisum(W, all_axes)
				vector_counts_2d = gavi.utils.multisum(vector_counts, all_axes)
				if vx is not None and vy is not None:
					count_max = vector_counts_2d.max()
					mask = (vector_counts_2d > (self.vector_level_min * count_max)) & \
					        (vector_counts_2d <= (self.vector_level_max * count_max))
					x = vector_positions[axes.xaxis_index]
					y = vector_positions[axes.yaxis_index]
					x2d, y2d = np.meshgrid(x, y)
					if vz is not None and self.vectors_color_code_3rd:
						colors = vz
						axes.quiver(x2d[mask], y2d[mask], vx[mask], vy[mask], colors[mask], cmap=self.colormap_vector)#, scale=1)
					else:
						axes.quiver(x2d[mask], y2d[mask], vx[mask], vy[mask], color="black")
						colors = None
		if self.coordinates_picked_row is not None:
			if self.dimensions >= 2:
				for axes in axes_list:
					axes.scatter([self.coordinates_picked_row[axes.xaxis_index]], [self.coordinates_picked_row[axes.yaxis_index]], color='red')


	def getVariableDict(self):
		return {} # TODO: remove this? of replace

	def plot_density_imshow(self, axes, amplitude, amplitude_selection, stack_image):
		if not self.visible:
			return
		ranges = []
		for minimum, maximum in self.ranges_grid:
			ranges.append(minimum)
			ranges.append(maximum)
		use_selection = amplitude_selection is not None
		#if isinstance(self.colormap, basestring):
		def normalize(amplitude):
			I = amplitude#self.contrast(amplitude)
			# scale to [0,1]
			mask = ~(np.isnan(I) | np.isinf(I))
			if np.sum(mask) == 0:
				return np.zeros(I.shape, dtype=np.float64)
			I -= I[mask].min()
			I /= I[mask].max()
			return I

		def to_rgb(intensity, color, pre_alpha=1.):
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
				cmap = matplotlib.cm.cmap_d[self.colormap]
				rgba = cmap(I * 1.00)
				rgba[...,3] = (np.clip((I**1.0) * self.alpha, 0, 1))
			if self.transparancy == "intensity":
				rgba[...,3] = (np.clip((I**1.0) * self.alpha, 0, 1)) * self.alpha * pre_alpha
			elif self.transparancy == "constant":
				rgba[alpha_mask,3] = 1. * self.alpha * pre_alpha
				rgba[~alpha_mask,3] = 0
			elif self.transparancy == "none":
				rgba[...,3] = pre_alpha
			else:
				raise "not implemented"
			return rgba

		levels = (np.arange(self.contour_count) + 1. ) / (self.contour_count + 1)
		ranges = list(self.ranges_grid[0]) + list(self.ranges_grid[1])


		amplitude_marginalized = amplitude
		amplitude_marginalized_selected = amplitude_selection
		if self.display_type == "contour":
			if self.contour_count > 0:
				if use_selection:
					axes.contour(normalize(amplitude_marginalized), origin="lower", extent=ranges, levels=levels, linewidths=1, colors=self.color, alpha=0.4*self.alpha)
					axes.contour(normalize(amplitude_marginalized_selected), origin="lower", extent=ranges, levels=levels, linewidths=1, colors=self.color_alt, alpha=self.alpha)
				else:
					axes.contour(normalize(amplitude_marginalized), origin="lower", extent=ranges, levels=levels, linewidths=1, colors=self.color, alpha=self.alpha)
		else:
			I = normalize(amplitude_marginalized)
			axes.rgb_images.append(to_rgb(I, color=self.color, pre_alpha=0.4 if use_selection else 1.0))
			if use_selection:
				I = normalize(amplitude_marginalized_selected)
				axes.rgb_images.append(to_rgb(I, color=self.color_alt))


	def onSelectMask(self, mask):
		self.check_selection_undo_redo()
		self.add_jobs()
		self.label_selection_info_update()
		#self.plot()

	def onSelectRow(self, row):
		self.coordinates_picked_row = None
		#self.plot()
		self.signal_plot_dirty.emit(self)

	def onSerieIndexSelect(self, sequence_index):
		if sequence_index != self.sequence_index: # avoid unneeded event
			self.sequence_index = sequence_index
			#self.seriesbox.setCurrentIndex(self.sequence_index)
		else:
			self.sequence_index = sequence_index
		#self.compute()
		#self.signal_plot_update.emit(delay=0)
		self.add_jobs()

	def get_options(self):
		options = collections.OrderedDict()
		#options["type-names"] = map(str.strip, self.names.split(","))
 		options["expressions"] = self.expressions
 		options["expression_weight"] = self.weight_expression
		options["amplitude_expression"] = self.amplitude_expression
		options["ranges_grid"] = self.ranges_grid
		options["weight_x_expression"] = self.weight_x_expression
		options["weight_y_expression"] = self.weight_y_expression
		options["weight_z_expression"] = self.weight_z_expression
		for plugin in self.plugins:
			options.update(plugin.get_options())
		# since options contains reference (like a list of expressions)
		# changes in the gui might reflect previously stored options
		options = copy.deepcopy(options)
		return dict(options)

	def apply_options(self, options, update=True):
		#map = {"expressions",}
		recognize = "expressions expression_weight amplitude_expression ranges_grid aspect weight_x_expression weight_y_expression weight_z_expression".split()
		for key in recognize:
			if key in options.keys():
				value = options[key]
				setattr(self, key, copy.copy(value))
				if key == "amplitude_expression":
					self.amplitude_box.lineEdit().setText(value)
				if key == "expression_weight":
					self.weight_box.lineEdit().setText(value or "")
				if key == "weight_x_expression":
					self.weight_x_box.lineEdit().setText(value or "")
				if key == "weight_y_expression":
					self.weight_y_box.lineEdit().setText(value or "")
				if key == "weight_y_expression":
					self.weight_y_box.lineEdit().setText(value or "")
				if key == "expressions":
					for expr, box in zip(value, self.axisboxes):
						box.lineEdit().setText(expr)
		for plugin in self.plugins:
			plugin.apply_options(options)
		for key in options.keys():
			if key not in recognize:
				logger.error("option %s not recognized, ignored" % key)
		if update:
			self.plot_window.queue_update()



	def plug_toolbar(self, callback, order):
		self.plugin_queue_toolbar.append((callback, order))

	def plug_page(self, callback, pagename, pageorder, order):
		self.plugin_queue_page.append((callback, pagename, pageorder, order))

	def plug_grids(self, callback_define, callback_draw):
		self.plugin_grids_defines.append(callback_define)
		self.plugin_grids_draw.append(callback_draw)

	def apply_mask(self, mask):
		self.dataset.selectMask(mask)
		self.execute()
		self.check_selection_undo_redo()
		self.label_selection_info_update()

	def execute(self):
		error_text = self.jobs_manager.execute()
		if error_text is not None:
			logger.error("error while executing: %r" % error_text)
			dialog_error(self.plot_window, "Error when executing", error_text)


	def message(self, *args, **kwargs):
		pass

	def add_jobs(self):
		#import traceback
		#print "updating compute counter", ''.join(traceback.format_stack())
		compute_counter = self.compute_counter = self.compute_counter + 1
		t0 = time.time()


		def calculate_range(info, block, axisIndex):
			if compute_counter < self.compute_counter:
				return True
			if info.error:
				self.message(info.error_text, index=-1)
				return True
			subblock_size = math.ceil(len(block)/self.thread_pool.nthreads)
			subblock_count = math.ceil(len(block)/subblock_size)
			def subblock(index):
				sub_i1, sub_i2 = index * subblock_size, (index +1) * subblock_size
				if len(block) < sub_i2: # last one can be a bit longer
					sub_i2 = len(block)
				return gavifast.find_nan_min_max(block[sub_i1:sub_i2])
			self.message("min/max[%d] at %.1f%% (%.2fs)" % (axisIndex, info.percentage, time.time() - info.time_start), index=50+axisIndex )
			QtCore.QCoreApplication.instance().processEvents()
			if info.first:
				results = self.thread_pool.run_parallel(subblock)
				self.ranges_grid[axisIndex] = min([result[0] for result in results]), max([result[1] for result in results])
			else:
				results = self.thread_pool.run_parallel(subblock)
				self.ranges_grid[axisIndex] = min([self.ranges_grid[axisIndex][0]] + [result[0] for result in results]), max([self.ranges_grid[axisIndex][1]] + [result[1] for result in results])
			if info.last:
				if self.plot_window.layers.index(self) == 0: # we are the first layer
					print "first layer", self.ranges_grid[axisIndex], self.plot_window.layers[1:]
					for layer in self.plot_window.layers[1:]:
						layer.ranges_grid[axisIndex] = copy.deepcopy(self.ranges_grid[axisIndex])
						layer.grids.ranges[axisIndex] = copy.deepcopy(self.ranges_grid[axisIndex])
				else:
					raise Exception, "should not happen"
				self.grids.ranges[axisIndex] = list(self.ranges_grid[axisIndex])
				self.message("min/max[%d] %.2fs" % (axisIndex, time.time() - t0), index=50+axisIndex)
				self.message(None, index=-1) # clear error msg

		for axisIndex in range(self.dimensions):
			if self.plot_window.layers.index(self) == 0: # we are the first layer, only this one needs range computations
				if self.ranges_grid[axisIndex] is None:
					self.jobs_manager.addJob(0, functools.partial(calculate_range, axisIndex=axisIndex), self.dataset, self.expressions[axisIndex]) #, **self.getVariableDict())
				else:
					self.grids.ranges[axisIndex] = copy.deepcopy(self.ranges_grid[axisIndex])
			else:
				first_layer = self.plot_window.layers[0]
				if first_layer.ranges_grid[axisIndex] is not None:
					self.ranges_grid[axisIndex] = copy.deepcopy(first_layer.ranges_grid[axisIndex])
				if self.ranges_grid[axisIndex] is not None:
					self.grids.ranges[axisIndex] = copy.deepcopy(self.ranges_grid[axisIndex])

		#if self.expression_weight is None or len(self.expression_weight.strip()) == 0:
		#	self.jobs_manager.addJob(1, self.calculate_visuals, self.dataset, *self.expressions, **self.getVariableDict())
		#else:
		#all_expressions = self.expressions + [self.expression_weight, self.weight_x_expression, self.weight_y_expression, self.weight_z_expression]
		self.grids.set_expressions(self.expressions)
		self.grids.define_grid("counts", self.plot_window.grid_size, None)
		self.grids.define_grid("weighted", self.plot_window.grid_size, self.weight_expression)
		if self.dimensions >= 2:
			self.expressions_vector[0] = self.weight_x_expression
			self.expressions_vector[1] = self.weight_y_expression
			self.expressions_vector[2] = self.weight_z_expression
			for i, expression in enumerate(self.expressions_vector):
				name = "xyzw"[i]
				self.grids.define_grid("weight"+name, self.plot_window.vector_grid_size, expression)

		for callback in self.plugin_grids_defines:
			callback(self.grids)
		self.grids.add_jobs(self.jobs_manager)

	def build_widget_qt(self, parent):
		self.widget_build = True

		# create plugins
		self.plugin_grids_defines = []
		self.plugin_grids_draw = []
		self.plugin_queue_toolbar = [] # list of tuples (callback, order)
		self.plugin_queue_page = []
		self.plugins = [cls(parent, self) for cls in gavi.vaex.plugin.PluginLayer.registry if cls.useon(self.plot_window.__class__)]
		logger.debug("PLUGINS: %r " % self.plugins)
		self.plugins_map = {plugin.name:plugin for plugin in self.plugins}
		#self.plugin_zoom = plugin.zoom.ZoomPlugin(self)


		self.toolbox = QtGui.QToolBox(parent)
		self.toolbox.setMinimumWidth(250)



		self.plug_page(self.page_main, "Main", 1., 1.)
		self.plug_page(self.page_visual, "Visual", 1.5, 1.)
		if self.dimensions >= 2:
			self.plug_page(self.page_vector, "Vector field", 2., 1.)
		#self.plug_page(self.page_display, "Display", 3., 1.)
		self.plug_page(self.page_selection, "Selection", 3.5, 1.)

		# first get unique page orders
		pageorders = {}
		for callback, pagename, pageorder, order in self.plugin_queue_page:
			pageorders[pagename] = pageorder
		self.pages = {}
		for pagename, order in sorted(pageorders.items(), key=operator.itemgetter(1)):
			page_frame = QtGui.QFrame(self.toolbox)
			self.pages[pagename] = page_frame
			self.toolbox.addItem(page_frame, pagename)
			logger.debug("created page: "+pagename)
		for pagename, order in sorted(pageorders.items(), key=operator.itemgetter(1)):
			logger.debug("filling page: %sr %r" % (pagename, filter(lambda x: x[1] == pagename, self.plugin_queue_page)))
			for callback, pagename_, pageorder, order in sorted(filter(lambda x: x[1] == pagename, self.plugin_queue_page), key=operator.itemgetter(3)):
	 			logger.debug("filling page: "+pagename +" order=" +str(order) + " callback=" +str(callback))
				callback(self.pages[pagename])
		page_name = self.options.get("page", "Main")
		page_frame = self.pages.get(page_name, None)
		if page_frame:
			self.toolbox.setCurrentWidget(page_frame)

		self.widget = self.toolbox

		if "selection" in self.options:
			filename = self.options["selection"]
			mask = np.load(filename)
			action = gavi.vaex.undo.ActionMask(self.dataset.undo_manager, "selection from %s" % filename, mask, self.apply_mask)
			action.do()
			#self.apply_mask(mask)
			#self.dataset.selectMask(mask)

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
		return self.dataset.column_names

	def onExpressionChanged(self, axisIndex):
		text = str(self.axisboxes[axisIndex].lineEdit().text())
		if text == self.expressions[axisIndex]:
			logger.debug("same expression, will not update")
			return
		self.expressions[axisIndex] = text
		# TODO: range reset as option?
		self.ranges_grid[axisIndex] = None
		self.plot_window.ranges_show[axisIndex] = None
		# TODO: how to handle axis lock.. ?
		if not self.plot_window.axis_lock:
			self.ranges_grid[axisIndex] = None
		linkButton = self.linkButtons[axisIndex]
		link = linkButton.link
		if link:
			logger.debug("sending link messages")
			link.sendRanges(self.ranges[axisIndex], linkButton)
			link.sendRangesShow(self.ranges_show[axisIndex], linkButton)
			link.sendExpression(self.expressions[axisIndex], linkButton)
			gavi.dataset.Link.sendCompute([link], [linkButton])
		else:
			logger.debug("not linked")
		# let any event handler deal with redraw etc
		self.coordinates_picked_row = None
		self.add_jobs()
		self.execute()
		#self.signal_expression_change.emit(self, axisIndex, text)
		#self.compute()
		#error_text = self.jobsManager.execute()
		#if error_text:
		#	dialog_error(self, "Error in expression", "Error: " +error_text)

	def onWeightExpr(self):
		text = str(self.weight_box.lineEdit().text())
		if (text == self.weight_expression) or (text == "" and self.weight_expression == None):
			logger.debug("same weight expression, will not update")
			return
		self.weight_expression = text
		if self.weight_expression.strip() == "":
			self.weight_expression = None
		self.range_level = None
		self.plot_window.range_level_show = None
		self.add_jobs()
		self.execute()
		#self.plot()

	def onTitleExpr(self):
		self.title_expression = str(self.title_box.lineEdit().text())
		self.plot()

	def onWeightXExpr(self):
		text = str(self.weight_x_box.lineEdit().text())
		if (text == self.weight_x_expression):
			logger.debug("same weight_x expression, will not update")
			return
		# is we set the text to "", check if some of the grids are existing, and simply 'disable' the and replot
		# otherwise check if it changed, if it did, see if we should do the grid computation, since
		# if only 1 grid is defined, we don't need it
		if text == "":
			self.weight_x_expression = ""
			if "weightx" in self.grids.grids:
				grid = self.grids.grids["weightx"]
				if grid is not None and grid.weight_expression is not None and len(grid.weight_expression) > 0:
					grid.weight_expression = ""
					self.plot_window.plot()
					return

		self.weight_x_expression = text
		if self.weight_x_expression.strip() == "":
			self.weight_x_expression = None
		self.range_level = None
		self.plot_window.range_level_show = None
		self.check_vector_expressions()

	def check_vector_expressions(self):
		expressions = [self.weight_x_expression, self.weight_y_expression, self.weight_z_expression]
		non_none_expressions = [k for k in expressions if k is not None and len(k) > 0]
		if len(non_none_expressions) >= 2:
			self.add_jobs()
			self.execute()


	def onWeightYExpr(self):
		text = str(self.weight_y_box.lineEdit().text())
		if (text == self.weight_y_expression):
			logger.debug("same weight_x expression, will not update")
			return
		# is we set the text to "", check if some of the grids are existing, and simply 'disable' the and replot
		# otherwise check if it changed, if it did, see if we should do the grid computation, since
		# if only 1 grid is defined, we don't need it
		if text == "":
			self.weight_y_expression = ""
			if "weighty" in self.grids.grids:
				grid = self.grids.grids["weighty"]
				if grid is not None and grid.weight_expression is not None and len(grid.weight_expression) > 0:
					grid.weight_expression = ""
					self.plot_window.plot()
					return

		self.weight_y_expression = text
		if self.weight_y_expression.strip() == "":
			self.weight_y_expression = None
		self.range_level = None
		self.plot_window.range_level_show = None
		self.check_vector_expressions()

	def onWeightZExpr(self):
		text = str(self.weight_z_box.lineEdit().text())
		if (text == self.weight_z_expression):
			logger.debug("same weight_x expression, will not update")
			return
		# is we set the text to "", check if some of the grids are existing, and simply 'disable' the and replot
		# otherwise check if it changed, if it did, see if we should do the grid computation, since
		# if only 1 grid is defined, we don't need it
		if text == "":
			self.weight_z_expression = ""
			if "weightz" in self.grids.grids:
				grid = self.grids.grids["weightz"]
				if grid is not None and grid.weight_expression is not None and len(grid.weight_expression) > 0:
					grid.weight_expression = ""
					self.plot_window.plot()
					return

		self.weight_z_expression = text
		if self.weight_z_expression.strip() == "":
			self.weight_z_expression = None
		self.range_level = None
		self.plot_window.range_level_show = None
		self.check_vector_expressions()

	def onAmplitudeExpr(self):
		text = str(self.amplitude_box.lineEdit().text())
		if len(text) == 0 or text == self.amplitude_expression:
			logger.debug("same expression, skip")
			return
		self.amplitude_expression = text
		self.range_level = None
		self.plot_window.range_level_show = None
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
				self.expressions.reverse()
				self.ranges_grid.reverse()
				# TODO: how to handle layers?
				self.plot_window.ranges_show.reverse()
				for box, expr in zip(self.axisboxes, self.expressions):
					box.lineEdit().setText(expr)
				self.add_jobs()
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
		axisIndex = 0

		self.grid_layout = QtGui.QGridLayout()
		self.grid_layout.setColumnStretch(2, 1)
		#row = 0
		self.linkButtons = []
		for axis_name in self.axis_names:
			row = axisIndex
			axisbox = QtGui.QComboBox(page)
			axisbox.setEditable(True)
			axisbox.setMinimumContentsLength(10)
			#self.form_layout.addRow(axis_name + '-axis:', axisbox)
			self.grid_layout.addWidget(QtGui.QLabel(axis_name + '-axis:', page), row, 1)
			self.grid_layout.addWidget(axisbox, row, 2, QtCore.Qt.AlignLeft)
			linkButton = LinkButton("link", self.dataset, axisIndex, page)
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
				templates = ["log(%s)", "sqrt(%s)", "1/(%s)", "abs(%s)"]

				for template in templates:
					action = QtGui.QAction(template % "...", page)
					def add(checked=None, axis_index=axisIndex, template=template):
						logger.debug("adding template %r to axis %r" % (template, axis_index))
						expression = self.expressions[axis_index].strip()
						if "#" in expression:
							expression = expression[:expression.index("#")].strip()
						self.expressions[axis_index] = template % expression
						# this doesn't cause an event causing jobs to be added?
						self.axisboxes[axis_index].lineEdit().setText(self.expressions[axis_index])
						self.ranges_grid[axis_index] = None
						self.coordinates_picked_row = None
						if not self.plot_window.axis_lock:
							self.plot_window.ranges_show[axis_index] = None
						# to add them
						self.add_jobs()
						self.execute()
					action.triggered.connect(add)
					menu.addAction(action)
				self.grid_layout.addWidget(functionButton, row, 3, QtCore.Qt.AlignLeft)
				#menu.addAction(unlink_action)
				#self.grid_layout.addWidget(functionButton, row, 2)
			#self.grid_layout.addWidget(linkButton, row, 0)
			#if axisIndex == 0:
			extra_expressions = []
			expressionList = self.get_expression_list()
			for prefix in ["", "v", "v_"]:
				names = "x y z".split()
				allin = True
				for name in names:
					if prefix + name not in expressionList:
						allin = False
				# if all items found, add it
				if allin:
					expression = "l2(%s) # l2 norm" % (",".join([prefix+name for name in names]))
					extra_expressions.append(expression)

				if 0: # this gives too much clutter
					for name1 in names:
						for name2 in names:
							if name1 != name2:
								if name1 in expressionList and name2 in expressionList:
									expression = "d(%s)" % (",".join([prefix+name for name in [name1, name2]]))
									extra_expressions.append(expression)


			axisbox.addItems(extra_expressions + self.get_expression_list())
			#axisbox.setCurrentIndex(self.expressions[axisIndex])
			#axisbox.currentIndexChanged.connect(functools.partial(self.onAxis, axisIndex=axisIndex))
			axisbox.lineEdit().setText(self.expressions[axisIndex])
			# keep a list to be able to disconnect
			self.onExpressionChangedPartials.append(functools.partial(self.onExpressionChanged, axisIndex=axisIndex))
			axisbox.lineEdit().editingFinished.connect(self.onExpressionChangedPartials[axisIndex])
			# if the combox pulldown is clicked, execute the same command
			axisbox.currentIndexChanged.connect(lambda _, axisIndex=axisIndex: self.onExpressionChangedPartials[axisIndex]())
			axisIndex += 1
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
		self.amplitude_box.addItems(["gf(log(counts+1),1) # gaussian filter"])
		self.amplitude_box.addItems(["gf(log(counts+1),2) # gaussian filter with higher sigma" ])
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
		self.grid_layout.addWidget(QtGui.QLabel("amplitude="), row, 1)
		self.grid_layout.addWidget(self.amplitude_box, row, 2, QtCore.Qt.AlignLeft)
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
			self.grid_layout.addWidget(QtGui.QLabel("title="), row, 1)
			self.grid_layout.addWidget(self.title_box, row, 2)
			self.title_box.lineEdit().editingFinished.connect(self.onTitleExpr)
			self.title_box.currentIndexChanged.connect(lambda _: self.onTitleExpr())
			self.title_expression = str(self.title_box.lineEdit().text())
			row += 1

		self.weight_box = QtGui.QComboBox(page)
		self.weight_box.setEditable(True)
		self.weight_box.addItems([self.options.get("weight", "")] + self.get_expression_list())
		self.weight_box.setMinimumContentsLength(10)
		self.grid_layout.addWidget(QtGui.QLabel("weight="), row, 1)
		self.grid_layout.addWidget(self.weight_box, row, 2)
		self.weight_box.lineEdit().editingFinished.connect(self.onWeightExpr)
		self.weight_box.currentIndexChanged.connect(lambda _: self.onWeightExpr())
		self.weight_expression = str(self.weight_box.lineEdit().text())
		if len(self.weight_expression.strip()) == 0:
			self.weight_expression = None

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
		if self.dimensions >= 2:
			transparancies = ["intensity", "constant", "none"]
			self.transparancy = self.options.get("transparancy", "constant")
			self.option_transparancy = Option(page_widget, "transparancy", transparancies, getter=attrgetter(self, "transparancy"), setter=attrsetter(self, "transparancy"), update=self.signal_plot_dirty.emit)
			row = self.option_transparancy.add_to_grid_layout(row, grid_layout)

		self.slider_layer_alpha = Slider(page_widget, "opacity", 0, 1, 1000, getter=attrgetter(self, "alpha"), setter=attrsetter(self, "alpha"), update=self.signal_plot_dirty.emit)
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


		if self.dimensions > 1:
			gavi.vaex.colormaps.process_colormaps()
			self.colormap_box = QtGui.QComboBox(page_widget)
			self.colormap_box.setIconSize(QtCore.QSize(16, 16))
			model = QtGui.QStandardItemModel(self.colormap_box)
			for colormap_name in gavi.vaex.colormaps.colormaps:
				colormap = matplotlib.cm.get_cmap(colormap_name)
				pixmap = gavi.vaex.colormaps.colormap_pixmap[colormap_name]
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
				self.colormap = colormap_name
				if hasattr(self, "widget_volume"):
					self.plugins_map["transferfunction"].tool.colormap = self.colormap
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
				if cmap not in gavi.vaex.colormaps.colormaps:
					colormaps_sorted = sorted(gavi.vaex.colormaps.colormaps)
					colormaps_string = " ".join(colormaps_sorted)
					dialog_error(self, "Wrong colormap name", "colormap {cmap} does not exist, choose between: {colormaps_string}".format(**locals()))
					index = 0
				else:
					index = gavi.vaex.colormaps.colormaps.index(cmap)
				self.colormap_box.setCurrentIndex(index)
				self.colormap = gavi.vaex.colormaps.colormaps[index]
			self.colormap_box.currentIndexChanged.connect(onColorMap)

		row += 1

		self.contour_count = int(self.options.get("contour_count", 4))
		self.slider_contour_count = Slider(page_widget, "contour_count", 0, 20, 20, getter=attrgetter(self, "contour_count"), setter=attrsetter(self, "contour_count"), update=self.signal_plot_dirty.emit, format="{0:<3d}", numeric_type=int)
		row = self.slider_contour_count.add_to_grid_layout(row, grid_layout)

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
			self.dataset.undo_manager.undo()
			self.check_selection_undo_redo()
		def on_redo(checked=False):
			self.dataset.undo_manager.redo()
			self.check_selection_undo_redo()
		self.button_selection_undo.clicked.connect(on_undo)
		self.button_selection_redo.clicked.connect(on_redo)
		self.check_selection_undo_redo()

		self.label_selection_info = QtGui.QLabel("should not see me", page)
		self.layout_page_selection.addWidget(self.label_selection_info)
		self.label_selection_info_update()

		def on_select_expression():
			all = storage_expressions.get_all("selection", self.dataset)
			expressions = []
			for stored in all:
				expressions.extend(stored["options"]["expressions"])
			for column in self.dataset.get_column_names():
				expressions.append("%s < 0" % column)
			expression = choose(self.plot_window, "Give expression", "Expression for selection: ", expressions, 0, True)
			if expression:
				expression = str(expression)
				if expression not in expressions:
					expressions.insert(0, expression)
				#dialog_info(self.plot_window, "expr", expression)
				storage_expressions.add("", "selection", self.dataset, {"expressions": expressions} )

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
					action = gavi.vaex.undo.ActionMask(self.dataset.undo_manager, "expression: " + expression, mask, self.apply_mask)
					action.do()
					self.check_selection_undo_redo()

					#self.plot_window.checkUndoRedo()
					#self.setMode(self.lastAction)


		self.button_selection_expression = QtGui.QPushButton(QtGui.QIcon(iconfile('undo')), "Add expression", page )
		self.button_selection_expression.clicked.connect(on_select_expression)
		self.layout_page_selection.addWidget(self.button_selection_expression)




	def label_selection_info_update(self):
		if self.dataset.mask is None:
			self.label_selection_info.setText("no selection")
		else:
			N_sel = int(np.sum(self.dataset.mask))
			N_total = len(self.dataset)
			self.label_selection_info.setText("selected {:,} ({:.2f}%)".format(N_sel, N_sel*100./float(N_total)))

	def check_selection_undo_redo(self):
		#if self.widget_build:
		self.button_selection_undo.setEnabled(self.dataset.undo_manager.can_undo())
		self.button_selection_redo.setEnabled(self.dataset.undo_manager.can_redo())



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


		if self.dimensions > -1:
			self.weight_x_box = QtGui.QComboBox(page)
			self.weight_x_box.setMinimumContentsLength(10)
			self.weight_x_box.setEditable(True)
			self.weight_x_box.addItems([self.options.get("vx", "")] + self.get_expression_list())
			self.weight_x_box.setMinimumContentsLength(10)
			self.grid_layout_vector.addWidget(QtGui.QLabel("vx="), row, 0)
			self.grid_layout_vector.addWidget(self.weight_x_box, row, 1)
			#def onWeightXExprLine(*args, **kwargs):
			#	if len(str(self.weight_x_box.lineEdit().text())) == 0:
			#		self.onWeightXExpr()
			self.weight_x_box.lineEdit().editingFinished.connect(lambda _=None: self.onWeightXExpr())
			self.weight_x_box.currentIndexChanged.connect(lambda _=None: self.onWeightXExpr())
			self.weight_x_expression = str(self.weight_x_box.lineEdit().text())
			if 0:
				for name in "x y z".split():
					if name in self.expressions[0]:
						for prefix in "v v_".split():
							expression = (prefix+name)
							if expression in self.get_expression_list():
								self.weight_x_box.lineEdit().setText(expression)
								self.weight_x_expression = expression

			row += 1

		if self.dimensions > -1:
			self.weight_y_box = QtGui.QComboBox(page)
			self.weight_y_box.setEditable(True)
			self.weight_y_box.addItems([self.options.get("vy", "")] + self.get_expression_list())
			self.weight_y_box.setMinimumContentsLength(10)
			self.grid_layout_vector.addWidget(QtGui.QLabel("vy="), row, 0)
			self.grid_layout_vector.addWidget(self.weight_y_box, row, 1)
			#def onWeightYExprLine(*args, **kwargs):
			#	if len(str(self.weight_y_box.lineEdit().text())) == 0:
			#		self.onWeightYExpr()
			self.weight_y_box.lineEdit().editingFinished.connect(lambda _=None: self.onWeightYExpr())
			self.weight_y_box.currentIndexChanged.connect(lambda _=None: self.onWeightYExpr())
			self.weight_y_expression = str(self.weight_y_box.lineEdit().text())
			if 0:
				for name in "x y z".split():
					if self.dimensions > 1:
						if name in self.expressions[1]:
							for prefix in "v v_".split():
								expression = (prefix+name)
								if expression in self.get_expression_list():
									self.weight_y_box.lineEdit().setText(expression)
									self.weight_y_expression = expression

			row += 1

			self.weight_z_box = QtGui.QComboBox(page)
			self.weight_z_box.setEditable(True)
			self.weight_z_box.addItems([self.options.get("vz", "")] + self.get_expression_list())
			self.weight_z_box.setMinimumContentsLength(10)
			self.grid_layout_vector.addWidget(QtGui.QLabel("vz="), row, 0)
			self.grid_layout_vector.addWidget(self.weight_z_box, row, 1)
			#def onWeightZExprLine(*args, **kwargs):
			#	if len(str(self.weight_z_box.lineEdit().text())) == 0:
			#		self.onWeightZExpr()
			self.weight_z_box.lineEdit().editingFinished.connect(lambda _=None: self.onWeightZExpr())
			self.weight_z_box.currentIndexChanged.connect(lambda _=None: self.onWeightZExpr())
			self.weight_z_expression = str(self.weight_z_box.lineEdit().text())

			row += 1

		if self.dimensions > -1:
			gavi.vaex.colormaps.process_colormaps()
			self.colormap_vector_box = QtGui.QComboBox(page)
			self.colormap_vector_box.setIconSize(QtCore.QSize(16, 16))
			model = QtGui.QStandardItemModel(self.colormap_vector_box)
			for colormap_name in gavi.vaex.colormaps.colormaps:
				colormap = matplotlib.cm.get_cmap(colormap_name)
				pixmap = gavi.vaex.colormaps.colormap_pixmap[colormap_name]
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
				self.colormap_vector = colormap_name
				#self.plot()
				self.signal_plot_dirty.emit()

			cmapnames = "vz_cmap vz_colormap vz_colourmap".split()
			if not set(cmapnames).isdisjoint(self.options):
				for name in cmapnames:
					if name in self.options:
						break
				cmap = self.options[name]
				if cmap not in gavi.vaex.colormaps.colormaps:
					colormaps_sorted = sorted(gavi.vaex.colormaps.colormaps)
					colormaps_string = " ".join(colormaps_sorted)
					dialog_error(self, "Wrong colormap name", "colormap {cmap} does not exist, choose between: {colormaps_string}".format(**locals()))
					index = 0
				else:
					index = gavi.vaex.colormaps.colormaps.index(cmap)
				self.colormap_vector_box.setCurrentIndex(index)
				self.colormap_vector = gavi.vaex.colormaps.colormaps[index]
			else:
				index = gavi.vaex.colormaps.colormaps.index(self.colormap_vector)
				self.colormap_vector_box.setCurrentIndex(index)
			self.colormap_vector_box.currentIndexChanged.connect(onColorMap)

			row += 1

		#self.toolbox.addItem(self.frame_options_main, "Main")
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

