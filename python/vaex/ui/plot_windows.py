# -*- coding: utf-8 -*-
import collections
from operator import itemgetter
import functools
import time
import copy
import json
import traceback
import numpy as np
from functools import reduce


from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
import matplotlib.colors
import matplotlib.widgets
import matplotlib.cm


from vaex.multithreading import ThreadPool
import vaex
import vaex.logging
import vaex.events
import vaex.kld
import vaex.ui.plugin.zoom
import vaex.ui.plugin.vector3d
import vaex.ui.plugin.animation
import vaex.ui.imageblending
import vaex.ui.colormaps
import vaex.ui.templates
import vaex.promise

from vaex.ui.qt import *
from vaex.ui.icons import iconfile
import vaex.vaexfast
from vaex.ui import qt, undo


logger = vaex.logging.getLogger("vaex")

class Slicer(matplotlib.widgets.Widget):
	"""
	"""
	def __init__(self, plot_window, axes, canvas, useblit=True, horizOn=False, vertOn=True,
				 radius=0.05, **lineprops):

		self.plot_window = plot_window
		self.canvas = canvas
		self.axes = axes
		self.horizOn = horizOn
		self.vertOn = vertOn

		xmin, xmax = axes[-1].get_xlim()
		ymin, ymax = axes[-1].get_ylim()
		xmid = 0.5 * (xmin + xmax)
		ymid = 0.5 * (ymin + ymax)

		self.visible = True
		self.useblit = useblit and self.canvas.supports_blit
		self.background = None
		self.needclear = False
		self.radius = radius

		if self.useblit:
			lineprops['animated'] = True

		import matplotlib.patches

		self.ellipse = matplotlib.patches.Ellipse([xmid, ymid], 1, 1, alpha=0.2, visible=False)
		self.update_ellipse(xmid, ymid)
		axes[0].add_patch(self.ellipse)

		if vertOn:
			self.vlines = [ax.axvline(xmid, visible=False, **lineprops)
						   for ax in axes]
		else:
			self.vlines = []

		if horizOn:
			self.hlines = [ax.axhline(ymid, visible=False, **lineprops)
						   for ax in axes]
		else:
			self.hlines = []

		self.connect()

	def connect(self):
		"""connect events"""
		self._cidmotion = self.canvas.mpl_connect('motion_notify_event',
												  self.onmove)
		self._ciddraw = self.canvas.mpl_connect('draw_event', self.clear)

	def disconnect(self):
		"""disconnect events"""
		self.canvas.mpl_disconnect(self._cidmotion)
		self.canvas.mpl_disconnect(self._ciddraw)

	def clear(self, event):
		"""clear the cursor"""
		if self.useblit:
			self.background = (
				self.canvas.copy_from_bbox(self.canvas.figure.bbox))
		for line in self.vlines + self.hlines:
			line.set_visible(False)
		self.ellipse.set_visible(False)

	def onmove(self, event):
		if event.inaxes is None:
			self.plot_window.slice_none()
			return
		if not self.canvas.widgetlock.available(self):
			return
		self.needclear = True
		if not self.visible:
			return
		if self.vertOn:
			for line in self.vlines:
				line.set_xdata((event.xdata, event.xdata))
				line.set_visible(self.visible)
		if self.horizOn:
			for line in self.hlines:
				line.set_ydata((event.ydata, event.ydata))
				line.set_visible(self.visible)
		self.plot_window.slice_circle(event.xdata, event.ydata, self.radius)

		self.update_ellipse(event.xdata, event.ydata)
		self.ellipse.set_visible(True)
		self._update()

	def update_ellipse(self, x, y):
		xlim, ylim = self.plot_window.ranges_show
		width = abs(xlim[1] - xlim[0])
		height = abs(ylim[1] - ylim[0])
		scale = self.radius * 2

		self.ellipse.center = [x, y]
		self.ellipse.width = width * scale
		self.ellipse.height = height * scale

	def _update(self):
		if self.useblit:
			if self.background is not None:
				self.canvas.restore_region(self.background)
			if self.vertOn:
				for ax, line in zip(self.axes, self.vlines):
					ax.draw_artist(line)
			if self.horizOn:
				for ax, line in zip(self.axes, self.hlines):
					ax.draw_artist(line)
			for ax in self.axes:
				ax.draw_artist(self.ellipse)
			self.canvas.blit(self.canvas.figure.bbox)
		else:

			self.canvas.draw_idle()

	def disconnect_events(self):
		self.ellipse.set_visible(False)
		self.canvas.restore_region(self.background)
		self.canvas.blit(self.canvas.figure.bbox)
		self.disconnect()



class PlotDialog(QtGui.QWidget):
	"""
	:type app: VaexApp
	:type current_layer: LayerTable
	:type layers: list[LayerTable]
	"""
	def __init__(self, parent, dataset, dimensions, axisnames, app, width=5, height=4, dpi=100, **options):
		super(PlotDialog, self).__init__()
		self.parent_widget = parent
		self.data_panel = parent
		self.options = options
		self.dataset = dataset
		self.app = app
		self.layers = []
		index = len(self.app.windows)
		self.name = options.get("window_name", "%s-%d" % (self.dataset.name, index))
		self.dimensions = dimensions
		if "fraction" in self.options:
			dataset.set_active_fraction(float(self.options["fraction"]))

		self.xlabel = options.get("xlabel")
		self.ylabel = options.get("ylabel")


		self.menu_bar = QtGui.QMenuBar(self)
		self.menu_file = QtGui.QMenu("&File", self.menu_bar)
		self.menu_bar.addMenu(self.menu_file)
		self.menu_view = QtGui.QMenu("&View", self.menu_bar)
		self.menu_bar.addMenu(self.menu_view)
		self.menu_mode = QtGui.QMenu("&Mode", self.menu_bar)
		self.menu_bar.addMenu(self.menu_mode)
		self.menu_selection = QtGui.QMenu("&Selection", self.menu_bar)
		self.menu_bar.addMenu(self.menu_selection)

		self.menu_samp = QtGui.QMenu("SAM&P", self.menu_bar)
		self.menu_bar.addMenu(self.menu_samp)


		self.undoManager = parent.undoManager
		self.setWindowTitle(self.name)
		self.dataset = dataset
		self.axisnames = axisnames
		self.pool = ThreadPool()
		#self.expressions = expressions
		#self.dimensions = len(self.expressions)
		#self.grids = Grids(self.dataset, self.pool, *self.expressions)

		if self.dimensions == 3:
			self.resize(800+400,700)
		else:
			self.resize(800,700)


		self.plugin_queue_toolbar = [] # list of tuples (callback, order)
		self.plugins = [cls(self) for cls in vaex.ui.plugin.PluginPlot.registry if cls.useon(self.__class__)]
		self.plugins_map = {plugin.name:plugin for plugin in self.plugins}
		#self.plugin_zoom = plugin.zoom.ZoomPlugin(self)

		self.aspect = None
		self.axis_lock = False

		self.update_counter = 0
		self.t_0 = 0
		self.t_last = 0
		self.slice_radius = 0.05

		self.shortcuts = []
		self.messages = {}

		default_grid_size = 256 if self.dimensions == 2 else 128
		self.grid_size = eval(self.options.get("grid_size", str(default_grid_size)))
		self.vector_grid_size = eval(self.options.get("vector_grid_size", "16"))


		self.fig = Figure(figsize=(width, height), dpi=dpi)
		self.canvas =  FigureCanvas(self.fig)
		self.canvas.setParent(self)
		self.add_axes()

		# if an update is scheduled, cancel any replot, since the preperations for the update may leave the grids
		# in the layers in a undefined state
		self.queue_redraw = Queue("redraw", 5, self.canvas.draw) # only draw the canvas again
		self.queue_replot = Queue("replot", 10, self.plot) # redo the whole plot, but no computation
		self.queue_update = Queue("update", 100, self.update_direct, pre=self.queue_replot.cancel) # a complete recalculation and refresh of the plot

		self.layout_main = QtGui.QVBoxLayout()
		self.layout_content = QtGui.QHBoxLayout()
		self.layout_main.setContentsMargins(0, 0, 0, 0)
		self.layout_content.setContentsMargins(0, 0, 0, 0)
		self.layout_main.setSpacing(0)
		self.layout_content.setSpacing(0)
		self.layout_main.addWidget(self.menu_bar)

		#self.button_layout.setSpacing(0)

		self.boxlayout = QtGui.QVBoxLayout()
		self.boxlayout_right = QtGui.QVBoxLayout()
		self.boxlayout.setContentsMargins(0, 0, 0, 0)
		self.boxlayout_right.setContentsMargins(0, 0, 0, 0)

		#self.ranges = [None for _ in range(self.dimensions)] # min/max for the data
		self.ranges_show = [None for _ in range(self.dimensions)] # min/max for the plots
		self.range_level_show = None
		# if zooming in with for instance pinching, we like to know what the ranges were
		# before zooming for the undo, and not all ranges in between
		self.last_ranges_show = None
		self.last_range_level_show = None

		#self.ranges_previous = None
		#self.ranges_show_previous = None
		#self.ranges_level_previous = None


		self.currentModes = None
		self.lastAction = None

		self.beforeCanvas(self.layout_main)
		self.layout_main.addLayout(self.layout_content, 1.)
		self.layout_plot_region = QtGui.QHBoxLayout()
		self.layout_plot_region.addWidget(self.canvas, 1)

		self.boxlayout.addLayout(self.layout_plot_region, 1)
		self.addToolbar2(self.layout_main)
		self.afterCanvas(self.boxlayout_right)
		self.layout_content.addLayout(self.boxlayout, 1.)

		self.status_bar = QtGui.QStatusBar(self)
		self.button_cancel = QtGui.QToolButton(self.status_bar)
		self.button_cancel.setText("cancel")
		self.button_cancel.setContentsMargins(0, 0, 0, 0)
		self.status_bar.layout().setContentsMargins(0,0,0,0)
		self.status_bar.layout().setSpacing(0)

		self.progress_bar = QtGui.QProgressBar(self.status_bar)
		self.progress_bar.setMaximum(1000)
		self.progress_bar.setMinimumWidth(100)
		self.progress_bar.setFixedWidth(100)
		self.button_cancel.setEnabled(False)

		self.label_time = QtGui.QLabel("", self.toolbar)

		#index = self.status_bar.
		self.status_bar.addPermanentWidget(self.progress_bar)
		self.status_bar.addPermanentWidget(self.button_cancel)
		self.status_bar.addPermanentWidget(self.label_time)
		def begin():
			self.time_begin = time.time()
			self.progress_bar.setValue(0)
			self.cancelled = False
			self.button_cancel.setEnabled(True)
		def end():
			self.progress_bar.setValue(1000)
			self.cancelled = False
			self.button_cancel.setEnabled(False)
			time_total = time.time() - self.time_begin
			self.label_time.setText("%.2fs" % time_total)
		def progress(fraction):
			self.progress_bar.setValue(fraction*1000)
			QtCore.QCoreApplication.instance().processEvents()
			logger.debug("queue: %r %r", self.queue_update.counter, self.queue_update.counter_processed)
			return (not self.cancelled) and (not self.queue_update.in_queue(2))
		def cancel():
			self.progress_bar.setValue(0)
			self.button_cancel.setEnabled(False)
			self.label_time.setText("cancelled")
		def on_click_cancel():
			self.cancelled = True
		self.button_cancel.clicked.connect(on_click_cancel)

		self._begin_signal = self.dataset.executor.signal_begin.connect(begin)
		self._progress_signal = self.dataset.executor.signal_progress.connect(progress)
		self._end_signal = self.dataset.executor.signal_end.connect(end)
		self._cancel_signal = self.dataset.executor.signal_cancel.connect(cancel)

		self.layout_main.addWidget(self.status_bar)

		self.layout_content.addLayout(self.boxlayout_right, 0)
		self.setLayout(self.layout_main)

		#self.plot()
		FigureCanvas.setSizePolicy(self,
									QtGui.QSizePolicy.Expanding,
									QtGui.QSizePolicy.Expanding)
		FigureCanvas.updateGeometry(self)
		self.currentMode = None
		self.shortcuts = []

		self.grabGesture(QtCore.Qt.PinchGesture);
		self.grabGesture(QtCore.Qt.PanGesture);
		self.grabGesture(QtCore.Qt.SwipeGesture);

		self.signal_samp_send_selection = vaex.events.Signal("samp send selection")
		self.signal_closed = vaex.events.Signal("close plot window")

		self.signal_plot_finished = vaex.events.Signal("plot finished")

		self.canvas.mpl_connect('resize_event', self.on_resize_event)
		self.canvas.mpl_connect('motion_notify_event', self.onMouseMove)
		#self.pinch_ranges_show = [None for i in range(self.dimension)]

	def slice_none(self):
		mask = np.ones((self.grid_size,) * self.dimensions, dtype=np.bool)
		layer = self.current_layer
		if layer:
			layer.signal_slice_change.emit(mask, False)

	def slice_circle(self, x, y, radius):
		logger.debug("slice circle: %r %r", x, y)
		xlim, ylim = self.ranges_show
		width = abs(xlim[1] - xlim[0])
		height = abs(ylim[1] - ylim[0])
		xrel = (x - xlim[0]) / width
		yrel = (y - ylim[0]) / height

		import vaex.utils
		x = vaex.utils.linspace_centers(0, 1, self.grid_size)
		y = vaex.utils.linspace_centers(0, 1, self.grid_size)
		x, y = np.meshgrid(x, y)
		distance = np.sqrt((x-xrel)**2 + (y-yrel)**2)
		mask = distance < radius
		layer = self.current_layer
		if layer:
			layer.signal_slice_change.emit(mask, False)

	def update_all_layers(self):
		for layer in self.layers:
			layer.flag_needs_update()
		self.queue_update()

	def add_axes(self):
		self.axes = self.fig.add_subplot(111)
		self.axes.xaxis_index = 0
		if self.dimensions > 1:
			self.axes.yaxis_index = 1
		self.axes.hold(True)

	def plug_toolbar(self, callback, order):
		self.plugin_queue_toolbar.append((callback, order))

	def plug_page(self, callback, pagename, pageorder, order):
		self.plugin_queue_page.append((callback, pagename, pageorder, order))

	def plug_grids(self, callback_define, callback_draw):
		self.plugin_grids_defines.append(callback_define)
		self.plugin_grids_draw.append(callback_draw)

	def getAxesList(self):
		return [self.axes]

	def __repr__(self):
		return "<%s at 0x%x layers=%r>" % (self.__class__.__name__, id(self), self.layers)

	def get_options(self):
		options = collections.OrderedDict()
		options["grid_size"] = self.grid_size
		options["vector_grid_size"] = self.vector_grid_size
		options["ranges_show"] = self.ranges_show
		options["aspect"] = self.aspect
		layer = self.current_layer
		if layer is not None:
			options["layer"] = layer.get_options()
		options = copy.deepcopy(options)
		return dict(options)

	def apply_options(self, options, update=True):
		#map = {"expressions",}
		#recognize = "ranges_show grid_size vector_grid_size aspect".split()
		recognize = "ranges_show  aspect".split()
		for key in recognize:
			if key in list(options.keys()):
				value = options[key]
				setattr(self, key, copy.copy(value))
				if key == "aspect":
					self.action_aspect_lock_one.setChecked(bool(value))
		for plugin in self.plugins:
			plugin.apply_options(options)
		for key in list(options.keys()):
			if key not in recognize:
				logger.error("option %s not recognized, ignored" % key)
		layer = self.current_layer
		if layer is not None:
			layer.apply_options(options["layer"], update=False)
		if update:
			self.queue_update()

	def load_options(self, name):
		self.plugins_map["favorites"].load_options(name, update=False)

	def add_layer(self, expressions, dataset=None, name=None, **options):
		if dataset is None:
			dataset = self.dataset
		if name is None:
			name = options.get("layer_name", "Layer: " + str(len(self.layers)+1))
		ranges = copy.deepcopy(self.ranges_show)
		logger.debug("adding layer {name} with expressions {expressions} for dataset {dataset} and options {options}".format(**locals()))

		if len(self.layers) > 0:
			first_layer = self.layers[0]
			assert len(expressions) == first_layer.dimensions
			for i in range(self.dimensions):
				if ranges[i] is None and first_layer.ranges_grid[i] is not None:
					ranges[i] = copy.copy(first_layer.ranges_grid[i])
		layer = vaex.ui.layers.LayerTable(self, name, dataset, expressions, self.axisnames, options, self.pool, self.fig, self.canvas, ranges)
		self.layers.append(layer)
		layer.build_widget_qt(self.widget_layer_stack) # layer.widget is the widget build
		self.widget_layer_stack.addWidget(layer.widget)

		#layer.build_widget_qt_layer_control(self.frame_layer_controls)
		#self.layout_frame_layer_controls.addWidget(layer.widget_layer_control)

		layer.widget.setVisible(False)
		self.layer_selection.addItem(name)
		self.layer_selection.setCurrentIndex(len(self.layers))



		def on_expression_change(layer, axis_index, expression):
			if not self.axis_lock: # and len(self.layers) == 1:
				self.ranges_show[axis_index] = None
			self.compute()
			error_text = self.dataset.executor.execute()
			if error_text:
				dialog_error(self, "Error in expression", "Error: " +error_text)

		def on_plot_dirty(layer=None):
			logger.debug("received signal plot dirty, layer=%r" % layer)
			self.queue_replot()

		layer.signal_expression_change.connect(on_expression_change)
		layer.signal_plot_dirty.connect(on_plot_dirty)
		layer.signal_plot_update.connect(self.queue_update)

		if "options" in options:
			assert self.current_layer == layer
			self.load_options(options["options"])
		#$layer.add_jobs()
		self.queue_update(layer=layer)
		#self.dataset.executor.execute()
		#self.queue_update()
		logger.debug("added layer")
		return layer

	def _wait(self):
		"""Used for unittesting to make sure the plots are all done"""
		self.queue_update._wait()
		self.queue_replot._wait()
		self.queue_redraw._wait()

	def plot_to_png(self, filename=None):
		if filename is None:
			import tempfile
			handle, filename = tempfile.mkstemp(".png")
			logger.debug("write to %s" % filename)
		self.fig.savefig(filename)
		return filename


	def on_resize_event(self, event):
		if not self.action_mini_mode_ultra.isChecked():
			logger.debug("resize event")
			self.fig.tight_layout()
			self.queue_redraw()

	def event(self, event):
		if isinstance(event, QtGui.QGestureEvent):
			for gesture in event.activeGestures():
				if isinstance(gesture, QtGui.QPinchGesture):
					center = gesture.centerPoint()
					x, y =  center.x(), center.y()
					geometry = self.canvas.geometry()
					if geometry.contains(x, y):
						rx = x - geometry.x()
						ry = y - geometry.y()
						#print self.canvas.geometry, self.canvas.mouse_grabber
						axes_list = [ax for ax in self.getAxesList() if ax.contains_point((rx, geometry.height()-1-ry))]
						#nx, ny = rx/geometry.width(), y/geometry.height()
						if len(axes_list) > 0:
							axes = axes_list[0]
							transform = axes.transData.inverted().transform
							x_data, y_data = transform([rx, geometry.height()-1-ry])
							if gesture.lastScaleFactor() != 0:
								scale = (gesture.scaleFactor()/gesture.lastScaleFactor())
							else:
								scale = (gesture.scaleFactor())
							#@scale = gesture.totalScaleFactor()
							scale = 1./(scale)
							self.zoom(scale, axes, x_data, y_data)
			return True
		else:
			return super(PlotDialog, self).event(event)
			#return True


	def closeEvent(self, event):
		# disconnect this event, otherwise we get an update/redraw for nothing
		# since closing a dialog causes this event to fire otherwise
		#self.parent_widget.plot_dialogs.remove(self)

		self.dataset.executor.signal_begin.disconnect(self._begin_signal)
		self.dataset.executor.signal_progress.disconnect(self._progress_signal)
		self.dataset.executor.signal_end.disconnect(self._end_signal)
		self.dataset.executor.signal_cancel.disconnect(self._cancel_signal)

		self.pool.close()
		for layer in self.layers:
			layer.removed()
		for plugin in self.plugins:
			plugin.clean_up()
		super(PlotDialog, self).closeEvent(event)
		self.signal_closed.emit(self)

	def getExpressionList(self):
		return self.dataset.column_names

	def add_pages(self, toolbox):
		pass


	def fill_menu_layer_new(self):
		self.menu_layer_new.clear()
		for dataset in self.data_panel.dataset_list:
			menu_dataset = QtGui.QMenu(dataset.name, self.menu_layer_new)
			self.menu_layer_new.addMenu(menu_dataset)
			for column1 in dataset.get_column_names():
				if self.dimensions == 1:
					action_col1 = QtGui.QAction(column1, menu_dataset)
					menu_dataset.addAction(action_col1)
					def add_layer_1(ignore=None, column1=column1, dataset=dataset):
						self.add_layer([column1], dataset=dataset)
						self.dataset.executor.execute()
					action_col1.triggered.connect(add_layer_1)
				else:
					menu_col1 = QtGui.QMenu(column1, menu_dataset)
					menu_dataset.addMenu(menu_col1)
					for column2 in dataset.get_column_names():
						if self.dimensions == 2:
							action_col2 = QtGui.QAction(column2, menu_dataset)
							menu_col1.addAction(action_col2)
							def add_layer_2(_ignore=None, column1=column1, column2=column2, dataset=dataset):
								self.add_layer([column1, column2], dataset=dataset)
								self.dataset.executor.execute()
							action_col2.triggered.connect(add_layer_2)
						else:
							pass

					#action = QtGui.QAction()
	def remove_layer(self):
		layer = self.current_layer
		logger.debug("remove layer: %r" % layer)
		if layer is not None:
			index = self.layers.index(layer)
			layer.removed()
			self.layers.remove(layer)
			self.layer_selection.removeItem(index+1) # index 0 is the layer control
			self.widget_layer_stack.removeWidget(layer.widget)
			self.plot()

	def afterCanvas(self, layout):

		layout.setContentsMargins(0, 0, 0, 0)
		layout.setSpacing(0)


		self.layer_box = QtGui.QGroupBox("Layers", self)
		self.layout_layer_box = QtGui.QVBoxLayout()
		self.layout_layer_box.setSpacing(0)
		self.layout_layer_box.setContentsMargins(0,0,0,0)
		self.layer_box.setLayout(self.layout_layer_box)



		self.layout_layer_buttons = QtGui.QHBoxLayout()


		self.button_layer_new = QtGui.QToolButton(self)
		self.button_layer_new.setIcon(QtGui.QIcon(iconfile("layer--plus")))
		self.button_layer_new.setPopupMode(QtGui.QToolButton.InstantPopup)
		self.button_layer_new.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)

		toolbuttonSizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
		self.button_layer_new.setSizePolicy(toolbuttonSizePolicy)

		self.button_layer_new.setText("add")
		self.button_layer_new.setEnabled(self.dimensions < 3)
		self.menu_layer_new = QtGui.QMenu()
		self.fill_menu_layer_new()
		self.button_layer_new.setMenu(self.menu_layer_new)
		self.button_layer_delete = QtGui.QToolButton(self)
		self.button_layer_delete.setIcon(QtGui.QIcon(iconfile("layer--minus")))
		self.button_layer_delete.setText("remove")
		self.button_layer_delete.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
		self.button_layer_delete.setSizePolicy(toolbuttonSizePolicy)
		self.button_layer_delete.setEnabled(self.dimensions < 3)
		def on_layer_remove(_ignore=None):
			logger.debug("remove layer")
			self.remove_layer()
		self.button_layer_delete.clicked.connect(on_layer_remove)
		self.layout_layer_buttons.addWidget(self.button_layer_new, 0)
		self.layout_layer_buttons.addWidget(self.button_layer_delete, 0)



		self.layer_selection = QtGui.QComboBox(self)
		self.layer_selection.addItems(["Layer controls"])
		self.layout_layer_box.addLayout(self.layout_layer_buttons)
		self.layout_layer_box.addWidget(self.layer_selection)
		self.current_layer = None
		def onSwitchLayer(index):
			logger.debug("switch to layer: %r %r" % (index, self.layers))
			#if self.current_layer is not None:
			#	self.bottom_layout.removeWidget(self.current_layer.widget)
			#	self.current_layer.widget.setVisible(False)
			#self.current_layer = self.layers[index]
			#self.current_layer.widget.setVisible(True)
			#self.bottom_layout.addWidget(self.current_layer.widget)
			#self.bottom_layout.update()
			#self.bottomFrame.updateGeometry()
			layer_index = index - 1
			if index == 0:
				self.current_layer = None
				for layer in self.layers:
					layer_control_widget = layer.grab_layer_control(self.frame_layer_controls)
					self.layout_frame_layer_controls.addWidget(layer_control_widget)
					#layer.build_widget_qt_layer_control(self.frame_layer_controls)
					#
			else:
				self.layers[layer_index].release_layer_control(self.frame_layer_controls)
				self.current_layer = self.layers[index-1]


			self.widget_layer_stack.setCurrentIndex(index)
			#self.widget_layer_stack.setCurrentIndex(0)

		self.layer_selection.currentIndexChanged.connect(onSwitchLayer)

		self.bottomFrame = QtGui.QFrame(self)
		layout.addWidget(self.bottomFrame, 0)

		self.bottom_layout = QtGui.QVBoxLayout()
		self.bottom_layout.setContentsMargins(0, 0, 0, 0)
		self.bottom_layout.setSpacing(0)

		self.bottomFrame.setLayout(self.bottom_layout)
		self.bottom_layout.addWidget(self.layer_box)

		self.widget_layer_stack = QtGui.QStackedWidget(self)
		self.bottom_layout.addWidget(self.widget_layer_stack)

		self.frame_layer_controls = QtGui.QGroupBox("Layer controls", self.widget_layer_stack)
		self.layout_frame_layer_controls = QtGui.QVBoxLayout(self.frame_layer_controls)
		self.layout_frame_layer_controls.setAlignment(QtCore.Qt.AlignTop)
		self.frame_layer_controls.setLayout(self.layout_frame_layer_controls)
		self.widget_layer_stack.addWidget(self.frame_layer_controls)

		self.frame_layer_controls_result = QtGui.QGroupBox("Layer result", self.frame_layer_controls)
		self.layout_frame_layer_controls_result = QtGui.QGridLayout()
		self.frame_layer_controls_result.setLayout(self.layout_frame_layer_controls_result)
		self.layout_frame_layer_controls_result.setSpacing(0)
		self.layout_frame_layer_controls_result.setContentsMargins(0,0,0,0)
		self.layout_frame_layer_controls.addWidget(self.frame_layer_controls_result)


		row = 0
		attr_name = "layer_brightness"
		self.layer_brightness = 1.
		self.slider_layer_brightness = Slider(self.frame_layer_controls_result, "brightness", 10**-1, 10**1, 1000, attrgetter(self, attr_name), attrsetter(self, attr_name), uselog=True, update=self.plot)
		row = self.slider_layer_brightness.add_to_grid_layout(row, self.layout_frame_layer_controls_result)
		attr_name = "layer_gamma"
		self.layer_gamma = 1.
		self.slider_layer_gamma = Slider(self.frame_layer_controls_result, "gamma", 10**-1, 10**1, 1000, attrgetter(self, attr_name), attrsetter(self, attr_name), uselog=True, update=self.plot)
		row = self.slider_layer_gamma.add_to_grid_layout(row, self.layout_frame_layer_controls_result)
		#self.frame_layer_controls_result


		self.blend_modes = list(vaex.ui.imageblending.modes.keys())
		self.blend_mode = self.blend_modes[0]
		self.option_layer_blend_mode = Option(self.frame_layer_controls_result, "blend", self.blend_modes, getter=attrgetter(self, "blend_mode"), setter=attrsetter(self, "blend_mode"), update=self.plot)
		row = self.option_layer_blend_mode.add_to_grid_layout(row, self.layout_frame_layer_controls_result)

		self.background_colors = ["white", "black"]
		self.background_color = self.background_colors[0]
		self.option_layer_background_color = Option(self.frame_layer_controls_result, "background", self.background_colors, getter=attrgetter(self, "background_color"), setter=attrsetter(self, "background_color"), update=self.plot)
		row = self.option_layer_background_color.add_to_grid_layout(row, self.layout_frame_layer_controls_result)

		#row = self.checkbox_intensity_as_opacity.add_to_grid_layout(row, self.layout_layer_control)

		#self.checkbox_intensity_as_opacity = Checkbox(self.group_box_layer_control, "use_intensity", getter=attrgetter(self, "use_intensity"), setter=attrsetter(self, "use_intensity"), update=self.plot)
		#row = self.checkbox_intensity_as_opacity.add_to_grid_layout(row, self.layout_layer_control)



	def add_shortcut(self, action, key):
		def trigger(action):
			def call(action=action):
				action.toggle()
				action.trigger()
			return call
		if action.isEnabled():
			shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
			shortcut.activated.connect(trigger(action))
			self.shortcuts.append(shortcut)

	def checkUndoRedo(self):
		self.action_undo.setEnabled(self.undoManager.can_undo())
		if self.undoManager.can_undo():
			self.action_undo.setToolTip("Undo: "+self.undoManager.actions_undo[-1].description())

		self.action_redo.setEnabled(self.undoManager.can_redo())
		if self.undoManager.can_redo():
			self.action_redo.setToolTip("Redo: "+self.undoManager.actions_redo[0].description())

	def onActionUndo(self):
		logger.debug("undo")
		self.undoManager.undo()
		self.checkUndoRedo()

	def onActionRedo(self):
		logger.debug("redo")
		self.undoManager.redo()
		self.checkUndoRedo()

	def onMouseMove(self, event):
		x, y = event.xdata, event.ydata
		if x is not None:
			extra_text = self.getExtraText(x, y)
			#extra_text = "TODO:"
			if extra_text:
				self.message("x,y=%5.4e,%5.4e %s" % (x, y, extra_text), index=0)
			else:
				self.message("x,y=%5.4e,%5.4e" % (x, y), index=0)
		else:
			self.message(None)

	def getExtraText(self, x, y):
		layer = self.current_layer
		if hasattr(layer, "amplitude_grid"):
			amplitude = layer.amplitude_grid
			if len(amplitude.shape) == 1:
					#if self.ranges[0]:
					N = amplitude.shape[0]
					if layer.ranges_grid[0] is not None:
						xmin, xmax = layer.ranges_grid[0]
						index = (x-xmin)/(xmax-xmin) * N
						if index >= 0 and index < N:
							index = int(index)
							return "value = %f" % (amplitude[index])
			if len(amplitude.shape) == 2:
					#if self.ranges[0] and self.ranges[1]:
					Nx, Ny = amplitude.shape
					if layer.ranges_grid[0] != None and layer.ranges_grid[1] != None:
						xmin, xmax = layer.ranges_grid[0]
						ymin, ymax = layer.ranges_grid[1]
						xindex = (x-xmin)/(xmax-xmin) * Nx
						yindex = (y-ymin)/(ymax-ymin) * Ny
						if xindex >= 0 and xindex < Nx and yindex >= 0 and yindex < Nx:
							return "value = %f" % (amplitude[int(yindex), int(xindex)])


	def message(self, text, index=0):
		if text is None:
			if index in self.messages:
				del self.messages[index]
		else:
			self.messages[index] = text
		text = ""
		keys = list(self.messages.keys())
		keys.sort()
		text_parts = [self.messages[key] for key in keys]
		self.status_bar.showMessage(" | ".join(text_parts))


	def beforeCanvas(self, layout):
		self.addToolbar(layout) #, yselect=True, lasso=False)

	def setMode(self, action, force=False):
		logger.debug("set mode %r %r %r" % (action, action.text(), action.isChecked()))
		#if not (action.isChecked() or force):
		if not action.isEnabled():
			logger.error("action selected that was disabled: %r" % action)
			self.setMode(self.lastAction)
			return
		if not (action.isChecked()):
			logger.debug("ignore action")
		else:
			self.lastAction = action
			axes_list = self.getAxesList()
			if self.currentModes is not None:
				logger.debug("disconnect %r" % (self.currentModes, ))
				for mode in self.currentModes:
					mode.disconnect_events()
					mode.active = False
			useblit = True
			if action == self.action_move:
				self.currentModes = [Mover(self, axes) for axes in axes_list]
			if action == self.action_pick:
				#hasy = hasattr(self, "getdatay")
				#hasx = hasattr(self, "getdatax")
				layer = self.current_layer
				if layer is not None:
					hasx = True
					hasy = len(layer.expressions) > 1
					self.currentModes = [matplotlib.widgets.Cursor(axes, hasy, hasx, color="red", linestyle="dashed", useblit=useblit) for axes in axes_list]
					for cursor in self.currentModes:
						def onmove(event, current=cursor, cursors=self.currentModes):
							if event.inaxes:
								for other_cursor in cursors:
									if current != other_cursor:
										other_cursor.onmove(event)
						cursor.connect_event('motion_notify_event', onmove)
					if hasx and hasy:
						for mode in self.currentModes:
							mode.connect_event('button_press_event', self.onPickXY)
					elif hasx:
						for mode in self.currentModes:
							mode.connect_event('button_press_event', self.onPickX)
					elif hasy:
						for mode in self.currentModes:
							mode.connect_event('button_press_event', self.onPickY)
				if useblit:
					self.canvas.draw() # buggy otherwise
			if action == self.action_xrange:
				logger.debug("setting last select action to xrange")
				self.lastActionSelect = self.action_xrange
				self.currentModes = [matplotlib.widgets.SpanSelector(axes, functools.partial(self.onSelectX, axes=axes), 'horizontal', useblit=useblit) for axes in axes_list]
				if useblit:
					self.canvas.draw() # buggy otherwise
			if action == self.action_yrange:
				logger.debug("setting last select action to yrange")
				self.lastActionSelect = self.action_yrange
				self.currentModes = [matplotlib.widgets.SpanSelector(axes, functools.partial(self.onSelectY, axes=axes), 'vertical', useblit=useblit) for axes in axes_list]
				if useblit:
					self.canvas.draw() # buggy otherwise
			if action == self.action_lasso:
				logger.debug("setting last select action to lasso")
				self.lastActionSelect = self.action_lasso
				self.currentModes =[ matplotlib.widgets.LassoSelector(axes, functools.partial(self.onSelectLasso, axes=axes)) for axes in axes_list]
				if useblit:
					self.canvas.draw() # buggy otherwise
			if action == self.action_slice:
				logger.debug("setting mode to slice")
				#self.lastActionSelect = self.action_slice
				self.currentModes =[ Slicer(self, [axes], canvas=self.canvas, horizOn=False, vertOn=False, radius=self.slice_radius) for axes in axes_list]
				if useblit:
					self.canvas.draw() # buggy otherwise
			#self.plugin_zoom.setMode(action)
			for plugin in self.plugins:
				logger.debug("plugin %r %r.setMode" % (plugin, plugin.name))
				plugin.setMode(action)
		self.syncToolbar()

		#if self.action_lasso
		#pass
		#self.


	def onPickX(self, event):
		x, y = event.xdata, event.ydata
		self.selected_point = None

		layer = self.current_layer
		axes = event.inaxes
		logger.debug("pickx %r %r" % (layer, axes))
		if layer is not None and  axes is not None:
			layer.coordinates_picked_row = None
			promise = layer.subspace.nearest([x])
			layer.dataset.executor.execute()
			def set(value):
				if value is None:
					logger.error("could not find nearest")
				else:
					index, distance, point = value
					layer.dataset.set_current_row(index)
			promise.then(set).end()
			self.setMode(self.lastAction)

	def onPickY(self, event):
		# TODO: not sure when we want this?
		x, y = event.xdata, event.ydata
		self.selected_point = None

		layer = self.current_layer
		axes = event.inaxes
		logger.debug("pickx %r %r" % (layer, axes))
		if layer is not None and  axes is not None:
			layer.coordinates_picked_row = None
			promise = layer.subspace.nearest([y])
			layer.dataset.executor.execute()
			def set(value):
				if value is None:
					logger.error("could not find nearest")
				else:
					index, distance, point = value
					layer.dataset.set_current_row(index)
			promise.then(set).end()
			self.setMode(self.lastAction)

	def onPickXY(self, event):
		x, y = event.xdata, event.ydata
		wx = self.ranges_show[0][1] - self.ranges_show[0][0]
		wy = self.ranges_show[1][1] - self.ranges_show[1][0]

		x, y = event.xdata, event.ydata
		self.selected_point = None

		layer = self.current_layer
		axes = event.inaxes
		logger.debug("pickx %r %r" % (layer, axes))
		if layer is not None and  axes is not None:
			layer.coordinates_picked_row = None
			promise = layer.subspace.nearest([x, y], metric=[1./wx, 1./wy])
			layer.dataset.executor.execute()
			def set(value):
				if value is None:
					logger.error("could not find nearest")
				else:
					index, distance, point = value
					layer.dataset.set_current_row(index)
			promise.then(set).end()
		return

	def select(self, name, vmin, vmax):
		values = [vmin, vmax]
		vmin, vmax = min(values), max(values)

		layer = self.current_layer
		if layer is not None:
			# xaxis is stored in the matplotlib object / axes.xaxis_index
			expr = getattr(layer, name)
			boolean_expression = "((%s) >= %f) & ((%s) < %f)" % (expr, vmin, expr, vmax)
			logger.debug("expression: %s", boolean_expression)
			print boolean_expression
			layer.dataset.select(boolean_expression, self.select_mode)
			mask = layer.dataset.mask
			action = undo.ActionMask(layer.dataset.undo_manager, "select %s range[%f,%f]" % (name, vmin, vmax), mask, layer.apply_mask)

	def onSelectX(self, xmin, xmax, axes):
		self.select("x", xmin, xmax)

	def onSelectY(self, ymin, ymax, axes):
		self.select("y", ymin, ymax)

	def onSelectLasso(self, vertices, axes):
		x, y = np.array(vertices).T

		x = np.ascontiguousarray(x, dtype=np.float64)
		y = np.ascontiguousarray(y, dtype=np.float64)
		layer = self.current_layer
		if layer is not None:
			self.dataset.lasso_select(layer.x, layer.y, x, y, mode=self.select_mode)
			#self.dataset.evaluate(select, layer.expressions[axes.xaxis_index], layer.expressions[axes.yaxis_index], **self.getVariableDict())
			meanx = x.mean()
			meany = y.mean()
			mask = layer.dataset.mask
			action = undo.ActionMask(layer.dataset.undo_manager, "lasso around [%f,%f]" % (meanx, meany), mask, layer.apply_mask)
			#action.do()
			self.checkUndoRedo()
			self.queue_update()
			#self.setMode(self.lastAction)
		return

	def set_ranges(self, axis_indices, ranges_show=None, range_level=None):
		logger.debug("set axis/ranges_show: %r / %r" % (axis_indices, ranges_show))
		if axis_indices is None: # signals a 'reset'
			for axis_index in range(self.dimensions):
				self.ranges_show[axis_index] = None
				for layer in self.layers:
					layer.ranges_grid[axis_index] = None
				#self.ranges[axis_index] = None
		else:
			for i, axis_index in enumerate(axis_indices):
				if ranges_show:
					self.ranges_show[axis_index] = ranges_show[i]
					for layer in self.layers:
						layer.ranges_grid[axis_index] = ranges_show[i]
				i#f ranges:
				#	self.ranges[axis_index] = ranges[i]
		logger.debug("set range_level: %r" % (range_level, ))
		self.range_level_show = range_level
		if len(axis_indices) > 0:
			self.check_aspect(axis_indices[0]) # maybe we should use the widest or smallest one
		self.update_all_layers()
		#self.update_plot()

	def update_plot(self):
		# default value
		self.update_direct()

	def update_direct(self, layer=None):
		logger.debug("update direct: ranges_show=%r" % (self.ranges_show, ))
		if layer:
			logger.debug("only update layer %r (index %d)" % (layer, self.layers.index(layer)))
			layers = [layer]
		else:
			logger.debug("updating all layers")
			layers = self.layers
		layers = [layer for layer in self.layers if layer.get_needs_update()]
		if not layers:
			logger.error("update requested while no layer needs it")

		timelog("begin computation", reset=True)
		# this can be
		promises = [layer.add_tasks_ranges() for layer in layers]
		# execute may do things async, like at a server
		try:
			self.dataset.executor.execute()
		except SyntaxError as e:
			msg = "%s: %r" % (e.args[0], e.args[1][3])
			qt.dialog_error(self, "Syntax error", "Syntax error: %s " % msg)
		except KeyError as e:
			msg = e.args[0]
			qt.dialog_error(self, "Unknown variable", "Unknown variable or column: %s " % msg)

		promise_ranges_done = vaex.promise.listPromise(promises)
		promise_ranges_done.then(self._update_step2, self.on_error_or_cancel).then(None, self.on_error_or_cancel).end()

	def on_error_or_cancel(self, error):
		logger.exception("error occured: %r", error, exc_info=error)
		import traceback
		traceback.print_exc()
		#raise exception
		#raise error

	def _update_step2(self, layers):
		"""Each layer has it's own ranges_grid computed now, unless something went wrong
		But all layers are shown with the same ranges (ranges_show)
		If any of the ranges is None, take the min/max of each layer
		"""
		logger.debug("done with ranges, now update step2 for layers: %r", layers)


		for dimension in range(self.dimensions):
			if self.ranges_show[dimension] is None:
				vmin = min([layer.ranges_grid[dimension][0] for layer in layers])
				vmax = max([layer.ranges_grid[dimension][1] for layer in layers])
				self.ranges_show[dimension] = [vmin, vmax]

		logger.debug("ranges before aspect check: %r", self.ranges_show)
		self.check_aspect(0)
		logger.debug("ranges after aspect check: %r", self.ranges_show)
		# now make sure the layers all have the same ranges_grid
		for layer in layers:
			layer.ranges_grid = copy.deepcopy(self.ranges_show)


		# now we are ready to calculate histograms
		promises = [layer.add_tasks_histograms() for layer in layers]
		self.dataset.executor.execute()

		promises_histograms_done = vaex.promise.listPromise(promises)
		promises_histograms_done.then(self._update_step3, self.on_error_or_cancel).end()

	def _update_step3(self, layers):
		logger.debug("done with histograms, now update step3, layers = %r" % layers)
		# all histograms are computed, and anything needed to visualize it
		# for 1d histograms, we for instance want to have similar levels
		if self.range_level_show is None:
			self.calculate_range_level_show()
		timelog("computation done")
		# now we can do the plot
		self.queue_replot()

	def calculate_range_level_show(self):
		layers = [layer for layer in self.layers if layer.range_level is not None]
		if layers:
			for layer in layers:
				logger.debug("layer %r has range_level %r" % (layer, layer.range_level))
			vmin = min([layer.range_level[0] for layer in layers])
			vmax = max([layer.range_level[1] for layer in layers])
			self.range_level_show = [vmin, vmax]
		logger.debug("range_level_show = %r" % (self.range_level_show, ))

	def zoom(self, factor, axes, x=None, y=None, delay=300, *args):
		if self.last_ranges_show is None:
			self.last_ranges_show = copy.deepcopy(self.ranges_show)
		if self.last_range_level_show is None:
			self.last_range_level_show = copy.deepcopy(self.range_level_show)
		xmin, xmax = axes.get_xlim()
		width = xmax - xmin

		if x is None:
			x = xmin + width/2

		fraction = (x-xmin)/width

		range_level_show = None
		ranges_show = []
		ranges = []
		axis_indices = []

		ranges_show.append((x - width *fraction *factor , x + width * (1-fraction)*factor))
		axis_indices.append(axes.xaxis_index)

		ymin, ymax = axes.get_ylim()
		height = ymax - ymin
		if y is None:
			y = ymin + height/2
		fraction = (y-ymin)/height
		ymin_show, ymax_show = y - height*fraction*factor, y + height*(1-fraction)*factor
		ymin_show, ymax_show = min(ymin_show, ymax_show), max(ymin_show, ymax_show)
		if len(self.ranges_show) == 1: # if 1d, y refers to range_level
			range_level_show = ymin_show, ymax_show
			if (QtGui.QApplication.keyboardModifiers() == QtCore.Qt.AltModifier) or (QtGui.QApplication.keyboardModifiers() == QtCore.Qt.ControlModifier):
				range_level_show = ymin, ymax
			else:
				range_level_show = ymin_show, ymax_show
		else:
			ranges_show.append((ymin_show, ymax_show))
			axis_indices.append(axes.yaxis_index)

		def delayed_zoom():
			action = undo.ActionZoom(self.undoManager, "zoom " + ("out" if factor > 1 else "in"),
							self.set_ranges,
							list(range(self.dimensions)), self.last_ranges_show, self.last_range_level_show,
							axis_indices, ranges_show=ranges_show, range_level_show=range_level_show)
			self.last_ranges_show = None
			self.last_range_level_show = None
			action.do()
			self.checkUndoRedo()
		for layer in self.layers:
			layer.flag_needs_update()
		self.queue_update(delayed_zoom, delay=delay)


		if 1:
			self.check_aspect(axes.xaxis_index)
			if self.dimensions in [2,3]:
				self.ranges_show[axes.xaxis_index] = list(ranges_show[0])
				self.ranges_show[axes.yaxis_index] = list(ranges_show[1])
				for ax in self.getAxesList():
					ax.set_xlim(self.ranges_show[ax.xaxis_index])
					ax.set_ylim(self.ranges_show[ax.yaxis_index])
			if self.dimensions == 1:
				self.ranges_show[axis_indices[0]] = list(ranges_show[0])
				self.range_level_show = list(range_level_show)
				axes.set_xlim(self.ranges_show[0])
				axes.set_ylim(self.range_level_show)
			self.queue_redraw()
			#self.plot()

	def onActionSaveFigure(self, *ignore_args):
		filetypes = dict(self.fig.canvas.get_supported_filetypes()) # copy, otherwise we lose png support :)
		pngtype = [("png", filetypes["png"])]
		del filetypes["png"]
		filetypes = [value + "(*.%s)" % key for (key, value) in pngtype + list(filetypes.items())]
		import string
		def make_save(expr):
			save_expr = ""
			for char in expr:
				if char not in string.whitespace:
					if char in string.ascii_letters or char in string.digits or char in "._":
						save_expr += char
					else:
						save_expr += "_"
			return save_expr
		layer = self.current_layer
		if layer != None:
			save_expressions = list(map(make_save, layer.expressions))
			type = "histogram" if self.dimensions == 1 else "density"
			filename = layer.dataset.name +"_%s_" % type  +"-vs-".join(save_expressions) + ".png"
			filename = QtGui.QFileDialog.getSaveFileName(self, "Export to figure", filename, ";;".join(filetypes))
			if isinstance(filename, tuple):
				filename = filename[0]
			filename = str(filename)
			if filename:
				logger.debug("saving to figure: %s" % filename)
				self.fig.savefig(filename)
				self.filename_figure_last = filename
				self.action_save_figure_again.setEnabled(True)

	def onActionSaveFigureAgain(self, *ignore_args):
		logger.debug("saving to figure: %s" % self.filename_figure_last)
		self.fig.savefig(self.filename_figure_last)

	def get_aspect(self):
		if 0:
			xmin, xmax = self.axes.get_xlim()
			ymin, ymax = self.axes.get_ylim()
			height = ymax - ymin
			width = xmax - xmin
		return 1 #width/height

	def onActionAspectLockOne(self, *ignore_args):
		self.aspect = self.get_aspect() if self.action_aspect_lock_one.isChecked() else None
		logger.debug("set aspect to: %r" % self.aspect)
		self.check_aspect(0)
		self.queue_update()
		#self.compute()
		#self.dataset.executor.execute()
		#self.plot()

	def _onActionAspectLockOne(self, *ignore_args):
		self.aspect = 1 #self.get_aspect() if self.action_aspect_lock.isEnabled() else None
		logger.debug("set aspect to: %r" % self.aspect)

	def onActionExport(self):
		if self.dimensions != 2:
			dialog_info(self, "Export failure", "Oops sorry, export only supported for 2d plots at the moment")
			return
		ok, mask = select_many(self, "Choose export options", ["Export matplotlib python script", "Export grid", "Export selection", "Export vector grid"])
		if ok:
			name = self.current_layer.dataset.name
			name = gettext(self, "Export name", "Give a base name for the export files", name)
			if name:
				dir_path = getdir(self, "Choose directory where to save")
				if dir_path is None:
					return
				msg_list = []

				yesall = False
				if mask[0]:
					scriptname = os.path.join(dir_path, name + "_plot.py")
					template = vaex.ui.templates.matplotlib
					if not os.path.exists(scriptname):
						yes, yesall = True, False
					else:
						yes, yesall = dialog_confirm(self, "Overwrite", "Overwrite: " +scriptname, to_all=True)
					if yes or yesall:
						file(scriptname, "w").write(template.format(name=name))
						msg_list.append("wrote: " + scriptname)


				optionsname = os.path.join(dir_path, name + "_meta.json")
				options = {}
				options["extent"] = list(self.current_layer.ranges_grid[0]) + list(self.current_layer.ranges_grid[1])
				json.dump(options, file(optionsname, "w"), indent=4)
				msg_list.append("wrote: " + optionsname)

				if mask[1]:
					gridname = os.path.join(dir_path, name + "_grid.npy")
					if not os.path.exists(gridname) or yesall:
						yes, yesall = True, yesall
					else:
						yes, yesall = dialog_confirm(self, "Overwrite", "Overwrite: " +gridname, to_all=True)
					if yes or yesall:
						np.save(gridname, self.current_layer.amplitude_grid)
						msg_list.append("wrote: " + gridname)
					if mask[2]:
						if self.current_layer.dataset.mask is not None:
							gridname = os.path.join(dir_path, name + "_grid_selection.npy")
							if not os.path.exists(gridname) or yesall:
								yes, yesall = True, yesall
							else:
								yes, yesall = dialog_confirm(self, "Overwrite", "Overwrite: " +gridname, to_all=True)
							if yes or yesall:
								np.save(gridname, self.current_layer.amplitude_selection)
								msg_list.append("wrote: " + gridname)
					if mask[3]:
						if hasattr(self.current_layer, "vector_grids"):
							gridname = os.path.join(dir_path, name + "_grid_vector.npy")
							if not os.path.exists(gridname) or yesall:
								yes, yesall = True, yesall
							else:
								yes, yesall = dialog_confirm(self, "Overwrite", "Overwrite: " +gridname, to_all=True)
							if yes or yesall:
								np.save(gridname, self.current_layer.vector_grids)
								msg_list.append("wrote: " + gridname)
					msg = "\n".join(msg_list)
					dialog_info(self, "Finished export", msg)



	def addToolbar2(self, layout, contrast=True, gamma=True):
		self.toolbar2 = QtGui.QToolBar(self)
		self.toolbar2.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
		self.toolbar2.setIconSize(QtCore.QSize(16, 16))

		layout.addWidget(self.toolbar2)



		def on_store_selection():
			if self.dataset.mask is None:
				dialog_error(self, "No selection", "No selection made")
			else:
				path = self.dataset.name + "-selection.npy"
				path = get_path_save(self, "Save selection as numpy array", path, "numpy array *.npy")
				if path:
					np.save(path, self.dataset.mask)
		self.action_selection_store = QtGui.QAction(QtGui.QIcon(iconfile('table_save')), '&Store selection', self)
		self.action_selection_store.triggered.connect(on_store_selection)
		#self.action_selection_store.setCheckable(True)
		#self.toolbar2.addAction(self.action_selection_store)
		self.menu_selection.addAction(self.action_selection_store)

		def on_load_selection():
			path = self.dataset.name + "-selection.npy"
			path = get_path_open(self, "Open selection as numpy array", path, "numpy array *.npy")
			if path:
				mask = np.load(path)
				if len(mask) != len(self.dataset):
					dialog_error(self, "Error opening selection", "Selection is not of same length (%d) as dataset (%d)" % (len(mask), len(self.dataset) ))
					return
				if mask.dtype != np.bool:
					dialog_error(self, "Error opening selection", "Expected type numpy.bool, got %r" % (mask.dtype ))
					return
				layer = self.current_layer
				if layer is not None:
					action = undo.ActionMask(layer.dataset.undo_manager, "loaded selection", mask, layer.apply_mask)
					action.do()
				#self.dataset.selectMask(mask)
				#self.dataset.executor.execute()
		self.action_selection_load = QtGui.QAction(QtGui.QIcon(iconfile('table_save')), '&Load selection', self)
		self.action_selection_load.triggered.connect(on_load_selection)
		#self.action_selection_load.setCheckable(True)
		#self.toolbar2.addAction(self.action_selection_load)
		self.menu_selection.addAction(self.action_selection_load)



		self.action_save_figure = QtGui.QAction(QtGui.QIcon(iconfile('table_save')), '&Export figure', self)
		self.action_save_figure_again = QtGui.QAction(QtGui.QIcon(iconfile('table_save')), '&Export figure again', self)
		#self.menu_save = QtGui.QMenu(self)
		#self.action_save_figure.setMenu(self.menu_save)
		#self.menu_save.addAction(self.action_save_figure_again)
		#self.toolbar2.addAction(self.action_save_figure)
		self.menu_file.addAction(self.action_save_figure)
		self.menu_file.addAction(self.action_save_figure_again)


		self.action_save_figure.triggered.connect(self.onActionSaveFigure)
		self.action_save_figure_again.triggered.connect(self.onActionSaveFigureAgain)
		self.action_save_figure_again.setEnabled(False)

		self.action_export = QtGui.QAction(QtGui.QIcon(iconfile('table_save')), '&Export data/script', self)
		self.menu_file.addSeparator()
		self.menu_file.addAction(self.action_export)
		self.action_export.triggered.connect(self.onActionExport)



		self.action_aspect_lock_one = QtGui.QAction(QtGui.QIcon(iconfile('control-stop-square')), 'Aspect=1', self)
		#self.action_aspect_lock_one = QtGui.QAction(QtGui.QIcon(iconfile('table_save')), '&Set aspect to one', self)
		#self.menu_aspect = QtGui.QMenu(self)
		#self.action_aspect_lock.setMenu(self.menu_aspect)
		#self.menu_aspect.addAction(self.action_aspect_lock_one)
		self.toolbar2.addAction(self.action_aspect_lock_one)
		self.menu_view.insertAction(self.action_mini_mode_normal, self.action_aspect_lock_one)
		self.menu_view.insertSeparator(self.action_mini_mode_normal)

		#self.action_aspect_lock.triggered.connect(self.onActionAspectLock)
		self.action_aspect_lock_one.setCheckable(True)
		self.action_aspect_lock_one.triggered.connect(self.onActionAspectLockOne)
		#self.action_save_figure_again.setEnabled(False)






		self.action_undo = QtGui.QAction(QtGui.QIcon(iconfile('arrow-curve-180-left')), 'Undo', self)
		self.action_redo = QtGui.QAction(QtGui.QIcon(iconfile('arrow-curve-000-left')), 'Redo', self)

		self.toolbar2.addAction(self.action_undo)
		self.toolbar2.addAction(self.action_redo)
		self.action_undo.triggered.connect(self.onActionUndo)
		self.action_redo.triggered.connect(self.onActionRedo)
		self.checkUndoRedo()

		self.action_disjoin = QtGui.QAction(QtGui.QIcon(iconfile('sql-join-outer-exclude')), 'Disjoined', self)
		self.action_disjoin.setCheckable(True)
		self.action_disjoin.triggered.connect(self.onActionDisjoin)
		self.toolbar2.addAction(self.action_disjoin)


		self.action_axes_lock = QtGui.QAction(QtGui.QIcon(iconfile('lock')), 'Lock axis', self)
		self.action_axes_lock.setCheckable(True)
		self.action_axes_lock.triggered.connect(self.onActionAxesLock)
		self.toolbar2.addAction(self.action_axes_lock)

	def onActionAxesLock(self, ignore=None):
		self.axis_lock = self.action_axes_lock.isChecked()

	def onActionShuffled(self, ignore=None):
		self.xoffset = 1 if self.action_shuffled.isChecked() else 0
		self.compute()
		self.dataset.executor.execute()
		logger.debug("xoffset = %r" % self.xoffset)

	def onActionDisjoin(self, ignore=None):
		#self.xoffset = 1 if self.action_shuffled.isChecked() else 0
		layer = self.current_layer
		if layer:
			layer.show_disjoined = self.action_disjoin.isChecked()
			layer.calculate_amplitudes()
			logger.debug("show_disjoined = %r" % layer.show_disjoined)
			self.queue_replot()


	def addToolbar(self, layout, pick=True, xselect=True, yselect=True, lasso=True):

		self.toolbar = QtGui.QToolBar(self)
		self.toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
		self.action_group_main = QtGui.QActionGroup(self)
		self.action_group_mainSelectMode = QtGui.QActionGroup(self)


		self.action_group_display = QtGui.QActionGroup(self)

		self.actiongroup_display_mode = QtGui.QActionGroup(self)

		#self.action_displmini_mode = QtGui.QAction(QtGui.QIcon(iconfile('picture_empty')), '&Mini screen(should not see)', self)
		self.action_mini_mode_normal = QtGui.QAction(QtGui.QIcon(iconfile('picture_empty')), 'Normal', self)
		self.action_mini_mode_compact = QtGui.QAction(QtGui.QIcon(iconfile('picture_empty')), 'Compact', self)
		self.action_mini_mode_ultra  = QtGui.QAction(QtGui.QIcon(iconfile('picture_empty')), 'Ultra compact', self)
		self.action_mini_mode_normal.setShortcut("Ctrl+Shift+N")
		self.action_mini_mode_compact.setShortcut("Ctrl+Shift+C")
		self.action_mini_mode_ultra.setShortcut("Ctrl+Shift+U")

		self.action_group_mini_mode = QtGui.QActionGroup(self)
		self.action_group_mini_mode.addAction(self.action_mini_mode_normal)
		self.action_group_mini_mode.addAction(self.action_mini_mode_compact)
		self.action_group_mini_mode.addAction(self.action_mini_mode_ultra)

		self.action_fullscreen = QtGui.QAction(QtGui.QIcon(iconfile('picture_empty')), '&Fullscreen', self)
		self.action_fullscreen.setCheckable(True)
		self.action_fullscreen.setShortcut(("Ctrl+F"))

		self.action_toolbar_toggle = QtGui.QAction(QtGui.QIcon(iconfile('ui-toolbar')), '&toolbars', self)
		self.action_toolbar_toggle.setCheckable(True)
		self.action_toolbar_toggle.setChecked(True)
		self.action_toolbar_toggle.setShortcut(("Ctrl+Shift+T"))

		#self.actiongroup_mini_mode.addAction(self.action_fullscreen)


		def toggle_fullscreen(ignore=None):
			fullscreen = self.windowState() & QtCore.Qt.WindowFullScreen
			fullscreen = not fullscreen # toggle
			self.action_fullscreen.setChecked(fullscreen)
			if fullscreen:
				self.setWindowState(self.windowState() | QtCore.Qt.WindowFullScreen);
			else:
				self.setWindowState(self.windowState() ^ QtCore.Qt.WindowFullScreen);

		self.action_fullscreen.triggered.connect(toggle_fullscreen)


		self.action_toolbar_toggle.triggered.connect(self.on_toolbar_toggle)

		self.action_move = QtGui.QAction(QtGui.QIcon(iconfile('edit-move')), '&Move', self)
		self.action_move.setShortcut("Ctrl+M")
		self.menu_mode.addAction(self.action_move)

		self.action_pick = QtGui.QAction(QtGui.QIcon(iconfile('cursor')), '&Pick', self)
		self.action_pick.setShortcut("Ctrl+P")
		self.menu_mode.addAction(self.action_pick)

		self.action_select = QtGui.QAction(QtGui.QIcon(iconfile('glue_lasso16')), '&Select(you should not read this)', self)
		self.action_xrange = QtGui.QAction(QtGui.QIcon(iconfile('glue_xrange_select16')), '&x-range', self)
		self.action_yrange = QtGui.QAction(QtGui.QIcon(iconfile('glue_yrange_select16')), '&y-range', self)
		self.action_lasso = QtGui.QAction(QtGui.QIcon(iconfile('glue_lasso16')), '&Lasso', self)
		self.action_slice = QtGui.QAction(QtGui.QIcon(iconfile('cutlery-knife')), '&Slice', self)
		self.action_select_none = QtGui.QAction(QtGui.QIcon(iconfile('dialog-cancel-3')), '&No selection', self)
		self.action_select_invert = QtGui.QAction(QtGui.QIcon(iconfile('dialog-cancel-3')), '&Invert', self)

		self.action_xrange.setShortcut("Ctrl+Shift+X")
		self.menu_mode.addAction(self.action_xrange)
		self.action_yrange.setShortcut("Ctrl+Shift+Y")
		self.menu_mode.addAction(self.action_yrange)
		self.action_lasso.setShortcut("Ctrl+L")
		self.menu_mode.addAction(self.action_lasso)
		self.action_slice.setShortcut("Ctrl+S")
		self.menu_mode.addAction(self.action_slice)
		self.action_select_none.setShortcut("Ctrl+N")
		self.menu_mode.addAction(self.action_select_none)
		self.action_select_invert.setShortcut("Ctrl+I")
		self.menu_mode.addAction(self.action_select_invert)


		self.menu_mode.addSeparator()
		self.action_select_invert.setShortcut("Ctrl+I")
		#self.menu_mode_select_mode = QtGui.QMenu("Select mode")
		#self.menu_mode.addMenu(self.menu_mode_select_mode)

		self.action_select_mode_replace = QtGui.QAction(QtGui.QIcon(iconfile('sql-join-right')), '&Replace', self)
		self.action_select_mode_and = QtGui.QAction(QtGui.QIcon(iconfile('sql-join-inner')), '&And', self)
		self.action_select_mode_or = QtGui.QAction(QtGui.QIcon(iconfile('sql-join-outer')), '&Or', self)
		self.action_select_mode_xor = QtGui.QAction(QtGui.QIcon(iconfile('sql-join-outer-exclude')), 'Xor', self)
		self.action_select_mode_subtract = QtGui.QAction(QtGui.QIcon(iconfile('sql-join-left-exclude')), 'Subtract', self)
		self.action_select_mode_replace.setShortcut("Ctrl+Shift+=")
		self.action_select_mode_and.setShortcut("Ctrl+Shift+&")
		self.action_select_mode_or.setShortcut("Ctrl+Shift+|")
		self.action_select_mode_xor.setShortcut("Ctrl+Shift+^")
		self.action_select_mode_subtract.setShortcut("Ctrl+Shift+-")
		self.menu_mode.addAction(self.action_select_mode_replace)
		self.menu_mode.addAction(self.action_select_mode_and)
		self.menu_mode.addAction(self.action_select_mode_or)
		self.menu_mode.addAction(self.action_select_mode_xor)
		self.menu_mode.addAction(self.action_select_mode_subtract)

		self.action_samp_send_table_select_row_list = QtGui.QAction(QtGui.QIcon(iconfile('block--arrow')), 'Broadcast selection over SAMP', self)
		self.action_samp_send_table_select_row_list.setShortcut('Ctrl+Shift+B')
		#self.toolbar.addAction(self.action_samp_sand_table_select_row_list)
		self.menu_samp.addAction(self.action_samp_send_table_select_row_list)
		def send_samp_selection(ignore=None):
			self.signal_samp_send_selection.emit(self.dataset)
		self.send_samp_selection_reference = send_samp_selection # does this fix the bug that clicking the buttons doesn't do anything?
		self.action_samp_send_table_select_row_list.triggered.connect(send_samp_selection)

		self.action_display_mode_both = QtGui.QAction(QtGui.QIcon(iconfile('picture_empty')), 'Show both', self)
		self.action_display_mode_full = QtGui.QAction(QtGui.QIcon(iconfile('picture_empty')), 'Show full', self)
		self.action_display_mode_selection = QtGui.QAction(QtGui.QIcon(iconfile('picture_empty')), 'Show selection', self)
		self.action_display_mode_both_contour = QtGui.QAction(QtGui.QIcon(iconfile('picture_empty')), 'Show contour', self)



		#self.actions_resolution = []

		self.actions_display = [self.action_display_mode_both, self.action_display_mode_full, self.action_display_mode_selection, self.action_display_mode_both_contour]
		for action in self.actions_display:
			self.action_group_display.addAction(action)
			action.setCheckable(True)
		action = self.actions_display[0]
		action.setChecked(True)
		self.action_display_current = action
		self.action_group_display.triggered.connect(self.onActionDisplay)
		#self.zoomButton = QtGui.QToolButton(self, )
		#$self.zoomButton.setIcon(QtGui.QIcon(iconfile('glue_zoom_to_rect')))
		#self.zoomMenu = QtGui.QMenu(self)
		#self.zoomMenu.addAction(self.action_zoom_x)
		#self.zoomMenu.addAction(self.action_zoom_y)
		#self.zoomMenu.addAction(self.action_zoom_out)
		#self.action_zoom.setMenu(self.zoomMenu)
		#self.action_zoom = self.toolbar.addWidget(self.zoomButton)

		#self.action_zoom = QtGui.QAction(QtGui.QIcon(iconfile('glue_zoom_to_rect')), '&Zoom', self)
		#exitAction.setShortcut('Ctrl+Q')
		#onExAction.setStatusTip('Exit application')

		#self.action_group_main.setToggleAction(True)
		#self.action_group_main.setExclusive(True)
		self.action_group_mainSelectMode.addAction(self.action_select_mode_replace)
		self.action_group_mainSelectMode.addAction(self.action_select_mode_and)
		self.action_group_mainSelectMode.addAction(self.action_select_mode_or)
		self.action_group_mainSelectMode.addAction(self.action_select_mode_xor)
		self.action_group_mainSelectMode.addAction(self.action_select_mode_subtract)

		self.action_group_main.addAction(self.action_move)
		self.action_group_main.addAction(self.action_pick)
		self.action_group_main.addAction(self.action_xrange)
		self.action_group_main.addAction(self.action_yrange)
		self.action_group_main.addAction(self.action_lasso)
		self.action_group_main.addAction(self.action_slice)
		#self.action_group_main.addAction(self.action_zoom_out)



		self.menu_view.addAction(self.action_mini_mode_normal)
		self.menu_view.addAction(self.action_mini_mode_compact)
		self.menu_view.addAction(self.action_mini_mode_ultra)
		self.menu_view.addSeparator()
		self.menu_view.addAction(self.action_fullscreen)
		self.menu_view.addAction(self.action_toolbar_toggle)

		self.menu_view.addSeparator()
		self.action_group_resolution = QtGui.QActionGroup(self)
		self.action_resolution_list = []
		for index, resolution in enumerate([32, 64, 128, 256, 512, 1024]):
			action_resolution = QtGui.QAction(QtGui.QIcon(iconfile('picture_empty')), 'Grid Resolution: %d' % resolution, self)
			def do(ignore=None, resolution=resolution):
				self.grid_size = resolution
				self.update_all_layers()
			action_resolution.setCheckable(True)
			# TODO: this need to move to a layer change event
			if resolution == int(self.grid_size):
				action_resolution.setChecked(True)
			#action_resolution.setEnabled(False)
			action_resolution.triggered.connect(do)
			action_resolution.setShortcut("Ctrl+Alt+%d" % (index+1))
			self.menu_view.addAction(action_resolution)
			self.action_group_resolution.addAction(action_resolution)
			self.action_resolution_list.append(action_resolution)

		self.menu_view.addSeparator()
		self.action_group_resolution_vector = QtGui.QActionGroup(self)
		self.action_resolution_vector_list = []
		for index, resolution in enumerate([8,16,32, 64, 128, 256]):
			action_resolution = QtGui.QAction(QtGui.QIcon(iconfile('picture_empty')), 'Grid Resolution: %d' % resolution, self)
			def do(ignore=None, resolution=resolution):
				self.vector_grid_size = resolution
				self.queue_update()
				#self.compute()
				#self.dataset.executor.execute()
			action_resolution.setCheckable(True)
			# TODO: this need to move to a layer change event
			if resolution == int(self.vector_grid_size):
				action_resolution.setChecked(True)
			#action_resolution.setEnabled(False)
			action_resolution.triggered.connect(do)
			action_resolution.setShortcut("Ctrl+Shift+Alt+%d" % (index+1))
			self.menu_view.addAction(action_resolution)
			self.action_group_resolution_vector.addAction(action_resolution)
			self.action_resolution_vector_list.append(action_resolution)


		#self.mini_mode_button = QtGui.QToolButton()
		#self.mini_mode_button.setPopupMode(QtGui.QToolButton.InstantPopup)
		#self.mini_mode_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
		#self.menu_mini_mode = QtGui.QMenu()
		#self.action_mini_mode.setMenu(self.menu_mini_mode)
		#self.mini_mode_button.setMenu(self.mini_mode_button_menu)
		#self.menu_mini_mode.addAction(self.action_mini_mode_normal)
		#self.menu_mini_mode.addAction(self.action_mini_mode_ultra)
		#self.mini_mode_button.setDefaultAction(self.action_miniscreen)
		#self.mini_mode_button.setCheckable(True)
		#self.mini_mode_button.setIcon(self.action_miniscreen.icon())
		#self.mini_mode_button.setText(self.action_miniscreen.text())

		#self.toolbar.addAction(self.action_mini_mode)
		self.toolbar.addAction(self.action_fullscreen)
		self.toolbar.addAction(self.action_toolbar_toggle)

		self.toolbar.addAction(self.action_move)
		if pick:
			self.toolbar.addAction(self.action_pick)
			#self.action_pick.setChecked(True)
			#self.setMode(self.action_pick, force=True)
			self.lastAction = self.action_pick
		self.toolbar.addAction(self.action_select)
		self.select_menu = QtGui.QMenu()
		self.action_select.setMenu(self.select_menu)
		self.select_menu.addAction(self.action_lasso)
		if yselect:
			#self.toolbar.addAction(self.action_yrange)
			self.select_menu.addAction(self.action_yrange)
			if self.dimensions > 1:
				self.lastActionSelect = self.action_yrange
		if xselect:
			#self.toolbar.addAction(self.action_xrange)
			self.select_menu.addAction(self.action_xrange)
			self.lastActionSelect = self.action_xrange
		if lasso:
			#self.toolbar.addAction(self.action_lasso)
			if self.dimensions > 1:
				self.lastActionSelect = self.action_lasso
		else:
			self.action_lasso.setEnabled(False)
		self.toolbar.addAction(self.action_slice)
		self.select_menu.addSeparator()
		self.select_menu.addAction(self.action_select_none)
		self.select_menu.addAction(self.action_select_invert)
		self.select_menu.addSeparator()


		self.select_mode_button = QtGui.QToolButton()
		self.select_mode_button.setPopupMode(QtGui.QToolButton.InstantPopup)
		self.select_mode_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
		self.select_mode_button_menu = QtGui.QMenu()
		self.select_mode_button.setMenu(self.select_mode_button_menu)

		self.select_mode_button_menu.addAction(self.action_select_mode_replace)
		self.select_mode_button_menu.addAction(self.action_select_mode_or)
		self.select_mode_button_menu.addAction(self.action_select_mode_and)
		self.select_mode_button_menu.addAction(self.action_select_mode_xor)
		self.select_mode_button_menu.addAction(self.action_select_mode_subtract)
		self.select_mode_button.setDefaultAction(self.action_select_mode_replace)
		self.toolbar.addWidget(self.select_mode_button)


		#self.toolbar.addAction(action_select_mode)
		if 0:
			self.toolbar.addAction(self.action_zoom)
			self.zoom_menu = QtGui.QMenu()
			self.action_zoom.setMenu(self.zoom_menu)
			self.zoom_menu.addAction(self.action_zoom_rect)
			self.zoom_menu.addAction(self.action_zoom_x)
			self.zoom_menu.addAction(self.action_zoom_y)
			if self.dimensions == 1:
				self.lastActionZoom = self.action_zoom_x # this makes more sense for histograms as default
			else:
				self.lastActionZoom = self.action_zoom_rect

			self.toolbar.addSeparator()
			self.toolbar.addAction(self.action_zoom_out)
			self.toolbar.addAction(self.action_zoom_fit)
			#self.toolbar.addAction(self.action_zoom_use)
		else:
			#self.plugin_zoom.plug()
			#
			plugin_chain_toolbar = sorted(self.plugin_queue_toolbar, key=itemgetter(1)) # sort by order field
			for plug, order in plugin_chain_toolbar:
				plug()

		#self.zoomButton.setPopupMode(QtCore.QToolButton.DelayedPopup)


		self.action_group_main.triggered.connect(self.setMode)
		self.action_group_mainSelectMode.triggered.connect(self.setSelectMode)

		self.action_mini_mode_normal.triggered.connect(self.onActionMiniModeNormal)
		self.action_mini_mode_compact.triggered.connect(self.onActionMiniModeCompact)
		self.action_mini_mode_ultra.triggered.connect(self.onActionMiniModeUltra)
		self.action_select.triggered.connect(self.onActionSelect)
		self.action_select_none.triggered.connect(self.onActionSelectNone)
		self.action_select_invert.triggered.connect(self.onActionSelectInvert)
		#action_zoom_out

		self.action_select_mode_replace.setCheckable(True)
		self.action_select_mode_and.setCheckable(True)
		self.action_select_mode_or.setCheckable(True)
		self.action_select_mode_xor.setCheckable(True)
		self.action_select_mode_subtract.setCheckable(True)

		#self.action_mini_mode.setCheckable(True)
		self.action_mini_mode_normal.setCheckable(True)
		self.action_mini_mode_normal.setChecked(True)
		self.action_mini_mode_compact.setCheckable(True)
		self.action_mini_mode_ultra.setCheckable(True)
		#self.action_mini_mode.setIcon(self.action_mini_mode_ultra.icon())
		#self.action_mini_mode.setText(self.action_mini_mode_ultra.text())

		self.action_move.setCheckable(True)
		self.action_pick.setCheckable(True)
		self.action_move.setChecked(True)
		self.action_select.setCheckable(True)
		self.action_xrange.setCheckable(True)
		self.action_yrange.setCheckable(True)
		self.action_lasso.setCheckable(True)
		self.action_slice.setCheckable(True)
		#self.action_zoom_out.setCheckable(True)
		#self.action_group_main.

		#action = self.toolbar.addAction(icon
		self.syncToolbar()
		#self.action_select_mode_replace.setChecked(True)
		self.select_mode = "replace"
		self.setMode(self.action_move)
		self.toolbar.setIconSize(QtCore.QSize(16, 16))
		layout.addWidget(self.toolbar)

	def onActionDisplay(self, action):
		logger.debug("display = %r" % action.text())
		self.action_display_current = action
		self.plot()

	def onActionMiniMode(self):
		#targetAction = self.mini_mode_button.defaultAction()
		enabled_mini_mode = self.action_mini_mode_compact.isChecked() or self.action_mini_mode_ultra.isChecked()
		#enabled_mini_mode = self.action_mini_mode_normal.isChecked() or self.action_mini_mode_ultra.isChecked()
		ultra_mode = self.action_mini_mode_ultra.isChecked()

		logger.debug("mini screen: %r (ultra: %r)" % (enabled_mini_mode, ultra_mode))
		toolbuttons = self.toolbar.findChildren(QtGui.QToolButton)
		for toolbutton in toolbuttons:
			toolbutton.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly if enabled_mini_mode else QtCore.Qt.ToolButtonTextUnderIcon)

		if enabled_mini_mode:
			values = self.fig.subplotpars
			self.subplotpars_values = {"left":values.left, "right":values.right, "bottom":values.bottom, "top":values.top}
			self.bottomHeight = self.bottomFrame.height()

		self.bottomFrame.setVisible(not enabled_mini_mode)
		if 0:
			if enabled_mini_mode:
				self.resize(QtCore.QSize(self.width(), self.height() - self.bottomHeight))
			else:
				self.resize(QtCore.QSize(self.width(), self.height() + self.bottomHeight))
		self.fig.tight_layout()
		if enabled_mini_mode:
			if ultra_mode:
				self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1.)
		else:
			self.fig.subplots_adjust(**self.subplotpars_values)
		self.canvas.draw()

	def on_toolbar_toggle(self, ignore=None):
		#self.action_toolbar_toggle.toggle()
		visible = self.action_toolbar_toggle.isChecked()
		logger.debug("toolbar visible? %r" % (visible, ))
		for widget in [self.toolbar, self.toolbar2, self.status_bar]:
			widget.setVisible(visible)

	def onActionMiniModeNormal(self, *args):
		self.onActionMiniMode()
		#self.mini_mode_button.setDefaultAction(self.action_miniscreen)
		#self.action_miniscreen.setChecked(True)
		#self.action_miniscreen_ultra.setChecked(False)
		#self.on
		#logger.debug("normal mini screen: %r" % self.action_miniscreen.isChecked())
		#self.action_mini_mode.setIcon(self.action_mini_mode_normal.icon())
		#self.action_mini_mode.setText(self.action_mini_mode_normal.text())
		#self.onActionMiniMode()
		#self.action_mini_mode.trigger()
		pass

	def onActionMiniModeCompact(self, *args):
		self.onActionMiniMode()
	def onActionMiniModeUltra(self, *args):
		self.onActionMiniMode()
		#self.mini_mode_button.setDefaultAction(self.action_miniscreen_ultra)
		#logger.debug("ultra mini screen: %r" % self.action_miniscreen_ultra.isChecked())
		#self.action_mini_mode.setIcon(self.action_mini_mode_ultra.icon())
		#self.action_mini_mode.setText(self.action_mini_mode_ultra.text())
		#self.action_mini_mode.trigger()
		#self.onActionMiniMode()
		#self.onActionMiniScreen()
		#self.action_miniscreen.setChecked(False)
		#self.action_miniscreen_ultra.setChecked(True)

	def setSelectMode(self, action):
		self.select_mode_button.setDefaultAction(action)
		if action == self.action_select_mode_replace:
			self._select_mode = "replace"
		if action == self.action_select_mode_and:
			self._select_mode = "and"
		if action == self.action_select_mode_or:
			self._select_mode = "or"
		if action == self.action_select_mode_xor:
			self._select_mode = "xor"
		if action == self.action_select_mode_subtract:
			self._select_mode = "subtract"

	@property
	def select_mode(self):
		return self._select_mode

	@select_mode.setter
	def select_mode(self, value):
		logger.debug("set to: %r", value)
		action = getattr(self, "action_select_mode_%s" % value)
		self.setSelectMode(action)

	def onActionSelectNone(self):
		#self.dataset.selectMask(None)
		#self.dataset.executor.execute()
		layer = self.current_layer
		if layer is not None:
			action = undo.ActionMask(layer.dataset.undo_manager, "clear selection", None, layer.apply_mask)
			action.do()
		#self.checkUndoRedo()

	def onActionSelectInvert(self):
		layer = self.current_layer
		if layer is not None:
			mask = layer.dataset.mask
			if mask is not None:
				mask = ~mask
			else:
				mask = np.ones(len(self.dataset), dtype=np.bool)
			action = undo.ActionMask(layer.dataset.undo_manager, "invert selection", mask, layer.apply_mask)
			action.do()

	def onActionSelect(self):
		self.lastActionSelect.setChecked(True)
		self.setMode(self.lastActionSelect)
		self.syncToolbar()

	def syncToolbar(self):
		for plugin in self.plugins:
			plugin.syncToolbar()
		for action in [self.action_select]: #, self.action_zoom]:
			logger.debug("sync action: %r" % action.text())
			subactions = action.menu().actions()
			subaction_selected = [subaction for subaction in subactions if subaction.isChecked()]
			#if len(subaction_selected) > 0:
			#	action.setText(subaction_selected[0].text())
			#	action.setIcon(subaction_selected[0].icon())
			logger.debug(" subaction_selected: %r" % subaction_selected)
			logger.debug(" action was selected?: %r" % action.isChecked())
			action.setChecked(len(subaction_selected) > 0)
			logger.debug(" action  is selected?: %r" % action.isChecked())
		logger.debug("last select action: %r" % self.lastActionSelect.text())
		#logger.debug("last zoom action: %r" % self.lastActionZoom.text())
		self.action_select.setText(self.lastActionSelect.text())
		self.action_select.setIcon(self.lastActionSelect.icon())
		#self.action_zoom.setText(self.lastActionZoom.text())
		#self.action_zoom.setIcon(self.lastActionZoom.icon())
		#self.action_select.update()

	def check_aspect(self, axis_follow):
		if self.aspect is not None:
			if self.ranges_show is None:
				return
			if any([k is None for k in self.ranges_show]):
				return
			width = self.ranges_show[axis_follow][1] - self.ranges_show[axis_follow][0]
			centers = [(self.ranges_show[i][1] + self.ranges_show[i][0])/2. for i in range(self.dimensions)]
			for i in range(self.dimensions):
				print i, self.ranges_show[i]
				if i != axis_follow:
					self.ranges_show[i] = [None, None]
					self.ranges_show[i][0] = centers[i] - width/2
					self.ranges_show[i][1] = centers[i] + width/2
				print i, self.ranges_show[i]
			return



			otheraxes = list(range(self.dimensions))
			allaxes = list(range(self.dimensions))
			otheraxes.remove(axis_follow)
			#ranges = [self.ranges_show[i] if self.ranges_show[i] is not None else self.ranges[i] for i in otheraxes]
			ranges = [self.ranges_show[i] for i in otheraxes]

			logger.debug("aspect 1")
			if None in ranges:
				return
			width = self.ranges_show[axis_follow][1] - self.ranges_show[axis_follow][0]
			#width = ranges[axis_follow][1] - ranges[axis_follow][0]
			center = (self.ranges_show[axis_follow][1] + self.ranges_show[axis_follow][0])/2.
			logger.debug("aspect 2")

			#widths = [ranges[i][1] - ranges[i][0] for i in range(self.dimensions-1)]
			centers = [(self.ranges_show[i][1] + self.ranges_show[i][0])/2. for i in range(self.dimensions)]
			logger.debug("aspect 3")

			#xmin, xmax = self.ranges[0]
			#ymin, ymax = self.ranges[1]
			for i in range(self.dimensions-1):
				axis_index = otheraxes[i]
				#if self.ranges_show[i] is None:
				#	self.ranges_show[i] = self.ranges[i]
				print self.ranges_show
				self.ranges_show[axis_index] = [None, None]
				self.ranges_show[axis_index][0] = centers[axis_index] - width/2
				self.ranges_show[axis_index][1] = centers[axis_index] + width/2
				logger.debug("aspect i=%d,%d", i, axis_index)
				print self.ranges_show
			for layer in self.layers:
				for i in range(self.dimensions-1):
					axis_index = otheraxes[i]
					layer.ranges_grid[axis_index] = list(self.ranges_show[axis_index])
				layer.ranges_grid[axis_follow] = list(self.ranges_show[axis_follow])




class HistogramPlotDialog(PlotDialog):
	type_name = "histogram"
	#names = "histogram,1d"
	def __init__(self, parent, dataset, app, **kwargs):
		super(HistogramPlotDialog, self).__init__(parent, dataset, 1, ["X"], app, **kwargs)

	def beforeCanvas(self, layout):
		self.addToolbar(layout, yselect=False, lasso=False)

	def _afterCanvas(self, layout):
		self.addToolbar2(layout, contrast=False, gamma=False)
		super(HistogramPlotDialog, self).afterCanvas(layout)

	def add_histogram(self, x, counts, selection):
		self.histograms.append((x, counts))


	def plot(self):
		t0 = time.time()
		self.axes.cla()
		self.axes.autoscale(False)
		#if self.expression_error:
		#	return
		#P.hist(x, 50, normed=1, histtype='stepfilled')
		#values =
		if len(self.layers) == 0:
			return
		if self.range_level_show is None:
			logger.error("cannot plot when range_level_show is None")
			return
		first_layer = self.layers[0]

		#for i in range(self.dimensions):
		#	if self.ranges_show[i] is None:
		#		self.ranges_show[i] = copy.copy(first_layer.ranges_grid[i])


		for layer in self.layers:
			layer.plot([self.axes], self.add_histogram)
			#layer.prepare_histograms()


		self.axes.set_xlabel(first_layer.expressions[0])
		xmin_show, xmax_show = self.ranges_show[0]
		self.axes.set_xlim(xmin_show, xmax_show)
		#if self.range_level_show is None:
		#	self.range_level_show = first_layer.range_level
		ymin_show, ymax_show = self.range_level_show
		if not first_layer.weight_expression:
			self.axes.set_ylabel("counts")
		else:
			self.axes.set_ylabel(first_layer.weight_expression)
		self.axes.set_ylim(ymin_show, ymax_show)
		if not self.action_mini_mode_ultra.isChecked():
			self.fig.tight_layout(pad=0.0)
		self.canvas.draw()
		self.update()
		self.message("plotting %.2fs" % (time.time() - t0), index=100)
		self.signal_plot_finished.emit(self, self.fig)
		return



		Nvector = self.grid_size
		width = self.ranges_show[0][1] - self.ranges_show[0][0]
		x = np.arange(0, Nvector)/float(Nvector) * width + self.ranges_show[0][0]# + width/(Nvector/2.)
		xmin, xmax = self.ranges_show[0]
		xmin, xmax = self.ranges_show[0]
		if self.ranges_show[0] is None:
			self.ranges_show[0] = xmin, xmax

		self.delta = (xmax - xmin) / self.grid_size
		self.centers = (np.arange(self.grid_size)+0.5) * self.delta + xmin

		logger.debug("expr for amplitude: %r" % self.amplitude_expression)
		grid_map = self.create_grid_map(self.grid_size, False)
		amplitude = self.eval_amplitude(self.amplitude_expression, locals=grid_map)
		use_selection = self.dataset.mask is not None
		if use_selection:
			grid_map_selection = self.create_grid_map(self.grid_size, True)
			amplitude_selection = self.eval_amplitude(self.amplitude_expression, locals=grid_map_selection)

		if use_selection:
			self.axes.bar(self.centers, amplitude, width=self.delta, align='center')
			self.axes.bar(self.centers, amplitude_selection, width=self.delta, align='center', color="red", alpha=0.8)
		else:
			self.axes.bar(self.centers, amplitude, width=self.delta, align='center')

		if self.range_level is None:
			if self.weight_expression:
				self.range_level = np.nanmin(amplitude) * 1.1, np.nanmax(amplitude) * 1.1
			else:
				self.range_level = 0, np.nanmax(amplitude) * 1.1
		if self.action_mini_mode_compact:
			self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1.)

class ScatterPlotDialog(PlotDialog):
	type_name = "density2d"
	#names = "heatmap,density2d,2d"
	def __init__(self, parent, dataset, app, **options):
		super(ScatterPlotDialog, self).__init__(parent, dataset, 2, "X Y".split(), app, **options)



	def _afterCanvas(self, layout):
		self.addToolbar2(layout)
		super(ScatterPlotDialog, self).afterCanvas(layout)

	def add_image_layer(self, rgba, intensity):
		self.image_layers.append(rgba)

	def plot(self):
		self.image_layers = []
		self.axes.rgb_images = self.image_layers
		self.axes.cla()
		if len(self.layers) == 0:
			self.canvas.draw()
			self.update()
			return
		first_layer = self.layers[0]

		N = self.grid_size
		background = np.ones((N, N, 4), dtype=np.float64)
		background[:,:,0:3] = matplotlib.colors.colorConverter.to_rgb(self.background_color)
		background[:,:,3] = 1.

		ranges = []
		for minimum, maximum in first_layer.ranges_grid:
			ranges.append(minimum)
			ranges.append(maximum)

		placeholder = self.axes.imshow(background, extent=ranges, origin="lower")
		self.add_image_layer(background, None)

		for i in range(self.dimensions):
			if self.ranges_show[i] is None:
				self.ranges_show[i] = copy.copy(first_layer.ranges_grid[i])
		#extent =
		#ranges = np.nanmin(datax), np.nanmax(datax), np.nanmin(datay), np.nanmax(datay)

		xmin, xmax = self.ranges_show[0]
		ymin, ymax = self.ranges_show[1]
		width = xmax - xmin
		height = ymax - ymin
		extent = [xmin-width, xmax+width, ymin-height, ymax+height]
		Z1 = np.array(([0,1]*8 + [1,0]*8)*8); Z1.shape = 16,16  # chessboard
		#im1 = self.axes.imshow(Z1, cmap="gray", interpolation='nearest', extent=extent, vmin=-4, vmax=1.)



		for layer in self.layers:
			layer.plot([self.axes], self.add_image_layer)



		rgba = vaex.ui.imageblending.blend(self.image_layers, self.blend_mode)
		rgba[...,3] = rgba[...,3] * 0 + 1
		for c in range(4):
			#rgba_dest[:,:,c] = np.clip((rgba_dest[:,:,c] ** 3.5)*2.6, 0., 1.)
			rgba[:,:,c] = np.clip((rgba[:,:,c] ** self.layer_gamma)*self.layer_brightness, 0., 1.)
		placeholder.set_data((rgba * 255).astype(np.uint8))



		if self.aspect is None:
			self.axes.set_aspect('auto')
		else:
			self.axes.set_aspect(self.aspect)
			#if self.dataset.selected_row_index is not None:
				#self.axes.autoscale(False)
		#index = self.dataset.selected_row_index

		if 1:
			self.axes.set_xlabel(self.xlabel if self.xlabel is not None else first_layer.x)
			self.axes.set_ylabel(self.ylabel if self.ylabel is not None else first_layer.y)
		self.axes.set_xlim(*self.ranges_show[0])
		self.axes.set_ylim(*self.ranges_show[1])
		#self.fig.texts = []
		if 0:
			title_text = self.title_expression.format(**self.getVariableDict())
			if hasattr(self, "title"):
				self.title.set_text(title_text)
			else:
				self.title = self.fig.suptitle(title_text)
			self.canvas.draw()
			#self.fig.tight_layout(pad=0.0)#1.008) #pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)
			#self.fig.tight_layout(pad=0.01)#1.008) #pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)
		self.fig.tight_layout()#1.008) #pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)
		if self.action_mini_mode_ultra.isChecked():
			logger.debug("ultra compact mode")
			self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1.)
		self.canvas.draw()
		self.update()
		if 0:
			if self.first_time:
				self.first_time = False
				if "filename" in self.options:
					self.filename_figure_last = self.options["filename"]
					self.fig.savefig(self.filename_figure_last)

		self.signal_plot_finished.emit(self, self.fig)
		return
		if 1:
			ranges = []
			logger.debug("self.ranges == %r" % (self.ranges, ))
			for minimum, maximum in self.ranges:
				ranges.append(minimum)
				ranges.append(maximum)


			#amplitude = self.grids.grids["counts"].get_data(self.gridsize)

			logger.debug("expr for amplitude: %r" % self.amplitude_expression)
			grid_map = self.create_grid_map(self.grid_size, False)
			try:
				amplitude = self.eval_amplitude(self.amplitude_expression, locals=grid_map)
			except Exception as e:
				self.error_in_field(self.amplitude_box, "amplitude", e)
				return
			use_selection = self.dataset.mask is not None
			if use_selection:
				grid_map_selection = self.create_grid_map(self.grid_size, True)
				amplitude_selection = self.eval_amplitude(self.amplitude_expression, locals=grid_map_selection)


		if self.action_display_current == self.action_display_mode_both:
			self.axes.imshow(self.contrast(amplitude), origin="lower", extent=ranges, alpha=0.4 if use_selection else 1.0, cmap=self.colormap)
			if use_selection:
				self.axes.imshow(self.contrast(amplitude_selection), origin="lower", extent=ranges, alpha=1, cmap=self.colormap)
		if self.action_display_current == self.action_display_mode_full:
			self.axes.imshow(self.contrast(amplitude), origin="lower", extent=ranges, cmap=self.colormap)
		if self.action_display_current == self.action_display_mode_selection:
			if self.counts_mask is not None:
				self.axes.imshow(self.contrast(amplitude_mask), origin="lower", extent=ranges, alpha=1, cmap=self.colormap)
		if 1:
			#locals = {key:None if grid is None else vaex.vaexfast.resize(grid, 64) for key, grid in locals}
			locals = {}
			for name in list(self.grids.grids.keys()):
				grid = self.grids.grids[name]
				if name == "counts" or (grid.weight_expression is not None and len(grid.weight_expression) > 0):
					if grid.max_size >= self.vector_grid_size:
						locals[name] = grid.get_data(self.vector_grid_size, use_selection)
				else:
					locals[name] = None

			if 1:
				grid_map_vector = self.create_grid_map(self.vector_grid_size, use_selection)
				if grid_map_vector["weightx"] is not None and grid_map_vector["weighty"] is not None:
					mask = grid_map_vector["counts"] > (self.min_level_vector2d * grid_map_vector["counts"].max())
					x = grid_map_vector["x"]
					y = grid_map_vector["y"]
					x2d, y2d = np.meshgrid(x, y)
					vx = self.eval_amplitude("weightx/counts", locals=grid_map_vector)
					vy = self.eval_amplitude("weighty/counts", locals=grid_map_vector)
					meanvx = 0 if self.vectors_subtract_mean is False else vx[mask].mean()
					meanvy = 0 if self.vectors_subtract_mean is False else vy[mask].mean()
					vx -= meanvx
					vy -= meanvy
					if grid_map_vector["weightz"] is not None and self.vectors_color_code_3rd:
						colors = self.eval_amplitude("weightz/counts", locals=grid_map_vector)
						self.axes.quiver(x2d[mask], y2d[mask], vx[mask], vy[mask], colors[mask], cmap=self.colormap_vector)#, scale=1)
					else:
						self.axes.quiver(x2d[mask], y2d[mask], vx[mask], vy[mask], color="black")
						colors = None
				#self.axes.quiver(x, y, U, V)
		if self.action_display_current == self.action_display_mode_both_contour:
			#self.axes.imshow(amplitude, origin="lower", extent=ranges, alpha=1 if self.counts_mask is None else 0.4, cmap=cm_plusmin)
			#self.axes.contour(amplitude, origin="lower", extent=ranges, levels=levels, linewidths=2, colors="red")
			self.axes.imshow(amplitude, origin="lower", extent=ranges, cmap=self.colormap)
			if self.counts_mask is not None:
				values = amplitude_mask[~np.isinf(amplitude_mask)]
				levels = np.linspace(values.min(), values.max(), 5)
				#self.axes.imshow(amplitude_mask, origin="lower", extent=ranges, alpha=1, cmap=cm_plusmin)
				self.axes.contour(amplitude_mask, origin="lower", extent=ranges, levels=levels, linewidths=2, colors="red")

		for callback in self.plugin_grids_draw:
			callback(self.axes, grid_map, grid_map_vector)


		if self.aspect is None:
			self.axes.set_aspect('auto')
		else:
			self.axes.set_aspect(self.aspect)
			#if self.dataset.selected_row_index is not None:
				#self.axes.autoscale(False)
		if 0:
			index = self.dataset.selected_row_index
			if index is not None and self.selected_point is None:
				logger.debug("point selected but after computation")
				# TODO: optimize
				def find_selected_point(info, blockx, blocky):
					if index >= info.i1 and index < info.i2: # selected point is in this block
						self.selected_point = blockx[index-info.i1], blocky[index-info.i1]
				self.dataset.evaluate(find_selected_point, *self.expressions, **self.getVariableDict())


			if self.selected_point:
				#x, y = self.getdatax()[self.dataset.selected_row_index],  self.getdatay()[self.dataset.selected_row_index]
				x, y = self.selected_point
				self.axes.scatter([x], [y], color='red') #, scalex=False, scaley=False)
			#if dataxsel is not None:
			#	self.axes.scatter(dataxsel, dataysel)
		self.axes.set_xlabel(self.expressions[0])
		self.axes.set_ylabel(self.expressions[1])
		self.axes.set_xlim(*self.ranges_show[0])
		self.axes.set_ylim(*self.ranges_show[1])
		#self.fig.texts = []
		title_text = self.title_expression.format(**self.getVariableDict())
		if hasattr(self, "title"):
			self.title.set_text(title_text)
		else:
			self.title = self.fig.suptitle(title_text)
		if not self.action_mini_mode_ultra.isChecked():
			self.fig.tight_layout(pad=0.0)#1.008) #pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)
		#self.fig.tight_layout(pad=0.01)#1.008) #pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)
		self.fig.tight_layout()#1.008) #pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)
		self.canvas.draw()
		self.update()
		if self.first_time:
			self.first_time = False
			if "filename" in self.options:
				self.filename_figure_last = self.options["filename"]
				self.fig.savefig(self.filename_figure_last)



class ScatterPlotMatrixDialog(PlotDialog):
	def __init__(self, parent, dataset, expressions):
		super(ScatterPlotMatrixDialog, self).__init__(parent, dataset, list(expressions), "X Y Z W V U T S R Q P".split()[:len(expressions)])

	def getAxesList(self):
		return reduce(lambda x,y: x + y, self.axes_grid, [])

	def addAxes(self):
		self.axes_grid = [[None,] * self.dimensions for _ in range(self.dimensions)]
		index = 0
		for i in range(self.dimensions)[::1]:
			for j in range(self.dimensions)[::1]:
				index = ((self.dimensions-1)-j) * self.dimensions + i + 1
				axes = self.axes_grid[i][j] = self.fig.add_subplot(self.dimensions,self.dimensions,index)
#													   sharey=self.axes_grid[0][j] if j > 0 else None,
#													   sharex=self.axes_grid[i][0] if i > 0 else None
#													   )
				# store the axis index in matplotlib object
				axes.xaxis_index = i
				axes.yaxis_index = j
				if i > 0:
					axes.yaxis.set_visible(False)
					#for label in axes.get_yticklabels():
					#	label.set_visible(False)
					#axes.yaxis.offsetText.set_visible(False)
				if j > 0:
					axes.xaxis.set_visible(False)
					#for label in axes.get_xticklabels():
					#	label.set_visible(False)
					#axes.xaxis.offsetText.set_visible(False)
				self.axes_grid[i][j].hold(True)
				index += 1
		self.fig.subplots_adjust(hspace=0, wspace=0)

	def calculate_visuals(self, info, *blocks):
		data_blocks = blocks[:self.dimensions]
		if len(blocks) > self.dimensions:
			weights_block = blocks[self.dimensions]
		else:
			weights_block = None
		elapsed = time.time() - info.time_start
		self.message("computation %.2f%% (%f seconds)" % (info.percentage, elapsed), index=9)
		QtCore.QCoreApplication.instance().processEvents()
		self.expression_error = False

		N = self.grid_size
		mask = self.dataset.mask
		if info.first:
			self.counts = np.zeros((N,) * self.dimensions, dtype=np.float64)
			self.counts_weights = self.counts
			if weights_block is not None:
				self.counts_weights = np.zeros((N,) * self.dimensions, dtype=np.float64)

			self.selected_point = None
			if mask is not None:
				self.counts_mask = np.zeros((N,) * self.dimensions, dtype=np.float64) #mab.utils.numpy.mmapzeros((128), dtype=np.float64)
				self.counts_weights_mask = self.counts_mask
				if weights_block is not None:
					self.counts_weights_mask = np.zeros((N,) * self.dimensions, dtype=np.float64)
			else:
				self.counts_mask = None
				self.counts_weights_mask = None

		if info.error:
			self.expression_error = True
			self.message(info.error_text)
			return


		xmin, xmax = self.ranges[0]
		ymin, ymax = self.ranges[1]
		for i in range(self.dimensions):
			if self.ranges_show[i] is None:
				self.ranges_show[i] = self.ranges[i]


		index = self.dataset.selected_row_index
		if index is not None:
			if index >= info.i1 and index < info.i2: # selected point is in this block
				self.selected_point = blockx[index-info.i1], blocky[index-info.i1]

		t0 = time.time()
		#histo2d(blockx, blocky, self.counts, *self.ranges)
		ranges = []
		for minimum, maximum in self.ranges:
			ranges.append(minimum)
			if minimum == maximum:
				maximum += 1
			ranges.append(maximum)
		try:
			args = data_blocks, self.counts, ranges
			if self.dimensions == 2:
				vaex.histogram.hist3d(data_blocks[0], data_blocks[1], self.counts, *ranges)
			if self.dimensions == 3:
				vaex.histogram.hist3d(data_blocks[0], data_blocks[1], data_blocks[2], self.counts, *ranges)
			if weights_block is not None:
				args = data_blocks, weights_block, self.counts, ranges
				vaex.histogram.hist2d_weights(blockx, blocky, self.counts_weights, weights_block, *ranges)
		except:
			raise

		if mask is not None:
			subsets = [block[mask[info.i1:info.i2]] for block in data_blocks]
			if self.dimensions == 2:
				vaex.histogram.hist2d(subsets[0], subsets[1], self.counts_weights_mask, *ranges)
			if self.dimensions == 3:
				vaex.histogram.hist3d(subsets[0], subsets[1], subsets[2], self.counts_weights_mask, *ranges)
			if weights_block is not None:
				subset_weights = weights_block[mask[info.i1:info.i2]]
				if self.dimensions == 2:
					vaex.histogram.hist2d_weights(subsets[0], subsets[1], self.counts_weights_mask, subset_weights, *ranges)
				if self.dimensions == 3:
					vaex.histogram.hist3d_weights(subsets[0], subsets[1], subsets[2], self.counts_weights_mask, subset_weights, *ranges)
		if info.last:
			elapsed = time.time() - info.time_start
			self.message("computation (%f seconds)" % (elapsed), index=9)


	def plot(self):
		t0 = time.time()
		#self.axes.cla()
		#extent =
		#ranges = np.nanmin(datax), np.nanmax(datax), np.nanmin(datay), np.nanmax(datay)
		ranges = []
		for minimum, maximum in self.ranges:
			ranges.append(minimum)
			ranges.append(maximum)

		amplitude = self.counts
		logger.debug("expr for amplitude: %r" % self.amplitude_expression)
		if self.amplitude_expression is not None:
			locals = {"counts":self.counts_weights, "counts1": self.counts}
			globals = np.__dict__
			amplitude = eval(self.amplitude_expression, globals, locals)
		#if self.ranges_level[0] is None:
		#	self.ranges_level[0] = 0, amplitude.max() * 1.1


		def multisum(a, axes):
			correction = 0
			for axis in axes:
				a = np.sum(a, axis=axis-correction)
				correction += 1
			return a
		for i in range(self.dimensions):
			for j in range(self.dimensions):
				axes = self.axes_grid[i][j]
				ranges = self.ranges[i] + self.ranges[j]
				axes.clear()
				allaxes = list(range(self.dimensions))
				if 0 :#i > 0:
					for label in axes.get_yticklabels():
						label.set_visible(False)
					axes.yaxis.offsetText.set_visible(False)
				if 0: #j > 0:
					for label in axes.get_xticklabels():
						label.set_visible(False)
					axes.xaxis.offsetText.set_visible(False)
				if i != j:
					allaxes.remove(i)
					allaxes.remove(j)
					counts_mask = None
					counts = multisum(self.counts, allaxes)
					if self.counts_mask is not None:
						counts_mask = multisum(self.counts_mask, allaxes)
					if i > j:
						counts = counts.T
					axes.imshow(np.log10(counts), origin="lower", extent=ranges, alpha=1 if counts_mask is None else 0.4)
					if counts_mask is not None:
						if i > j:
							counts_mask = counts_mask.T
						axes.imshow(np.log10(counts_mask), origin="lower", extent=ranges)
					axes.set_aspect('auto')
					if self.dataset.selected_row_index is not None:
						#self.axes.autoscale(False)
						x, y = self.getdatax()[self.dataset.selected_row_index],  self.getdatay()[self.dataset.selected_row_index]
						axes.scatter([x], [y], color='red') #, scalex=False, scaley=False)

					axes.set_xlim(self.ranges_show[i][0], self.ranges_show[i][1])
					axes.set_ylim(self.ranges_show[j][0], self.ranges_show[j][1])
				else:
					allaxes.remove(j)
					counts = multisum(self.counts, allaxes)
					N = len(counts)
					xmin, xmax = self.ranges[i]
					delta = (xmax - xmin) / N
					centers = np.arange(N) * delta + xmin

					#axes.autoscale(False)
					#P.hist(x, 50, normed=1, histtype='stepfilled')
					#values =
					if 1: #if self.counts_mask is None:
						axes.bar(centers, counts, width=delta, align='center')
					else:
						self.axes.bar(self.centers, self.counts, width=self.delta, align='center', alpha=0.5)
						self.axes.bar(self.centers, self.counts_mask, width=self.delta, align='center', color="red")
					axes.set_xlim(self.ranges_show[i][0], self.ranges_show[i][1])
					axes.set_ylim(0, np.max(counts)*1.1)

		if 0:

			self.axes.imshow(amplitude.T, origin="lower", extent=ranges, alpha=1 if self.counts_mask is None else 0.4, cmap=cm_plusmin)
			if 1:
				if self.counts_mask is not None:
					if self.amplitude_expression is not None:
						#locals = {"counts":self.counts_mask}
						locals = {"counts":self.counts_weights_mask, "counts1": self.counts_mask}
						globals = np.__dict__
						amplitude_mask = eval(self.amplitude_expression, globals, locals)
					self.axes.imshow(amplitude_mask.T, origin="lower", extent=ranges, alpha=1, cmap=cm_plusmin)
				#self.axes.imshow((I), origin="lower", extent=ranges)
			self.axes.set_aspect('auto')
				#if self.dataset.selected_row_index is not None:
					#self.axes.autoscale(False)
			index = self.dataset.selected_row_index
			if index is not None and self.selected_point is None:
				logger.debug("point selected but after computation")
				# TODO: optimize
				def find_selected_point(info, blockx, blocky):
					if index >= info.i1 and index < info.i2: # selected point is in this block
						self.selected_point = blockx[index-info.i1], blocky[index-info.i1]
				self.dataset.evaluate(find_selected_point, *self.expressions, **self.getVariableDict())


			if self.selected_point:
				#x, y = self.getdatax()[self.dataset.selected_row_index],  self.getdatay()[self.dataset.selected_row_index]
				x, y = self.selected_point
				self.axes.scatter([x], [y], color='red') #, scalex=False, scaley=False)
			#if dataxsel is not None:
			#	self.axes.scatter(dataxsel, dataysel)
			self.axes.set_xlabel(self.expressions[0])
			self.axes.set_ylabel(self.expressions[0])
			self.axes.set_xlim(*self.ranges_show[0])
			self.axes.set_ylim(*self.ranges_show[1])
		self.canvas.draw()
		self.message("ploting %f" % (time.time() - t0), index=5)

time_previous = time.time()
time_start = time.time()
def timelog(msg, reset=False):
	global time_previous, time_start
	now = time.time()
	if reset:
		time_start = now
	T = now - time_start
	deltaT = now - time_previous
	logger.info("*** TIMELOG: %s (T=%f deltaT=%f)" % (msg, T, deltaT))
	time_previous = now

class VolumeRenderingPlotDialog(PlotDialog):
	type_name = "volumerendering"
	#names = "volumerendering,3d"
	def __init__(self, parent, dataset, **options):
		super(VolumeRenderingPlotDialog, self).__init__(parent, dataset, 3, "X Y Z".split(), **options)
		#[xname, yname, zname]

	def closeEvent(self, event):
		self.widget_volume.orbit_stop()
		super(VolumeRenderingPlotDialog, self).closeEvent(event)

	def afterCanvas(self, layout):

		self.widget_volume = vaex.ui.volumerendering.VolumeRenderWidget(self)
		self.layout_plot_region.insertWidget(0, self.widget_volume, 1)


		#self.addToolbar2(layout)
		super(VolumeRenderingPlotDialog, self).afterCanvas(layout)

		self.menu_view.addSeparator()
		self.action_group_quality_3d = QtGui.QActionGroup(self)
		self.actions_quality_3d = []
		for index, (name, resolution, iterations) in enumerate([("Fast", 256, 200), ("Medium", 256+128, 300), ("Best", 512, 500)]):
			if "vr_quality" in self.options:
				vr_index = int(eval(self.options.get("vr_quality")))
				if index == vr_index:
					#self.actions_quality_3d[index].trigger()
					self.widget_volume.ray_iterations = iterations
					self.widget_volume.texture_size = resolution

			action_quality_3d = QtGui.QAction(QtGui.QIcon(iconfile('picture_empty')), name + ' volume rendering', self)
			def do(ignore=None, resolution=resolution, iterations=iterations):
				self.widget_volume.set_iterations(iterations)
				self.widget_volume.setResolution(resolution)
				self.widget_volume.update()
			action_quality_3d.setCheckable(True)
			if resolution == self.widget_volume.texture_size and iterations == self.widget_volume.ray_iterations:
				action_quality_3d.setChecked(True)
			action_quality_3d.triggered.connect(do)
			action_quality_3d.setShortcut("Ctrl+Shift+Meta+%d" % (index+1))
			self.menu_view.addAction(action_quality_3d)
			self.action_group_quality_3d.addAction(action_quality_3d)
			self.actions_quality_3d.append(action_quality_3d)


	def getAxesList(self):
		#return reduce(lambda x,y: x + y, self.axes_grid, [])
		return [self.axis_top, self.axis_bottom]

	def add_pages(self, toolbox):
		self.frame_options_volume_rendering = QtGui.QFrame(self)

		toolbox.addItem(self.frame_options_volume_rendering, "Volume rendering")
		toolbox.setCurrentIndex(3)
		self.fill_page_volume_rendering(self.frame_options_volume_rendering)

	def add_axes(self):
		self.axes_grid = [[None,] * self.dimensions for _ in range(self.dimensions)]
		self.axis_top = self.fig.add_subplot(2,1,1)
		self.axis_bottom = self.fig.add_subplot(2,1,2)
		self.axis_top.xaxis_index = 0
		self.axis_top.yaxis_index = 1
		self.axis_bottom.xaxis_index = 0
		self.axis_bottom.yaxis_index = 2
		#self.fig.subplots_adjust(hspace=0, wspace=0)


	def add_image_layer(self, rgba, intensity):
		self.image_layers.append(intensity)

	def plot(self):
		self.image_layers = []
		#self.image_layers = []

		axes_list = self.getAxesList()
		for axes in axes_list:
			axes.cla()
		if len(self.layers) == 0:
			return
		first_layer = self.layers[0]

		for i in range(self.dimensions):
			if self.ranges_show[i] is None:
				self.ranges_show[i] = copy.copy(first_layer.ranges_grid[i])
		#extent =
		for axes in axes_list:
			ranges = []
			for minimum, maximum in [self.ranges_show[axes.xaxis_index], self.ranges_show[axes.yaxis_index], ]:
				ranges.append(minimum)
				ranges.append(maximum)
			axes.rgb_images = []
			N = self.grid_size
			background = np.ones((N, N, 4), dtype=np.float64)
			background[:,:,0:3] = matplotlib.colors.colorConverter.to_rgb(self.background_color)
			background[:,:,3] = 1.
			axes.placeholder = axes.imshow(background, extent=ranges, origin="lower")
			axes.rgb_images.append(background)
			colors = "red green blue".split()
			axes.spines['bottom'].set_color(colors[axes.xaxis_index])
			axes.spines['left'].set_color(colors[axes.yaxis_index])
			linewidth = 2.
			axes.spines['bottom'].set_linewidth(linewidth)
			axes.spines['left'].set_linewidth(linewidth)
			if self.aspect is None:
				axes.set_aspect('auto')
			else:
				axes.set_aspect(self.aspect)


		for layer in self.layers:
			layer.plot(axes_list, self.add_image_layer)
		#for image in self.image_layers:
		if first_layer.amplitude_selection is not None:
			self.widget_volume.setGrid(first_layer.amplitude_selection, first_layer.amplitude, first_layer.vector_grid)
		else:
			self.widget_volume.setGrid(first_layer.amplitude, vectorgrid=first_layer.vector_grid)

		for axes in axes_list:
			rgba = vaex.ui.imageblending.blend(axes.rgb_images, self.blend_mode)
			rgba[...,3] = rgba[...,3] * 0 + 1
			for c in range(4):
				#rgba_dest[:,:,c] = np.clip((rgba_dest[:,:,c] ** 3.5)*2.6, 0., 1.)
				rgba[:,:,c] = np.clip((rgba[:,:,c] ** self.layer_gamma)*self.layer_brightness, 0., 1.)
			axes.placeholder.set_data((rgba * 255).astype(np.uint8))

		#if self.aspect is None:
		#	self.axes.set_aspect('auto')
		#else:
		#	self.axes.set_aspect(self.aspect)
		#self.axes.set_xlim(*self.ranges_show[0])
		#self.axes.set_ylim(*self.ranges_show[1])
		self.fig.tight_layout()#1.008) #pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)
		self.canvas.draw()
		self.update()


	def plot_(self):
		timelog("plot start", reset=False)
		t0 = time.time()
		if 1:
			ranges = []
			for minimum, maximum in self.ranges_show:
				ranges.append(minimum)
				ranges.append(maximum)

			timelog("creating grid map")
			grid_map = self.create_grid_map(self.grid_size, False)
			timelog("eval amplitude")
			amplitude = self.eval_amplitude(self.amplitude_expression, locals=grid_map)
			timelog("eval amplitude done")
			use_selection = self.dataset.mask is not None
			if use_selection:
				timelog("repeat for selection")
				grid_map_selection = self.create_grid_map(self.grid_size, True)
				amplitude_selection = self.eval_amplitude(self.amplitude_expression, locals=grid_map_selection)

			timelog("creating grid map vector")
			grid_map_vector = self.create_grid_map(self.vector_grid_size, use_selection)
			vector_grid = None
			vector_counts = grid_map_vector["counts"]
			vector_mask = vector_counts > 0
			if grid_map_vector["weightx"] is not None:
				vector_x = grid_map_vector["x"]
				vx = self.eval_amplitude("weightx/counts", locals=grid_map_vector)
			else:
				vector_x = None
				vx = None
			if grid_map_vector["weighty"] is not None:
				vector_y = grid_map_vector["y"]
				vy = self.eval_amplitude("weighty/counts", locals=grid_map_vector)
			else:
				vector_y = None
				vy = None
			if grid_map_vector["weightz"] is not None:
				vector_z = grid_map_vector["z"]
				vz = self.eval_amplitude("weightz/counts", locals=grid_map_vector)
			else:
				vector_z = None
				vz = None
			if vx is not None and vy is not None and vz is not None:
				timelog("making vector grid")
				vector_grid = np.zeros((4, ) + ((vx.shape[0],) * 3), dtype=np.float32)
				mask = vector_counts > 0
				meanvx = 0 if self.vectors_subtract_mean is False else vx[mask].mean()
				meanvy = 0 if self.vectors_subtract_mean is False else vy[mask].mean()
				meanvz = 0 if self.vectors_subtract_mean is False else vz[mask].mean()
				vector_grid[0] = vx - meanvx
				vector_grid[1] = vy - meanvy
				vector_grid[2] = vz - meanvz
				vector_grid[3] = vector_counts
				vector_grid = np.swapaxes(vector_grid, 0, 3)
				vector_grid = vector_grid * 1.
			timelog("setting grid")
			if use_selection:
				self.widget_volume.setGrid(amplitude_selection, amplitude, vectorgrid=vector_grid)
			else:
				self.widget_volume.setGrid(amplitude, vectorgrid=vector_grid)
			timelog("grid")
			if 0:
				self.tool.grid = amplitude
				self.tool.update()
			
			#if self.ranges_level[0] is None:
			#	self.ranges_level[0] = 0, amplitude.max() * 1.1
			#return


			def multisum(a, axes):
				correction = 0
				for axis in axes:
					a = np.nansum(a, axis=axis-correction)
					correction += 1
				return a		
			axeslist = self.getAxesList()
			vector_values = [vx, vy, vz]
			vector_positions = [vector_x, vector_y, vector_z]
			for i in range(2):
					timelog("axis: " +str(i))
					axes = axeslist[i]
					i1 = 0
					i2 = i + 1
					i3 = 2- i
					ranges = list(self.ranges[i1]) + list(self.ranges[i2])
					axes.clear()
					allaxes = list(range(self.dimensions))
					if 0 :#i > 0:
						for label in axes.get_yticklabels():
							label.set_visible(False)
						axes.yaxis.offsetText.set_visible(False)
					if 0: #j > 0:
						for label in axes.get_xticklabels():
							label.set_visible(False)
						axes.xaxis.offsetText.set_visible(False)
					if 1:
						allaxes.remove(2-(0))
						allaxes.remove(2-(1+i))
						counts_mask = None
						colors = "red green blue".split()
						axes.spines['bottom'].set_color(colors[i1])
						axes.spines['left'].set_color(colors[i2])
						linewidth = 2.
						axes.spines['bottom'].set_linewidth(linewidth)
						axes.spines['left'].set_linewidth(linewidth)

						grid_map_2d = {key:None if grid is None else (grid if grid.ndim != 3 else multisum(grid, allaxes)) for key, grid in list(grid_map.items())}
						amplitude = self.eval_amplitude(self.amplitude_expression, locals=grid_map_2d)
						if use_selection:
							grid_map_selection_2d = {key:None if grid is None else (grid if grid.ndim != 3 else multisum(grid, allaxes)) for key, grid in list(grid_map_selection.items())}
							amplitude_selection = self.eval_amplitude(self.amplitude_expression, locals=grid_map_selection_2d)

						axes.imshow(self.contrast(amplitude), origin="lower", extent=ranges, alpha=0.4 if use_selection else 1.0, cmap=self.colormap)
						if use_selection:
							axes.imshow(self.contrast(amplitude_selection), origin="lower", extent=ranges, alpha=1, cmap=self.colormap)

						#vector_positions1, vector_positions2,  = vector_positions[i1],  vector_positions[i2]
						#vector_values1, vector_values2 = vector_values[i1],  vector_values[i2]
						if vector_positions[i1] is not None and vector_positions[i2] is not None:
							mask = multisum(vector_counts, allaxes) > 0
							x, y = np.meshgrid(vector_positions[i1], vector_positions[i2])
							U = multisum(vector_values[i1], allaxes)
							V = multisum(vector_values[i2], allaxes)

							if np.any(mask):
								meanU = 0 if self.vectors_subtract_mean is False else np.nanmean(U[mask])
								meanV = 0 if self.vectors_subtract_mean is False else np.nanmean(V[mask])
								U -= meanU
								V -= meanV

							if vector_positions[i3] is not None and self.vectors_color_code_3rd:
								W = multisum(vector_values[i3], allaxes)
								if np.any(mask):
									meanW = 0 if self.vectors_subtract_mean is False else np.nanmean(W[mask])
									W -= meanW
								axes.quiver(x[mask], y[mask], U[mask], V[mask], W[mask], cmap=self.colormap_vector)
							else:
								axes.quiver(x[mask], y[mask], U[mask], V[mask], color="black")
	

						if 0: # TODO: self.dataset.selected_row_index is not None:
							#self.axes.autoscale(False)
							x, y = self.getdatax()[self.dataset.selected_row_index],  self.getdatay()[self.dataset.selected_row_index]
							axes.scatter([x], [y], color='red') #, scalex=False, scaley=False)
					if self.aspect is None:
						axes.set_aspect('auto')
					else:
						axes.set_aspect(self.aspect)
						
					axes.set_xlim(self.ranges_show[i1][0], self.ranges_show[i1][1])
					axes.set_ylim(self.ranges_show[i2][0], self.ranges_show[i2][1])
					axes.set_xlabel(self.expressions[i1])
					axes.set_ylabel(self.expressions[i2])
			if 0:
					
				self.axes.imshow(amplitude.T, origin="lower", extent=ranges, alpha=1 if self.counts_mask is None else 0.4, cmap=cm_plusmin)
				if 1:
					if self.counts_mask is not None:
						if self.amplitude_expression is not None:
							#locals = {"counts":self.counts_mask}
							locals = {"counts":self.counts_weights_mask, "counts1": self.counts_mask}
							globals = np.__dict__
							amplitude_mask = eval(self.amplitude_expression, globals, locals)
						self.axes.imshow(amplitude_mask.T, origin="lower", extent=ranges, alpha=1, cmap=cm_plusmin)
					#self.axes.imshow((I), origin="lower", extent=ranges)
				self.axes.set_aspect('auto')
					#if self.dataset.selected_row_index is not None:
						#self.axes.autoscale(False)
				index = self.dataset.selected_row_index
				if index is not None and self.selected_point is None:
					logger.debug("point selected but after computation")
					# TODO: optimize
					def find_selected_point(info, blockx, blocky):
						if index >= info.i1 and index < info.i2: # selected point is in this block
							self.selected_point = blockx[index-info.i1], blocky[index-info.i1]
					self.dataset.evaluate(find_selected_point, *self.expressions, **self.getVariableDict())
					

				if self.selected_point:
					#x, y = self.getdatax()[self.dataset.selected_row_index],  self.getdatay()[self.dataset.selected_row_index]
					x, y = self.selected_point
					self.axes.scatter([x], [y], color='red') #, scalex=False, scaley=False)
				#if dataxsel is not None:
				#	self.axes.scatter(dataxsel, dataysel)
				self.axes.set_xlabel(self.expressions[0])
				self.axes.set_ylabel(self.expressions[0])
				self.axes.set_xlim(*self.ranges_show[0])
				self.axes.set_ylim(*self.ranges_show[1])
		self.canvas.draw()
		timelog("plot end")
		self.message("ploting %f" % (time.time() - t0), index=5)
		

class Rank1ScatterPlotDialog(ScatterPlotDialog):
	type_name = "sequence-density2d"
	def __init__(self, parent, dataset, xname=None, yname=None):
		self.nSlices = dataset.rank1s[list(dataset.rank1s.keys())[0]].shape[0]
		self.serieIndex = dataset.selected_serie_index if dataset.selected_serie_index is not None else 0
		self.record_frames = False
		super(Rank1ScatterPlotDialog, self).__init__(parent, dataset, xname, yname)

	def getTitleExpressionList(self):
		#return []
		return ["%s: {%s: 4f}" % (name, name) for name in self.dataset.axis_names]
		
	def addToolbar2(self, layout, contrast=True, gamma=True):
		super(Rank1ScatterPlotDialog, self).addToolbar2(layout, contrast, gamma)
		self.action_save_frames = QtGui.QAction(QtGui.QIcon(iconfile('film')), '&Export frames', self)
		self.menu_save.addAction(self.action_save_frames)
		self.action_save_frames.triggered.connect(self.onActionSaveFrames)
		
	def onActionSaveFrames(self, ignore=None):
		#directory = QtGui.QFileDialog.getExistingDirectory(self, "Choose where to save frames", "",  QtGui.QFileDialog.ShowDirsOnly | QtGui.QFileDialog.DontResolveSymlinks)
		directory = qt.getdir(self, "Choose where to save frames", "")
		self.frame_template = os.path.join(directory, "%s_{index:05}.png" % self.dataset.name)
		self.frame_template = qt.gettext(self, "template for frame filenames", "template:", self.frame_template)
		self.record_frames = True
		self.onPlayOnce()
		
	def plot(self):
		super(Rank1ScatterPlotDialog, self).plot()
		if self.record_frames:
			index = self.serieIndex
			path = self.frame_template.format(**locals())
			self.fig.savefig(path)
			if self.serieIndex == self.nSlices-1:
				self.record_frames = False
				

	def onSerieIndexSelect(self, serie_index):
		if serie_index != self.serieIndex: # avoid unneeded event
			self.serieIndex = serie_index
			self.seriesbox.setCurrentIndex(self.serieIndex)
		else:
			self.serieIndex = serie_index
		self.compute()
		#self.jobsM
		#self.plot()
	
		
	def getExpressionList(self):
		names = []
		for rank1name in self.dataset.rank1names:
			names.append(rank1name + "[index]")
		return names
	
	def getVariableDict(self):
		vars = {"index": self.serieIndex}
		for name in self.dataset.axis_names:
			vars[name] = self.dataset.axes[name][self.serieIndex]
		return vars
		

	def _getVariableDictMinMax(self):
		return {"index": slice(None, None, None)}

	def afterCanvas(self, layout):
		super(Rank1ScatterPlotDialog, self).afterCanvas(layout)
		#return

		self.seriesbox = QtGui.QComboBox(self)
		self.seriesbox.addItems([str(k) for k in range(self.nSlices)])
		self.seriesbox.setCurrentIndex(self.serieIndex)
		self.seriesbox.currentIndexChanged.connect(self.onSerieIndex)
		
		self.grid_layout.addWidget(self.seriesbox, 10, 1)
		#self.form_layout = QtGui.QFormLayout(self)
		#self.form_layout.addRow("index", self.seriesbox)
		#self.buttonLoop = QtGui.QToolButton(self)
		#self.buttonLoop.setText("one loop")
		#self.buttonLoop.clicked.connect(self.onPlayOnce)
		#self.form_layout.addRow("movie", self.buttonLoop)
		#layout.addLayout(self.form_layout, 0)
		
	def onPlayOnce(self):
		#self.timer = QtCore.QTimer(self)
		#self.timer.timeout.connect(self.onNextFrame)
		self.delay = 10
		if not self.axis_lock:
			for i in range(self.dimensions):
				self.ranges[i] = None
			for i in range(self.dimensions):
				self.ranges_show[i] = None
		self.dataset.selectSerieIndex(0)
		self.dataset.executor.execute()
		QtCore.QTimer.singleShot(self.delay if not self.record_frames else 0, self.onNextFrame);
		
	def onNextFrame(self, *args):
		step = 1
		next = self.serieIndex +step
		if next >= self.nSlices:
			next = self.nSlices-1
		if not self.axis_lock:
			for i in range(self.dimensions):
				self.ranges[i] = None
			for i in range(self.dimensions):
				self.ranges_show[i] = None
		self.dataset.selectSerieIndex(next)
		self.dataset.executor.execute()
		if self.serieIndex < self.nSlices-1 : # not last frame
			QtCore.QTimer.singleShot(self.delay, self.onNextFrame);
			
			
	def onSerieIndex(self, index):
		if index != self.dataset.selected_serie_index: # avoid unneeded event
			if not self.axis_lock:
				for i in range(self.dimensions):
					self.ranges[i] = None
				for i in range(self.dimensions):
					self.ranges_show[i] = None
			self.dataset.selectSerieIndex(index)
			#self.compute()
			self.dataset.executor.execute()

class Mover(object):
	def __init__(self, plot, axes):
		self.plot = plot
		self.axes = axes
		self.canvas = self.axes.figure.canvas
		self.axes = None

		self.canvas.mpl_connect('scroll_event', self.mpl_scroll)
		self.last_x, self.last_y = None, None
		self.handles = []
		self.handles.append(self.canvas.mpl_connect('motion_notify_event', self.mouse_move))
		self.handles.append(self.canvas.mpl_connect('button_press_event', self.mouse_down))
		self.handles.append(self.canvas.mpl_connect('button_release_event', self.mouse_up))
		self.begin_x, self.begin_y = None, None
		self.moved = False
		self.zoom_queue = []
		self.zoom_counter = 0

	def disconnect_events(self):
		for handle in self.handles:
			self.canvas.mpl_disconnect(handle)

	def mouse_up(self, event):
		self.last_x, self.last_y = None, None
		if self.moved:
			# self.plot.ranges = list(self.plot.ranges_show)
			#for layer in self.plot.layers:
			#	layer.ranges = list(self.plot)
			#self.plot.compute()
			#self.plot.jobsManager.execute()
			self.plot.update_all_layers()
			self.moved = False

	def mouse_down(self, event):
		self.moved = False
		if event.dblclick:
			factor = 0.333
			if event.button != 1:
				factor = 1/factor
			self.plot.zoom(factor, axes=event.inaxes, x=event.xdata, y=event.ydata)
		else:
			self.begin_x, self.begin_y = event.xdata, event.ydata
			self.last_x, self.last_y = event.xdata, event.ydata
			self.current_axes = event.inaxes
			self.plot.ranges_begin = list(self.plot.ranges_show)

	def mouse_move(self, event):
		#return
		if self.last_x is not None and event.xdata is not None and self.current_axes is not None:
			#axes = event.inaxes
			transform = self.current_axes.transData.inverted().transform
			x_data, y_data = event.xdata, event.ydata
			self.moved = True
			dx = self.last_x - x_data
			dy = self.last_y - y_data
			xmin, xmax = self.plot.ranges_show[self.current_axes.xaxis_index][0] + dx, self.plot.ranges_show[self.current_axes.xaxis_index][1] + dx
			if self.plot.dimensions == 1:
				ymin, ymax = self.plot.range_level_show[0] + dy, self.plot.range_level_show[1] + dy
			else:
				ymin, ymax = self.plot.ranges_show[self.current_axes.yaxis_index][0] + dy, self.plot.ranges_show[self.current_axes.yaxis_index][1] + dy
			#self.plot.ranges_show = [[xmin, xmax], [ymin, ymax]]
			self.plot.ranges_show[self.current_axes.xaxis_index] = [xmin, xmax]
			if self.plot.dimensions == 1:
				self.plot.range_level_show = [ymin, ymax]
			else:
				self.plot.ranges_show[self.current_axes.yaxis_index] = [ymin, ymax]
			# TODO: maybe the dimension should be stored in the axes, not in the plotdialog
			for axes in self.plot.getAxesList():
				if self.plot.dimensions == 1:
					# ftm we assume we only have 1 histogram, meabning axes == self.current_axes
					axes.set_xlim(*self.plot.ranges_show[self.current_axes.xaxis_index])
					axes.set_ylim(*self.plot.range_level_show)
				else:
					if axes.xaxis_index == self.current_axes.xaxis_index:
						axes.set_xlim(*self.plot.ranges_show[self.current_axes.xaxis_index])
					if axes.yaxis_index == self.current_axes.xaxis_index:
						axes.set_ylim(*self.plot.ranges_show[self.current_axes.xaxis_index])
					if axes.xaxis_index == self.current_axes.yaxis_index:
						axes.set_xlim(*self.plot.ranges_show[self.current_axes.yaxis_index])
					if axes.yaxis_index == self.current_axes.yaxis_index:
						axes.set_ylim(*self.plot.ranges_show[self.current_axes.yaxis_index])

			# transform again after we changed the axes limits
			transform = self.current_axes.transData.inverted().transform
			x_data, y_data = transform([event.x*1., event.y*1])
			self.last_x, self.last_y = x_data, y_data

			self.canvas.draw_idle()



	def mpl_scroll(self, event):
		factor = 10**(-event.step/8)
		self.zoom_counter += 1
		if event.inaxes is None:
			return

		if factor < 1:
			self.plot.zoom(factor, event.inaxes, event.xdata, event.ydata)
			#self.zoom_queue.append((factor, event.xdata, event.ydata))
		else:
			self.plot.zoom(factor, event.inaxes, event.xdata, event.ydata) #, event.xdata, event.ydata)
			#self.zoom_queue.append((factor, None, None))
		return
		def idle_zoom(ignore=None, zoom_counter=None, axes=None):

			if zoom_counter < self.zoom_counter:
				pass # ignore, a later event will come
			else:
				#zoom_queue = list((self.zoom_queue) # make copy to ensure it doesn't get modified in
				for i, (factor, x, y) in enumerate(self.zoom_queue):
					# only redraw at last call
					is_last = i==len(self.zoom_queue)-1
					self.plot.zoom(factor, axes=axes, x=x, y=y, recalculate=False, history=False, redraw=is_last)
				self.zoom_queue = []
		if event.axes:
			QtCore.QTimer.singleShot(1, functools.partial(idle_zoom, zoom_counter=self.zoom_counter, axes=event.inaxes))

class Queue(object):
	logger = vaex.logging.getLogger("vaex.ui.queue")
	def __init__(self, name, default_delay, default_callable, pre=lambda: None):
		self.name = name
		self.default_delay = default_delay
		self.counter = 0
		self.counter_processed = 0
		self.default_callable = default_callable
		self.pre = pre

	def is_empty(self):
		return (self.counter_processed == self.counter)

	def in_queue(self, minimum=1):
		return (self.counter_processed <= (self.counter - minimum))


	def _wait(self, sleep=10):
		if self.counter > 0:
			qt_app = QtCore.QCoreApplication.instance()
			logger.debug("*** waiting for queue %r" % self.name)
			while self.counter_processed != self.counter:
				qt_app.processEvents()
				QtTest.QTest.qSleep(sleep)
			logger.debug("*** done with queue %r" %self.name)

	def cancel(self):
		def nop():
			self.logger.debug("nop called in queue %r" % self.name)
		self.logger.debug("cancelling last call in queue %r by inserting a nop" % self.name)
		self(nop)

	def __call__(self, callable=None, delay=None, *args, **kwargs):
		self.pre()
		#print self.name
		#import traceback
		#	traceback.print_stack()
		if delay is None:
			delay = self.default_delay
		callable = callable or self.default_callable
		def call(_=None, counter=None, callable=None):
			try:
				if counter < self.counter:
					pass # ignore, more events coming
					self.logger.debug("ignoring this event in queue %r, since a new one is scheduled" % self.name)
				else:
					self.logger.debug("calling callback in queue %r" % self.name)
					self.counter_processed = self.counter -1 # to make the queue 'empty'
					callable()
			finally:
				self.counter_processed = counter
		callable = functools.partial(callable, *args, **kwargs)
		self.counter += 1
		self.logger.debug("add in queue %r %r" % (self.name, delay))
		#import traceback
		#traceback.print_stack()
		if delay == 0:
			call(counter=self.counter, callable=callable)
		else:
			QtCore.QTimer.singleShot(delay, functools.partial(call, counter=self.counter, callable=callable))

from vaex.ui.layers import LayerTable
import vaex.ui.layers
try:
	from vaex.ui.main import VaexApp
except: pass
