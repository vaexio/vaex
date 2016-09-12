import functools

import matplotlib.widgets

import vaex.ui.plugin
from vaex.ui import undo
from vaex.ui.qt import *
from vaex.ui.icons import iconfile
import logging
import vaex.ui.undo as undo
import vaex.ui.qt as dialogs
import re
logger = logging.getLogger("vaex.plugin.tasks")

def loglog(plot_window):
	for layer in plot_window.layers:
		for i, expression in enumerate(layer.state.expressions):
			layer.set_expression("log10(%s)" % expression, i)
	plot_window.queue_history_change("task: log/log")

def removelog(plot_window):
	def remove_log(expression):
		return re.sub("^\s*(log|log2|log10)\((.*?)\)\s*$", "\\2", expression)
	for layer in plot_window.layers:
		for i, expression in enumerate(layer.state.expressions):
			layer.set_expression(remove_log(expression), i)
	plot_window.queue_history_change("task: remove log/log")

def loglog_and_sigma3(plot_window):
	removelog(plot_window)
	loglog(plot_window)
	plot_window.queue_history_change(None)
	sigma3(plot_window)
	plot_window.queue_history_change("task: log/log and 3 sigma region")

def sigma3(plot_window):
	if plot_window.layers:
		layer = plot_window.layers[0]


		if layer.dataset.is_local():
			executor = vaex.execution.Executor()
		else:
			executor = vaex.remote.ServerExecutor()
		subspace = layer.dataset.subspace(*layer.state.expressions, executor=executor, async=True)
		means = subspace.mean()
		with dialogs.ProgressExecution(plot_window, "Calculating mean", executor=executor) as progress:
			progress.add_task(means)
			progress.execute()
		logger.debug("get means")
		means = means.get()
		logger.debug("got means")

		vars = subspace.var(means=means)
		with dialogs.ProgressExecution(plot_window, "Calculating variance", executor=executor) as progress:
			progress.add_task(vars)
			progress.execute()
		#limits  = limits.get()
		vars = vars.get()
		stds = vars**0.5
		sigmas = 3
		limits = list(zip(means-sigmas*stds, means+sigmas*stds))
		#plot_window.ranges_show = limits
		plot_window.set_ranges(range(len(limits)), limits, add_to_history=True, reason="3 sigma region")
		#plot_window.update_all_layers()
		#for layer in plot_window.layers:
		#	layer.flag_needs_update()
		logger.debug("means=%r", means)
		logger.debug("vars=%r", vars)
		logger.debug("limits=%r", limits)
		plot_window.queue_history_change("task: 3 sigma region")
		#plot_window.queue_update()

def subtract_mean(plot_window):
	if plot_window.layers:
		layer = plot_window.layers[0]

		executor = vaex.execution.Executor()
		subspace = layer.dataset.subspace(*layer.state.expressions, executor=executor, async=True)
		means = subspace.mean()
		with dialogs.ProgressExecution(plot_window, "Calculating mean", executor=executor):
			executor.execute()
		means = means.get()
		new_expressions = ["(%s) - %s" % (expression, mean) for expression, mean in zip(layer.state.expressions, means)]
		for i in range(len(new_expressions)):
			vmin, vmax = layer.plot_window.state.ranges_viewport[i]
			vmin -= means[i]
			vmax -= means[i]
			layer.plot_window.set_range(vmin, vmax, i)
		for i in range(len(new_expressions)):
			layer.set_expression(new_expressions[i], i)
		plot_window.update_all_layers()
		plot_window.queue_history_change("task: remove mean")


@vaex.ui.plugin.pluginclass
class TasksPlugin(vaex.ui.plugin.PluginPlot):
	name = "tasks"
	def __init__(self, dialog):
		super(TasksPlugin, self).__init__(dialog)
		dialog.plug_toolbar(self.plug_toolbar, 1.6)


	def plug_toolbar(self):
		logger.info("adding %s plugin" % self.name)

		self.menu = QtGui.QMenu("&Tasks")
		self.dialog.menu_bar.addMenu(self.menu)

		self.tasks_button = QtGui.QToolButton()
		self.tasks_button.setIcon(QtGui.QIcon(iconfile('gear')))
		self.tasks_button.setText("Tasks")
		self.tasks_button.setPopupMode(QtGui.QToolButton.InstantPopup)
		self.tasks_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
		self.tasks_button.setMenu(self.menu)

		#self.action_tasks = QtGui.QAction(QtGui.QIcon(iconfile('gear')), 'Tasks', self.dialog)
		self.dialog.toolbar.addWidget(self.tasks_button)


		self.action_tasks_loglog = QtGui.QAction(QtGui.QIcon(iconfile('gear')), 'Log-Log', self.dialog)
		self.menu.addAction(self.action_tasks_loglog)
		self.action_tasks_loglog.triggered.connect(lambda *arg: loglog(self.dialog))

		self.action_tasks_loglog = QtGui.QAction(QtGui.QIcon(iconfile('gear')), '3 sigma region', self.dialog)
		self.menu.addAction(self.action_tasks_loglog)
		self.action_tasks_loglog.triggered.connect(lambda *arg: sigma3(self.dialog))

		self.action_tasks_loglog_3sigma = QtGui.QAction(QtGui.QIcon(iconfile('gear')), 'Log-Log and 3 sigma region', self.dialog)
		self.menu.addAction(self.action_tasks_loglog_3sigma)
		self.action_tasks_loglog_3sigma.triggered.connect(lambda *arg: loglog_and_sigma3(self.dialog))

		self.action_tasks_removelog = QtGui.QAction(QtGui.QIcon(iconfile('gear')), 'Remove log', self.dialog)
		self.menu.addAction(self.action_tasks_removelog)
		self.action_tasks_removelog.triggered.connect(lambda *arg: removelog(self.dialog))

		self.action_tasks_subtract_mean = QtGui.QAction(QtGui.QIcon(iconfile('gear')), 'Subtract mean', self.dialog)
		self.menu.addAction(self.action_tasks_subtract_mean)
		self.action_tasks_subtract_mean.triggered.connect(lambda *arg: subtract_mean(self.dialog))


	def __(self):

		self.action_store  = QtGui.QAction(QtGui.QIcon(iconfile('gear')), 'Store', self.dialog)
		self.action_store.setShortcut("Ctrl+B")
		self.action_store_toolbar  = QtGui.QAction(QtGui.QIcon(iconfile('star')), 'Store', self.dialog)


		self.dialog.toolbar.addAction(self.action_store_toolbar)
		self.action_store.triggered.connect(self.on_store)
		self.action_store_toolbar.triggered.connect(self.on_store)
		self.changed_handle = storage_plots.changed.connect(self.load_options_menu)
		self.load_options_menu()

		self.dialog.menu_mode.addSeparator()
		self.action_zoom_rect = QtGui.QAction(QtGui.QIcon(iconfile('zoom')), '&Zoom to rect', self.dialog)
		self.action_zoom_rect.setShortcut("Ctrl+Alt+Z")
		self.dialog.menu_mode.addAction(self.action_zoom_rect)

		self.action_zoom_x = QtGui.QAction(QtGui.QIcon(iconfile('zoom_x')), '&Zoom x', self.dialog)
		self.action_zoom_y = QtGui.QAction(QtGui.QIcon(iconfile('zoom_y')), '&Zoom y', self.dialog)
		self.action_zoom = QtGui.QAction(QtGui.QIcon(iconfile('zoom')), '&Zoom(you should not read this)', self.dialog)

		self.action_zoom_x.setShortcut("Ctrl+Alt+X")
		self.action_zoom_y.setShortcut("Ctrl+Alt+Y")
		self.dialog.menu_mode.addAction(self.action_zoom_x)
		self.dialog.menu_mode.addAction(self.action_zoom_y)


		self.dialog.menu_mode.addSeparator()
		self.action_zoom_out = QtGui.QAction(QtGui.QIcon(iconfile('zoom_out')), '&Zoom out', self.dialog)
		self.action_zoom_in = QtGui.QAction(QtGui.QIcon(iconfile('zoom_in')), '&Zoom in', self.dialog)
		self.action_zoom_fit = QtGui.QAction(QtGui.QIcon(iconfile('arrow_out')), '&Reset view', self.dialog)
		#self.action_zoom_use = QtGui.QAction(QtGui.QIcon(iconfile('chart_bar')), '&Use zoom area', self.dialog)
		self.action_zoom_out.setShortcut("Ctrl+Alt+-")
		self.action_zoom_in.setShortcut("Ctrl+Alt++")
		self.action_zoom_fit.setShortcut("Ctrl+Alt+0")
		self.dialog.menu_mode.addAction(self.action_zoom_out)
		self.dialog.menu_mode.addAction(self.action_zoom_in)
		self.dialog.menu_mode.addAction(self.action_zoom_fit)



		self.dialog.action_group_main.addAction(self.action_zoom_rect)
		self.dialog.action_group_main.addAction(self.action_zoom_x)
		self.dialog.action_group_main.addAction(self.action_zoom_y)

		#self.dialog.toolbar.addAction(self.action_zoom_out)
		#self.dialog.add_shortcut(self.action_zoom_in,"+")
		#self.dialog.add_shortcut(self.action_zoom_out,"-")

		#self.dialog.add_shortcut(self.action_zoom_rect,"Z")
		#self.dialog.add_shortcut(self.action_zoom_x,"Alt+X")
		#self.dialog.add_shortcut(self.action_zoom_y,"Alt+Y")
		#self.dialog.add_shortcut(self.action_zoom_fit, "0")

		self.dialog.toolbar.addAction(self.action_zoom)
		self.zoom_menu = QtGui.QMenu()
		self.action_zoom.setMenu(self.zoom_menu)
		self.zoom_menu.addAction(self.action_zoom_rect)
		self.zoom_menu.addAction(self.action_zoom_x)
		self.zoom_menu.addAction(self.action_zoom_y)
		if self.dialog.dimensions == 1:
			self.lastActionZoom = self.action_zoom_x # this makes more sense for histograms as default
		else:
			self.lastActionZoom = self.action_zoom_rect

		self.dialog.toolbar.addSeparator()
		#self.dialog.toolbar.addAction(self.action_zoom_out)
		self.dialog.toolbar.addAction(self.action_zoom_fit)

		self.action_zoom.triggered.connect(self.onActionZoom)
		self.action_zoom_out.triggered.connect(self.onZoomOut)
		self.action_zoom_in.triggered.connect(self.onZoomIn)
		self.action_zoom_fit.triggered.connect(self.onZoomFit)
		#self.action_zoom_use.triggered.connect(self.onZoomUse)


		self.action_zoom.setCheckable(True)
		self.action_zoom_rect.setCheckable(True)
		self.action_zoom_x.setCheckable(True)
		self.action_zoom_y.setCheckable(True)

