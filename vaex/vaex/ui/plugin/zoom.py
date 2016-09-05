import functools

import matplotlib.widgets

import vaex.ui.plugin
from vaex.ui import undo
from vaex.ui.qt import *
from vaex.ui.icons import iconfile
import logging
import vaex.ui.undo as undo


logger = logging.getLogger("plugin.zoom")

@vaex.ui.plugin.pluginclass
class ZoomPlugin(vaex.ui.plugin.PluginPlot):
	name = "zoom"
	def __init__(self, dialog):
		super(ZoomPlugin, self).__init__(dialog)
		dialog.plug_toolbar(self.plug_toolbar, 1.2)
		
		
	def plug_toolbar(self):
		logger.info("adding zoom plugin")
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
		
		
	def setMode(self, action):
		useblit = True
		axes_list = self.dialog.getAxesList()
		if action == self.action_zoom_x:
			print("zoom x")
			self.lastActionZoom = self.action_zoom_x
			self.dialog.currentModes = [matplotlib.widgets.SpanSelector(axes, functools.partial(self.onZoomX, axes=axes), 'horizontal', useblit=useblit) for axes in axes_list] #, rectprops={"color":"blue"})
			if useblit:
				self.dialog.canvas.draw() # buggy otherwise
		if action == self.action_zoom_y:
			self.lastActionZoom = self.action_zoom_y
			self.dialog.currentModes = [matplotlib.widgets.SpanSelector(axes, functools.partial(self.onZoomY, axes=axes), 'vertical', useblit=useblit)  for axes in axes_list] #, rectprops={"color":"blue"})
			if useblit:
				self.dialog.canvas.draw() # buggy otherwise
		if action == self.action_zoom_rect:
			print("zoom rect")
			self.lastActionZoom = self.action_zoom_rect
			self.dialog.currentModes = [matplotlib.widgets.RectangleSelector(axes, functools.partial(self.onZoomRect, axes=axes), useblit=useblit) for axes in axes_list] #, rectprops={"color":"blue"})
			if useblit:
				self.dialog.canvas.draw() # buggy otherwise

		

	def onZoomIn(self, *args):
		axes = self.getAxesList()[0] # TODO: handle propery multiple axes
		self.dialog.zoom(0.5, axes)
		self.dialog.queue_history_change("zoom in")
		
	def onZoomOut(self):
		axes = self.dialog.getAxesList()[0] # TODO: handle propery multiple axes
		self.dialog.zoom(2., axes)
		self.dialog.queue_history_change("zoom out")
		

	def onActionZoom(self):
		print("onactionzoom")
		self.lastActionZoom.setChecked(True)
		self.dialog.setMode(self.lastActionZoom)
		self.syncToolbar()
		
	def onZoomFit(self, *args):
		#for i in range(self.dimensions):
		#	self.dialog.ranges[i] = None
		#	self.dialog.state.ranges_viewport[i] = None
		#	self.range_level = None
		if 0:
			for axisIndex in range(self.dimensions):
				linkButton = self.linkButtons[axisIndex]
				link = linkButton.link
				if link:
					logger.debug("sending link messages")
					link.sendRanges(self.dialog.ranges[axisIndex], linkButton)
					link.sendRangesShow(self.dialog.state.ranges_viewport[axisIndex], linkButton)
				
		
		action = undo.ActionZoom(self.dialog.undoManager, "zoom to fit", self.dialog.set_ranges,
		                         list(range(self.dialog.dimensions)),
		                         self.dialog.state.ranges_viewport, self.dialog.state.range_level_show,
						   		 list(range(self.dialog.dimensions)),
							     ranges_viewport=[None] * self.dialog.dimensions, range_level_show=None)
		#for layer in dialog.layers:
		#	layer.range_level = None # reset these... is this the right place?
		action.do()
		self.dialog.checkUndoRedo()
		self.dialog.queue_history_change("zoom to fit")
		
		if 0:
			linked_buttons = [button for button in self.linkButtons if button.link is not None]
			links = [button.link for button in linked_buttons]
			if len(linked_buttons) > 0:
				logger.debug("sending compute message")
				vaex.dataset.Link.sendCompute(links, linked_buttons)
			#linked_buttons[0].sendCompute(blacklist)
		#if linkButtonLast: # only send once
		#	link = linkButtonLast.link
		#	logger.debug("sending compute message")
		#	link.sendCompute(linkButton)

		#self.compute()
		#self.dataset.executor.execute()
		
	def onZoomUse(self, *args):
		# TODO: when this will be an option again, implement this as action
		# TODO: will we ever use this again? auto updates are much better
		for i in range(self.dimensions):
			self.dialog.ranges[i] = self.dialog.state.ranges_viewport[i]
		self.range_level = None
		for axisIndex in range(self.dimensions):
			linkButton = self.linkButtons[axisIndex]
			link = linkButton.link
			if link:
				logger.debug("sending link messages")
				link.sendRanges(self.dialog.ranges[axisIndex], linkButton)
				#link.sendRangesShow(self.dialog.state.ranges_viewport[axisIndex], linkButton)
		linked_buttons = [button for button in self.linkButtons if button.link is not None]
		links = [button.link for button in linked_buttons]
		if len(linked_buttons) > 0:
			logger.debug("sending compute message")
			vaex.dataset.Link.sendCompute(links, linked_buttons)
		self.compute()
		self.dataset.executor.execute()
		
	def onZoomX(self, xmin, xmax, axes):
		
		axisIndex = axes.xaxis_index
		#self.dialog.state.ranges_viewport[axisIndex] = xmin, xmax
		# move the link code to the set ranges
		if 0:
			linkButton = self.linkButtons[axisIndex]
			link = linkButton.link
			if link:
				logger.debug("sending link messages")
				link.sendRangesShow(self.dialog.state.ranges_viewport[axisIndex], linkButton)
				link.sendPlot(linkButton)
		action = undo.ActionZoom(self.dialog.undoManager, "zoom x [%f,%f]" % (xmin, xmax),
						   self.dialog.set_ranges, list(range(self.dialog.dimensions)),
						   self.dialog.state.ranges_viewport,self.dialog.state.range_level_show, [axisIndex], ranges_viewport=[[xmin, xmax]])
		action.do()
		self.dialog.checkUndoRedo()
		self.dialog.queue_history_change("zoom x")

	def onZoomY(self, ymin, ymax, axes):
		if len(self.dialog.state.ranges_viewport) == 1: # if 1d, y refers to range_level
			#self.range_level = ymin, ymax
			action = undo.ActionZoom(self.dialog.undoManager, "change level [%f,%f]" % (ymin, ymax), self.dialog.set_ranges, list(range(self.dialog.dimensions)),
							self.dialog.state.ranges_viewport, self.dialog.state.range_level_show, [], range_level_show=[ymin, ymax])
		else:
			#self.dialog.state.ranges_viewport[axes.yaxis_index] = ymin, ymax
			action = undo.ActionZoom(self.dialog.undoManager, "zoom y [%f,%f]" % (ymin, ymax), self.dialog.set_ranges, list(range(self.dialog.dimensions)),
							self.dialog.state.ranges_viewport, self.dialog.state.range_level_show, [axes.yaxis_index], ranges_viewport=[[ymin, ymax]])
			
		action.do()
		self.dialog.checkUndoRedo()
		self.dialog.queue_history_change("zoom y")

	def onZoomRect(self, eclick, erelease, axes):
		x1, y1 = (eclick.xdata, eclick.ydata)
		x2, y2 = (erelease.xdata, erelease.ydata)
		x = [x1, x2]
		y = [y1, y2]

		range_level = None
		ranges_show = []
		ranges = []
		axis_indices = []
		xmin_show, xmax_show = min(x), max(x)
		ymin_show, ymax_show = min(y), max(y)
		if self.dialog.state.ranges_viewport[0][0] > self.dialog.state.ranges_viewport[0][1]:
			xmin_show, xmax_show = xmax_show, xmin_show
		if len(self.dialog.state.ranges_viewport) == 1 and self.dialog.state.range_level_show[0] > self.dialog.state.range_level_show[1]:
			ymin_show, ymax_show = ymax_show, ymin_show
		elif self.dialog.state.ranges_viewport[1][0] > self.dialog.state.ranges_viewport[1][1]:
			ymin_show, ymax_show = ymax_show, ymin_show

		#self.dialog.state.ranges_viewport[axes.xaxis_index] = xmin_show, xmax_show
		axis_indices.append(axes.xaxis_index)
		ranges_show.append([xmin_show, xmax_show])
		if len(self.dialog.state.ranges_viewport) == 1: # if 1d, y refers to range_level
			#self.range_level = ymin_show, ymax_show
			range_level =  ymin_show, ymax_show
			logger.debug("range refers to level: %r" % (self.dialog.state.range_level_show,))
		else:
			#self.dialog.state.ranges_viewport[axes.yaxis_index] = ymin_show, ymax_show
			axis_indices.append(axes.yaxis_index)
			ranges_show.append([ymin_show, ymax_show])
			
			
		def delayed_zoom():
			action = undo.ActionZoom(self.dialog.undoManager, "zoom to rect", self.dialog.set_ranges, list(range(self.dialog.dimensions)),
							self.dialog.state.ranges_viewport,
							self.dialog.state.range_level_show, axis_indices, ranges_viewport=ranges_show, range_level_show=range_level)
			action.do()
			self.dialog.checkUndoRedo()
		#self.dialog.queue_update(delayed_zoom, delay=300)
		delayed_zoom()
		self.dialog.queue_history_change("zoom to rectangle")

		if 1:
			#self.dialog.state.ranges_viewport = list(ranges_show)
			self.dialog.state.ranges_viewport[axes.xaxis_index] = list(ranges_show[0])
			if self.dialog.dimensions == 2:
				self.dialog.state.ranges_viewport[axes.yaxis_index] = list(ranges_show[1])
				self.dialog.check_aspect(1)
				axes.set_xlim(self.dialog.state.ranges_viewport[0])
				axes.set_ylim(self.dialog.state.ranges_viewport[1])
			if self.dialog.dimensions == 1:
				self.dialog.state.range_level_show = range_level
				axes.set_xlim(self.dialog.state.ranges_viewport[0])
				axes.set_ylim(self.dialog.state.range_level_show)
			self.dialog.queue_redraw()
			
		if 0:

			for axisIndex in range(self.dimensions):
				linkButton = self.linkButtons[axisIndex]
				link = linkButton.link
				if link:
					logger.debug("sending link messages")
					link.sendRangesShow(self.dialog.state.ranges_viewport[axisIndex], linkButton)
					link.sendPlot(linkButton)
			
			#self.axes.set_xlim(self.xmin_show, self.xmax_show)
			#self.axes.set_ylim(self.ymin_show, self.ymax_show)
			#self.canvas.draw()
			action = undo.ActionZoom(self.undoManager, "zoom to rect", self.set_ranges, list(range(self.dimensions)), self.dialog.ranges, self.dialog.state.ranges_viewport,  self.range_level, axis_indices, ranges_viewport=ranges_show, range_level=range_level)
			action.do()
			self.checkUndoRedo()

			if 0:
				if self.autoRecalculate():
					for i in range(self.dimensions):
						self.dialog.ranges[i] = self.dialog.state.ranges_viewport[i]
						self.range_level = None
					self.compute()
					self.dataset.executor.execute()
				else:
					self.plot()
		

	def syncToolbar(self):
		for action in [self.action_zoom]:
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
		#logger.debug("last select action: %r" % self.lastActionSelect.text())
		logger.debug("last zoom action: %r" % self.lastActionZoom.text())
		#self.action_select.setText(self.lastActionSelect.text())
		#self.action_select.setIcon(self.lastActionSelect.icon())
		self.action_zoom.setText(self.lastActionZoom.text())
		self.action_zoom.setIcon(self.lastActionZoom.icon())

