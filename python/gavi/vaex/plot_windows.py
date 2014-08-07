# -*- coding: utf-8 -*-
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbarQt
from matplotlib.figure import Figure
import matplotlib
from matplotlib.widgets import Lasso, LassoSelector
import matplotlib.widgets 
import os
import gavi
import numpy as np
import functools
import time

import gavi.logging

from numba import jit

logger = gavi.logging.getLogger("gavi.vaex")

try:
	from PyQt4 import QtGui, QtCore
	import sip
	sip.setapi('QVariant', 1)
except ImportError, e1:
	try:
		from PySide import QtGui, QtCore
	except ImportError, e2:
		print >>sys.stderr, "could not import PyQt4 or PySide, please install"
		print >>sys.stderr, "errors: ", repr(e1), repr(e2)
		sys.exit(1)

def iconfile(name):
	path = os.path.join(os.path.dirname(gavi.__file__), "icons", name+".png")
	print "icon path:", path
	return path

import math
@jit(nopython=True)
def find_nearest_index_(datax, datay, x, y, wx, wy):
	N = len(datax)
	index = 0
	mindistance = math.sqrt((datax[0]-x)**2/wx**2 + (datay[0]-y)**2/wy**2)
	for i in range(1,N):
		distance = math.sqrt((datax[i]-x)**2/wx**2 + (datay[i]-y)**2/wy**2)
		if distance < mindistance:
			mindistance = distance
			index = i
	return index
		

def find_nearest_index(datax, datay, x, y, wx, wy):
	index = find_nearest_index_(datax, datay, x, y, wx, wy)
	distance = math.sqrt((datax[index]-x)**2/wx**2 + (datay[index]-y)**2/wy**2)
	return index, distance


@jit(nopython=True)
def find_nearest_index1d_(datax, x):
	N = len(datax)
	index = 0
	mindistance = math.sqrt((datax[0]-x)**2)
	for i in range(1,N):
		distance = math.sqrt((datax[i]-x)**2)
		if distance < mindistance:
			mindistance = distance
			index = i
	return index

def find_nearest_index1d(datax, x):
	index = find_nearest_index1d_(datax, x)
	distance = math.sqrt((datax[index]-x)**2)
	return index, distance
		
		
		
class LinkButton(QtGui.QToolButton):
	def __init__(self, title, dataset, axisIndex, parent):
		super(LinkButton, self).__init__(parent)
		self.plot = parent
		self.dataset = dataset
		self.axisIndex = axisIndex
		self.setText(title)
		self.setAcceptDrops(True)
		self.disconnect_icon = QtGui.QIcon(iconfile('network-disconnect-2'))
		self.connect_icon = QtGui.QIcon(iconfile('network-connect-3'))
		self.setIcon(self.disconnect_icon)
		
		self.action_link_global = QtGui.QAction(self.connect_icon, '&Global link', self)
		self.action_unlink = QtGui.QAction(self.connect_icon, '&Unlink', self)
		self.menu = QtGui.QMenu()
		self.menu.addAction(self.action_link_global)
		self.menu.addAction(self.action_unlink)
		self.action_link_global.triggered.connect(self.onLinkGlobal)
		self.setMenu(self.menu)
		self.setDefaultAction(self.action_link_global)
		self.link = None
		
	def onLinkGlobal(self):
		self.link = self.dataset.link(self.plot.expressions[self.axisIndex], self)
		logger.debug("made global link: %r" % self.link)
		#self.parent.links[self.axisIndex] = self.linkHandle
		
	def onChangeRangeShow(self, range_):
		self.plot.ranges_show[self.axisIndex] = range_
		
	def onChangeRange(self, range_):
		self.plot.ranges[self.axisIndex] = range_
		
	def onCompute(self):
		self.plot.compute()
	
	def onPlot(self):
		self.plot.plot()
	
	def onLinkLimits(self, min, max):
		self.plot.expressions[self.axisIndex] = expression
	
	def onChangeExpression(self, expression):
		self.plot.expressions[self.axisIndex] = expression
		
		

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



class PlotDialog(QtGui.QDialog):
	def addAxes(self):
		self.axes = self.fig.add_subplot(111)
		self.axes.hold(True)
		
	def getAxesList(self):
		return [self.axes]
	
	def __init__(self, parent, jobsManager, dataset, expressions, axisnames, width=5, height=4, dpi=100):
		super(PlotDialog, self).__init__(parent)
		self.resize(700,700)
		self.jobsManager = jobsManager
		self.dataset = dataset
		self.axisnames = axisnames
		
		self.fig = Figure(figsize=(width, height), dpi=dpi)
		self.addAxes()
		
		self.canvas =  FigureCanvas(self.fig)
		self.canvas.setParent(self)

		self.boxlayout = QtGui.QVBoxLayout(self)
		
		self.expressions = expressions
		self.dimensions = len(self.expressions)
		self.ranges = [None for _ in range(self.dimensions)] # min/max for the data
		self.ranges_show = [None for _ in range(self.dimensions)] # min/max for the plots
		self.ranges_level = [None for _ in range(self.dimensions)] # min/max for the histogram/densities
		#self.xmin_show, self.xmax_show = None, None
		#self.ymin_show, self.ymax_show = None, None
		#self.xmin, self.xmax = None, None
		#self.ymin, self.ymax = None, None
		self.currentModes = None
		self.lastAction = None

		self.beforeCanvas(self.boxlayout)
		self.boxlayout.addWidget(self.canvas, 1)
		self.afterCanvas(self.boxlayout)
		self.setLayout(self.boxlayout)
		
		self.compute()
		self.jobsManager.after_execute.append(self.plot)
		#self.plot()
		FigureCanvas.setSizePolicy(self,
									QtGui.QSizePolicy.Expanding,
									QtGui.QSizePolicy.Expanding)
		FigureCanvas.updateGeometry(self)
		self.currentMode = None
		self.dataset.mask_listeners.append(self.onSelectMask)
		self.dataset.row_selection_listeners.append(self.onSelectRow)
		self.dataset.serie_index_selection_listeners.append(self.onSerieIndexSelect)
		
	
	def closeEvent(self, event):
		print "close event"
		# disconnect this event, otherwise we get an update/redraw for nothing
		# since closing a dialog causes this event to fire otherwise
		for axisbox, func in zip(self.axisboxes, self.onExpressionChangedPartials):
			axisbox.lineEdit().editingFinished.disconnect(func)
		self.dataset.mask_listeners.remove(self.onSelectMask)
		self.dataset.row_selection_listeners.remove(self.onSelectRow)
		self.dataset.serie_index_selection_listeners.remove(self.onSerieIndexSelect)
		super(PlotDialog, self).closeEvent(event)

	def onSerieIndexSelect(self, serie_index):
		pass
	
	def getExpressionList(self):
		return self.dataset.column_names
	
	def afterCanvas(self, layout):
		self.form_layout = QtGui.QFormLayout(self)

		self.axisboxes = []
		self.onExpressionChangedPartials = []
		axisIndex = 0
		
		self.grid_layout = QtGui.QGridLayout(self)
		#row = 0
		self.linkButtons = []
		for axisname in self.axisnames:
			row = axisIndex
			axisbox = QtGui.QComboBox(self)
			axisbox.setEditable(True)
			#self.form_layout.addRow(axisname + '-axis:', axisbox)
			self.grid_layout.addWidget(QtGui.QLabel(axisname + '-axis:', self), row, 0)
			self.grid_layout.addWidget(axisbox, row, 1, QtCore.Qt.AlignLeft)
			functionButton = QtGui.QToolButton(self)
			linkButton = LinkButton("link", self.dataset, axisIndex, self)
			self.linkButtons.append(linkButton)
			
			menu = QtGui.QMenu()
			functionButton.setMenu(menu)
			functionButton.setPopupMode(QtGui.QToolButton.InstantPopup)
			link_action = QtGui.QAction(QtGui.QIcon(iconfile('network-connect-3')), '&Link axis', self)
			unlink_action = QtGui.QAction(QtGui.QIcon(iconfile('network-disconnect-2')), '&Unlink axis', self)
			menu.addAction(link_action)
			menu.addAction(unlink_action)
			self.grid_layout.addWidget(functionButton, row, 2)
			self.grid_layout.addWidget(linkButton, row, 3)
			axisbox.addItems(self.getExpressionList())
			#axisbox.setCurrentIndex(self.expressions[axisIndex])
			#axisbox.currentIndexChanged.connect(functools.partial(self.onAxis, axisIndex=axisIndex))
			axisbox.lineEdit().setText(self.expressions[axisIndex])
			# keep a list to be able to disconnect
			self.onExpressionChangedPartials.append(functools.partial(self.onExpressionChanged, axisIndex=axisIndex))
			axisbox.lineEdit().editingFinished.connect(self.onExpressionChangedPartials[axisIndex])
			axisIndex += 1
			self.axisboxes.append(axisbox)
		
		layout.addLayout(self.grid_layout, 0)
		layout.addLayout(self.form_layout, 0)

		self.amplitude_box = QtGui.QComboBox(self)
		self.amplitude_box.setEditable(True)
		self.amplitude_box.addItems(["log(counts)", "counts", "counts**2", "sqrt(counts)"])
		self.amplitude_box.setMinimumContentsLength(40)
		self.form_layout.addRow("amplitude=", self.amplitude_box)
		self.amplitude_box.lineEdit().editingFinished.connect(self.onAmplitudeExpr)
		self.amplitude_expression = str(self.amplitude_box.lineEdit().text())
		
		self.weight_box = QtGui.QComboBox(self)
		self.weight_box.setEditable(True)
		self.weight_box.addItems(["", "vx"])
		self.weight_box.setMinimumContentsLength(40)
		self.form_layout.addRow("weight=", self.weight_box)
		self.weight_box.lineEdit().editingFinished.connect(self.onWeightExpr)
		self.weight_expression = str(self.weight_box.lineEdit().text())
		
		
		
	def onWeightExpr(self):
		self.weight_expression = str(self.weight_box.lineEdit().text())
		print self.weight_expression
		if self.weight_expression.strip() == "":
			self.weight_expression = None
		self.ranges_level[0] = None
		self.compute()
		self.jobsManager.execute()
		self.plot()
	
	def onAmplitudeExpr(self):
		self.amplitude_expression = str(self.amplitude_box.lineEdit().text())
		print self.amplitude_expression
		self.ranges_level[0] = None
		self.plot()

	def beforeCanvas(self, layout):
		self.addToolbar(layout) #, yselect=True, lasso=False)
		
	def onExpressionChanged(self, axisIndex):
		text = str(self.axisboxes[axisIndex].lineEdit().text())
		print "expr", repr(text)
		if text == self.expressions[axisIndex]:
			logger.debug("same expression, will not update")
			return
		self.expressions[axisIndex] = text
		# TODO: range reset as option?
		self.ranges[axisIndex] = None
		self.ranges_show[axisIndex] = None
		linkButton = self.linkButtons[axisIndex]
		link = linkButton.link
		if link:
			logger.debug("sending link messages")
			link.sendRanges(self.ranges[axisIndex], linkButton)
			link.sendRangesShow(self.ranges_show[axisIndex], linkButton)
			link.sendExpression(self.expressions[axisIndex], linkButton)
			link.sendCompute(linkButton)
		else:
			logger.debug("not linked")
		self.compute()
		self.jobsManager.execute()


	def compute(self):
		def calculate_range(info, block, axisIndex):
			#print "block", info.index, info.size, block
			if info.error:
				print "error", info.error_text
				return
			if info.first:
				self.ranges[axisIndex] = [np.nanmin(block), np.nanmax(block)]
			else:
				self.ranges[axisIndex] = [min(self.ranges[axisIndex][0], np.nanmin(block)), max(self.ranges[axisIndex][1], np.nanmax(block)),]
			print "min/max for axis", axisIndex, self.ranges[axisIndex]

		for axisIndex in range(self.dimensions):
			if self.ranges[axisIndex] is None:
				self.jobsManager.addJob(0, functools.partial(calculate_range, axisIndex=axisIndex), self.dataset, self.expressions[axisIndex], **self.getVariableDict())
		if self.weight_expression is None or len(self.weight_expression.strip()) == 0:
			self.jobsManager.addJob(1, self.calculate_visuals, self.dataset, *self.expressions, **self.getVariableDict())
		else:
			all_expressions = self.expressions + [self.weight_expression]
			self.jobsManager.addJob(1, self.calculate_visuals, self.dataset, *all_expressions, **self.getVariableDict())
	
	def getVariableDict(self):
		return {}
			
	def __getVariableDictMinMax(self):
		return {}
			
	def onSelectMask(self, mask):
		self.compute()
		#self.plot()
		
	def onSelectRow(self, row):
		print "row selected", row
		self.selected_point = None
		self.plot()
			
	
	def _beforeCanvas(self, layout):
		pass
	
	def _afterCanvas(self, layout):
		pass
	
	def setMode(self, action, force=False):
		print "set mode", action, action.text(), action.isChecked()
		#if not (action.isChecked() or force):
		if not (action.isChecked()):
			print "ignore action"
		else:
			self.lastAction = action
			if self.currentModes is not None:
				print "disconnect", self.currentModes
				for mode in self.currentModes:
					mode.disconnect_events()
					mode.active = False
			if action == self.action_pick:
				#hasy = hasattr(self, "getdatay")
				#hasx = hasattr(self, "getdatax")
				#print "pick", hasx, hasy
				axes_list = self.getAxesList()
				hasx = True
				hasy = len(self.expressions) > 1
				self.currentModes = [matplotlib.widgets.Cursor(axes, hasy, hasx, color="red", linestyle="dashed") for axes in axes_list]
				if hasx and hasy:
					for mode in self.currentModes:
						mode.connect_event('button_press_event', self.onPickXY)
				elif hasx:
					for mode in self.currentModes:
						mode.connect_event('button_press_event', self.onPickX)
				elif hasy:
					for mode in self.currentModes:
						mode.connect_event('button_press_event', self.onPickY)
			if action == self.action_xrange:
				logger.debug("setting last select action to xrange")
				self.lastActionSelect = self.action_xrange
				self.currentModes = [matplotlib.widgets.SpanSelector(self.axes, self.onSelectX, 'horizontal')]
			if action == self.action_yrange:
				logger.debug("setting last select action to yrange")
				self.lastActionSelect = self.action_yrange
				self.currentModes = [matplotlib.widgets.SpanSelector(self.axes, self.onSelectY, 'vertical')]
			if action == self.action_lasso:
				logger.debug("setting last select action to lasso")
				self.lastActionSelect = self.action_lasso
				self.currentModes =[ matplotlib.widgets.LassoSelector(self.axes, self.onSelectLasso)]
				self.canvas.draw() # bug fix for LassoSelector
			if action == self.action_zoom_x:
				self.lastActionZoom = self.action_zoom_x
				self.currentModes = [matplotlib.widgets.SpanSelector(self.axes, self.onZoomX, 'horizontal')] #, rectprops={"color":"blue"})
			if action == self.action_zoom_y:
				self.lastActionZoom = self.action_zoom_y
				self.currentModes = [matplotlib.widgets.SpanSelector(self.axes, self.onZoomY, 'vertical')] #, rectprops={"color":"blue"})
			if action == self.action_zoom_rect:
				self.lastActionZoom = self.action_zoom_rect
				self.currentModes = [matplotlib.widgets.RectangleSelector(self.axes, self.onZoomRect)] #, rectprops={"color":"blue"})
		self.syncToolbar()
				
		#if self.action_lasso
		#pass
		#self.
		
	def onPickX(self, event):
		x, y = event.xdata, event.ydata
		self.selected_point = None
		class Scope(object):
			pass
		# temp scope object
		scope = Scope()
		scope.index = None
		scope.distance = None
		def pick(block, info, scope=scope):
			if info.first:
				scope.index, scope.distance = find_nearest_index1d(block, x)
			else:
				scope.block_index, scope.block_distance = find_nearest_index1d(block, x)
				if scope.block_distance < scope.distance:
					scope.index = scope.block_index
		self.dataset.evaluate(pick, self.expressions[0], **self.getVariableDict())
		index, distance = scope.index, scope.distance
		print "nearest row", index, distance
		self.dataset.selectRow(index)
		self.setMode(self.lastAction)		
		
	def onPickY(self, event):
		x, y = event.xdata, event.ydata
		self.selected_point = None
		class Scope(object):
			pass
		# temp scope object
		scope = Scope()
		scope.index = None
		scope.distance = None
		def pick(block, info, scope=scope):
			if info.first:
				scope.index, scope.distance = find_nearest_index1d(block, y)
			else:
				scope.block_index, scope.block_distance = find_nearest_index1d(block, y)
				if scope.block_distance < scope.distance:
					scope.index = scope.block_index
		self.dataset.evaluate(pick, self.expressions[1], **self.getVariableDict())
		index, distance = scope.index, scope.distance
		print "nearest row", index, distance
		self.dataset.selectRow(index)
		self.setMode(self.lastAction)		
		

		
	def onPickXY(self, event):
		x, y = event.xdata, event.ydata
		wx = self.ranges_show[0][1] - self.ranges_show[0][0]
		wy = self.ranges_show[1][1] - self.ranges_show[1][0]
		
		self.selected_point = None
		class Scope(object):
			pass
		# temp scope object
		scope = Scope()
		scope.index = None
		scope.distance = None
		def pick(blockx, blocky, info, scope=scope):
			if info.first:
				scope.index, scope.distance = find_nearest_index(blockx, blocky, x, y, wx, wy)
			else:
				scope.block_index, scope.block_distance = find_nearest_index(blockx, blocky, x, y, wx, wy)
				if scope.block_distance < scope.distance:
					scope.index = scope.block_index
		self.dataset.evaluate(pick, *self.expressions[:2], **self.getVariableDict())
		index, distance = scope.index, scope.distance
		print "nearest row", index, distance
		self.dataset.selectRow(index)
		self.setMode(self.lastAction)		
		
		
	def onSelectX(self, xmin, xmax):
		#data = self.getdatax()
		x = [xmin, xmax]
		xmin, xmax = min(x), max(x)
		print "selectx", xmin, xmax
		#xmin = xmin if not self.useLogx() else 10**xmin
		#xmax = xmax if not self.useLogx() else 10**xmax
		#mask = np.zeros(self.dataset._length, dtype=np.bool)
		length = self.dataset.current_slice[1] - self.dataset.current_slice[0]
		mask = np.zeros(length, dtype=np.bool)
		#for block, info in self.dataset.evaluate(self.expressions[0]):
		#	mask[info.i1:info.i2] = (block >= xmin) & (block < xmax)
		#	print ">>>>>>>>>>>>>>> block", info.i1,info.i2, "selected", sum(mask[info.i1:info.i2])
		def putmask(info, block):
			print ">>>>>>>>>>>>>>> block", info.i1,info.i2, "selected", sum(mask[info.i1:info.i2])
			mask[info.i1:info.i2] = self.select_mode(None if self.dataset.mask is None else self.dataset.mask[info.i1:info.i2], (block >= xmin) & (block < xmax))
		self.dataset.evaluate(putmask, self.expressions[0], **self.getVariableDict())
		
		print "selectx", xmin, xmax, "selected", np.sum(mask)
		self.dataset.selectMask(mask)
		self.jobsManager.execute()
		self.setMode(self.lastAction)
		
	def onSelectY(self, ymin, ymax):
		y = [ymin, ymax]
		ymin, ymax = min(y), max(y)
		#mask = (data >= ymin) & (data < ymax)
		mask = np.zeros(self.dataset._length, dtype=np.bool)
		def putmask(info, block):
			mask[info.i1:info.i2] = self.select_mode(None if self.dataset.mask is None else self.dataset.mask[info.i1:info.i2], (block >= ymin) & (block < ymax))
		self.dataset.evaluate(putmask, self.expressions[1], **self.getVariableDict())
		#for block, info in self.dataset.evaluate(self.expressions[1]):
		#	mask[info.i1:info.i2] = (block >= ymin) & (block < ymax)
		print "selecty", ymin, ymax, "selected", np.sum(mask)
		self.dataset.selectMask(mask)
		self.jobsManager.execute()
		self.setMode(self.lastAction)
		
	def onSelectLasso(self, vertices):
		# TODO: new system
		x, y = np.array(vertices).T
		x = np.ascontiguousarray(x, dtype=np.float64)
		y = np.ascontiguousarray(y, dtype=np.float64)
		#mask = np.zeros(len(self.dataset._length), dtype=np.uint8)
		mask = np.zeros(self.dataset._fraction_length, dtype=np.bool)
		meanx = x.mean()
		meany = y.mean()
		radius = np.sqrt((meanx-x)**2 + (meany-y)**2).max()
		#print (x, y, self.parent.datax, self.parent.datay, mask, meanx, meany, radius)
		#for (blockx, blocky), info in self.dataset.evaluate(*self.expressions[:2]):
		def select(info, blockx, blocky):
			gavi.selection.pnpoly(x, y, blockx, blocky, mask[info.i1:info.i2], meanx, meany, radius)
			mask[info.i1:info.i2] = self.select_mode(None if self.dataset.mask is None else self.dataset.mask[info.i1:info.i2], mask[info.i1:info.i2])
		self.dataset.evaluate(select, *self.expressions[:2], **self.getVariableDict())
		if 0:
			try:
				gavi.selection.pnpoly(x, y, self.getdatax(), self.getdatay(), mask, meanx, meany, radius)
			except:
				print gavi.selection.pnpoly.inspect_types()
				args = (x, y, self.getdatax(), self.getdatay(), mask, meanx, meany, radius)
				print "issue with pnppoly, arguments: "
				for i, arg in enumerate(args):
					print i, repr(arg), arg.dtype if hasattr(arg, "dtype") else ""
				raise
		self.dataset.selectMask(mask)
		self.jobsManager.execute()
		self.setMode(self.lastAction)
		
	def onZoomX(self, xmin, xmax):
		self.ranges_show[0] = xmin, xmax
		
		axisIndex = 0
		linkButton = self.linkButtons[axisIndex]
		link = linkButton.link
		if link:
			logger.debug("sending link messages")
			link.sendRangesShow(self.ranges_show[axisIndex], linkButton)
			link.sendPlot(linkButton)
		
		self.plot()
		
	def onZoomY(self, ymin, ymax):
		if len(self.ranges_show) == 1: # if 1d, y refers to range_level
			self.ranges_level[0] = ymin, ymax
		else:
			self.ranges_show[1] = ymin, ymax
		self.plot()
		
	def onZoomRect(self, eclick, erelease):
		x1, y1 = (eclick.xdata, eclick.ydata)
		x2, y2 = (erelease.xdata, erelease.ydata)
		x = [x1, x2]
		y = [y1, y2]

		
		
		
		xmin_show, xmax_show = min(x), max(x)
		ymin_show, ymax_show = min(y), max(y)
		self.ranges_show[0] = xmin_show, xmax_show

		if len(self.ranges_show) == 1: # if 1d, y refers to range_level
			self.ranges_level[0] = ymin_show, ymax_show
			logger.debug("range refers to level: %r" % (self.ranges_level[0],))
		else:
			self.ranges_show[1] = ymin_show, ymax_show
		
		for axisIndex in range(self.dimensions):
			linkButton = self.linkButtons[axisIndex]
			link = linkButton.link
			if link:
				logger.debug("sending link messages")
				link.sendRangesShow(self.ranges_show[axisIndex], linkButton)
				link.sendPlot(linkButton)
		
		#self.axes.set_xlim(self.xmin_show, self.xmax_show)
		#self.axes.set_ylim(self.ymin_show, self.ymax_show)
		#self.canvas.draw()
		self.plot()
		
	def onZoomOut(self, *args):
		xmin, xmax = self.axes.get_xlim()
		width = xmax - xmin
		self.ranges_show[0] = xmin - width , xmax + width

		ymin, ymax = self.axes.get_ylim()
		height = ymax - ymin
		ymin_show, ymax_show = ymin - height, ymax + height
		if len(self.ranges_show) == 1: # if 1d, y refers to range_level
			self.ranges_level[0] = ymin_show, ymax_show
		else:
			self.ranges_show[1] = ymin_show, ymax_show
		
		for axisIndex in range(self.dimensions):
			linkButton = self.linkButtons[axisIndex]
			link = linkButton.link
			if link:
				logger.debug("sending link messages")
				link.sendRangesShow(self.ranges_show[axisIndex], linkButton)
				link.sendPlot(linkButton)
		#self.axes.set_ylim(self.ymin_show, self.ymax_show)
		#self.canvas.draw()
		self.plot()
		
	def onZoomFit(self, *args):
		for i in range(self.dimensions):
			self.ranges[i] = None
			self.ranges_show[i] = None
			self.ranges_level[i] = None
		sendCompute = False
		for axisIndex in range(self.dimensions):
			linkButton = self.linkButtons[axisIndex]
			link = linkButton.link
			if link:
				logger.debug("sending link messages")
				link.sendRanges(self.ranges[axisIndex], linkButton)
				link.sendRangesShow(self.ranges_show[axisIndex], linkButton)
				if not sendCompute: # only send once
					link.sendCompute(linkButton)
					sendCompute  = True

		self.compute()
		self.jobsManager.execute()
		
	def onZoomUse(self, *args):
		for i in range(self.dimensions):
			self.ranges[i] = self.ranges_show[i]
			self.ranges_level[i] = None
		sendCompute = False
		for axisIndex in range(self.dimensions):
			linkButton = self.linkButtons[axisIndex]
			link = linkButton.link
			if link:
				logger.debug("sending link messages")
				link.sendRanges(self.ranges[axisIndex], linkButton)
				link.sendRangesShow(self.ranges_show[axisIndex], linkButton)
				if not sendCompute: # only send once
					link.sendCompute(linkButton)
					sendCompute  = True
		self.compute()
		self.jobsManager.execute()
		
	def addToolbar(self, layout, pick=True, xselect=True, yselect=True, lasso=True):
		
		self.toolbar = QtGui.QToolBar(self)
		self.toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
		self.actionGroup = QtGui.QActionGroup(self)
		self.actionGroupSelectMode = QtGui.QActionGroup(self)
		self.action_pick = QtGui.QAction(QtGui.QIcon(iconfile('glue_cross')), '&Pick', self)
		
		self.action_select = QtGui.QAction(QtGui.QIcon(iconfile('glue_lasso')), '&Select(you should not read this)', self)
		self.action_xrange = QtGui.QAction(QtGui.QIcon(iconfile('glue_xrange_select')), '&x-range', self)
		self.action_yrange = QtGui.QAction(QtGui.QIcon(iconfile('glue_yrange_select')), '&y-range', self)
		self.action_lasso = QtGui.QAction(QtGui.QIcon(iconfile('glue_lasso')), '&Lasso', self)
		self.action_select_none = QtGui.QAction(QtGui.QIcon(iconfile('dialog-cancel-3')), '&No selection', self)

		self.action_select_mode_replace = QtGui.QAction(QtGui.QIcon(iconfile('glue_replace')), '&Replace', self)
		self.action_select_mode_and = QtGui.QAction(QtGui.QIcon(iconfile('glue_and')), '&And', self)
		self.action_select_mode_or = QtGui.QAction(QtGui.QIcon(iconfile('glue_or')), '&Or', self)
		
		self.action_zoom_rect = QtGui.QAction(QtGui.QIcon(iconfile('glue_zoom_to_rect')), '&Zoom to rect', self)
		#self.action_zoom_rect.setIconText("rect")

		#self.actionGroupZoom = QtGui.QActionGroup(self)
		self.action_zoom_x = QtGui.QAction(QtGui.QIcon(iconfile('glue_zoom_to_rect')), '&Zoom x', self)
		self.action_zoom_y = QtGui.QAction(QtGui.QIcon(iconfile('glue_zoom_to_rect')), '&Zoom y', self)
		self.action_zoom = QtGui.QAction(QtGui.QIcon(iconfile('glue_lasso')), '&Zoom(you should not read this)', self)


		self.action_zoom_out = QtGui.QAction(QtGui.QIcon(iconfile('glue_zoom_to_rect')), '&Zoom out', self)
		self.action_zoom_fit = QtGui.QAction(QtGui.QIcon(iconfile('glue_move')), '&Zoom to fit', self)
		self.action_zoom_use = QtGui.QAction(QtGui.QIcon(iconfile('glue_move')), '&Use zoom area', self)

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
		#exitAction.setStatusTip('Exit application')

		#self.actionGroup.setToggleAction(True)
		#self.actionGroup.setExclusive(True)
		self.actionGroupSelectMode.addAction(self.action_select_mode_replace)
		self.actionGroupSelectMode.addAction(self.action_select_mode_and)
		self.actionGroupSelectMode.addAction(self.action_select_mode_or)
		
		self.actionGroup.addAction(self.action_pick)
		self.actionGroup.addAction(self.action_xrange)
		self.actionGroup.addAction(self.action_yrange)
		self.actionGroup.addAction(self.action_lasso)
		self.actionGroup.addAction(self.action_zoom_rect)
		self.actionGroup.addAction(self.action_zoom_x)
		self.actionGroup.addAction(self.action_zoom_y)
		#self.actionGroup.addAction(self.action_zoom_out)
		if pick:
			self.toolbar.addAction(self.action_pick)
			#self.action_pick.setChecked(True)
			#self.setMode(self.action_pick, force=True)
			self.lastAction = self.action_pick
		self.toolbar.addAction(self.action_select)
		self.select_menu = QtGui.QMenu()
		self.action_select.setMenu(self.select_menu)
		if xselect:
			#self.toolbar.addAction(self.action_xrange)
			self.select_menu.addAction(self.action_xrange)
			self.lastActionSelect = self.action_xrange
		if yselect:
			#self.toolbar.addAction(self.action_yrange)
			self.select_menu.addAction(self.action_yrange)
			if self.dimensions > 1:
				self.lastActionSelect = self.action_yrange
		if lasso:
			#self.toolbar.addAction(self.action_lasso)
			self.select_menu.addAction(self.action_lasso)
			if self.dimensions > 1:
				self.lastActionSelect = self.action_lasso
		self.select_menu.addSeparator()
		self.select_menu.addAction(self.action_select_none)
		self.select_menu.addSeparator()
		
		
		self.select_mode_button = QtGui.QToolButton()
		self.select_mode_button.setPopupMode(QtGui.QToolButton.InstantPopup)
		#self.select_mode_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
		self.select_mode_button_menu = QtGui.QMenu()
		self.select_mode_button.setMenu(self.select_mode_button_menu)
		
		self.select_mode_button_menu.addAction(self.action_select_mode_replace)
		self.select_mode_button_menu.addAction(self.action_select_mode_or)
		self.select_mode_button_menu.addAction(self.action_select_mode_and)
		self.select_mode_button.setDefaultAction(self.action_select_mode_replace)
		self.toolbar.addWidget(self.select_mode_button)
		
		#self.toolbar.addAction(action_select_mode)
		
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
		self.toolbar.addAction(self.action_zoom_use)
		
		
		#self.zoomButton.setPopupMode(QtCore.QToolButton.DelayedPopup)
		
		
		self.actionGroup.triggered.connect(self.setMode)
		self.actionGroupSelectMode.triggered.connect(self.setSelectMode)
		self.action_select.triggered.connect(self.onActionSelect)
		self.action_zoom.triggered.connect(self.onActionZoom)
		self.action_zoom_out.triggered.connect(self.onZoomOut)
		self.action_zoom_fit.triggered.connect(self.onZoomFit)
		self.action_zoom_use.triggered.connect(self.onZoomUse)
		self.action_select_none.triggered.connect(self.onActionSelectNone)
		#action_zoom_out
		
		self.action_select_mode_replace.setCheckable(True)
		self.action_select_mode_and.setCheckable(True)
		self.action_select_mode_or.setCheckable(True)
		
		self.action_pick.setCheckable(True)
		self.action_pick.setChecked(True)
		self.action_select.setCheckable(True)
		self.action_xrange.setCheckable(True)
		self.action_yrange.setCheckable(True)
		self.action_lasso.setCheckable(True)
		self.action_zoom.setCheckable(True)
		self.action_zoom_rect.setCheckable(True)
		self.action_zoom_x.setCheckable(True)
		self.action_zoom_y.setCheckable(True)
		#self.action_zoom_out.setCheckable(True)
		#self.actionGroup.
		
		#action = self.toolbar.addAction(icon
		self.syncToolbar()
		self.action_select_mode_replace.setChecked(True)
		self.select_mode = self.select_replace
		layout.addWidget(self.toolbar)

	def setSelectMode(self, action):
		self.select_mode_button.setDefaultAction(action)
		if action == self.action_select_mode_replace:
			self.select_mode = self.select_replace
		if action == self.action_select_mode_and:
			self.select_mode = self.select_and
		if action == self.action_select_mode_or:
			self.select_mode = self.select_or
			
	def select_replace(self, maskold, masknew):
		return masknew
		
	def select_and(self, maskold, masknew):
		return masknew if maskold is None else maskold & masknew
		
	def select_or(self, maskold, masknew):
		return masknew if maskold is None else maskold | masknew
		
	def onActionSelectNone(self):
		self.dataset.selectMask(None)
		self.jobsManager.execute()
		
	def onActionSelect(self):
		self.lastActionSelect.setChecked(True)
		self.setMode(self.lastActionSelect)
		self.syncToolbar()
		
	def onActionZoom(self):
		self.lastActionZoom.setChecked(True)
		self.setMode(self.lastActionZoom)
		self.syncToolbar()
		
	def syncToolbar(self):
		for action in [self.action_select, self.action_zoom]:
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
		logger.debug("last zoom action: %r" % self.lastActionZoom.text())
		self.action_select.setText(self.lastActionSelect.text())
		self.action_select.setIcon(self.lastActionSelect.icon())
		self.action_zoom.setText(self.lastActionZoom.text())
		self.action_zoom.setIcon(self.lastActionZoom.icon())
		#self.action_select.update()
		
		


class HistogramPlotDialog(PlotDialog):
	def __init__(self, parent, jobsManager, dataset, expression):
		super(HistogramPlotDialog, self).__init__(parent, jobsManager, dataset, [expression], ["X"])
		
	def beforeCanvas(self, layout):
		self.addToolbar(layout, yselect=False, lasso=False)
		
	def calculate_visuals(self, info, block, weights_block=None):
		self.expression_error = False
		#return
		xmin, xmax = self.ranges[0]
		if self.ranges_show[0] is None:
			self.ranges_show[0] = xmin, xmax
		N = 128
		mask = self.dataset.mask
		if info.first:
			self.selected_point = None
			self.counts = np.zeros(N, dtype=np.float64)
			if weights_block is not None:
				self.counts_weights = np.zeros(N, dtype=np.float64)
			else:
				self.counts_weights = None
			if mask is not None:
				self.counts_mask = np.zeros(N, dtype=np.float64) #mab.utils.numpy.mmapzeros((128), dtype=np.float64)
				if weights_block is not None:
					self.counts_mask_weights = np.zeros(N, dtype=np.float64) #mab.utils.numpy.mmapzeros((128), dtype=np.float64)
				else:
					self.counts_mask_weights = None
			else:
				self.counts_mask = None
				self.counts_mask_weights = None
		
		if info.error:
			print "error", info.error_text
			self.expression_error = True
			return
		#totalxmin, totalxmax = self.gettotalxrange()
		#print repr(self.data), repr(self.counts), repr(xmin), repr(xmax)
		t0 = time.time()
		try:
			args = (block, self.counts, xmin, xmax)
			gavi.histogram.hist1d(block, self.counts, xmin, xmax)
			if weights_block is not None:
				args = (block, self.counts, xmin, xmax, weights_block)
				gavi.histogram.hist1d_weights(block, self.counts_weights, weights_block, xmin, xmax)
		except:
			logger.exception("error with hist1d, arguments: %r" % (args,))
		if mask is not None:
			subset = block[mask[info.i1:info.i2]]
			gavi.histogram.hist1d(subset, self.counts_mask, xmin, xmax)
			if weights_block is not None:
				subset_weights = weights_block[mask[info.i1:info.i2]]
				gavi.histogram.hist1d_weights(subset, self.counts_mask_weights, subset_weights, xmin, xmax)
		print "it took", time.time()-t0
		
		index = self.dataset.selected_row_index
		if index is not None:
			if index >= info.i1 and index < info.i2: # selected point is in this block
				self.selected_point = block[index-info.i1]

		self.delta = (xmax - xmin) / N
		self.centers = np.arange(N) * self.delta + xmin
		#print xmin, xmax, self.centers
		
		
	def plot(self):
		self.axes.cla()
		self.axes.autoscale(False)
		if self.expression_error:
			return
		#P.hist(x, 50, normed=1, histtype='stepfilled')
		#values = 
		amplitude = self.counts
		logger.debug("expr for amplitude: %r" % self.amplitude_expression)
		if self.amplitude_expression is not None:
			locals = {"counts":self.counts, "counts_weights":self.counts_weights}
			globals = np.__dict__
			amplitude = eval(self.amplitude_expression, globals, locals)

		if self.ranges_level[0] is None:
			if self.weight_expression:
				self.ranges_level[0] = np.nanmin(amplitude) * 1.1, np.nanmax(amplitude) * 1.1
			else:
				self.ranges_level[0] = 0, np.nanmax(amplitude) * 1.1


		if self.counts_mask is None:
			self.axes.bar(self.centers, amplitude, width=self.delta, align='center')
		else:
			if self.amplitude_expression is not None:
				locals = {"counts":self.counts_mask}
				globals = np.__dict__
				amplitude_mask = eval(self.amplitude_expression, globals, locals)
			self.axes.bar(self.centers, amplitude, width=self.delta, align='center', alpha=0.5)
			self.axes.bar(self.centers, amplitude_mask, width=self.delta, align='center', color="red")
		
		index = self.dataset.selected_row_index
		if index is not None and self.selected_point is None:
			logger.debug("point selected but after computation")
			# TODO: optimize
			# TODO: optimize
			def find_selected_point(info, block):
				if index >= info.i1 and index < info.i2: # selected point is in this block
					self.selected_point = block[index-info.i1]
			self.dataset.evaluate(find_selected_point, *self.expressions, **self.getVariableDict())
		
		if self.selected_point is not None:
			#x = self.getdatax()[self.dataset.selected_row_index]
			print "drawing vline at", self.selected_point
			self.axes.axvline(self.selected_point, color="red")
		
		self.axes.set_xlabel(self.expressions[0])
		xmin_show, xmax_show = self.ranges_show[0]
		print "plot limits:", xmin_show, xmax_show
		self.axes.set_xlim(xmin_show, xmax_show)
		ymin_show, ymax_show = self.ranges_level[0]
		print "level limits:", ymin_show, ymax_show
		self.axes.set_ylim(ymin_show, ymax_show)
		self.canvas.draw()

cols = []
for x in np.linspace(0,1, 256):
	rcol = 0.237 - 2.13*x + 26.92*x**2 - 65.5*x**3 + 63.5*x**4 - 22.36*x**5
	gcol = ((0.572 + 1.524*x - 1.811*x**2)/(1 - 0.291*x + 0.1574*x**2))**2
	bcol = 1/(1.579 - 4.03*x + 12.92*x**2 - 31.4*x**3 + 48.6*x**4 - 23.36*x**5)
	cols.append((rcol, gcol, bcol))

cm_plusmin = matplotlib.colors.LinearSegmentedColormap.from_list("PaulT_plusmin", cols)


class ScatterPlotDialog(PlotDialog):
	def __init__(self, parent, jobsManager, dataset, xname=None, yname=None):
		super(ScatterPlotDialog, self).__init__(parent, jobsManager, dataset, [xname, yname], "X Y".split())
		
	def calculate_visuals(self, info, blockx, blocky, weights_block=None):
		self.expression_error = False

		xmin, xmax = self.ranges[0]
		ymin, ymax = self.ranges[1]
		for i in range(self.dimensions):
			if self.ranges_show[i] is None:
				self.ranges_show[i] = self.ranges[i]

		N = 128
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
			print "error", info.error_text
			self.expression_error = True
			return
		
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
			args = blockx, blocky, self.counts, ranges
			gavi.histogram.hist2d(blockx, blocky, self.counts, *ranges)
			if weights_block is not None:
				args = blockx, blocky, weights_block, self.counts, ranges
				gavi.histogram.hist2d_weights(blockx, blocky, self.counts_weights, weights_block, *ranges)
		except:
			print "args", args	
			print blockx.shape, blockx.dtype
			print blocky.shape, blocky.dtype
			print self.counts.shape, self.counts.dtype
			raise
		print "it took", time.time()-t0

		if mask is not None:
			subsetx = blockx[mask[info.i1:info.i2]]
			subsety = blocky[mask[info.i1:info.i2]]
			#print subx, suby, mask[info.i1:info.i2]
			#histo2d(subsetx, subsety, self.counts_mask, *self.ranges)
			gavi.histogram.hist2d(subsetx, subsety, self.counts_mask, *ranges)
			if weights_block is not None:
				subset_weights = weights_block[mask[info.i1:info.i2]]
				gavi.histogram.hist2d_weights(subsetx, subsety, self.counts_weights_mask, subset_weights, *ranges)
		
		
	def plot(self):
		self.axes.cla()
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
		print "amplitude", np.nanmin(amplitude), np.nanmax(amplitude)
		#if self.ranges_level[0] is None:
		#	self.ranges_level[0] = 0, amplitude.max() * 1.1

			
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
			print "drawing selected point at", x, y
			self.axes.scatter([x], [y], color='red') #, scalex=False, scaley=False)
		#if dataxsel is not None:
		#	self.axes.scatter(dataxsel, dataysel)
		#self.axes.set_xlabel(self.xaxisName)
		#self.axes.set_ylabel(self.yaxisName)
		print "plot limits:", self.ranges
		self.axes.set_xlim(*self.ranges_show[0])
		self.axes.set_ylim(*self.ranges_show[1])
		self.canvas.draw()
		
		
class Rank1ScatterPlotDialog(ScatterPlotDialog):
	def __init__(self, parent, jobsManager, dataset, xname=None, yname=None):
		self.nSlices = dataset.rank1s[dataset.rank1s.keys()[0]].shape[0]
		self.serieIndex = dataset.selected_serie_index if dataset.selected_serie_index is not None else 0
		super(Rank1ScatterPlotDialog, self).__init__(parent, jobsManager, dataset, xname, yname)

	def onSerieIndexSelect(self, serie_index):
		if serie_index != self.serieIndex: # avoid unneeded event
			self.serieIndex = serie_index
			self.seriesbox.setCurrentIndex(self.serieIndex)
		else:
			self.serieIndex = serie_index
		#print "%" * 200
		self.compute()
		#self.jobsM
		#self.plot()
	
		
	def getExpressionList(self):
		names = []
		for rank1name in self.dataset.rank1names:
			names.append(rank1name + "[index]")
		return names
	
	def getVariableDict(self):
		return {"index": self.serieIndex}

	def _getVariableDictMinMax(self):
		return {"index": slice(None, None, None)}

	def afterCanvas(self, layout):
		super(Rank1ScatterPlotDialog, self).afterCanvas(layout)

		self.seriesbox = QtGui.QComboBox(self)
		self.seriesbox.addItems([str(k) for k in range(self.nSlices)])
		self.seriesbox.setCurrentIndex(self.serieIndex)
		self.seriesbox.currentIndexChanged.connect(self.onSerieIndex)
		
		self.form_layout.addRow("index", self.seriesbox)
		self.buttonLoop = QtGui.QToolButton(self)
		self.buttonLoop.setText("one loop")
		self.buttonLoop.clicked.connect(self.onPlayOnce)
		self.form_layout.addRow("movie", self.buttonLoop)
		layout.addLayout(self.form_layout, 0)
		
	def onPlayOnce(self):
		#self.timer = QtCore.QTimer(self)
		#self.timer.timeout.connect(self.onNextFrame)
		self.delay = 10
		for i in range(self.dimensions):
			self.ranges[i] = None
		for i in range(self.dimensions):
			self.ranges_show[i] = None
		self.dataset.selectSerieIndex(0)
		self.jobsManager.execute()
		QtCore.QTimer.singleShot(self.delay, self.onNextFrame);
		
	def onNextFrame(self, *args):
		#print args
		step = 15
		next = self.serieIndex +step
		if next >= self.nSlices:
			next = self.nSlices-1
		for i in range(self.dimensions):
			self.ranges[i] = None
		for i in range(self.dimensions):
			self.ranges_show[i] = None
		self.dataset.selectSerieIndex(next)
		self.jobsManager.execute()
		if self.serieIndex < self.nSlices-1 : # not last frame
			QtCore.QTimer.singleShot(self.delay, self.onNextFrame);
			
			
	def onSerieIndex(self, index):
		if index != self.dataset.selected_serie_index: # avoid unneeded event
			for i in range(self.dimensions):
				self.ranges[i] = None
			for i in range(self.dimensions):
				self.ranges_show[i] = None
			self.dataset.selectSerieIndex(index)
			#self.compute()
			self.jobsManager.execute()


class SequencePlot(PlotDialog):
	def __init__(self, parent, jobsManager, dataset, expression="x[:,index]"):
		self.index = 0
		super(SequencePlot, self).__init__(parent, jobsManager, dataset, [expression], ["X"])
		
	def beforeCanvas(self, layout):
		self.addToolbar(layout, xselect=True, lasso=False)
		
	def getExpressionList(self):
		names = []
		for rank1name in self.dataset.rank1names:
			names.append(rank1name + "[:,index]")
		return names
	
	def getVariableDict(self):
		return {"index": self.index}

	def calculate_visuals(self, info, block):
		#print "nothing to calculate"
		print block, block.shape
	
	def _calculate_visuals(self, info, block):
		self.expression_error = False
		#return
		xmin, xmax = self.ranges[0]
		if self.ranges_show[0] is None:
			self.ranges_show[0] = xmin, xmax
		N = 128
		mask = self.dataset.mask
		if info.first:
			self.selected_point = None
			self.counts = np.zeros(N, dtype=np.float64)
			if mask is not None:
				self.counts_mask = np.zeros(N, dtype=np.float64) #mab.utils.numpy.mmapzeros((128), dtype=np.float64)
			else:
				self.counts_mask = None
		
		if info.error:
			print "error", info.error_text
			self.expression_error = True
			return
		#totalxmin, totalxmax = self.gettotalxrange()
		#print repr(self.data), repr(self.counts), repr(xmin), repr(xmax)
		t0 = time.time()
		try:
			gavi.histogram.hist1d(block, self.counts, xmin, xmax)
		except:
			args = (block, self.counts, xmin, xmax, self.useLog())
			logging.exception("error with hist1d, arguments: %r" % (args,))
		if mask is not None:
			subset = block[mask[info.i1:info.i2]]
			gavi.histogram.hist1d(subset, self.counts_mask, xmin, xmax)
		print "it took", time.time()-t0
		
		index = self.dataset.selected_row_index
		if index is not None:
			if index >= info.i1 and index < info.i2: # selected point is in this block
				self.selected_point = block[index-info.i1]

		self.delta = (xmax - xmin) / N
		self.centers = np.arange(N) * self.delta + xmin
		#print xmin, xmax, self.centers
		
		
	def plot(self):
		self.axes.cla()
		self.axes.autoscale(False)
		if self.expression_error:
			return
		#P.hist(x, 50, normed=1, histtype='stepfilled')
		#values = 
		amplitude = self.counts
		logger.debug("expr for amplitude: %r" % self.amplitude_expression)
		if self.amplitude_expression is not None:
			locals = {"counts":self.counts}
			globals = np.__dict__
			amplitude = eval(self.amplitude_expression, globals, locals)

		if self.ranges_level[0] is None:
			self.ranges_level[0] = 0, amplitude.max() * 1.1


		if self.counts_mask is None:
			self.axes.bar(self.centers, amplitude, width=self.delta, align='center')
		else:
			if self.amplitude_expression is not None:
				locals = {"counts":self.counts_mask}
				globals = np.__dict__
				amplitude_mask = eval(self.amplitude_expression, globals, locals)
			self.axes.bar(self.centers, amplitude, width=self.delta, align='center', alpha=0.5)
			self.axes.bar(self.centers, amplitude_mask, width=self.delta, align='center', color="red")
		
		index = self.dataset.selected_row_index
		if index is not None and self.selected_point is None:
			logger.debug("point selected but after computation")
			# TODO: optimize
			# TODO: optimize
			def find_selected_point(info, block):
				if index >= info.i1 and index < info.i2: # selected point is in this block
					self.selected_point = block[index-info.i1]
			self.dataset.evaluate(find_selected_point, *self.expressions, **self.getVariableDict())
		
		if self.selected_point is not None:
			#x = self.getdatax()[self.dataset.selected_row_index]
			print "drawing vline at", self.selected_point
			self.axes.axvline(self.selected_point, color="red")
		
		self.axes.set_xlabel(self.expressions[0])
		xmin_show, xmax_show = self.ranges_show[0]
		print "plot limits:", xmin_show, xmax_show
		self.axes.set_xlim(xmin_show, xmax_show)
		ymin_show, ymax_show = self.ranges_level[0]
		print "level limits:", ymin_show, ymax_show
		self.axes.set_ylim(ymin_show, ymax_show)
		self.canvas.draw()
	
		
class PlotDialogNd(PlotDialog):
	def __init__(self, parent, dataset, names, axisnames):
		self.axisnames = axisnames
		self.dataset = dataset
		assert len(names) == len(axisnames)
		column_names = self.getColumnNames()
		self.currentAxes = [column_names.index(name) for name in names]
		self.axisNames = [column_names[index] for index in self.currentAxes]
		column_dict = self.getColumnDict()
		self.datalist = [column_dict[name] for name in self.axisNames]
		self.nAxes = len(self.datalist)
		self.axeslist = range(self.nAxes)
		self.ranges_show = None
		self.ranges = None
		super(PlotDialogNd, self).__init__(parent, dataset)
		
	def addAxes(self):
		self.axes_grid = [[None,] * self.nAxes for _ in self.axeslist]
		index = 0
		for i in self.axeslist[::1]:
			for j in self.axeslist[::1]:
				index = ((self.nAxes-1)-j) * self.nAxes + i + 1
				axes = self.axes_grid[i][j] = self.fig.add_subplot(self.nAxes,self.nAxes,index)
#													   sharey=self.axes_grid[0][j] if j > 0 else None,
#													   sharex=self.axes_grid[i][0] if i > 0 else None
#													   )
				if i > 0:
					for label in axes.get_yticklabels():
						label.set_visible(False)
					axes.yaxis.offsetText.set_visible(False)
				if j > 0:
					for label in axes.get_xticklabels():
						label.set_visible(False)
					axes.xaxis.offsetText.set_visible(False)
				self.axes_grid[i][j].hold(True)
				index += 1
		self.fig.subplots_adjust(hspace=0, wspace=0)

		
	def afterCanvas(self, layout):
		#self.mpl_toolbar = NavigationToolbar(self.canvas, self.axes, self)
		
		self.form_layout = QtGui.QFormLayout(self)

		self.axisboxes = []
		axisIndex = 0
		for axisname in self.axisnames:
			axisbox = QtGui.QComboBox(self)
			self.form_layout.addRow(axisname + '-axis:', axisbox)
			axisbox.addItems(self.getColumnNames())
			axisbox.setCurrentIndex(self.currentAxes[axisIndex])
			axisbox.currentIndexChanged.connect(functools.partial(self.onAxis, axisIndex=axisIndex))
			axisIndex += 1
		layout.addLayout(self.form_layout, 0)

	def onAxis(self, index, axisIndex=0):
		self.currentAxes[axisIndex] = index
		self.axisNames[axisIndex] = self.getColumnNames()[self.currentAxes[axisIndex]]
		print "axis index %d (%s) changed, index=%d, name=%s" % (axisIndex, self.axisnames[axisIndex], index, self.axisNames[axisIndex])
		self.datalist[axisIndex] = self.getColumnDict()[self.axisNames[axisIndex]][slice(*self.dataset.current_slice)]
		self.ranges[axisIndex] = None
		self.ranges_show[axisIndex] = None
		#self.onXDataSelected(self.datax)
		self.compute()
		self.plot()
		
	def getrange(self, axisIndex):
		return np.nanmin(self.datalist[axisIndex]), np.nanmax(self.datalist[axisIndex])

	def onZoomFit(self, *args):
		self.ranges_show = [self.getrange(axis) for axis in self.axes]
		self.ranges = [range_ for range_ in self.ranges_show]
		#self.axes.set_xlim(self.xmin_show, self.xmax_show)
		#self.axes.set_ylim(self.ymin_show, self.ymax_show)
		#self.canvas.draw()
		self.compute()
		self.plot()

	def onZoomUse(self, *args):
		self.xmin, self.xmax = self.xmin_show, self.xmax_show
		self.ymin, self.ymax = self.ymin_show, self.ymax_show
		#self.axes.set_xlim(self.xmin, self.xmax)
		#self.axes.set_ylim(self.ymin, self.ymax)
		#self.canvas.draw()
		self.compute()
		self.plot()
		
	def beforeCanvas(self, layout):
		self.addToolbar(layout) #, yselect=True, lasso=False)
		
	def getdatax(self):
		return None
	def getdatay(self):
		return None



class PlotDialog3d(PlotDialogNd):
	def __init__(self, parent, dataset, xname=None, yname=None, zname=None):
		super(PlotDialog3d, self).__init__(parent, dataset, [xname, yname, zname], "X Y Z".split())
		
	def getAxesList(self):
		return reduce(lambda x,y: x + y, self.axes_grid, [])

	def getColumnNames(self):
		return self.dataset.column_names
	
	def getColumnDict(self):
		return self.dataset.columns
	
	def compute(self):
		if self.ranges_show is None:
			self.ranges_show = [self.getrange(axis) for axis in self.axeslist]
		for i in self.axeslist:
			if self.ranges_show[i] is None:
				self.ranges_show[i] = self.getrange(i)
		if self.ranges is None:
			self.ranges = [range_ for range_ in self.ranges_show]
		for i in self.axeslist:
			if self.ranges[i] is None:
				self.ranges[i] = self.ranges_show[i]
			
		Nhisto = 128
		self.counts = np.zeros((Nhisto, ) * self.nAxes, dtype=np.float64)
		assert self.nAxes == 3
		print self.datalist[0], self.datalist[1], self.datalist[2], self.counts, (self.ranges[0] + self.ranges[1] + self.ranges[2])
		gavi.histogram.hist3d(self.datalist[0], self.datalist[1], self.datalist[2], self.counts, *(self.ranges[0] + self.ranges[1] + self.ranges[2]))
		
	def plot(self):
		def multisum(a, axes):
			correction = 0
			for axis in axes:
				a = np.sum(a, axis=axis-correction)
				correction += 1
			return a		
		for i in self.axeslist:
			for j in self.axeslist:
				axes = self.axes_grid[i][j]
				ranges = self.ranges[i] + self.ranges[j]
				axes.clear()
				allaxes = range(self.nAxes)
				if i > 0:
					for label in axes.get_yticklabels():
						label.set_visible(False)
					axes.yaxis.offsetText.set_visible(False)
				if j > 0:
					for label in axes.get_xticklabels():
						label.set_visible(False)
					axes.xaxis.offsetText.set_visible(False)
				if i != j:
					allaxes.remove(i)
					allaxes.remove(j)
					counts = multisum(self.counts, allaxes)
					if i < j:
						counts = counts.T
					axes.imshow(np.log10(counts), origin="lower", extent=ranges) #, alpha=1 if self.counts_mask is None else 0.4)
					axes.set_aspect('auto')
					if self.dataset.selected_row_index is not None:
						#self.axes.autoscale(False)
						x, y = self.getdatax()[self.dataset.selected_row_index],  self.getdatay()[self.dataset.selected_row_index]
						print "drawing selected point at", x, y
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
					
				#index += 1
		#return
		#self.axes.cla()
		#self.axes.imshow(np.log10(self.counts.T), origin="lower", extent=self.ranges, alpha=1 if self.counts_mask is None else 0.4)
		#if self.counts_mask is not None:
		#	self.axes.imshow(np.log10(self.counts_mask.T), origin="lower", extent=self.ranges, alpha=1)
		#self.axes.imshow((I), origin="lower", extent=ranges)
		#if dataxsel is not None:
		#	self.axes.scatter(dataxsel, dataysel)
		#self.axes.set_xlabel(self.xaxisName)
		#self.axes.set_ylabel(self.yaxisName)
		#print "plot limits:", self.xmin, self.xmax
		#self.axes.set_xlim(self.xmin_show, self.xmax_show)
		#self.axes.set_ylim(self.ymin_show, self.ymax_show)
		self.canvas.draw()
	
class PlotDialog1d(PlotDialog):
	def __init__(self, parent, jobsManager, dataset, name=None, axisname="X"):
		self.axisname = axisname
		#self.currentAxis = 0 if name is None else dataset.column_names.index(name)
		#self.axisName = dataset.column_names[self.currentAxis]
		#self.data = dataset.columns[self.axisName][slice(*dataset.current_slice)]
		self.expr = name
		#self.name = name
		super(PlotDialog1d, self).__init__(parent, jobsManager, dataset)
		
		
	def afterCanvas(self, layout):
		#self.mpl_toolbar = NavigationToolbar(self.canvas, self.axes, self)
		
		self.form_layout = QtGui.QFormLayout(self)

		self.axisbox = QtGui.QComboBox(self)
		self.axisbox.setEditable(True)
		self.form_layout.addRow(self.axisname + '-axis:', self.axisbox)
		self.axisbox.addItems(self.dataset.column_names)
		self.axisbox.lineEdit().setText(self.expr)
		#self.axisbox.setCurrentIndex(self.currentAxis)
		#self.axisbox.currentIndexChanged.connect(self.onAxis)
		#self.axisbox.editTextChanged.connect(self.onExpressionChanged)		
		self.axisbox.lineEdit().editingFinished.connect(self.onExpressionChanged)
		self.logcheckbox = QtGui.QCheckBox(self)
		self.logcheckbox.hide()
		#self.form_layout.addRow('log10 ' +self.axisname, self.logcheckbox)
		self.logcheckbox.stateChanged.connect(self.onChangeLog)


		#layout.addWidget(self.mpl_toolbar, 1)
		layout.addLayout(self.form_layout, 0)
		
	def closeEvent(self, event):
		print "close event"
		self.axisbox.lineEdit().editingFinished.disconnect(self.onExpressionChanged)
		super(PlotDialog1d, self).closeEvent(event)

	def useLog(self):
		return self.logcheckbox.checkState() == QtCore.Qt.Checked
		
	def useLogx(self):
		return self.useLog()
		
	def onChangeLog(self):
		self.compute()
		self.plot()

	def onAxis(self, index):
		if 0:
			print "index", index
			self.currentAxis = index
			#self.expr =
			self.axisName = self.dataset.column_names[self.currentAxis]
			print "axis changed, index=", self.currentAxis, "name is", self.axisName
			self.data = self.dataset.columns[self.axisName][slice(*self.dataset.current_slice)]
			self.onDataSelected(self.data)
			self.compute()
			self.plot()
		
	def onExpressionChanged(self):
		text = str(self.axisbox.lineEdit().text())
		print "expr", repr(text)
		if text == self.expr:
			logger.debug("same expression, will not update")
			return
		self.expr = str(text)
		self.xmin, self.xmax = None, None
		self.ymin, self.ymax = None, None
		self.xmin_show, self.xmax_show = None, None
		self.ymin_show, self.ymax_show = None, None
		self.compute()
		self.jobsManager.execute()
		#self.plot()
		
	def onDataSelected(self, data):
		pass



class HistogramPlotDialog_old(PlotDialog1d):
	def __init__(self, parent, jobsManager, dataset, name):
		super(HistogramPlotDialog, self).__init__(parent, jobsManager, dataset, name)
		
	def beforeCanvas(self, layout):
		self.addToolbar(layout, yselect=False, lasso=False)
		
	def onChangeLog(self):
		self.xmin, self.xmax = None, None
		self.ymin, self.ymax = None, None
		self.xmin_show, self.xmax_show = None, None
		self.ymin_show, self.ymax_show = None, None
		super(HistogramPlotDialog, self).onChangeLog()
		
	def onDataSelected(self, data):
		self.xmin, self.xmax = None, None
		self.ymin, self.ymax = None, None
		self.xmin_show, self.xmax_show = None, None
		self.ymin_show, self.ymax_show = None, None
		
	def _getdatax(self):
		return self.data[slice(*self.dataset.current_slice)]

	def getx_expr(self):
		return self.expr
	
	def getxrange(self):
		#return np.nanmin(self.data), np.nanmax(self.data)
		minvalue, maxvalue = None, None
		for block, info in self.dataset.evaluate(self.expr):
			print "block", info.index, info.size, block
			if info.first:
				minvalue = np.nanmin(block)
				maxvalue = np.nanmax(block)
			else:
				minvalue = min(minvalue, np.nanmin(block))
				maxvalue = max(maxvalue, np.nanmax(block))
			print "min/max", minvalue, maxvalue
		return minvalue, maxvalue
		
	def _gettotalxrange(self):
		return np.nanmin(self.data), np.nanmax(self.data)
		
	def getyrange(self):
		return 0., np.max(self.counts)
	
	def checkExpressions(self, *expressions):
		isValid = True
		if 0: #not self.dataset.validExpressions(*expressions)
			palette = QtCore.QPalette(self.axisbox.lineEdit().palette())
			palette.setColor(QtCore.QPalette.Background, QtCore.Qt.red);
			self.axisbox.lineEdit().setAutoFillBackground(True);
			self.axisbox.lineEdit().setPalette(pallete);
			self.axisbox.lineEdit().show();			
		return isValid
		
	#def calculate_xrange(self, blockx, info):

	def compute(self):
		def calculate_xrange(info, block):
			#print "block", info.index, info.size, block
			if info.error:
				print "error", info.error_text
				return
			if info.first:
				self.xmin = np.nanmin(block)
				self.xmax = np.nanmax(block)
			else:
				self.xmin = min(self.xmin, np.nanmin(block))
				self.xmax = max(self.xmax, np.nanmax(block))
			print "min/max", self.xmin, self.xmax
		
		if self.xmin is None or self.xmax is None:
			self.jobsManager.addJob(0, calculate_xrange, self.dataset, self.expr)
			#self.xmin, self.xmax = self.getxrange()
		#	print "compute: setting x limits", self.xmin, self.xmax
		self.jobsManager.addJob(1, self.calculate_visuals, self.dataset, self.expr)
		#self.dataset.addJob(2, self.plot)
		
	def calculate_visuals(self, block, info):
		self.error = False
		#return
		if self.xmin_show is None or self.xmax_show is None:
			self.xmin_show = self.xmin
			self.xmax_show = self.xmax
		N = 128 #len(self.counts) 
		if info.first:
			self.counts = np.zeros(N, dtype=np.float64)
		
		mask = self.dataset.mask
		if mask is not None and info.first:
			self.counts_mask = np.zeros(N, dtype=np.float64) #mab.utils.numpy.mmapzeros((128), dtype=np.float64)
		
		if info.error:
			print "error", info.error_text
			self.error = True
			return
		#totalxmin, totalxmax = self.gettotalxrange()
		#print repr(self.data), repr(self.counts), repr(xmin), repr(xmax)
		t0 = time.time()
		if 0:
			data = self.dataset.columns[self.expr]
			data_length = len(data)
			if 0:
				parts = 20
				#count_parts = np.zeros((parts, 128), dtype=np.float64)
				counts_parts = mab.utils.numpy.mmapzeros((parts, 128), dtype=np.float64)
				@parallelize(cores=QtCore.QThread.idealThreadCount())
				def histo1d_par(part):
					#N = len(datax)
					i1 = (data_length / parts) * part
					i2 = (data_length / parts) * (part+1)
					#print i1, i2,datax[i1:i2],datay[i1:i2], Iparts[part], ranges
					#histo2d(datax[i1:i2], datay[i1:i2], Iparts[part], *ranges)
					gavi.histogram.hist1d(data[i1:i2], counts_parts[part], self.xmin, self.xmax)
					#histo2d(datax[part::parts], datay[part::parts], Iparts[part], np.min(datax), np.max(datax), np.min(datay), np.max(datay))
				histo1d_par(range(parts))
				self.counts = np.sum(counts_parts, axis=0)
			else:
				self.counts = np.zeros(128, dtype=np.float64)
				gavi.histogram.hist1d(data, self.counts, self.xmin, self.xmax, self.useLog())
				#I = np.zeros((parts, 128, 128), dtype=np.int32)
				#histo2d(datax, datay, Iparts, np.min(datax), np.max(datax), np.min(datay), np.max(datay))
				#I = np.sum(I, axis=0)
				
			if self.dataset.mask is not None:
				data_masked = data[self.dataset.mask]
				data_length = len(data_masked)
				self.counts_mask = mab.utils.numpy.mmapzeros((128), dtype=np.float64)
				gavi.histogram.hist1d(data_masked, self.counts_mask, self.xmin, self.xmax, self.useLog())
			else:
				self.counts_mask = None
		else:
			#print "block", info.index, info.size, block
			try:
				gavi.histogram.hist1d(block, self.counts, self.xmin, self.xmax, self.useLog())
			except:
				args = (block, self.counts, self.xmin, self.xmax, self.useLog())
				logger.exception("error with hist1d, arguments: %r" % (args,))
			if mask is not None:
				#print "block", info.index, info.size, block, info.i1, info.i2
				sub = block[mask[info.i1:info.i2]]
				#print sub, mask[info.i1:info.i2]
				gavi.histogram.hist1d(sub, self.counts_mask, self.xmin, self.xmax, self.useLog())
			else:
				self.counts_mask = None
		print "it took", time.time()-t0
		
		
		if self.ymin is None or self.ymax is None:
			self.ymin, self.ymax = self.getyrange()
		if self.ymin_show is None or self.ymax_show is None:
			self.ymin_show = self.ymin
			self.ymax_show = self.ymax
		self.delta = (self.xmax - self.xmin) / N
		self.centers = np.arange(N) * self.delta + self.xmin
		#print xmin, xmax, self.centers
		
		
	def plot(self):
		self.axes.cla()
		self.axes.autoscale(False)
		if self.error:
			return
		#P.hist(x, 50, normed=1, histtype='stepfilled')
		#values = 
		if self.counts_mask is None:
			self.axes.bar(self.centers, self.counts, width=self.delta, align='center')
		else:
			self.axes.bar(self.centers, self.counts, width=self.delta, align='center', alpha=0.5)
			self.axes.bar(self.centers, self.counts_mask, width=self.delta, align='center', color="red")
		
		print "row index", self.dataset.selected_row_index
		if self.dataset.selected_row_index is not None:
			x = self.getdatax()[self.dataset.selected_row_index]
			print "drawing vline at", x
			self.axes.axvline(x, color="red")
		
		#width = self.xmax - self.xmin
		#self.axes.set_xlim(self.xmin - width * 0.01, self.xmax + width * 0.01)
		self.axes.set_xlabel(self.expr)
		print "plot limits:", self.xmin_show, self.xmax_show
		self.axes.set_xlim(self.xmin_show, self.xmax_show)
		self.axes.set_ylim(self.ymin_show, self.ymax_show)
		#if self.lastAction is not None:
		#	self.setMode(self.lastAction)
		self.canvas.draw()
		
class PlotDialog2d(PlotDialog):
	def __init__(self, parent, dataset, xname=None, yname=None, xaxisname="X", yaxisname="Y"):
		self.dataset = dataset
		self.xaxisname = xaxisname
		self.yaxisname = yaxisname
		#self.currentXAxis = 0 if xname is None else self.getColumnNames().index(xname)
		#self.currentYAxis = 1 if yname is None else self.getColumnNames().index(yname)
		##self.xaxisName = self.getColumnNames()[self.currentXAxis]
		#self.yaxisName = self.getColumnNames()[self.currentYAxis]
		#self.datax = self.getColumnDict()[self.xaxisName][slice(*dataset.current_slice)]
		#self.datay = self.getColumnDict()[self.yaxisName][slice(*dataset.current_slice)]
		#self.name = name
		self.expressions = [xname if xname is not None else self.dataset.column_names[0], yname if yname  is not None else self.dataset.column_names[1]]
		super(PlotDialog2d, self).__init__(parent, dataset)

	def getColumnNames(self):
		return self.__dict__['dataset'].column_names
	
	def getColumnDict(self):
		return self.dataset.columns
		
	def afterCanvas(self, layout):
		#self.mpl_toolbar = NavigationToolbar(self.canvas, self.axes, self)
		
		self.form_layout = QtGui.QFormLayout(self)

		self.xaxisbox = QtGui.QComboBox(self)
		self.form_layout.addRow(self.xaxisname + '-axis:', self.xaxisbox)
		self.xaxisbox.addItems(self.getColumnNames())
		self.xaxisbox.setEditable(True)
		self.xaxisbox.lineEdit().setText(self.expressions[0])
		self.xaxisbox.editTextChanged.connect(self.onExpressionChangedX)		

		
		self.yaxisbox = QtGui.QComboBox(self)
		self.form_layout.addRow(self.yaxisname + '-axis:', self.yaxisbox)
		self.yaxisbox.addItems(self.getColumnNames())
		self.yaxisbox.setEditable(True)
		self.yaxisbox.lineEdit().setText(self.expressions[1])
		self.yaxisbox.editTextChanged.connect(self.onExpressionChangedY)		
		#self.yaxisbox.setCurrentIndex(self.currentYAxis)
		#self.yaxisbox.currentIndexChanged.connect(self.onYAxis)
		
		#layout.addWidget(self.mpl_toolbar, 1)
		layout.addLayout(self.form_layout, 0)
		
	def onExpressionChangedX(self, text):
		self.expressions[0] = str(text)
		self.compute()
		self.plot()
		
		
	def onExpressionChangedY(self, text):
		self.expressions[1] = str(text)
		self.compute()
		self.plot()

	
	def onZoomFit(self, *args):
		self.xmin_show, self.xmax_show = self.xmin, self.xmax = self.getxrange()
		self.ymin_show, self.ymax_show = self.ymin, self.ymax = self.getyrange()
		self.axes.set_xlim(self.xmin_show, self.xmax_show)
		self.axes.set_ylim(self.ymin_show, self.ymax_show)
		#self.canvas.draw()
		self.compute()
		self.plot()

	def onZoomUse(self, *args):
		self.xmin, self.xmax = self.xmin_show, self.xmax_show
		self.ymin, self.ymax = self.ymin_show, self.ymax_show
		#self.axes.set_xlim(self.xmin, self.xmax)
		#self.axes.set_ylim(self.ymin, self.ymax)
		#self.canvas.draw()
		self.compute()
		self.plot()

class ScatterPlotDialog_old(PlotDialog2d):
	def __init__(self, parent, dataset, xname=None, yname=None):
		super(ScatterPlotDialog, self).__init__(parent, dataset, xname, yname)
		
	def beforeCanvas(self, layout):
		self.addToolbar(layout) #, yselect=True, lasso=False)
		
	def onExpressionChangedX(self, text):
		self.xmin, self.xmax = None, None
		self.xmin_show, self.xmax_show = None, None
		super(ScatterPlotDialog, self).onExpressionChangedX(text)
	
	def onExpressionChangedY(self, text):
		self.ymin, self.ymax = None, None
		self.ymin_show, self.ymax_show = None, None
		super(ScatterPlotDialog, self).onExpressionChangedY(text)
		
	def getrange(self, expr):
		minvalue, maxvalue = None, None
		for block, info in self.dataset.evaluate(expr):
			if info.first:
				minvalue = np.nanmin(block)
				maxvalue = np.nanmax(block)
			else:
				minvalue = min(minvalue, np.nanmin(block))
				maxvalue = max(maxvalue, np.nanmax(block))
		return minvalue, maxvalue


	def getx_expr(self):
		return self.expressions[0]
		
	def gety_expr(self):
		return self.expressions[1]
		
	def getxrange(self):
		return self.getrange(self.getx_expr())
	def getyrange(self):
		return self.getrange(self.gety_expr())
		
	def compute(self):
		if self.xmin is None or self.xmax is None:
			self.xmin, self.xmax = self.getxrange()
			if self.xmin == self.xmax:
				self.xmax += 1
		if self.ymin is None or self.ymax is None:
			self.ymin, self.ymax = self.getyrange()
			if self.ymin == self.ymax:
				self.ymax += 1
		if self.xmin_show is None or self.xmax_show is None:
			self.xmin_show, self.xmax_show = self.xmin, self.xmax
		if self.ymin_show is None or self.ymax_show is None:
			self.ymin_show, self.ymax_show = self.ymin, self.ymax
		#if self.dataset.mask is not None:

		self.ranges = self.xmin, self.xmax, self.ymin, self.ymax
		#self.ranges = np.nanmin(datax), np.nanmax(datax), np.nanmin(datay), np.nanmax(datay)
		#self.ranges
		t0 = time.time()
		if 0: # BUG: parallelize doesn't work well with numba, maybe precompile first?
			parts = 20
			Iparts = np.zeros((parts, 128, 128))
			Iparts = mab.utils.numpy.mmapzeros((parts, 128, 128), dtype=np.int32)
			@parallelize(cores=QtCore.QThread.idealThreadCount())
			def histo2d_par(part):
				N = len(datax)
				i1 = (N / parts) * part
				i2 = (N / parts) * (part+1)
				#print i1, i2,datax[i1:i2],datay[i1:i2], Iparts[part], ranges
				try:
					histo2d(datax[i1:i2], datay[i1:i2], Iparts[part], *self.ranges)
				except TypeError:
					print histo2d.inspect_types()
					args = (datax[i1:i2], datay[i1:i2], Iparts[part]) + self.ranges
					for i, arg in enumerate(args):
						print i, repr(arg), arg.dtype if hasattr(arg, "dtype") else ""
					
				#histo2d(datax[part::parts], datay[part::parts], Iparts[part], np.min(datax), np.max(datax), np.min(datay), np.max(datay))
			histo2d_par(range(parts))
			I = np.sum(Iparts, axis=0)
		else:
			I = np.zeros((128, 128), dtype=np.int32)
			for (blockx, blocky), info in self.dataset.evaluate(self.getx_expr(), self.gety_expr()):
				histo2d(blockx, blocky, I, *self.ranges)
		print "it took", time.time()-t0
		self.counts = I
		mask = self.dataset.mask
		if mask is not None: #dataxsel is not None:
			self.counts_mask = np.zeros((128, 128), dtype=np.int32)
			for (blockx, blocky), info in self.dataset.evaluate(self.getx_expr(), self.gety_expr()):
				print "block info", info.index, info.size, info.i1, info.i2
				subx = blockx[mask[info.i1:info.i2]]
				suby = blocky[mask[info.i1:info.i2]]
				print subx, suby, mask[info.i1:info.i2]
				histo2d(subx, suby, self.counts_mask, *self.ranges)
		else:
			self.counts_mask = None
		
		#I, x, y = np.histogram2d(self.dataset.columns[x], self.dataset.columns[y], bins=128)
		#print res
		#I = res
		
	def plot(self):
		self.axes.cla()
		#extent = 
		#ranges = np.nanmin(datax), np.nanmax(datax), np.nanmin(datay), np.nanmax(datay)
		self.axes.imshow(np.log10(self.counts.T), origin="lower", extent=self.ranges, alpha=1 if self.counts_mask is None else 0.4)
		if self.counts_mask is not None:
			self.axes.imshow(np.log10(self.counts_mask.T), origin="lower", extent=self.ranges, alpha=1)
		#self.axes.imshow((I), origin="lower", extent=ranges)
		self.axes.set_aspect('auto')
		if self.dataset.selected_row_index is not None:
			#self.axes.autoscale(False)
			x, y = self.getdatax()[self.dataset.selected_row_index],  self.getdatay()[self.dataset.selected_row_index]
			print "drawing selected point at", x, y
			self.axes.scatter([x], [y], color='red') #, scalex=False, scaley=False)
		#if dataxsel is not None:
		#	self.axes.scatter(dataxsel, dataysel)
		#self.axes.set_xlabel(self.xaxisName)
		#self.axes.set_ylabel(self.yaxisName)
		print "plot limits:", self.xmin, self.xmax
		self.axes.set_xlim(self.xmin_show, self.xmax_show)
		self.axes.set_ylim(self.ymin_show, self.ymax_show)
		self.canvas.draw()
	
class SerieSlicePlotDialog(ScatterPlotDialog):
	def __init__(self, parent, dataset, xname=None, yname=None):
		self.serieIndex = 0 if dataset.selected_serie_index is None else dataset.selected_serie_index
		super(SerieSlicePlotDialog, self).__init__(parent, dataset, xname, yname)
		self.dataset.serie_index_selection_listeners.append(self.onSerieIndexSelect)
	
	def afterCanvas(self, layout):
		super(SerieSlicePlotDialog, self).afterCanvas(layout)
		
		self.form_layout = QtGui.QFormLayout(self)

		
		self.nSlices = self.datax.shape[0]
		self.seriesbox = QtGui.QComboBox(self)
		self.seriesbox.addItems([str(k) for k in range(self.nSlices)])
		self.seriesbox.setCurrentIndex(self.serieIndex)
		self.seriesbox.currentIndexChanged.connect(self.onSerieIndex)
		
		self.form_layout.addRow("index", self.seriesbox)
		self.buttonLoop = QtGui.QToolButton(self)
		self.buttonLoop.setText("one loop")
		self.buttonLoop.clicked.connect(self.onPlayOnce)
		self.form_layout.addRow("movie", self.buttonLoop)
		layout.addLayout(self.form_layout, 0)
		
	def onPlayOnce(self):
		#self.timer = QtCore.QTimer(self)
		#self.timer.timeout.connect(self.onNextFrame)
		self.delay = 10
		self.dataset.selectSerieIndex(0)
		QtCore.QTimer.singleShot(self.delay, self.onNextFrame);
		
	def onNextFrame(self, *args):
		#print args
		step = 15
		next = self.serieIndex +step
		if next >= self.nSlices:
			next = self.nSlices-1
		self.dataset.selectSerieIndex(next)
		if self.serieIndex < self.nSlices-1 : # not last frame
			QtCore.QTimer.singleShot(self.delay, self.onNextFrame);
			
			
	def onSerieIndex(self, index):
		if index != self.dataset.selected_serie_index: # avoid unneeded event
			self.dataset.selectSerieIndex(index)
		
	def onSerieIndexSelect(self, serie_index):
		if serie_index != self.serieIndex: # avoid unneeded event
			self.serieIndex = serie_index
			self.seriesbox.setCurrentIndex(self.serieIndex)
		else:
			self.serieIndex = serie_index
		self.compute()
		self.plot()
		
	def getColumnNames(self):
		return self.__dict__['dataset'].rank1names
	
	def getColumnDict(self):
		return self.dataset.rank1s

	def getdatax(self):
		return self.datax[self.serieIndex,slice(*self.dataset.current_slice)]
	def getdatay(self):
		return self.datay[self.serieIndex,slice(*self.dataset.current_slice)]

class ScatterPlotDialog_old(QtGui.QDialog):
	def __init__(self, parent, dataset, xname=None, yname=None, width=5, height=4, dpi=100):
		super(ScatterPlotDialog, self).__init__(parent)
		self.resize(700,700)
		self.dataset = dataset
		
		self.fig = Figure(figsize=(width, height), dpi=dpi)
		self.axes = self.fig.add_subplot(111)
		self.axes.hold(True)
		
		print "scatter plot", xname, yname
		self.currentXIndex = 0 if xname is None else self.dataset.column_names.index(xname)
		self.currentYIndex = 1 if yname is None else self.dataset.column_names.index(yname) 
		print "scatter plot", self.currentXIndex, self.currentYIndex
		
		x = self.dataset.column_names[self.currentXIndex]
		y = self.dataset.column_names[self.currentYIndex]
		self.datax = self.dataset.columns[x][slice(*self.dataset.current_slice)]
		self.datay = self.dataset.columns[y][slice(*self.dataset.current_slice)]
		
		
		
		self.canvas =  FigureCanvas(self.fig)
		self.canvas.setParent(self)
		
		self.selected_row = None

		self.mask = None
		self.plot()
		FigureCanvas.setSizePolicy(self,
									QtGui.QSizePolicy.Expanding,
									QtGui.QSizePolicy.Expanding)
		FigureCanvas.updateGeometry(self)
		
		#self.mpl_toolbar = NavigationToolbar(self.canvas, self.axes, self)

		self.boxlist = QtGui.QVBoxLayout(self)
		
		self.form_layout = QtGui.QFormLayout()
		
		self.xname = QtGui.QComboBox(self)
		self.form_layout.addRow('X-axis:', self.xname)
		self.xname.addItems(self.dataset.column_names)
		self.xname.setCurrentIndex(self.currentXIndex)
		self.xname.currentIndexChanged.connect(self.onX)
		
		self.yname = QtGui.QComboBox(self)
		self.form_layout.addRow('Y-axis:', self.yname)
		self.yname.addItems(self.dataset.column_names)
		self.yname.setCurrentIndex(self.currentYIndex)
		self.yname.currentIndexChanged.connect(self.onY)
		
		self.boxlist.addWidget(self.canvas, 1)
		#self.boxlist.addWidget(self.mpl_toolbar, 1)
		self.boxlist.addLayout(self.form_layout, 0)
		self.setLayout(self.boxlist)
		
		#self.fig.canvas.mpl_connect('button_press_event', self.onclick)
		self.dataset.row_selection_listeners.append(self.onRowSelect)
		import matplotlib.widgets
		#self.lasso = matplotlib.widgets.LassoSelector(self.axes, self.onLassoSelect)
		#self.rectselect = matplotlib.widgets.RectangleSelector(self.axes, self.onLassoSelect)
		#self.spanselect = matplotlib.widgets.SpanSelector(self.axes, self.onLassoSelect, 'horizontal')
		self.dataset.mask_listeners.append(self.onMaskSelect)
		
	def onMaskSelect(self, mask):
		self.mask = mask
		self.plot()

	def set_mask(self, mask):
		self.dataset.selectMask(mask)
		
	def onLassoSelect(self, *args):
		print args
		
	def press_select_point(self, event):
		print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'% (event.button, event.x, event.y, event.xdata, event.ydata)
		px, py = event.xdata, event.ydata
		x = self.dataset.column_names[self.currentXIndex]
		self.datax = self.dataset.columns[x][slice(*self.dataset.current_slice)]
		y = self.dataset.column_names[self.currentYIndex]
		self.datay = self.dataset.columns[y][slice(*self.dataset.current_slice)]
		#print self.datax, self.datay
		selected_row = find_nearest_index(self.datax, self.datay, px, py)
		print "nearest row", selected_row, self.datax[selected_row], self.datay[selected_row]
		self.dataset.selectRow(selected_row)
		
	def onRowSelect(self, row):
		print "row selected", row
		self.selected_row = row
		x = self.dataset.column_names[self.currentXIndex]
		self.datax = self.dataset.columns[x][slice(*self.dataset.current_slice)]
		y = self.dataset.column_names[self.currentYIndex]
		self.datay = self.dataset.columns[y][slice(*self.dataset.current_slice)]
		self.plot()
		
	def onX(self, index):
		print "x changed", index
		self.currentXIndex = index
		x = self.dataset.column_names[self.currentXIndex]
		self.datax = self.dataset.columns[x][slice(*self.dataset.current_slice)]
		self.plot()
	
	def onY(self, index):
		print "y changed", index
		self.currentYIndex = index
		y = self.dataset.column_names[self.currentYIndex]
		self.datay = self.dataset.columns[y][slice(*self.dataset.current_slice)]
		self.plot()
		


	def plot(self):
		self.axes.cla()
		#x = np.arange(0,10,0.01)
		#y = x**x
		
		#x = self.dataset.column_names[self.currentXIndex]
		#y = self.dataset.column_names[self.currentYIndex]

		dataxsel, dataysel = None, None
		datax = self.datax 
		datay = self.datay
		if self.mask is not None:
			#print self.mask
			#print "sum of mask", sum(self.mask)
			dataxsel = datax[self.mask]
			dataysel = datay[self.mask]
			#print dataxsel
			#print len(dataxsel)
		
		ranges = np.nanmin(datax), np.nanmax(datax), np.nanmin(datay), np.nanmax(datay)
		t0 = time.time()
		if 1:
			parts = 20
			Iparts = np.zeros((parts, 128, 128))
			Iparts = mab.utils.numpy.mmapzeros((parts, 128, 128), dtype=np.int32)
			@parallelize(cores=QtCore.QThread.idealThreadCount())
			def histo2d_par(part):
				N = len(datax)
				i1 = (N / parts) * part
				i2 = (N / parts) * (part+1)
				#print i1, i2,datax[i1:i2],datay[i1:i2], Iparts[part], ranges
				histo2d(datax[i1:i2], datay[i1:i2], Iparts[part], *ranges)
				#histo2d(datax[part::parts], datay[part::parts], Iparts[part], np.min(datax), np.max(datax), np.min(datay), np.max(datay))
			histo2d_par(range(parts))
			I = np.sum(Iparts, axis=0)
		else:
			I = np.zeros((parts, 128, 128), dtype=np.int32)
			histo2d(datax, datay, Iparts, np.min(datax), np.max(datax), np.min(datay), np.max(datay))
			I = np.sum(I, axis=0)
		print "it took", time.time()-t0
		if dataxsel is not None:
			parts = 20
			Iparts = np.zeros((parts, 128, 128))
			Iparts = mab.utils.numpy.mmapzeros((parts, 128, 128), dtype=np.int32)
			@parallelize(cores=QtCore.QThread.idealThreadCount())
			def histo2d_par(part):
				N = len(dataxsel)
				i1 = (N / parts) * part
				i2 = (N / parts) * (part+1)
				#print i1, i2,dataxsel[i1:i2],dataysel[i1:i2], Iparts[part], ranges
				histo2d(dataxsel[i1:i2], dataysel[i1:i2], Iparts[part], *ranges)
				#histo2d(datax[part::parts], datay[part::parts], Iparts[part], np.min(datax), np.max(datax), np.min(datay), np.max(datay))
			histo2d_par(range(parts))
			Is = np.sum(Iparts, axis=0)
		
		#I, x, y = np.histogram2d(self.dataset.columns[x], self.dataset.columns[y], bins=128)
		#print res
		#I = res
		
		self.axes.imshow(np.log10(I.T), origin="lower", extent=ranges, alpha=1 if dataxsel is None else 0.4)
		if dataxsel is not None:
			self.axes.imshow(np.log10(Is.T), origin="lower", extent=ranges, alpha=1)
		#self.axes.imshow((I), origin="lower", extent=ranges)
		self.axes.set_aspect('auto')
		if self.selected_row is not None:
			#self.axes.autoscale(False)
			x, y = self.datax[self.selected_row],  self.datay[self.selected_row]
			print "drawing selected point at", x, y
			self.axes.scatter([x], [y], color='red') #, scalex=False, scaley=False)
		#if dataxsel is not None:
		#	self.axes.scatter(dataxsel, dataysel)
		self.canvas.draw()
		
		
class SerieSlicePlotDialog_(QtGui.QDialog):
	def __init__(self, parent, data, width=5, height=4, dpi=100):
		super(SerieSlicePlotDialog, self).__init__(parent)
		self.resize(500,500)
		self.data = data
		
		self.fig = Figure(figsize=(width, height), dpi=dpi)
		self.axes = self.fig.add_subplot(111)
		self.axes.hold(True)
		
		self.currentXIndex = 0
		self.currentYIndex = 1
		
		self.serieIndex = 0
		
		x = self.data.rank1names[self.currentXIndex]
		y = self.data.rank1names[self.currentYIndex]
		self.datax = self.data.rank1s[x][self.serieIndex,slice(*self.data.current_slice)]
		self.datay = self.data.rank1s[y][self.serieIndex,slice(*self.data.current_slice)]
		
		
		
		self.canvas =  FigureCanvas(self.fig)
		self.canvas.setParent(self)
		
		self.selected_row = None

		self.plot()
		FigureCanvas.setSizePolicy(self,
									QtGui.QSizePolicy.Expanding,
									QtGui.QSizePolicy.Expanding)
		FigureCanvas.updateGeometry(self)

		self.boxlist = QtGui.QVBoxLayout(self)
		
		self.form_layout = QtGui.QFormLayout()
		
		self.xname = QtGui.QComboBox(self)
		self.form_layout.addRow('X-axis:', self.xname)
		self.xname.addItems(self.data.rank1names)
		self.xname.currentIndexChanged.connect(self.onX)
		
		self.yname = QtGui.QComboBox(self)
		self.form_layout.addRow('Y-axis:', self.yname)
		self.yname.addItems(self.data.rank1names)
		self.yname.setCurrentIndex(1)
		self.yname.currentIndexChanged.connect(self.onY)
		
		self.boxlist.addWidget(self.canvas, 1)
		self.boxlist.addLayout(self.form_layout, 0)
		self.setLayout(self.boxlist)
		
		self.fig.canvas.mpl_connect('button_press_event', self.onclick)
		self.data.row_selection_listeners.append(self.onRowSelect)
		self.data.serie_index_selection_listeners.append(self.onSerieIndexSelect)
		
	def onclick(self, event):
		print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'% (event.button, event.x, event.y, event.xdata, event.ydata)
		x, y = event.xdata, event.ydata
		selected_row = find_nearest_index(self.datax, self.datay, x, y)
		print "nearest row", selected_row, self.datax[selected_row], self.datay[selected_row]
		self.data.selectRow(selected_row)
		
	def onRowSelect(self, row):
		print "row selected", row
		self.selected_row = row
		self.plot()
		
	def onSerieIndexSelect(self, serie_index):
		print "series index selected", serie_index
		self.serieIndex = serie_index
		x = self.data.rank1names[self.currentXIndex]
		y = self.data.rank1names[self.currentYIndex]
		self.datax = self.data.rank1s[x][self.serieIndex,slice(*self.data.current_slice)]
		self.datay = self.data.rank1s[y][self.serieIndex,slice(*self.data.current_slice)]
		
		self.plot()
		
	def onX(self, index):
		print "x changed", index
		self.currentXIndex = index
		x = self.data.rank1names[self.currentXIndex]
		self.datax = self.data.rank1s[x][self.serieIndex,slice(*self.data.current_slice)]
		self.plot()
	
	def onY(self, index):
		print "y changed", index
		self.currentYIndex = index
		y = self.data.rank1names[self.currentYIndex]
		y = self.data.rank1names[self.currentYIndex]
		self.datay = self.data.rank1s[y][self.serieIndex,slice(*self.data.current_slice)]
		self.plot()
		


	def plot(self):
		self.axes.cla()
		#x = np.arange(0,10,0.01)
		#y = x**x
		
		#x = self.data.column_names[self.currentXIndex]
		#y = self.data.column_names[self.currentYIndex]

		datax = self.datax 
		datay = self.datay 
		ranges = np.nanmin(datax), np.nanmax(datax), np.nanmin(datay), np.nanmax(datay)
		t0 = time.time()
		if 1:
			parts = 20
			Iparts = np.zeros((parts, 128, 128))
			Iparts = mab.utils.numpy.mmapzeros((parts, 128, 128), dtype=np.int32)
			@parallelize(cores=QtCore.QThread.idealThreadCount())
			def histo2d_par(part):
				N = self.data._length
				i1 = (N / parts) * part
				i2 = (N / parts) * (part+1)
				#print i1, i2
				histo2d(datax[i1:i2], datay[i1:i2], Iparts[part], *ranges)
				#histo2d(datax[part::parts], datay[part::parts], Iparts[part], np.min(datax), np.max(datax), np.min(datay), np.max(datay))
			histo2d_par(range(parts))
			I = np.sum(Iparts, axis=0)
		else:
			I = np.zeros((parts, 128, 128), dtype=np.int32)
			histo2d(datax, datay, Iparts, np.min(datax), np.max(datax), np.min(datay), np.max(datay))
			I = np.sum(I, axis=0)
		print "it took", time.time()-t0
		
		#I, x, y = np.histogram2d(self.data.columns[x], self.data.columns[y], bins=128)
		#print res
		#I = res
		
		self.axes.imshow(np.log10(I.T), origin="lower", extent=ranges)
		#self.axes.imshow((I), origin="lower", extent=ranges)
		self.axes.set_aspect('auto')
		if self.selected_row is not None:
			#self.axes.autoscale(False)
			x, y = self.datax[self.selected_row],  self.datay[self.selected_row]
			print "drawing selected point at", x, y
			self.axes.scatter([x], [y], color='red') #, scalex=False, scaley=False)
		self.canvas.draw()		

class ScatterSeries2dPlotDialog(QtGui.QDialog):
	def __init__(self, parent, data, width=5, height=4, dpi=100):
		super(ScatterSeries2dPlotDialog, self).__init__(parent)
		self.resize(500,500)
		self.data = data
		
		self.fig = Figure(figsize=(width, height), dpi=dpi)
		self.axes = self.fig.add_subplot(111)
		self.axes.hold(True)
		
		self.currentXIndex = 0
		self.currentYIndex = 1
		
		x = self.data.rank1names[self.currentXIndex]
		y = self.data.rank1names[self.currentYIndex]
		self.selected_serie_index = None
		
		xseries = self.data.rank1s[x]
		yseries = self.data.rank1s[y]
		
		self.datax = xseries[:,0 if self.data.selected_row_index is None else self.data.selected_row_index]
		self.datay = yseries[:,0 if self.data.selected_row_index is None else self.data.selected_row_index]

		
		self.canvas =  FigureCanvas(self.fig)
		self.canvas.setParent(self)

		self.plot()
		FigureCanvas.setSizePolicy(self,
									QtGui.QSizePolicy.Expanding,
									QtGui.QSizePolicy.Expanding)
		FigureCanvas.updateGeometry(self)

		self.boxlist = QtGui.QVBoxLayout(self)
		
		self.form_layout = QtGui.QFormLayout()
		
		self.xname = QtGui.QComboBox(self)
		self.form_layout.addRow('X-axis:', self.xname)
		self.xname.addItems(self.data.rank1names)
		self.xname.currentIndexChanged.connect(self.onX)
		
		self.yname = QtGui.QComboBox(self)
		self.form_layout.addRow('Y-axis:', self.yname)
		self.yname.addItems(self.data.rank1names)
		self.yname.setCurrentIndex(1)
		self.yname.currentIndexChanged.connect(self.onY)
		
		self.boxlist.addWidget(self.canvas, 1)
		self.boxlist.addLayout(self.form_layout, 0)
		self.setLayout(self.boxlist)
		self.data.row_selection_listeners.append(self.onRowSelect)
		
		self.fig.canvas.mpl_connect('button_press_event', self.onclick)
		self.data.row_selection_listeners.append(self.onRowSelect)
		self.data.serie_index_selection_listeners.append(self.onSerieSelect)
		
	def onclick(self, event):
		print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'% (event.button, event.x, event.y, event.xdata, event.ydata)
		x, y = event.xdata, event.ydata
		selected_index = find_nearest_index(self.datax, self.datay, x, y)
		print "nearest selected_index", selected_index, self.datax[selected_index], self.datay[selected_index]
		self.data.selectSerieIndex(selected_index)
		
	def onSerieSelect(self, serie_index):
		self.selected_serie_index = serie_index
		self.plot()
		
	def onRowSelect(self, row):
		x = self.data.rank1names[self.currentXIndex]
		y = self.data.rank1names[self.currentYIndex]
		xseries = self.data.rank1s[x]
		yseries = self.data.rank1s[y]
		self.datax = xseries[:,self.data.selected_row_index]
		self.datay = yseries[:,self.data.selected_row_index]
		self.plot()
		
	def onX(self, index):
		print "x changed", index
		self.currentXIndex = index
		x = self.data.rank1names[self.currentXIndex]
		xseries = self.data.rank1s[x]
		self.datax = xseries[:,self.data.selected_row_index]

		self.plot()
	
	def onY(self, index):
		print "y changed", index
		self.currentYIndex = index
		y = self.data.rank1names[self.currentYIndex]
		yseries = self.data.rank1s[y]
		self.datay = yseries[:,self.data.selected_row_index]
		self.plot()


	def plot(self):
		self.axes.cla()
		#x = np.arange(0,10,0.01)
		#y = x**x
		

		self.axes.plot(self.datax, self.datay)
		self.axes.scatter(self.datax, self.datay)
		#self.axes.imshow(xseries, origin="lower") #, extent=ranges)
		#self.axes.set_aspect('auto')
		if 0:
			#datax = self.data.columns[x][slice(*self.data.current_slice)]
			#datay = self.data.columns[y][slice(*self.data.current_slice)]
			#ranges = np.nanmin(datax), np.nanmax(datax), np.nanmin(datay), np.nanmax(datay)
			t0 = time.time()
			if 1:
				parts = 20
				Iparts = np.zeros((parts, 128, 128))
				Iparts = mab.utils.numpy.mmapzeros((parts, 128, 128), dtype=np.int32)
				@parallelize(cores=QtCore.QThread.idealThreadCount())
				def histo2d_par(part):
					N = self.data._length
					i1 = (N / parts) * part
					i2 = (N / parts) * (part+1)
					#print i1, i2
					histo2d(datax[i1:i2], datay[i1:i2], Iparts[part], *ranges)
					#histo2d(datax[part::parts], datay[part::parts], Iparts[part], np.min(datax), np.max(datax), np.min(datay), np.max(datay))
				histo2d_par(range(parts))
				I = np.sum(Iparts, axis=0)
			else:
				I = np.zeros((parts, 128, 128), dtype=np.int32)
				histo2d(datax, datay, Iparts, np.min(datax), np.max(datax), np.min(datay), np.max(datay))
				I = np.sum(I, axis=0)
			print "it took", time.time()-t0
			
			#I, x, y = np.histogram2d(self.data.columns[x], self.data.columns[y], bins=128)
			#print res
			#I = res
			
			#self.axes.imshow(np.log10(I), origin="lower", extent=ranges)
			self.axes.imshow((I), origin="lower", extent=ranges)
			self.axes.set_aspect('auto')
		if self.selected_serie_index is not None:
			#self.axes.autoscale(False)
			x, y = self.datax[self.selected_serie_index],  self.datay[self.selected_serie_index]
			print "drawing selected point at", x, y
			self.axes.scatter([x], [y], color='red') #, scalex=False, scaley=False)
		self.canvas.draw()


class ScatterSeries1dPlotDialog(QtGui.QDialog):
	def __init__(self, parent, data, width=5, height=4, dpi=100):
		super(ScatterSeries1dPlotDialog, self).__init__(parent)
		self.resize(500,500)
		self.data = data
		
		self.fig = Figure(figsize=(width, height), dpi=dpi)
		self.axes = self.fig.add_subplot(111)
		self.axes.hold(True)
		
		self.currentYIndex = 1
		self.selected_serie_index = None
		y = self.data.rank1names[self.currentYIndex]
		yseries = self.data.rank1s[y]
		self.datay = yseries[:,0 if self.data.selected_row_index is None else self.data.selected_row_index]
		
		
		self.canvas =  FigureCanvas(self.fig)
		self.canvas.setParent(self)

		self.plot()
		FigureCanvas.setSizePolicy(self,
									QtGui.QSizePolicy.Expanding,
									QtGui.QSizePolicy.Expanding)
		FigureCanvas.updateGeometry(self)

		self.boxlist = QtGui.QVBoxLayout(self)
		
		self.form_layout = QtGui.QFormLayout()
		
		self.yname = QtGui.QComboBox(self)
		self.form_layout.addRow('Y-axis:', self.yname)
		self.yname.addItems(self.data.rank1names)
		self.yname.setCurrentIndex(1)
		self.yname.currentIndexChanged.connect(self.onY)
		
		self.boxlist.addWidget(self.canvas, 1)
		self.boxlist.addLayout(self.form_layout, 0)
		self.setLayout(self.boxlist)
		self.fig.canvas.mpl_connect('button_press_event', self.onclick)
		self.data.row_selection_listeners.append(self.onRowSelect)
		self.data.serie_index_selection_listeners.append(self.onSerieIndexSelect)

	def onclick(self, event):
		print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'% (event.button, event.x, event.y, event.xdata, event.ydata)
		x, y = event.xdata, event.ydata
		##selected_index = find_nearest_index1d(self.datay, y)
		selected_index = int(x)
		print "nearest selected_index", selected_index, self.datay[selected_index]
		self.data.selectSerieIndex(selected_index)
		
	def onRowSelect(self, row):
		y = self.data.rank1names[self.currentYIndex]
		yseries = self.data.rank1s[y]
		self.datay = yseries[:,self.data.selected_row_index]
		self.plot()
		
	def onSerieIndexSelect(self, serie_index):
		self.selected_serie_index = serie_index
		self.plot()
		
	def onY(self, index):
		print "y changed", index
		self.currentYIndex = index
		y = self.data.rank1names[self.currentYIndex]
		yseries = self.data.rank1s[y]
		self.datay = yseries[:,0 if self.data.selected_row_index is None else self.data.selected_row_index]
		print ">", self.datay.shape, self.data.selected_row_index
		print yseries.shape
		print yseries[:,self.data.selected_row_index].shape
		self.plot()


	def plot(self):
		self.axes.cla()
		#x = np.arange(0,10,0.01)
		#y = x**x
		


		self.axes.plot(self.datay)
		#self.axes.imshow(xseries, origin="lower") #, extent=ranges)
		#self.axes.set_aspect('auto')
		if 0:
			#datax = self.data.columns[x][slice(*self.data.current_slice)]
			#datay = self.data.columns[y][slice(*self.data.current_slice)]
			#ranges = np.nanmin(datax), np.nanmax(datax), np.nanmin(datay), np.nanmax(datay)
			t0 = time.time()
			if 1:
				parts = 20
				Iparts = np.zeros((parts, 128, 128))
				Iparts = mab.utils.numpy.mmapzeros((parts, 128, 128), dtype=np.int32)
				@parallelize(cores=QtCore.QThread.idealThreadCount())
				def histo2d_par(part):
					N = self.data._length
					i1 = (N / parts) * part
					i2 = (N / parts) * (part+1)
					#print i1, i2
					histo2d(datax[i1:i2], datay[i1:i2], Iparts[part], *ranges)
					#histo2d(datax[part::parts], datay[part::parts], Iparts[part], np.min(datax), np.max(datax), np.min(datay), np.max(datay))
				histo2d_par(range(parts))
				I = np.sum(Iparts, axis=0)
			else:
				I = np.zeros((parts, 128, 128), dtype=np.int32)
				histo2d(datax, datay, Iparts, np.min(datax), np.max(datax), np.min(datay), np.max(datay))
				I = np.sum(I, axis=0)
			print "it took", time.time()-t0
			
			#I, x, y = np.histogram2d(self.data.columns[x], self.data.columns[y], bins=128)
			#print res
			#I = res
			
			#self.axes.imshow(np.log10(I), origin="lower", extent=ranges)
			self.axes.imshow((I), origin="lower", extent=ranges)
			self.axes.set_aspect('auto')
		if self.selected_serie_index is not None:
			x, y = self.selected_serie_index, self.datay[self.selected_serie_index]
			print "drawing selected point at", x, y
			self.axes.scatter([x], [y], color='red') #, scalex=False, scaley=False)
			
		self.canvas.draw()


