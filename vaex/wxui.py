# -*- coding: utf-8 -*-
import os
import wx
import matplotlib
matplotlib.use('WXAgg')
import numpy as np

# local fix 
maartenfix = os.environ["USER"] == "breddels"

if maartenfix:
	from backend_wxagg import FigureCanvasWxAgg as FigureCanvas
	from backend_wx import NavigationToolbar2Wx
else:
	from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
	from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.figure import Figure


class PlotWindow1d(wx.Panel):
	def __init__(self, parent):
		wx.Panel.__init__(self, parent)
		self.sizer = wx.BoxSizer(wx.VERTICAL)
		self.figure = Figure()
		self.axes = self.figure.add_subplot(111)
		self.canvas2d = FigureCanvas(self, -1, self.figure)
		#self.sizerInfo = wx.BoxSizer(wx.HORIZONTAL)
		#self.canvas2d.Hide()
		
		##t = np.arange(0.0, 3.0, 0.01)
		#s = np.sin(2 * np.pi * t)
		##self.axes.plot(t, s)
		
		
		self.sizer.Add(self.canvas2d)
		self.SetSizer(self.sizer)
		
	def plot1d(self, data, xmin, xmax, xlabel, ylabel="density", clear=True):
		if clear:
			self.axes.cla()
		N = len(data)
		dx = (xmax - xmin)/N
		x = np.linspace(xmin, xmax, N, endpoint=False) + dx/2
		self.axes.set_aspect('auto')
		self.axes.plot(x, data)
		self.axes.set_xlabel(xlabel)
		self.axes.set_ylabel(ylabel)
		

	def plot2d(self, density, xmin, xmax, ymin, ymax, xlabel, ylabel, clear=True, log=True):
		if clear:
			self.axes.cla()
		#Nx = len(x)
		
		minima = [xmin, ymin]
		maxima = [xmax, ymax]
		I = np.log10(density)
		I -= I.max()
		I[I<-4] = -4
		#print minima
		self.axes.imshow(I, origin="lower", extent=[minima[0], maxima[0], minima[1], maxima[1]])
		self.axes.set_aspect('auto')
		self.axes.set_xlabel(xlabel)
		self.axes.set_ylabel(ylabel)
		
		