import vaex.vaexfast

import numpy as np

from vaex.ui.qt import *
import vaex.ui.colormaps


class HistogramAndTransfer(QtGui.QWidget):
	def __init__(self, parent, colormap, function_count=3):
		super(HistogramAndTransfer, self).__init__(parent)
		self.colormap = colormap
		self.grid = None
		self.setMinimumHeight(32+100*0)
		self.function_count = function_count
		self.function_opacities = [0.2/2**(function_count-1-k) for k in range(function_count)]
		self.function_sigmas = [0.025] * function_count
		self.function_means = list((np.arange(function_count) / float(function_count-1)) * 0.8 + 0.10)
		
		
	def paintEvent(self, event):
		painter = QtGui.QPainter()
		painter.begin(self)
		painter.fillRect(event.rect(), QtGui.QBrush(QtCore.Qt.black))
		
		self.draw_colormap(painter)
		if self.grid is not None:
			self.draw_histogram(painter)
		painter.end()
		
		
	def draw_histogram(self, painter):
		return
		grid1d = np.log10(self.grid.sum(axis=1).sum(axis=0).reshape(-1).astype(np.float64)+1)
		grid1d = np.log10(self.grid.reshape(-1).astype(np.float64)+1)
		xmin, xmax = grid1d.min(), grid1d.max()
		width = self.width()
		counts = np.zeros(width, dtype=np.float64)
		vaex.vaexfast.histogram1d(grid1d, None, counts, xmin, xmax+1)
		#counts, _ = np.histogram(grid1d, bins=width, range=(xmin, xmax))
		print(("histogram", xmin, xmax, counts, grid1d.mean(), len(grid1d)))
		counts = np.log10(counts+1)
		counts -= counts.min()
		counts /= counts.max()
		counts *= 100
		painter.setPen(QtCore.Qt.white)
		for i in range(width):
			painter.drawLine(i, 100, i, 100-counts[i])
			
		
	def draw_colormap(self, painter):
		rect = self.size()
		Nx, Ny = rect.width(), 32
		image, stringdata = vaex.ui.colormaps.colormap_to_QImage(self.colormap, Nx, Ny)
		point = QtCore.QPoint(0, rect.height()-32)
		painter.drawImage(point, image)
		
		painter.setPen(QtCore.Qt.white)
		for i in range(self.function_count):
			x = np.arange(Nx)
			nx = x / (Nx-1.)
			y = np.exp(-((nx-self.function_means[i])/self.function_sigmas[i])**2) * (np.log10(self.function_opacities[i])+3)/3 * 32.
			x = x.astype(np.int32)
			y = 100*0+32-y.astype(np.int32)
			polygon = QtGui.QPolygon([QtCore.QPoint(*x) for x in zip(x, y)])
			painter.drawPolyline(polygon)
		

		