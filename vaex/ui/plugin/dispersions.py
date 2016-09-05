__author__ = 'maartenbreddels'

import functools

import matplotlib.patches as patches
import numpy as np
import matplotlib.artist as artist

import vaex.ui.plugin
from vaex.ui.qt import *
import logging


logger = logging.getLogger("plugin.dispersions")


import matplotlib.transforms as transforms
from matplotlib.path import Path

class DispersionEllipse(patches.Patch):
	"""
	This ellipse has it's center in user coordinates, and the width and height in device coordinates
	such that is is not deformed
	"""
	def __str__(self):
		return "DispersionEllipse(%s,%s;%sx%s)" % (self.center[0], self.center[1],
										 self.width, self.height)

	#@docstring.dedent_interpd
	def __init__(self, xy, width, height, scale=1.0, angle=0.0, **kwargs):
		"""
		*xy*
		  center of ellipse

		*width*
		  total length (diameter) of horizontal axis

		*height*
		  total length (diameter) of vertical axis

		*angle*
		  rotation in degrees (anti-clockwise)

		Valid kwargs are:
		%(Patch)s
		"""
		patches.Patch.__init__(self, **kwargs)

		self.center = xy
		self.width, self.height = width, height
		self.scale = scale
		self.angle = angle
		self._path = Path.unit_circle()
		# Note: This cannot be calculated until this is added to an Axes
		self._patch_transform = transforms.IdentityTransform()

	def _recompute_transform(self):
		"""NOTE: This cannot be called until after this has been added
				 to an Axes, otherwise unit conversion will fail. This
				 maxes it very important to call the accessor method and
				 not directly access the transformation member variable.
		"""
		center = (self.convert_xunits(self.center[0]),
				  self.convert_yunits(self.center[1]))
		width = self.width #self.convert_xunits(self.width)
		height = self.height #self.convert_yunits(self.height)
		trans = artist.Artist.get_transform(self)
		self._patch_transform = transforms.Affine2D() \
			.scale(width * 0.5 * self.scale, height * 0.5* self.scale) \
			.rotate_deg(self.angle) \
			.translate(*trans.transform(center))

	def get_path(self):
		"""
		Return the vertices of the rectangle
		"""
		return self._path

	def get_transform(self):
		"""
		Return the :class:`~matplotlib.transforms.Transform` applied
		to the :class:`Patch`.
		"""
		return self.get_patch_transform()

	def get_patch_transform(self):
		self._recompute_transform()
		return self._patch_transform

	def contains(self, ev):
		if ev.x is None or ev.y is None:
			return False, {}
		x, y = self.get_transform().inverted().transform_point((ev.x, ev.y))
		return (x * x + y * y) <= 1.0, {}


class DispersionPlugin(vaex.ui.plugin.PluginLayer):
	name = "dispersion"
	def __init__(self, parent, layer):
		super(DispersionPlugin, self).__init__(parent, layer)
		dialog.plug_page(self.plug_page, "Dispersions", 2.25, 1.0)
		dialog.plug_grids(self.define_grids, self.draw_grids)

	def define_grids(self, grids):
		#grids.define_grid("counts_vector", self.dialog.gridsize_vector, "VZ*0+1")

		# covariance matrix terms
		# diagonals
		for dimension in range(self.dialog.dimensions):
			axis_name = self.dialog.axisnames[dimension].lower()
			expression = self.expressions[dimension].strip()
			if len(expression) > 0:
				grids.define_grid(axis_name + "_mom1", self.dialog.vector_grid_size, expression)
				grids.define_grid(axis_name + "_mom2", self.dialog.vector_grid_size, "(" + expression + ")**2")
			else:
				grids.define_grid(axis_name + "_mom1", self.dialog.vector_grid_size, None)
				grids.define_grid(axis_name + "_mom2", self.dialog.vector_grid_size, None)

		if 1:
			for dimension1 in range(self.dialog.dimensions):
				for dimension2 in range(dimension1+1, self.dialog.dimensions):
					axis_name1 = self.dialog.axisnames[dimension1].lower()
					axis_name2 = self.dialog.axisnames[dimension2].lower()
					expression1 = self.expressions[dimension1].strip()
					expression2 = self.expressions[dimension2].strip()
					if len(expression1) > 0 and  len(expression2) > 0:
						grids.define_grid("cov_" + axis_name1 +"_" +axis_name2, self.dialog.vector_grid_size, "(" + expression1 + ")*(" + expression2 +")")

	def draw_grids(self, axes, grid_map, grid_map_vector):
		if not self.dispersions_draw:
			return
		self.ellipses = []
		dispersions = []
		counts = grid_map_vector["counts"]
		#print "counts check", np.sum(counts), np.sum(grid_map["counts"])
		#print counts
		#print grid_map_vector.keys()
		if self.dialog.dimensions == 2:
			axis_name1 = self.dialog.axisnames[0].lower()
			axis_name2 = self.dialog.axisnames[1].lower()
			if len(self.expressions[0]) > 0 and len(self.expressions[1]) > 0:
				meanx = grid_map_vector[axis_name1 + "_mom1"]/counts
				meany = grid_map_vector[axis_name2 + "_mom1"]/counts
				varx  = grid_map_vector[axis_name1 + "_mom2"]/counts
				vary  = grid_map_vector[axis_name2 + "_mom2"]/counts
				covxy = grid_map_vector["cov_" +axis_name1 + "_" +axis_name2]/counts - meanx*meany
				sigmax = (varx-meanx**2)**0.5
				sigmay = (vary-meany**2)**0.5
				mask = counts > 0
				x = grid_map_vector["x"]
				y = grid_map_vector["y"]
				x, y = np.meshgrid(x, y)


				vmax = np.nanmax(np.sqrt(sigmax.reshape(-1)**2 + sigmay.reshape(-1)**2))

				width, height = self.dialog.canvas.get_width_height()
				#print "width,height", width, height
				max_size = min(width, height) / float(self.dialog.vector_grid_size)# * 0.9
				#print max_size
				#identity_transform = matplotlib.transforms.IdentityTransform()
				#deltax = self.dialog.ranges_show[0][1] - self.dialog.ranges_show[0][0]
				#deltay = self.dialog.ranges_show[1][1] - self.dialog.ranges_show[1][0]
				#aspect = deltay / float(height) / (deltax/float(width))
				#for grid in [x, y, sigmax, sigmay, covxy, counts, mask]:
				#	print grid.shape
				for x, y, sigmax, sigmay, covxy in zip(x[mask].reshape(-1), y[mask].reshape(-1), sigmax[mask].reshape(-1), sigmay[mask].reshape(-1), covxy[mask].reshape(-1)):
					try:
						covmatrix = [[sigmax**2, covxy], [covxy, sigmay**2]]
						eigen_values, eigen_vectors = np.linalg.eig(covmatrix)
					except:
						pass
					else:
						scaling = 1./vmax * max_size
						device_width =  (np.sqrt(np.max(eigen_values)) * scaling)
						device_height = (np.sqrt(np.min(eigen_values)) * scaling)
						if self.dispersions_unit_length:
							length = np.sqrt(device_width**2+device_height**2)
							device_width  /= float(length) / max_size
							device_height /= float(length) / max_size
						#ellipse_width = np.sqrt(np.max(eigen_values)) * scaling / width * deltax
						#ellipse_height = np.sqrt(np.min(eigen_values)) * scaling / height * deltay
						#ellipse_height /= aspect
						if sigmax < sigmay: # if x was smaller, the largest eigenvalue corresponds to the y value
							device_width, device_height = device_height, device_width
							#ellipse_width, ellipse_height = ellipse_height, ellipse_width
						#ellipse_height /= aspect
						angle = np.arctan(2*covxy / (sigmax**2-sigmay**2))/2.
						#angle2 = np.arctan(2*covxy / (sigmax**2-sigmay**2))/2.
						#angle = angle2 = 0
						#print aspect, sigmax, sigmay, sigmax/sigmay, covxy/(sigmax*sigmay), ellipse_width/ellipse_height
						#aspect = 0.1
						#m = [[np.cos(angle2), np.sin(angle2)*aspect], [-np.sin(angle2), np.cos(angle2)*aspect]]
						#ellipse_width, ellipse_height = np.dot(m, [ellipse_width, ellipse_height])
						#print covxy/(sigmax*sigmay), angle, sigmax, sigmay, covxy
						#device_x, device_y = axes.transData.transform((x, y))
						#print device_x, device_y, device_width, device_height
						#ellipse = patches.Ellipse(xy=(device_x, device_y), width=device_width, height=device_height, angle=angle, transform=identity_transform,
						#                         alpha=0.4, color="blue") #rand()*360
						#ellipse = patches.Ellipse(xy=(x, y), width=ellipse_width, height=ellipse_height, angle=np.degrees(angle),
						#                         alpha=0.4, color="blue") #rand()*360
						ellipse = DispersionEllipse(xy=(x, y), width=device_width, height=device_height, angle=np.degrees(angle), scale=self.scale_dispersion,
												  alpha=0.4, facecolor="green", edgecolor="black") #rand()*360

						axes.add_artist(ellipse)
						self.ellipses.append(ellipse)
						#axes.quiver()

						#[Ellipse(xy=rand(2)*10, width=rand(), height=rand(), angle=rand()*360)




	#@staticmethod
	#def useon(dialog_class):
	#	return issubclass(dialog_class, vaex.plot_windows.VolumeRenderingPlotDialog)

	def plug_page(self, page):
		layout = self.layout = QtGui.QGridLayout()
		page.setLayout(self.layout)
		layout.setSpacing(0)
		layout.setContentsMargins(0,0,0,0)
		layout.setAlignment(QtCore.Qt.AlignTop)


		row = 0

		self.dispersions_draw = bool(eval(self.dialog.options.get("disp_draw", "True")))
		def setter(value):
			self.dispersions_draw = value
			self.dialog.plot()
		self.dispersions_draw_checkbox = self.dialog.create_checkbox(page, "Draw dispersion tensors", lambda : self.dispersions_draw, setter)
		layout.addWidget(self.dispersions_draw_checkbox, row, 1)
		row += 1

		self.dispersions_unit_length = bool(eval(self.dialog.options.get("disp_unit", "False")))
		def setter(value):
			self.dispersions_unit_length = value
			self.dialog.plot()
		self.dispersions_unit_lengthcheckbox = self.dialog.create_checkbox(page, "Unit length", lambda : self.dispersions_unit_length, setter)
		layout.addWidget(self.dispersions_unit_lengthcheckbox, row, 1)
		row += 1


		self.expressions = []
		self.expression_boxes = []


		for dimension in range(self.dialog.dimensions):
			axis_name = self.dialog.axisnames[dimension]
			expression_box = QtGui.QComboBox(page)
			expression_box.setEditable(True)
			expression_box.setMinimumContentsLength(10)
			self.expression_boxes.append(expression_box)

			self.layout.addWidget(QtGui.QLabel(axis_name + '-axis:', page), row, 0)
			self.layout.addWidget(expression_box, row, 1, QtCore.Qt.AlignLeft)

			expression = self.dialog.options.get("disp"+axis_name.lower(), "")
			expression_box.lineEdit().setText(expression)
			self.expressions.append(expression)
			#self.onExpressionChangedPartials.append()
			#expression_box.lineEdit().editingFinished.connect(self.onExpressionChangedPartials[axisIndex])
			calllback = functools.partial(self.onExpressionChanged, axis_index=dimension)
			expression_box.lineEdit().editingFinished.connect(calllback)
			row += 1

		self.scale_dispersion = eval(self.dialog.options.get("disp_scale", "1"))
		def setter(value):
			self.scale_dispersion = value
			for ellipse in self.ellipses:
				ellipse.scale = self.scale_dispersion
			self.dialog.canvas.draw()
			#self.dialog.plot()
		self.scale_dispersion_label, self.scale_dispersion_slider, self.scale_dispersion_value_label =\
				self.dialog.create_slider(page, "scale: ", 1./100, 100., lambda : self.scale_dispersion, setter, format=" {0:>05.2f}", transform=lambda x: 10**x, inverse=lambda x: np.log10(x))
		layout.addWidget(self.scale_dispersion_label, row, 0)
		layout.addWidget(self.scale_dispersion_slider, row, 1)
		layout.addWidget(self.scale_dispersion_value_label, row, 2)
		row += 1


	def onExpressionChanged(self, _=None, axis_index=-1):
		text = str(self.expression_boxes[axis_index].lineEdit().text())
		logger.debug("text set for axis %i: %s" % (axis_index, text))
		if text != self.expressions[axis_index]:
			axis_name = self.dialog.axisnames[axis_index].lower()
			self.expressions[axis_index] = text
			if text == "": # check if we can replot without doing the whole calculation
				self.dialog.plot()
			else:
				non_empty = [k for k in self.expressions if len(k) > 0]
				if len(non_empty) == len(self.expressions):
					self.dialog.compute()
					self.dialog.jobsManager.execute()
		else:
			logger.debug("nothing changed")


