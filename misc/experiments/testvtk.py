# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import vtk
try:
	from PyQt4 import QtCore, QtGui
except:
	from PySide import QtCore, QtGui
from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk.util.numpy_support
import numpy as np
import mmap
import h5py
import vaex.histogram
import scipy.spatial

		
class Hull(object):
	def __init__(self, pointlist):
		self.pointlist = pointlist
		
		
	def update(self):
		self.hull.Reset()
		for x, y in self.pointlist:
			self.hull.InsertNextPoint(x, y, 0)
		
		size = self.hull.GetSizeCCWHullZ();
		print(dir(self.hull))
		print(self.hull.GetCCWHullX())

if 0:
	points = zip(np.random.normal(100, 10, 25), np.random.normal(150, 10, 25))
	points = np.array(points)
	hull = scipy.spatial.ConvexHull(points)
	for simplex in hull.simplices:
		print(simplex)
	hull = scipy.spatial.ConvexHull(points)
	for simplex in hull.simplices:
		print(simplex)
	import matplotlib.pyplot as plt
	plt.scatter(points[:,0], points[:,1])
	plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
	plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
	plt.show()
	#plt.savefig("hull.png")
	#hull = Hull(points)
	#hull.update()
	sys.exit(0)

class MemoryMapped(object):
	def __init__(self, filename):
		self.filename = filename
		self.name = self.filename
		self.file = file(self.filename, "r")
		self.fileno = self.file.fileno()
		self.mapping = mmap.mmap(self.fileno, 0, prot=mmap.PROT_READ)
		self._length = None
		self.nColumns = 0
		self.columns = {}
		self.column_names = []
		
	def addColumn(self, name, offset, length, dtype=np.float64, stride=None):
		if self._length is not None and length != self._length:
			error("inconsistent length", "length of column %s is %d, while %d was expected" % (name, length, self._length))
		else:
			self._length = length
			mmapped_array = np.frombuffer(self.mapping, dtype=dtype, count=length if stride is None else length * 3, offset=offset)
			if stride:
				#import pdb
				#pdb.set_trace()
				mmapped_array = mmapped_array[::stride]
			self.columns[name] = mmapped_array
			self.column_names.append(name)
			#self.column_names.sort()
			self.nColumns += 1
			self.nRows = self._length

class Hdf5MemoryMapped(MemoryMapped):
	def __init__(self, filename):
		super(Hdf5MemoryMapped, self).__init__(filename)
		h5file = h5py.File(self.filename)
		data = h5file["/data"]
		for column_name in data:
			column = data[column_name]
			offset = column.id.get_offset() 
			self.addColumn(column_name, offset, len(column), dtype=column.dtype)


def createImageData():
	sphere =  vtk.vtkSphere()
	sphere.SetRadius(0.1);
	sphere.SetCenter(0.0,0.0,0.0);

	sampleFunction = vtk.vtkSampleFunction()
	sampleFunction.SetImplicitFunction(sphere);
	sampleFunction.SetOutputScalarTypeToDouble();
	sampleFunction.SetSampleDimensions(127,127,127);
	sampleFunction.SetModelBounds(-1.0,1.0,-1.0,1.0,-1.0,1.0);
	sampleFunction.SetCapping(False);
	sampleFunction.SetComputeNormals(False);
	sampleFunction.SetScalarArrayName("values");
	sampleFunction.Update();

	a = sampleFunction.GetOutput().GetPointData().GetScalars("values");
	range = a.GetRange();
	print(range)
	t = vtk.vtkImageShiftScale()
	t.SetInputConnection(sampleFunction.GetOutputPort());

	t.SetShift(-range[0]);
	magnitude=range[1]-range[0];
	if magnitude==0.0:
		magnitude=1.0
	t.SetScale(255.0/magnitude)
	t.SetOutputScalarTypeToUnsignedChar()
	t.Update();
	return t.GetOutput()
	#imageData->ShallowCopy(t->GetOutput());
	
data_matrix = np.zeros([256, 256, 256], dtype=np.float64)
#data_matrix[0:35, 0:35, 0:35] = 50
#data_matrix[25:55, 25:55, 25:55] = 100
#data_matrix[45:74, 45:74, 45:74] = 150

dataset = Hdf5MemoryMapped(sys.argv[1])
x = dataset.columns[sys.argv[2]]
y = dataset.columns[sys.argv[3]]
z = dataset.columns[sys.argv[4]]
Nrows = int(1e7)
x, y, z = [col[:Nrows] for col in [x,y,z]]

print("do histogram")

def minmax(x, fraction=0.005):
	indices = np.argsort(x)
	N = len(x)
	i1 = int(fraction*N)
	i2 = int((1-fraction)*N)
	return x[indices[i1]], x[indices[i2]]

xmin, xmax = minmax(x)
ymin, ymax = minmax(y)
zmin, zmax = minmax(z)

print("done")
import vaex.vaexfast
vaex.vaexfast.histogram3d(x, y, z, None, data_matrix , xmin, xmax, ymin, ymax, zmin, zmax)
data_matrix = np.log(data_matrix + 1)
#vaex.histogram.hist3d(x, y, z, data_matrix, xmin, xmax, ymin, ymax, zmin, zmax)
print("done")
maxvalue = data_matrix.max()
print(maxvalue)

ar = vtk.util.numpy_support.numpy_to_vtk(data_matrix.reshape((-1,)))
print(ar.GetNumberOfComponents(), ar.GetNumberOfTuples())
print(ar)
print(dir(ar))

grid = vtk.vtkImageData()
grid.SetOrigin(0, 0, 0) # default values
dx = 1
grid.SetSpacing(dx, dx, dx)
grid.SetDimensions(256, 256, 256)
print(grid.GetNumberOfPoints())
grid.GetPointData().SetScalars(ar)

def createImageData():
	#print dir(vtk.util.numpy_support)
	#dataImporter = vtk.vtkImageImport()
	#dataImporter.SetDataScalarTypeToUnsignedChar()
	
	print(grid.GetPointData().GetNumberOfArrays())
	return grid

class MyInteractorStyle(vtk.vtkInteractorStyleSwitch):
	def __init__(self, window):
		vtk.vtkInteractorStyleSwitch.__init__(self)
		self.window = window
		
	def OnMouseMove(self):
		print("mouse move")
		super(MyInteractorStyle, self).OnMouseMove()
		
	
class Lasso(object):
	def __init__(self, list):
		self.list = list

		self.points = vtk.vtkPoints()

		self.cells = vtk.vtkCellArray()
		self.cells.Initialize()
		
		self.line = vtk.vtkLine()
		

		self.update()

		self.polyData = vtk.vtkPolyData() 
		self.polyData.Initialize()
		self.polyData.SetPoints(self.points)
		self.polyData.SetLines(self.cells)
		self.polyData.Modified()
		
		self.coordinate = vtk.vtkCoordinate()
		self.coordinate.SetCoordinateSystemToDisplay()
		
		self.lassoMapper = vtk.vtkPolyDataMapper2D()
		self.lassoMapper.SetInput(self.polyData)
		self.lassoMapper.SetTransformCoordinate(self.coordinate)
		self.lassoMapper.ScalarVisibilityOn()
		self.lassoMapper.SetScalarModeToUsePointData()

		self.lassoMapper.Update()
		
		self.actor = vtk.vtkActor2D()
		self.actor.SetMapper(self.lassoMapper)
		self.actor.GetProperty().SetLineWidth(2.0);
		self.actor.GetProperty().SetColor(1,0,0);

	def update(self):
		self.points.SetNumberOfPoints(len(self.list)); 
		self.points.Allocate(len(self.list));
		self.line.GetPointIds().SetNumberOfIds(len(self.list))
		for i, (x, y) in enumerate(self.list):
			self.points.InsertPoint(i, x, y, 10)
			self.line.GetPointIds().SetId(i, i)
		self.cells.Reset()
		self.cells.InsertNextCell(self.line) 
		#self.polyData.Modified()
		#self.lassoMapper.Update()
		
		
		
class Overlay(QtGui.QWidget):
	def __init__(self, target, parent):
		super(Overlay, self).__init__(parent)
		self.target = target
		#self.setPalette(QtCore.Qt.transparent)
		#self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents);
		
		self.vl = QtGui.QVBoxLayout()
		self.button = QtGui.QPushButton("overlay", self)
		self.vl.addWidget(self.button)
		self.setLayout(self.vl)
		
		#self.original_resizeEvent = target.resizeEvent
		#target.resizeEvent = self.onResizeTarget
		
	def onResizeTarget(self, event):
		size = self.target.size()
		print(size, self.target.x(), self.target.y())
		self.original_resizeEvent(event)
		self.move(self.target.x(), self.target.y())
		self.resize(self.target.size())
		
	def paintEvent(self, event):
		print(event)
		painter = QtGui.QPainter(self)
		painter.drawLine(0, 0, 100, 200)
		
class MainWindow:#QtGui.QMainWindow):

	def _paintEvent(self, event):
		print(event)
		#self.vtkWidget._original_paintEvent(event)
		#painter = QtGui.QPainter(self.vtkWidget)
		#painter.drawLine(9, 0, 100, 200)
		
	def mousePressEvent(self, event):
		if 1:
			self.vtkWidget._original_mousePressEvent(event)
		else:
			self.lasso_screen = []
			self.dragging = True
			
	def mouseReleaseEvent(self, event):
		if 1:
			self.vtkWidget._original_mouseReleaseEvent(event)
		else:
			self.dragging = False
	def mouseMoveEvent(self, event):
		if 1:
			self.vtkWidget._original_mouseMoveEvent(event)
		else:
			if self.dragging:
				pos = event.x(), self.vtkWidget.size().height() - event.y() - 1 
				self.lasso_screen.append(pos)
			self.lasso.list = self.lasso_screen
			print(self.lasso.list)
			self.lasso.update()
			#self.lasso.line.Update()
			#self.vtkWidget.GetRenderWindow().Render()
			#self.vtkWidget.update()
			self.update()
			#print self.lasso_screen

	def __init__(self, parent = None):
		#QtGui.QMainWindow.__init__(self, parent)

		self.dragging = False
		self.lasso_screen = [(0, 0), (100, 50)]
		
		

		if 0:
			self.frame = QtGui.QFrame()

			self.vl = QtGui.QVBoxLayout()
			self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
			# monkey patch
			self.vtkWidget._original_mouseMoveEvent = self.vtkWidget.mouseMoveEvent
			self.vtkWidget.mouseMoveEvent = self.mouseMoveEvent

			self.vtkWidget._original_mousePressEvent = self.vtkWidget.mousePressEvent
			self.vtkWidget.mousePressEvent = self.mousePressEvent


			self.vtkWidget._original_mouseReleaseEvent = self.vtkWidget.mouseReleaseEvent
			self.vtkWidget.mouseReleaseEvent = self.mouseReleaseEvent
			
			if 0:
				self.vtkWidget._original_paintEvent = self.vtkWidget.paintEvent
				self.vtkWidget.paintEvent = self.paintEvent
				def test():
					print(QtGui.QPaintEngine.OpenGL);
					return 0;
				self.vtkWidget.paintEngine = test

			self.vl.addWidget(self.vtkWidget)
			
			#self.overlay = Overlay(self.vtkWidget, self.frame)
			#self.vl.addWidget(self.overlay)
			#self.vl.addWidget(QtGui.QPushButton("test", self))
		else:
			window = vtk.vtkXOpenGLRenderWindow()
			window.SetOffScreenRendering(True)


		self.ren = vtk.vtkRenderer()
		window.AddRenderer(self.ren)
		#self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
		#self.iren.GetInteractorStyle().SetCurrentStyleToTrackballCamera()
		#self.iren.GetInteractorStyle().AddObserver("MouseMoveEvent", self.mouseMoveEvent)
		#self.interactorStyle = MyInteractorStyle(self)
		#self.iren.SetInteractorStyle(self.interactorStyle)
		#import pdb
		#pdb.set_trace()
		# Create source
		#source = vtk.vtkSphereSource()
		#source.SetCenter(0, 0, 0)
		#source.SetRadius(5.0)
		if 0:
			self.levelSlider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
			self.levelSlider.setRange(0, 100)
			self.levelSlider.sliderReleased.connect(self.sliderReleased)
			self.isoCheckbox = QtGui.QCheckBox("Show isosurface", self)
			self.volumeCheckbox = QtGui.QCheckBox("Show volume rendering", self)
			self.vl.addWidget(self.isoCheckbox)
			self.vl.addWidget(self.volumeCheckbox)
			self.vl.addWidget(self.levelSlider)
			
			self.isoCheckbox.setCheckState(QtCore.Qt.Checked)
			self.volumeCheckbox.setCheckState(QtCore.Qt.Checked)
		
		#self.formLayout = QtGui.QFormLayout(self)
		#self.formLayout.addRow("show isosurface", QCheckBox
		#self.vl.addItem(self.formLayout)
		
		imageData = createImageData()
		volumeMapper = vtk.vtkSmartVolumeMapper()
		volumeMapper = vtk.vtkVolumeTextureMapper3D()
		volumeMapper.SetBlendModeToComposite();

		#compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
		# We can finally create our volume. We also have to specify the data for it, as well as how the data will be rendered.
		#volumeMapper = vtk.vtkVolumeRayCastMapper()
		#volumeMapper.SetVolumeRayCastFunction(compositeFunction)
		
		volumeMapper.SetInputConnection(imageData.GetProducerPort());

		volumeProperty = vtk.vtkVolumeProperty()
		volumeProperty.ShadeOff()
		volumeProperty.SetInterpolationType(vtk.VTK_LINEAR_INTERPOLATION)

		compositeOpacity = vtk.vtkPiecewiseFunction()
		compositeOpacity.AddPoint(  0., 0.0)
		compositeOpacity.AddPoint( 1., 0.105)
		compositeOpacity.AddPoint(2., 0.101)
		compositeOpacity.AddPoint(3., 0.102)
		volumeProperty.SetScalarOpacity(compositeOpacity)
		
		
		color = vtk.vtkColorTransferFunction()
		#color.AddRGBPoint(1, 1.0, 0.0, 0.0)
		color.AddRGBPoint(2, 0.0, 1.0, 0.0)
		color.AddRGBPoint(3, 0.0, 0.0, 1.0)
		volumeProperty.SetColor(color)


		volume =  vtk.vtkVolume()
		volume.SetMapper(volumeMapper)
		volume.SetProperty(volumeProperty)
		self.ren.AddViewProp(volume)
		
		outline = vtk.vtkOutlineFilter()
		outline.SetInput(imageData)
		outlineMapper = vtk.vtkPolyDataMapper()
		outlineMapper.SetInputConnection(outline.GetOutputPort())
		outlineActor = vtk.vtkActor()
		outlineActor.SetMapper(outlineMapper)
		outlineActor.GetProperty().SetColor(0,0,0)
		self.ren.AddActor(outlineActor)
		
		
		self.lasso = Lasso(self.lasso_screen)
		self.ren.AddActor(self.lasso.actor)

		if 0:
			def onVolume(state):
				checked = state == QtCore.Qt.Checked
				if checked:
					volume.VisibilityOn()
				else:
					volume.VisibilityOff()
				self.vtkWidget.GetRenderWindow().Render()
			self.volumeCheckbox.stateChanged.connect(onVolume)
		
		#self.ren.ResetCamera()
		
		if 0:
			self.surface = vtk.vtkMarchingCubes()
			self.surface.SetInput(imageData)
			self.surface.ComputeNormalsOn();
			self.surface.SetValue(0, 100);
			
			#print surface
			#print surface.GetOutput()
			#print surface.GetOutputPort()


			# Create a mapper
			mapper = vtk.vtkPolyDataMapper()
			mapper.SetInputConnection(self.surface.GetOutputPort())
			mapper.ScalarVisibilityOff()
			print(mapper)
			import pdb
			#pdb.set_trace()
			# Create an actor
			actor = vtk.vtkActor()
			actor.SetMapper(mapper)

			actor.GetProperty().SetColor(1,0,0)
			actor.GetProperty().SetOpacity(1)
			
			self.ren.AddActor(actor)
			def onIso(state):
				checked = state == QtCore.Qt.Checked
				if checked:
					actor.VisibilityOn()
				else:
					actor.VisibilityOff()
				self.vtkWidget.GetRenderWindow().Render()
			self.isoCheckbox.stateChanged.connect(onIso)

		self.ren.ResetCamera()

		#self.frame.setLayout(self.vl)
		#self.setCentralWidget(self.frame)
		self.ren.SetBackground(1,1,1)
		#self.ren.SetOffScreenRendering(True)
		arr = vtk.vtkUnsignedCharArray()
		print("size", arr.GetDataSize())
		window.Render()
		print(window.GetPixelData(0, 0, 10, 10, 0, arr))
		print("size", arr.GetDataSize())

		self.show()
		self.raise_()
		print("aap")
		self.iren.Initialize()
	
	def sliderReleased(self, value=None):
		if value is None:
			value = self.levelSlider.value()
		fraction = value/100.
		isovalue = maxvalue * 10**((fraction-1)*3)
		print(value, "of 100, corresponds to", isovalue, "of", maxvalue)
		self.surface.SetValue(0, isovalue)
		#self.surface.Update()
		#self.vtkWidget.GetRenderWindow().Render()

 
if __name__ == "__main__":

	app = QtGui.QApplication(sys.argv)

	window = MainWindow()


	sys.exit(app.exec_())