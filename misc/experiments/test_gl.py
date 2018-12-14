


#!/usr/bin/env python

"""PySide port of the opengl/hellogl example from Qt v4.x"""
from __future__ import print_function

import sys
import math
#from PySide import QtCore, QtGui, QtOpenGL
from PyQt4 import QtCore, QtGui, QtOpenGL
from OpenGL.arrays import vbo
try:
    from OpenGL import GL, GLU
except ImportError:
    app = QtGui.QApplication(sys.argv)
    QtGui.QMessageBox.critical(None, "OpenGL hellogl",
                            "PyOpenGL must be installed to run this example.",
                            QtGui.QMessageBox.Ok | QtGui.QMessageBox.Default,
                            QtGui.QMessageBox.NoButton)
    sys.exit(1)


import numpy as np
import mmap
import h5py
import ctypes
import scipy.spatial



from numba import jit
@jit(nopython=True)
def project(x, y, z, modelview, projection, viewport, winx, winy, winz):
	length = len(x)
	for i in range(length):
		objx = x[i]
		objy = y[i]
		objz = z[i]
		eye_x=modelview[0]*objx+modelview[4]*objy+modelview[8]*objz+modelview[12] # w is always 1
		eye_y=modelview[1]*objx+modelview[5]*objy+modelview[9]*objz+modelview[13]
		eye_z=modelview[2]*objx+modelview[6]*objy+modelview[10]*objz+modelview[14]
		eye_w=modelview[3]*objx+modelview[7]*objy+modelview[11]*objz+modelview[15]
		
		
		clip_x=projection[0]*eye_x+projection[4]*eye_y+projection[8]*eye_z+projection[12]*eye_w;
		clip_y=projection[1]*eye_x+projection[5]*eye_y+projection[9]*eye_z+projection[13]*eye_w;
		clip_z=projection[2]*eye_x+projection[6]*eye_y+projection[10]*eye_z+projection[14]*eye_w;
		clip_w=projection[3]*eye_x+projection[7]*eye_y+projection[11]*eye_z+projection[15]*eye_w;
		#clip_w=1#-eye_w;
		
		clip_x /= clip_w
		clip_y /= clip_w
		clip_z /= clip_w
		
		winx[i] = (clip_x*0.5 + 0.5) * viewport[2] + viewport[0]
		winy[i] = (clip_y*0.5 + 0.5) * viewport[3] + viewport[1]
		winz[i] = clip_z*0.5+0.5
		
		
# from http://www.ecse.rpi.edu/Homepages/wrf/Research/Short_Notes/pnpoly.html		
	
@jit('(f4[:], f4[:], f4[:], f4[:], u1[:], f8, f8, f8)', nopython=True)
def pnpoly(vertx, verty, testx, testy, inside, meanx, meany, radius):
	nvert = len(vertx)
	ntest = len(testx)
	for k in range(ntest):
		distancesq = (testx[k] - meanx)**2 + (testy[k] - meany)**2
		inside[k] = 0
		if distancesq < radius**2: # quick check
			inside[k] = 0
			if 1:
				j = nvert-1
				for i in range(nvert):
					if (((verty[i]>testy[k]) != (verty[j]>testy[k])) and (testx[k] < (vertx[j]-vertx[i]) * (testy[k]-verty[i]) / (verty[j]-verty[i]) + vertx[i]) ):
						inside[k] = not inside[k]
					j = i

lasso_test = [(102, 245), (101, 241), (98, 235), (96, 232), (94, 230), (93, 225), (92, 221), (92, 219), (94, 216), (95, 215), (99, 213), (104, 212), (106, 212), (112, 211), (119, 211), (126, 215), (134, 221), (137, 225), (140, 233), (141, 242), (141, 252), (137, 254), (123, 257), (109, 253), (104, 251), (99, 243), (98, 240)]
lasso_list = lasso_test
points = np.array(lasso_list, dtype=np.float32)
vertx = points[:,0]
verty = points[:,1]


testx, testy = np.array([vertx.mean(), 120], dtype='f32'), np.array([verty.mean(), 260], dtype='f32')
inside = np.zeros(2, dtype=np.uint8)
print((vertx, verty, testx, testy, inside, 0., 0., 10000.))
print(pnpoly(vertx, verty, testx, testy, inside, 0., 0., 10000.))
print(inside)
#sys.exit(0)
	
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

dataset = Hdf5MemoryMapped(sys.argv[1])
data_x = dataset.columns[sys.argv[2]]
data_y = dataset.columns[sys.argv[3]]
data_z = dataset.columns[sys.argv[4]]

length = 1000000

def norm(x):
	xmin = x.min()
	xmax = x.max()
	return (x - x.min())/(x.max() - x.min()) - 0.5 
data_x = norm(data_x) #[:length*3:3])
data_y = norm(data_y) #[:length*3:3])
data_z = norm(data_z) #[:length*3:3])
length = len(data_x)
buffer = np.zeros((length, 3), dtype='f')
buffer[:,0] = data_x
buffer[:,1] = data_y
buffer[:,2] = data_z
winx = (data_x * 0).astype(np.float32)
winy = (data_x * 0).astype(np.float32)
winz = (data_x * 0).astype(np.float32)
winx_inside = winx[:1] * 0
winy_inside = winy[:1] * 0
winz_inside = winz[:1] * 0
inside = np.zeros(len(winx), dtype=np.uint8)
indices_all = np.arange(len(data_x), dtype=np.uint32)
indices_selected = indices_all[0:0]

class Window(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)

        self.glWidget = GLWidget()

        self.xSlider = self.createSlider(QtCore.SIGNAL("xRotationChanged(int)"),
                                         self.glWidget.setXRotation)
        self.ySlider = self.createSlider(QtCore.SIGNAL("yRotationChanged(int)"),
                                         self.glWidget.setYRotation)
        self.zSlider = self.createSlider(QtCore.SIGNAL("zRotationChanged(int)"),
                                         self.glWidget.setZRotation)

        mainLayout = QtGui.QHBoxLayout()
        mainLayout.addWidget(self.glWidget)
        mainLayout.addWidget(self.xSlider)
        mainLayout.addWidget(self.ySlider)
        mainLayout.addWidget(self.zSlider)
        self.setLayout(mainLayout)

        self.xSlider.setValue(15 * 16)
        self.ySlider.setValue(345 * 16)
        self.zSlider.setValue(0 * 16)

        self.setWindowTitle(self.tr("Hello GL"))

    def createSlider(self, changedSignal, setterSlot):
        slider = QtGui.QSlider(QtCore.Qt.Vertical)

        slider.setRange(0, 360 * 16)
        slider.setSingleStep(16)
        slider.setPageStep(15 * 16)
        slider.setTickInterval(15 * 16)
        slider.setTickPosition(QtGui.QSlider.TicksRight)

        self.glWidget.connect(slider, QtCore.SIGNAL("valueChanged(int)"), setterSlot)
        self.connect(self.glWidget, changedSignal, slider, QtCore.SLOT("setValue(int)"))

        return slider


class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
		QtOpenGL.QGLWidget.__init__(self, parent)

		self.object = 0
		self.xRot = 0
		self.yRot = 0
		self.zRot = 0

		self.lastPos = QtCore.QPoint()

		self.trolltechGreen = QtGui.QColor.fromCmykF(0.40, 0.0, 1.0, 0.0)
		self.trolltechPurple = QtGui.QColor.fromCmykF(0.39, 0.39, 0.0, 0.0)
		self.dragging = False
		self.lasso_list = []
		self.lasso_list = lasso_test
		self.bbox = []

    def xRotation(self):
        return self.xRot

    def yRotation(self):
        return self.yRot

    def zRotation(self):
        return self.zRot

    def minimumSizeHint(self):
        return QtCore.QSize(50, 50)

    def sizeHint(self):
        return QtCore.QSize(400, 400)

    def setXRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.xRot:
            self.xRot = angle
            self.emit(QtCore.SIGNAL("xRotationChanged(int)"), angle)
            self.updateGL()

    def setYRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.yRot:
            self.yRot = angle
            self.emit(QtCore.SIGNAL("yRotationChanged(int)"), angle)
            self.updateGL()

    def setZRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.zRot:
            self.zRot = angle
            self.emit(QtCore.SIGNAL("zRotationChanged(int)"), angle)
            self.updateGL()

    def initializeGL(self):
        self.qglClearColor(self.trolltechPurple.darker())
        self.object = self.makeObject()
        GL.glShadeModel(GL.GL_FLAT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)

    def paintGL(self):
		self.resizeGL(self.width(), self.height())
		GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
		GL.glLoadIdentity()
		GL.glTranslated(0.0, 0.0, -10.0)
		GL.glRotated(self.xRot / 16.0, 1.0, 0.0, 0.0)
		GL.glRotated(self.yRot / 16.0, 0.0, 1.0, 0.0)
		GL.glRotated(self.zRot / 16.0, 0.0, 0.0, 1.0)
		#GL.glCallList(self.object)
		
		
		GL.glEnable( GL.GL_POINT_SMOOTH );
		GL.glEnable( GL.GL_BLEND );
		GL.glBlendFunc( GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA );
		GL.glPointSize( 0.5 );	
    
		if 0:
			GL.glBegin(GL.GL_POINTS)
			z = 1.
			s = 0.1
			GL.glVertex3d(-s, -s, z)
			GL.glVertex3d(s, -s, z)
			GL.glVertex3d(s, s, z)
			GL.glVertex3d(-s, s, z)
			#length = len(data_x)
			for i in range(length):
				#print data_x[i]/s, data_y[i]/s, -data_z[i]/s/100
				GL.glVertex3d(data_x[i], data_y[i], data_z[i])
			GL.glEnd()
		else:
			GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
			vertices = buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
			GL.glVertexPointer(3, GL.GL_FLOAT, 0, vertices);
			alpha = 1.
			GL.glColor4f( 0.95, 0.207, 0.031, alpha);
			#GL.glDrawArrays(GL.GL_POINTS, 0, length);
			indices_all_ptr = indices_all.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
			GL.glDrawElements(GL.GL_POINTS, len(indices_all), GL.GL_UNSIGNED_INT, indices_all_ptr);
			GL.glDisableClientState(GL.GL_VERTEX_ARRAY);

			if 1:
					GL.glColor4f( 0.95, 0.907, 0.831, alpha);
					GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
					vertices = buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
					GL.glVertexPointer(3, GL.GL_FLOAT, 0, vertices);
					#GL.glDrawArrays(GL.GL_POINTS, 0, length);
					indices_selected_ptr = indices_selected.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
					GL.glDrawElements(GL.GL_POINTS, len(indices_selected), GL.GL_UNSIGNED_INT, indices_selected_ptr);
					print(len(indices_selected), "selected")
					GL.glDisableClientState(GL.GL_VERTEX_ARRAY);

		if 1:
			self.model = GL.glGetDoublev(GL.GL_MODELVIEW_MATRIX)
			self.proj = GL.glGetDoublev(GL.GL_PROJECTION_MATRIX)
			self.view = GL.glGetIntegerv(GL.GL_VIEWPORT)
			print("view", self.view)
			#print model, proj, view
			#i = 0
			#winxi,winyi,winzi = GLU.gluProject(data_x[i], data_y[i], data_z[i],model.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),proj.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),view.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
			#print winxi, winyi, winzi
			#print winx[i], winy[i]
			#M = np.matrix(model)
			#P = np.matrix(proj)
			#import pdb
			#pdb.set_trace()
			#print "bla"
			# M * np.array([[data_x[i], data_y[i], data_z[i], 1]]).T
			#winx,winy,winz = np.vectorize(GLU.gluProject)(data_x, data_y, data_z,model,proj,view)
		
		if self.bbox:
			GL.glColor4f( 1, 1, 1, 1);
			GL.glBegin(GL.GL_LINES)
			#pairs = [(0,1), (0,2), (]
			for i in range(8):
				for j in range(8):
					if i != j:
						GL.glVertex3d(self.bbox[i][0], self.bbox[i][1], self.bbox[i][2])
						GL.glVertex3d(self.bbox[j][0], self.bbox[j][1], self.bbox[j][2])
			GL.glEnd()
		win_width, win_height = self.width(), self.height()
		GL.glViewport(0, 0, win_width, win_height);
		GL.glMatrixMode(GL.GL_PROJECTION);
		GL.glLoadIdentity();
		GL.glOrtho(0, win_width, 0, win_height, -1, 1);
		GL.glMatrixMode(GL.GL_MODELVIEW);
		GL.glLoadIdentity();
		GL.glDisable(GL.GL_DEPTH_TEST)
		GL.glLineWidth(2.5); 
		z = 1.
		print("lasso list:", self.lasso_list)
		if len(self.lasso_list) >= 3:
			points = np.array(self.lasso_list)
			hull = scipy.spatial.ConvexHull(points)
			hullx = points[hull.vertices,0].astype(np.float32)
			hully = points[hull.vertices,1].astype(np.float32)

			meanx = hullx.mean()
			meany = hully.mean()
			radius = np.sqrt((points[hull.vertices,0] - meanx)**2 + (points[hull.vertices,1] - meany)**2).max()
			GL.glColor4f(0.5, 1.0, 0.5, 0.5);
			GL.glBegin(GL.GL_TRIANGLE_FAN)
			N = 100
			GL.glVertex3d(meanx, self.height() - meany - 1, z)
			for i in range(N):
				x = meanx + np.sin(math.pi*2/N*i) * radius
				y = meany + np.cos(math.pi*2/N*i) * radius
				GL.glVertex3d(x, self.height() - y - 1, z)
			GL.glEnd()
				

			
			GL.glColor4f(1., 0.5, 0.5, 0.25);
			GL.glBegin(GL.GL_TRIANGLE_FAN)
			for vertex in hull.vertices[::-1]:
				#print vertex
				x, y = self.lasso_list[vertex]
				GL.glVertex3d(x, self.height() - y - 1, z)
			GL.glEnd()
			

			GL.glColor4f(1., 0.5, 0.5, 0.27);
			GL.glBegin(GL.GL_TRIANGLE_FAN)
			for vertex in hull.vertices[::-1]:
				#print vertex
				x, y = self.lasso_list[vertex]
				GL.glVertex3d(x, self.height() - y - 1, z)
			GL.glEnd()
			#print hullx, hully
			print("inside", np.sum(inside), len(inside))
			if 1:
				GL.glColor4f(1., 1.0, 1.0, 0.27);
				if 0:
					GL.glBegin(GL.GL_POINTS)
					z = 1.
					for i in range(len(winy_inside)):
						GL.glVertex3d(winx_inside[i], winy_inside[i], z)
					GL.glEnd()
				else:
					pass
		GL.glBegin(GL.GL_LINE_STRIP)
		GL.glColor3f(1., 1., 1.);
		for x, y in self.lasso_list:
			GL.glVertex3d(x, self.height() - y - 1, z)


		#GL.glVertex3d(0., 0., z)
		#GL.glVertex3d(100., 100., z)
		GL.glEnd()
		GL.glEnable(GL.GL_DEPTH_TEST)
		
    def resizeGL(self, width, height):
        side = min(width, height)
        GL.glViewport((width - side) / 2, (height - side) / 2, side, side)

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(-0.5, +0.5, +0.5, -0.5, 4.0, 15.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)

    def mouseReleaseEvent(self, event):
		global winx_inside, winy_inside, winz_inside, indices_selected, indices_all
		self.dragging = False
		if len(self.lasso_list) >= 3:
			points = np.array(self.lasso_list)
			hull = scipy.spatial.ConvexHull(points)
			hullx = points[hull.vertices,0].astype(np.float32)
			hully = points[hull.vertices,1].astype(np.float32)
			
			meanx = hullx.mean()
			meany = hully.mean()
			radius = np.sqrt((points[hull.vertices,0] - meanx)**2 + (points[hull.vertices,1] - meany)**2).max()

			project(data_x, data_y, data_z, self.model.reshape(-1), self.proj.reshape(-1), self.view.reshape(-1), winx, winy, winz)
			pnpoly(hullx, hully, winx, self.height() - winy - 1, inside, meanx, meany, radius)
			winx_inside = winx[inside==1]
			winy_inside = winy[inside==1]
			winz_inside = winz[inside==1]
			#zmin, zmax = winz_inside.min(), winz_inside.max()
			#counts = 
			#vaex.histogram.hist1d(
			fraction = 0.005
			N = len(winz_inside)
			indices = np.argsort(winz_inside)
			i1, i2 = indices[int(N*fraction)], indices[int(N*(1-fraction))]
			print(i1, i2)
			zmin = winz_inside[i1]
			zmax = winz_inside[i2]
			xmin, xmax = winx_inside.min(), winx_inside.max()
			ymin, ymax = winy_inside.min(), winy_inside.max()
			#zmin, zmax = winz_inside.min(), winz_inside.max()
			print() 
			print("x:", xmin, xmax)
			print("y:", ymin, ymax)
			print("z:", zmin, zmax)
			
			
			M = np.matrix(self.model)
			P = np.matrix(self.proj)
			T = (P * M)
			print("M", self.model)
			print("P", self.proj)
			print("v", self.view)
			#xmin, xmax = 0, 1
			#ymin, ymax = 0, 1
			#zmin, zmax = 0, 1
			self.bbox = []
			for z in [zmin, zmax]:
				for y in [ymin, ymax]:
					for x in [xmin, xmax]:
						xc = ((x - self.view[0])/self.view[2] * 2 - 1)
						yc = ((y - self.view[1])/self.view[3] * 2 - 1 )
						zc = z*2.-1.
						clip = np.array([[xc, yc, zc, 1]]).T
						#view = P.I * clip
						eye = (P.T*M.T).I * clip
						print(eye)
						eye[0:3] /= eye[3]
						print(x, y, z)
						print("->", eye)
						#print clip
						#print (P*M) * (P*M).I
						#print GLU.gluUnProject
						self.bbox.append((eye[0], eye[1], eye[2]))
					
						xu,yu,zu = GLU.gluUnProject(x, y, z, self.model.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), self.proj.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), self.view.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
						print("glu:", xu, yu, zu)
			indices_selected = np.arange(len(winx), dtype=np.uint32)[inside==1]
			indices_all = np.arange(len(winx), dtype=np.uint32)[inside==0]
			print(data_x[indices_selected].min(), data_x[indices_selected].max())
			print(data_y[indices_selected].min(), data_y[indices_selected].max())
			print(data_z[indices_selected].min(), data_z[indices_selected].max())
			#import pdb
			#pdb.set_trace()
			
			
			x, y, z = 1,1,1
			xu,yu,zu = GLU.gluUnProject(x, y, z, self.model.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), self.proj.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), self.view.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
			print()
			print(xu, yu, zu)
			print(GLU.gluProject(xu, yu, zu, self.model.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), self.proj.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), self.view.ctypes.data_as(ctypes.POINTER(ctypes.c_float))))
			
			
			
			self.lasso_list = []
			self.updateGL()
		
    def mousePressEvent(self, event):
		global winx_inside, winy_inside
		if 1:
			self.dragging = True
			self.lasso_list = []
			winx_inside = winx_inside[0:0]
			winy_inside = winx_inside[0:0]
		else:
			self.lastPos = QtCore.QPoint(event.pos())

    def mouseMoveEvent(self, event):
		if 1:
			if self.dragging:
				pos = event.x(), event.y() 
				self.lasso_list.append(pos)
			print(self.lasso_list)
			self.update()
		else:
			dx = event.x() - self.lastPos.x()
			dy = event.y() - self.lastPos.y()

			if event.buttons() & QtCore.Qt.LeftButton:
				self.setXRotation(self.xRot + 8 * dy)
				self.setYRotation(self.yRot + 8 * dx)
			elif event.buttons() & QtCore.Qt.RightButton:
				self.setXRotation(self.xRot + 8 * dy)
				self.setZRotation(self.zRot + 8 * dx)

			self.lastPos = QtCore.QPoint(event.pos())

    def makeObject(self):
        genList = GL.glGenLists(1)
        GL.glNewList(genList, GL.GL_COMPILE)

        GL.glBegin(GL.GL_QUADS)

        x1 = +0.06
        y1 = -0.14
        x2 = +0.14
        y2 = -0.06
        x3 = +0.08
        y3 = +0.00
        x4 = +0.30
        y4 = +0.22

        self.quad(x1, y1, x2, y2, y2, x2, y1, x1)
        self.quad(x3, y3, x4, y4, y4, x4, y3, x3)

        self.extrude(x1, y1, x2, y2)
        self.extrude(x2, y2, y2, x2)
        self.extrude(y2, x2, y1, x1)
        self.extrude(y1, x1, x1, y1)
        self.extrude(x3, y3, x4, y4)
        self.extrude(x4, y4, y4, x4)
        self.extrude(y4, x4, y3, x3)

        Pi = 3.14159265358979323846
        NumSectors = 200

        for i in range(NumSectors):
            angle1 = (i * 2 * Pi) / NumSectors
            x5 = 0.30 * math.sin(angle1)
            y5 = 0.30 * math.cos(angle1)
            x6 = 0.20 * math.sin(angle1)
            y6 = 0.20 * math.cos(angle1)

            angle2 = ((i + 1) * 2 * Pi) / NumSectors
            x7 = 0.20 * math.sin(angle2)
            y7 = 0.20 * math.cos(angle2)
            x8 = 0.30 * math.sin(angle2)
            y8 = 0.30 * math.cos(angle2)

            self.quad(x5, y5, x6, y6, x7, y7, x8, y8)

            self.extrude(x6, y6, x7, y7)
            self.extrude(x8, y8, x5, y5)

        GL.glEnd()
        GL.glEndList()

        return genList

    def quad(self, x1, y1, x2, y2, x3, y3, x4, y4):
        self.qglColor(self.trolltechGreen)

        GL.glVertex3d(x1, y1, -0.05)
        GL.glVertex3d(x2, y2, -0.05)
        GL.glVertex3d(x3, y3, -0.05)
        GL.glVertex3d(x4, y4, -0.05)

        GL.glVertex3d(x4, y4, +0.05)
        GL.glVertex3d(x3, y3, +0.05)
        GL.glVertex3d(x2, y2, +0.05)
        GL.glVertex3d(x1, y1, +0.05)

    def extrude(self, x1, y1, x2, y2):
        self.qglColor(self.trolltechGreen.darker(250 + int(100 * x1)))

        GL.glVertex3d(x1, y1, +0.05)
        GL.glVertex3d(x2, y2, +0.05)
        GL.glVertex3d(x2, y2, -0.05)
        GL.glVertex3d(x1, y1, -0.05)

    def normalizeAngle(self, angle):
        while angle < 0:
            angle += 360 * 16
        while angle > 360 * 16:
            angle -= 360 * 16
        return angle


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = Window()
    window.show()
    window.raise_()
    sys.exit(app.exec_())