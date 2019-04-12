# -*- coding: utf-8 -*-
from __future__ import print_function
import vaex.dataset as dataset
import numpy as np
import numpy
import math
import vaex.utils
import matplotlib.pyplot as plt
import scipy.ndimage
import matplotlib.animation as animation
import matplotlib
import time



def rotz(angle):
	matrix = np.identity(4)
	matrix[0,0] = np.cos(angle)
	matrix[0,1] = np.sin(angle)
	matrix[1,0] = -np.sin(angle)
	matrix[1,1] = np.cos(angle)
	return np.matrix(matrix).T

def rotx(angle):
	matrix = np.identity(4)
	matrix[1,1] = np.cos(angle)
	matrix[1,2] = np.sin(angle)
	matrix[2,1] = -np.sin(angle)
	matrix[2,2] = np.cos(angle)
	return np.matrix(matrix).T

def scale(factor):
	matrix = np.identity(4)
	for i in range(3):
		matrix[i,i] = float(factor)
	return np.matrix(matrix).T

def translate(x,y,z):
	matrix = np.identity(4)
	matrix[3,0:3] = x, y, z
	return np.matrix(matrix).T
	
def proj(size):
	matrix = np.identity(4)
	#return np.matrix(matrix).T
	right = float(size)
	left = 0
	top = float(N)
	bottom = 0
	far = 1.
	near = 0
	print() 
	
	matrix[0][0] = 2./(right-left)
	matrix[1][1] = 2./(top-bottom)
	matrix[2][2] = -2./(far-near)
	# col3 is only when left, bottom and near != 0
	#matrix[3][0] = - (right+left)/(right-left)
	return np.matrix(matrix).T

N = 256
N2d = 256
#m = scale_matrix(1./N, [N/2., N/2., N/2.])
#m = scale_matrix(1./N)
#m = rotation_matrix(np.radians(30), [0, 1, 0]) * m
if 0:
	print("t", translate(-1, -1, -1))
	print("s", scale(2./N))
	print("p", proj(N))
	print(np.dot(m, [0, 0, 0, 1]))
	print(np.dot(m, [N/2, N/2, N/2, 1]))
	print(np.dot(m, (N, N, N, 1)))
	print(np.dot(m, (N, N, 0, 1)))
#print rotation_matrix(np.radians(30), [0, 1, 0])
colormaps = []
colormap_pixmap = {}
colormaps_processed = False
cols = []
for x in np.linspace(0,1, 256):
	rcol = 0.237 - 2.13*x + 26.92*x**2 - 65.5*x**3 + 63.5*x**4 - 22.36*x**5
	gcol = ((0.572 + 1.524*x - 1.811*x**2)/(1 - 0.291*x + 0.1574*x**2))**2
	bcol = 1/(1.579 - 4.03*x + 12.92*x**2 - 31.4*x**3 + 48.6*x**4 - 23.36*x**5)
	cols.append((rcol, gcol, bcol))

name = 'PaulT_plusmin'
cm_plusmin = matplotlib.colors.LinearSegmentedColormap.from_list(name, cols)
matplotlib.cm.register_cmap(name=name, cmap=cm_plusmin)


#data = dataset.Hdf5MemoryMapped("data/dist/Aq-A-2-999-shuffled-fraction.hdf5")
data = dataset.Hdf5MemoryMapped("/home/data/vaex/Aq-A-2-999-shuffled.hdf5")

Nrows = int(1e7)
#x, y, z = [col[:Nrows] for col in [data.columns["x"], data.columns["y"], data.columns["z"]]]
x, y, z = [col for col in [data.columns["x"], data.columns["y"], data.columns["z"]]]
x = x - 54 #x.mean()
y = y - 50 #y.mean()
z = z - 50 #y.mean()

import vaex.histogram
density = np.zeros((N,N,N))
#vaex.histogram.hist3d(x, y, z, density, np.min(x), np.max(x), np.min(y), np.max(y), np.min(z), np.max(z))
w = 10
#vaex.histogram.hist3d(x, y, z, density, np.min(x)+w, np.max(x)-w, np.min(y)+w, np.max(y)-w, np.min(z)+w, np.max(z)-w)
#vaex.histogram.hist3d(x, y, z, density, -w, w, -w, w, -w, w)
import vaex.vaexfast

#for i in range(10):

t_prev = 0

import threading
class ThreadPool(object):
	def __init__(self, ntheads=8):
		self.ntheads = ntheads
		self.threads = [threading.Thread(target=self.execute, kwargs={"index":i}) for i in range(ntheads)]
		self.semaphore_in = threading.Semaphore(0)
		self.semaphore_out = threading.Semaphore(0)
		for thread in self.threads:
			thread.setDaemon(True)
			thread.start()
			
	def execute(self, index):
		print("index", index)
		while True:
			#print "waiting..", index
			self.semaphore_in.acquire()
			#print "running..", index
			self.callable(index)
			#print "done..", index
			self.semaphore_out.release()
		
	def run_parallel(self, callable):
		self.callable = callable
		for thread in self.threads:
			self.semaphore_in.release()
		for thread in self.threads:
			self.semaphore_out.acquire()
		
from vaex.multithreading import ThreadPool
		

thread_pool = ThreadPool(8)

#vaex.vaexfast.histogram3d(x, y, z, None, density, -w, w, -w, w, -w, w)
density_per_thread = np.zeros((thread_pool.nthreads, ) + density.shape)
def calc_dens(index, i1, i2):
	vaex.vaexfast.histogram3d(x[i1:i2], y[i1:i2], z[i1:i2], None, density[index], -w, w, -w, w, -w, w)
thread_pool.run_blocks(calc_dens, len(x))
density  = np.sum(density_per_thread, axis=0)
#density = np.log10(density + 1)

fig, ax = plt.subplots()


def frame(i):
	global image
	global t_prev
	print("frame", i)
	angle1 = i / 40. * 2 * math.pi/4#/2
	angle2 = i / 80. * 2 * math.pi/4#/2
	#angle1, angle2 = 0, 0
	m = translate(N2d/2., N2d/2., N2d/2.) * scale(N2d/2.) * rotz((angle1)) * rotx((angle2)) * translate(-1, -1, -1) * scale(2./N)

	px = np.array(list(m[2].flat))
	py = np.array(list(m[1].flat))
	#print px, py
	surface = np.zeros((N2d,N2d))
	surface_per_thread = np.zeros((thread_pool.nthreads, N2d,N2d))
	block = density.shape[0]/thread_pool.nthreads
	#density_per_thread = [np.ascontiguousarray(density[index*block:(index+1)*block,:,:] * 1) for index in range(thread_pool.ntheads)]
	#for i in range(8):
	#	print "shape", i, density_per_thread[index].shape, density_per_thread[index].strides
		
	with vaex.utils.Timer("proj"):
		if 0:
			vaex.histogram.proj(density, surface, px, py)
		else:
			projection = np.array(list(px) + list(py))
			#density_per_thread = [density[index*block:(index+1)*block,:,:] for index in range(thread_pool.ntrheads)]
			def execute(index, i1, i2):
				#print "execute", index, density_per_thread[index].shape, density_per_thread[index].strides
				#print index, i1, i2
				center = np.array([0., 0., index*block])
				#vaex.vaexfast.project(density[index*block:(index+1)*block], surface_per_thread[index], projection, center)
				vaex.vaexfast.project(density[i1:i2], surface_per_thread[index], projection, center)
			#print [(index*block, (index+1)*block) for index in range(thread_pool.ntheads)]
			#dsa
			if 1:
				#thread_pool.run_parallel(execute)
				thread_pool.run_blocks(execute, density.shape[0])
			else:
				center = np.array([0., 0., 6*block])
				vaex.vaexfast.project(density_per_thread[0], surface_per_thread[0], projection, center)
			surface = surface_per_thread.sum(axis=0)
	#print surface
	#I = density.sum(axis=1)

	I = np.log10(surface+1)
	I = scipy.ndimage.gaussian_filter(I, 1.)
	mi, ma = I.min(), I.max()
	mi = mi + (ma-mi) * 0.4
	ma = ma - (ma-mi) * 0.4
	if i == 0:
		image = plt.imshow(I, cmap='PaulT_plusmin', interpolation='none', vmin=mi, vmax=ma)
		t_prev = time.time()
	else:
		t_now = time.time()
		print("fps", 1/(t_now - t_prev))
		t_prev = t_now
		image.set_data(I)
	return [image]
	#plt.show()
	
ax.hold(False)
ani = animation.FuncAnimation(fig, frame, 10000, interval=10, blit=True)
plt.show()
#data = dict(density=(density, "counts"))
#bbox = np.array([[np.min(x), np.max(x)], [np.min(y), np.max(y)], [np.min(z), np.max(z)]])






