__author__ = 'breddels'

import sys
import platform

import numpy as np
import h5py
from optparse import OptionParser
import os
import sys
import sampy

import gavi.utils

import encodings

# help py2app, it was missing this import
import PIL._imaging

darwin = "darwin" in platform.system().lower()
frozen = getattr(sys, 'frozen', False)


#print "DEFAULT ENCODING is: %s"%(sys.getdefaultencoding())
#print "FILE SYSTEM ENCODING is: %s"%(sys.getfilesystemencoding())
#if darwin:
if sys.getfilesystemencoding() == None: # TODO: why does this happen in pyinstaller?
	def getfilesystemencoding_wrapper():
		return "UTF-8"
	sys.getfilesystemencoding = getfilesystemencoding_wrapper


# on osx 10.8 we sometimes get pipe errors while printing, ignore these
import signal
#signal.signal(signal.SIGPIPE, signal.SIG_DFL)

import thread
import threading
import time
try:
	import pdb
	import astropy.io.fits
	#pdb.set_trace()
except Exception, e:
	print e
	pdb.set_trace()
import gavi.vaex.plot_windows as vp
from gavi.vaex.ranking import *
import gavi.vaex.undo
import gavi.selection
import gavi.kld
import gavi.utils

#import subspacefind
#import ctypes
import numexpr as ne

import imp

import gavi.logging
logger = gavi.logging.getLogger("gavi")
import logging
logger.setLevel(logging.DEBUG)

import sys
#import locale
#locale.setlocale(locale.LC_ALL, )

# samp stuff
#import astropy.io.votable
import thread
import threading
import time
from gavi.samp import Samp


custom = None
custompath = path = os.path.expanduser('~/.vaex/custom.py')
#print path
if os.path.exists(path):
	customModule = imp.load_source('gavi.custom', path)
	custom = customModule.Custom()
else:
	custom = None
	print >>sys.stderr, path, "does not exist"

#print "root path is", gavi.utils.get_root_path()


if getattr(sys, 'frozen', False):
	application_path = os.path.dirname(sys.executable)
elif __file__:
	application_path = os.path.dirname(__file__)


if not frozen: # astropy not working :s
	pass
	#import pdb
	#pdb.set_trace()
	# fix from Chris Beaumont
	#import astropy.logger
	#astropy.logger.log.disable_warnings_logging()
	__import__("astropy.io.votable")



# for osx
if "darwin" in platform.system().lower():
	application_path = os.path.abspath(".")

if 0:
	N = 1e9
	array = np.arange(N)
	counts = np.zeros(10, dtype=np.float64)
	if 0:
		#ptr = array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
		col_a = subspacefind.make_column(array)
		#col_b = subspacefind.make_column(array)
		col_b = subspacefind.SquareColumn(col_a)
		col_b2 = subspacefind.SquareColumn(col_b)
		col_b3 = subspacefind.SquareColumn(col_b2)
		col_b4 = subspacefind.SquareColumn(col_b3)
		col_c = subspacefind.DivConstColumn(col_b4, N**2)

		print col_a.get(10)
		#array[:] = array**2/N**2
		print col_a.get(10)
		print col_c.get(10)
		print sum(counts)

		with gavi.utils.Timer("hist"):
			subspacefind.histogram1d(col_c, counts, 0., 1.)
	else:
		step = 1000
		res = array[:len(array)/step] * 0.0
		Nstep = len(res)
		col_c = subspacefind.make_column(res)
		#ne.set_num_threads(10)
		vmax = None
		with gavi.utils.Timer("hist"):
			for i in range(step):
				a = array[i*Nstep:(i+1)*Nstep]
				#ne.evaluate("log(a)**2/N**2", out=res)
				ne.evaluate("sqrt(a**2+a**3)", out=res)
				if vmax is None:
					vmax = res.max()
				else:
					vmax = max(vmax, res.max())
				subspacefind.histogram1d(col_c, counts, 0., 1.)
		print  vmax
	print counts
	#col_a = subspacefind.DoubleColumn(array)
	sys.exit(0)
if 0:
	class NavigationToolbar(NavigationToolbarQt):
		def __init__(self, canvas, axes, parent):
			self.toolitems = [k for k in self.toolitems if (k[0] is None) or (k[0].lower() in "home pan zoom")]
			#self.toolitems.append()
			super(NavigationToolbar, self).__init__(canvas, parent)
			self.parent = parent
			self.axes = axes
			#self.basedir = os.path.join(matplotlib.rcParams[ 'datapath' ],'images')
			#print self.basedir
			self.extra_toolitems = [
				('Select', 'Select point', 'filesave', 'select_point'),
				('Lasso', 'Lasso select', 'matplotlib', 'select_lasso'),
							]
			for text, tooltip_text, image_file, callback in self.extra_toolitems:
				a = self.addAction(self._icon(image_file + '.png'), text, getattr(self, callback))
				a.setCheckable(True)
				self._actions[callback] = a
				if tooltip_text is not None:
					a.setToolTip(tooltip_text)
			self._idPress = None
			self._idRelease = None
			self.lasso = None

		def sync_buttons(self):
			self._actions['select_point'].setChecked(self._active == 'SELECT_POINT')
			self._actions['select_lasso'].setChecked(self._active == 'SELECT_LASSO')

		def select_point(self, *args):
			print self._active
			name = 'SELECT_POINT'
			self._active = None if self._active == name else name
			self.sync_buttons()
			if self._idPress is not None:
				self._idPress = self.canvas.mpl_disconnect(self._idPress)
				self._idRelease = self.canvas.mpl_disconnect(self._idRelease)

			if self.lasso:
				self.lasso.active = False
				self.lasso.disconnect_events()
				self.lasso = None

			if self._active == name:
				self._idPress = self.canvas.mpl_connect(
					'button_press_event', self.parent.press_select_point)
				self.canvas.widgetlock(self)
				self.set_message('select point')
			else:
				self.canvas.widgetlock.release(self)
				self.set_message('')

		def select_lasso(self, *args):
			print self._active
			name = 'SELECT_LASSO'
			self._active = None if self._active == name else name
			self.sync_buttons()
			if self._idPress is not None:
				self._idPress = self.canvas.mpl_disconnect(self._idPress)
				self._idRelease = self.canvas.mpl_disconnect(self._idRelease)

			if self._active == name:
				#self._idPress = self.canvas.mpl_connect('button_press_event', self.press_select_lasso)
				#self._idPress = self.canvas.mpl_connect('button_release_event', self.release_select_lasso)
				#self.canvas.widgetlock(self)
				#self.lasso = LassoSelector(evt.inaxes, (evt.xdata, evt.ydata), self.lasso_callback)
				self.lasso = LassoSelector(self.axes, self.lasso_callback)
				self.canvas.draw()
				self.set_message('lasso select point')
			else:
				#self.canvas.widgetlock.release(self)
				#self.set_message('')
				self.lasso.active = False
				self.lasso.disconnect_events()
				self.lasso = None

		def press_select_lasso(self, evt):
			print "lasso", self.canvas.widgetlock.locked()
			self.canvas.draw()
			if not self.canvas.widgetlock.locked():
				self.canvas.widgetlock(self.lasso)

		def release_select_lasso(self, evt):
			if self.lasso:
				self.canvas.widgetlock.release(self.lasso)
				self.lasso = None
			self.canvas.draw()


		def lasso_callback(self, vertices):
			#print vertices
			x, y = np.array(vertices).T
			mask = np.zeros(len(self.parent.datax), dtype=np.uint8)
			meanx = x.mean()
			meany = y.mean()
			radius = np.sqrt((meanx-x)**2 + (meany-y)**2).max()
			#print (x, y, self.parent.datax, self.parent.datay, mask, meanx, meany, radius)
			gavi.selection.pnpoly(x, y, self.parent.datax, self.parent.datay, mask, meanx, meany, radius)
			self.parent.set_mask(mask==1)

#from PySide import QtGui, QtCore
from gavi.vaex.qt import *
from gavi.vaex.table import *

from gavi.samp import Samp

usage = """
Convert VO table from SAMP to hdf5 format:

Example:

gavi-data-samp2hdf5 -o photometry.hdf5

Now open topcat:
topcat -f csv $GAVI_DATA/scl_photo.csv

...

"""
#parser = OptionParser(usage=usage)

#parser.add_option("-n", "--name",
 #                 help="dataset name [default=%default]", default="data", type=str)
#parser.add_option("-o", "--output",
#                 help="dataset output filename [by default the suffix of input filename will be replaced by hdf5]", default=None, type=str)
#(options, args) = parser.parse_args()

#if len(args) != 1:
#	print "Program requires output filename as argument"
#	sys.exit(1)


import h5py
import mmap

def error(title, msg):
	print "Error", title, msg

from gavi.dataset import *

possibleFractions = [10**base * f for base in [-3,-2,-1,0] for f in [0.25, 0.5, 0.75, 1.]]
possibleFractions.insert(0,10**-4)
#print possibleFractions

class DataList(QtGui.QListWidget):
	def __init__(self, parent):
		super(DataList, self).__init__(parent)
		self.icon = QtGui.QIcon('icons/png/24x24/devices/memory.png')
		self.datasets = []
		self.signal_pick = gavi.events.Signal("pick")
		self.signal_add_dataset = gavi.events.Signal("add dataset")

		self.signal_add_dataset.connect(self.on_add_dataset)
		#self.items

	def on_add_dataset(self, dataset):
		#print "added dataset", dataset
		self.datasets.append(dataset)
		dataset.signal_pick.connect(self.on_pick, dataset=dataset)

	def on_pick(self, row, dataset=None):
		# broadcast
		logger.debug("broadcast pick")
		self.signal_pick.emit(dataset, row)

	def __testfill(self):
		self.addHdf5("/home/data/gavi/gaussian3d-1e8-b.hdf5")
		self.addHdf5("/home/data/gavi/gaussian3d-1e9-b.hdf5")
		self.addHdf5("/home/data/gavi/rave/rave-dr5-shuffled.hdf5")
		self.addHdf5("/home/data/gavi/helmi2000-FeH-s2.hdf5")

		#self.addGadgetHdf5("/Users/users/breddels/mab/models/nbody/gcsink/gadget/sink_nfw_soft/output/snapshot_213.hdf5")
		#self.addGadgetHdf5("/Users/users/breddels/mab/models/nbody/gcsink/gadget/sink_einasto_2kpc_fornax/IC.hdf5")
		#self.addGadgetHdf5("/Users/users/breddels/mab/models/nbody/hernquist/gadget/hernquist_half/output/snapshot_000.hdf5")

		self.addHdf5("/Users/maartenbreddels/gavi/src/SubspaceFinding/data/gaussian4d-1e7.hdf5")
		self.addHdf5("/Users/maartenbreddels/gavi/src/SubspaceFinding/data/helmi2000-FeH-s2.hdf5")

		#self.addGadget2("/home/data/gavi/egpbos/snap_008")
		self.addHdf5("/Users/users/breddels/gavi/src/SubspaceFinding/data/helmi2000-FeH-s2-shuffled.hdf5")

		try:
			hmmap = HansMemoryMapped("data/Orbitorb9.ac8.10000.100.5.orb.bin", "data/Orbitorb9.ac8.10000.100.5.orb.omega2")
			self.addDataset(hmmap)
		except:
			print "oops"
		self.addHdf5('/home/data/gavi/Aq-A-2-999-shuffled.hdf5')

		#self.addGadgetHdf5('/home/data/gavi/snap_800.hdf5')
		# 0 - gas
		# 1 - halo
		# 2 disk
		# 4 new stars
		# 5 sat

		for index, name in list(enumerate("gas halo disk stars sat".split()))[::-1]:
			self.addGadgetHdf5('data/disk2nv_N6N5_z0.1h_RfAs0.5H_no_H1_0.5_nH01_vw5s_ml50_st-snap_800.hdf5', name, index)
		for index, name in list(enumerate("gas halo disk stars sat".split()))[::-1]:
			self.addGadgetHdf5('/home/data/gavi/oldplanar_c15_md0.002_z0.1h_H4_0.5_nH01_vw5s_ml30_sM2e9-snap_400.hdf5', name, index)



	def setBestFraction(self, dataset):
		Nmax = 1000*1000*10
		for fraction in possibleFractions[::-1]:
			N  = dataset.current_slice[1] - dataset.current_slice[0]
			if N > Nmax:
				dataset.setFraction(fraction)
			else:
				break


	def addDataset(self, dataset):
		self.setBestFraction(dataset)
		item = QtGui.QListWidgetItem(self)
		item.setText(dataset.name)
		#self.icon = QIcon.fromTheme('document-open')
		item.setIcon(self.icon)
		item.setToolTip("file: " +dataset.filename)
		item.setData(QtCore.Qt.UserRole, dataset)
		self.setCurrentItem(item)
		self.signal_add_dataset.emit(dataset)

	def addGadget2(self, filename, auto_fraction=True):
		if not os.path.exists(filename):
			return

		self.hdf5file = MemoryMappedGadget(filename)
		if auto_fraction:
			self.setBestFraction(self.hdf5file)
		item = QtGui.QListWidgetItem(self)
		item.setText(self.hdf5file.name)
		item.setToolTip("file: " +self.hdf5file.filename)
		#self.icon = QIcon.fromTheme('document-open')
		item.setIcon(self.icon)
		item.setData(QtCore.Qt.UserRole, self.hdf5file)
		self.setCurrentItem(item)
		dataset = self.hdf5file
		self.signal_add_dataset.emit(dataset)



	def addGadgetHdf5(self, filename, name, particleType):
		if not os.path.exists(filename):
			print "does not exist", filename
			return
		try:
			self.hdf5file = Hdf5MemoryMappedGadget(filename, name, particleType)
		except KeyError, e:
			print "error", e
			logger.exception("loading gadget hdf5")
			return
		self.setBestFraction(self.hdf5file)
		item = QtGui.QListWidgetItem(self)
		item.setText(self.hdf5file.name)
		#self.icon = QIcon.fromTheme('document-open')
		item.setToolTip("file: " +self.hdf5file.filename)
		item.setIcon(self.icon)
		item.setData(QtCore.Qt.UserRole, self.hdf5file)
		self.setCurrentItem(item)
		dataset = self.hdf5file
		self.signal_add_dataset.emit(dataset)

	def addHdf5(self, filename, auto_fraction=True):
		if not os.path.exists(filename):
			return

		self.hdf5file = Hdf5MemoryMapped(filename)
		if auto_fraction:
			self.setBestFraction(self.hdf5file)
		item = QtGui.QListWidgetItem(self)
		item.setText(self.hdf5file.name)
		item.setToolTip("file: " +self.hdf5file.filename)
		#self.icon = QIcon.fromTheme('document-open')
		item.setIcon(self.icon)
		item.setData(QtCore.Qt.UserRole, self.hdf5file)
		self.setCurrentItem(item)
		dataset = self.hdf5file
		self.signal_add_dataset.emit(dataset)

	def addAmuse(self, filename, auto_fraction=True):
		if not os.path.exists(filename):
			return

		self.hdf5file = AmuseHdf5MemoryMapped(filename)
		if auto_fraction:
			self.setBestFraction(self.hdf5file)
		item = QtGui.QListWidgetItem(self)
		item.setText(self.hdf5file.name)
		item.setToolTip("file: " +self.hdf5file.filename)
		#self.icon = QIcon.fromTheme('document-open')
		item.setIcon(self.icon)
		item.setData(QtCore.Qt.UserRole, self.hdf5file)
		self.setCurrentItem(item)
		dataset = self.hdf5file
		self.signal_add_dataset.emit(dataset)

	def addFits(self, filename, auto_fraction=True):
		#if not os.path.exists(filename):
		#	return

		self.hdf5file = FitsBinTable(filename)
		if auto_fraction:
			self.setBestFraction(self.hdf5file)
		item = QtGui.QListWidgetItem(self)
		item.setText(self.hdf5file.name)
		item.setToolTip("file: " +self.hdf5file.filename)
		#self.icon = QIcon.fromTheme('document-open')
		item.setIcon(self.icon)
		item.setData(QtCore.Qt.UserRole, self.hdf5file)
		self.setCurrentItem(item)
		#self.setCurrentRow(0)
		dataset = self.hdf5file
		self.signal_add_dataset.emit(dataset)

	def _addHdf5(filename, columns):
		h5file = h5py.File(filename)


		print f
		print fileno
		mapping = mmap.mmap(fileno, 0, prot=mmap.PROT_READ)


class Worker(QtCore.QThread):
	def __init__(self, parent, name, func, *args, **kwargs):
		QtCore.QThread.__init__(self, parent=None)
		self.func = func
		self.args = args
		self.kwargs = kwargs
		self.name = name
		self.signal = QtCore.SIGNAL("signal")

	def run(self):
		time.sleep(0.1)
		print "in thread", self.currentThreadId()
		self.result = self.func(*self.args, **self.kwargs)
		print "result:", self.result
		#self.emit(self.signal, self.result)
		#self.exec_()
import multiprocessing

def MyStats(object):
	def __init__(self, data):
		self.data = data

	def __call___(self, args):
		print args
		#stat_name, column_name = args
		#print "do", stat_name, "on", column_name
		return 1
		#f = stats[stat_name]
		#return column_name, stat_name, f(self.data.columns[column_name])

#stats = {"minimum": lambda x: str(np.nanmin(x)), "maximum": lambda x: str(np.nanmax(x)), "mean": lambda x: str(np.mean(x)), "std": lambda x: str(np.std(x)), "median": lambda x: str(np.median(x))}
stats = {"minimum": lambda x: str(np.nanmin(x)), "maximum": lambda x: str(np.nanmax(x)), "mean": lambda x: str(np.mean(x)), "std": lambda x: str(np.std(x))}
def statsrun(args):
	columns, stat_name, column_name = args
	f = stats[stat_name]
	#print args
	return 1

class StatWorker(QtCore.QThread):
	def __init__(self, parent, data):
		QtCore.QThread.__init__(self, parent=parent)
		self.data = data

	def run(self):
		time.sleep(0.1)
		print "in thread", self.currentThreadId()
		jobs = [(stat_name, column_name) for stat_name in stats.keys() for column_name in self.data.columns.keys()]
		@parallelize(cores=QtCore.QThread.idealThreadCount())
		def dostats(args, data=self.data):
			stat_name, column_name = args
			columns = data.columns
			f = stats[stat_name]
			result = f(columns[column_name][slice(*data.current_slice)])
			print result
			return result
		values = dostats(jobs)
		self.results = {}
		for job, value in zip(jobs, values):
			stat_name, column_name = job
			if stat_name not in self.results:
				self.results[stat_name] = {}
			self.results[stat_name][column_name] = value
		print "results", self.results




from gavi.parallelize import parallelize


class StatisticsDialog(QtGui.QDialog):
	def __init__(self, parent, data):
		super(StatisticsDialog, self).__init__(parent)
		self.data = data

		#self.form_layout = QtGui.QFormLayout()
		#self.min = QtGui.QLabel('...computing...', self)
		#self.form_layout.addRow('Minimum:', self.min)
		#self.setLayout(self.form_layout)

		self.boxlist = QtGui.QHBoxLayout(self)

		self.headers = ['minimum', 'maximum', 'mean', 'std']

		#WorkerMinimum = lambda parent, data, column_name: Worker(parent, 'minimum', lambda data, column_name: str(min(data.columns[column_name])), data=data, column_name=column_name)
		#WorkerMaximum = lambda parent, data, column_name: Worker(parent, 'maximum', lambda data, column_name: str(max(data.columns[column_name])), data=data, column_name=column_name)
		#self.workers = {'minimum':WorkerMinimum, 'maximum': WorkerMaximum}

		self.table = QtGui.QTableWidget(data.nColumns, len(self.headers), self)
		self.table.setHorizontalHeaderLabels(self.headers)
		self.table.setVerticalHeaderLabels(self.data.columns.keys())




		#pool = multiprocessing.Pool() #processes=QtCore.QThread.idealThreadCount())
		#print "jobs:", jobs
		worker = StatWorker(self, self.data)
		def onFinish(worker=worker):
			for column, stat in enumerate(self.headers):
				for row, column_name in enumerate(self.data.columns.keys()):
					value = worker.results[stat][column_name]
					item = QtGui.QTableWidgetItem(value)
					self.table.setItem(row, column, item)


		worker.finished.connect(onFinish)
		worker.start()
		#for name in self.header:
		#for column_name in self.data.colums.keys():
		#	self.table.set
		#worker.finished.connect(onFinish)
		if 0:
			self.worker_list = [] # keep references
			def onFinish():
				for column, stat in enumerate(self.headers):
					for row, column_name in enumerate(self.data.columns.keys()):
						value = worker.results[stat][column_name]
						item = QtGui.QTableWidgetItem(worker.result)
						self.table.setItem(row, column, item)
			for column, stat in enumerate(self.headers):
				for row, column_name in enumerate(self.data.columns.keys()):
					worker = self.workers[stat](parent, data, column_name)
					def onFinish(worker=worker, row=row, column=column):
						print "finished running", worker.result
						item = QtGui.QTableWidgetItem(worker.result)
						self.table.setItem(row, column, item)
					worker.finished.connect(onFinish)
					print "starting", row, column
					worker.start(QtCore.QThread.IdlePriority)
					self.worker_list.append(worker) # keeps reference to avoid GC


		self.boxlist.addWidget(self.table)
		self.setLayout(self.boxlist)




		if 0:
			#w1 = Worker(self, lambda data: str(min(data.columns.items()[0])), self.data)
			self.w1 = Worker(self, self.test, self.data)
			#self.connect(self.w1, self.w1.signal, self.setmin)
			def setmin():
				print self.min.setText(self.w1.result)
			self.w1.finished.connect(setmin)
			self.w1.start()

	def test(self, data):
		print "test"
		data = data.columns.values()[0]
		return str(min(data))
		#return "test"
	def onFinish(self, worker):
		print "worker", worker
		#print "setting", result
		#self.min = str

import 	operator
import random

class MainPanel(QtGui.QFrame):
	def __init__(self, parent, dataset_list):
		super(MainPanel, self).__init__(parent)

		self.jobsManager = gavi.dataset.JobsManager()
		self.dataset = None
		self.dataset_list = dataset_list

		self.undoManager = gavi.vaex.undo.UndoManager()

		self.form_layout = QtGui.QFormLayout()

		self.name = QtGui.QLabel('', self)
		self.form_layout.addRow('Name:', self.name)

		self.columns = QtGui.QLabel('', self)
		self.form_layout.addRow('Columns:', self.columns)

		self.length = QtGui.QLabel('', self)
		self.form_layout.addRow('Length:', self.length)

		#self.histogramButton = QtGui.QPushButton('histogram (1d)', self)
		self.histogramButton = QtGui.QToolButton(self)
		self.histogramButton.setText('histogram (1d)')
		self.form_layout.addRow('Plotting:', self.histogramButton)

		self.scatterButton = QtGui.QToolButton(self)
		self.scatterButton.setText('x/y density')
		self.form_layout.addRow('', self.scatterButton)

		self.scatter3dButton = QtGui.QToolButton(self)
		self.scatter3dButton.setText('x/y/z density')
		self.form_layout.addRow('', self.scatter3dButton)
		if 0:


			self.scatter1dSeries = QtGui.QPushButton('series', self)
			self.form_layout.addRow('', self.scatter1dSeries)

			self.scatter2dSeries = QtGui.QPushButton('x/y series', self)
			self.form_layout.addRow('', self.scatter2dSeries)

		if 0:
			self.serieSlice = QtGui.QToolButton(self)
			self.serieSlice.setText('serie slice')
			self.form_layout.addRow('', self.serieSlice)

		self.statistics = QtGui.QPushButton('Statistics', self)
		self.form_layout.addRow('Data:', self.statistics)

		self.rank = QtGui.QPushButton('Rank subspaces', self)
		self.form_layout.addRow('', self.rank)

		self.table = QtGui.QPushButton('Open table', self)
		self.form_layout.addRow('', self.table)

		if 0:
			self.button_variables = QtGui.QPushButton('Variables', self)
			self.form_layout.addRow('', self.button_variables)


		self.fractionLabel = QtGui.QLabel('Fraction used: ...')
		self.fractionWidget = QtGui.QWidget(self)
		self.fractionLayout = QtGui.QHBoxLayout(self.fractionWidget)
		self.fractionSlider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
		self.fractionSlider.setMinimum(0)
		self.fractionSlider.setMaximum(len(possibleFractions)-1)
		self.numberLabel = QtGui.QLabel('')

		self.fractionLayout.addWidget(self.fractionSlider)
		self.fractionLayout.addWidget(self.numberLabel)
		self.fractionWidget.setLayout(self.fractionLayout)
		#self.fractionSlider.setTickInterval(len(possibleFractions))
		self.form_layout.addRow(self.fractionLabel, self.fractionWidget)


		self.fractionSlider.sliderReleased.connect(self.onFractionSet)
		self.fractionSlider.valueChanged.connect(self.onValueChanged)
		self.onValueChanged(0)


		self.histogramButton.clicked.connect(self.onOpenHistogram)
		self.statistics.clicked.connect(self.onOpenStatistics)
		self.scatterButton.clicked.connect(self.onOpenScatter)
		self.scatter3dButton.clicked.connect(self.onOpenScatter3d)
		#self.scatter1dSeries.clicked.connect(self.onOpenScatter1dSeries)
		#self.scatter2dSeries.clicked.connect(self.onOpenScatter2dSeries)
		#self.serieSlice.clicked.connect(self.onOpenSerieSlice)
		self.rank.clicked.connect(self.onOpenRank)
		self.table.clicked.connect(self.onOpenTable)

		self.setLayout(self.form_layout)
		self.plot_dialogs = []
		self.signal_open_plot = gavi.events.Signal("open plot")

	def onOpenStatistics(self):
		#print "open", self.dataset
		if self.dataset is not None:
			dialog = StatisticsDialog(self, self.dataset)
			dialog.show()
			#print "show"

	def onOpenScatter(self):
		#print "open", self.dataset
		if self.dataset is not None:
			xname, yname = self.dataset.column_names[:2]
			self.plotxy(xname, yname)

	def onOpenScatter3d(self):
		if self.dataset is not None:
			xname, yname, zname = self.dataset.column_names[:3]
			self.plotxyz(xname, yname, zname)

	def onOpenSerieSlice(self):
		if self.dataset is not None:
			xname, yname = self.dataset.rank1names[:2]
			self.plotseriexy(xname, yname)

	def onOpenScatter1dSeries(self):
		if self.dataset is not None:
			dialog = vp.SequencePlot(self, self.jobsManager, self.dataset)
			dialog.show()
			self.jobsManager.execute()

	def onOpenScatter2dSeries(self):
		if self.dataset is not None:
			dialog = vp.ScatterSeries2dPlotDialog(self, self.dataset)
			dialog.show()

	def onOpenHistogram(self):
		if self.dataset is not None:
			xname = self.dataset.column_names[0]
			self.histogram(xname)

	def plotxy(self, xname, yname, **kwargs):
		dialog = vp.ScatterPlotDialog(self, self.jobsManager, self.dataset, **kwargs)
		dialog.add_layer([xname, yname], self.dataset, **kwargs)
		dialog.show()
		self.plot_dialogs.append(dialog)
		self.jobsManager.execute()
		self.signal_open_plot.emit(dialog)
		return dialog

	def plotxyz(self, xname, yname, zname, **kwargs):
		dialog = vp.VolumeRenderingPlotDialog(self, self.jobsManager, self.dataset, **kwargs)
		dialog.add_layer([xname, yname, zname], **kwargs)
		dialog.show()
		self.plot_dialogs.append(dialog)
		self.jobsManager.execute()
		self.signal_open_plot.emit(dialog)
		return dialog

	def plotmatrix(self, *expressions):
		dialog = vp.ScatterPlotMatrixDialog(self, self.jobsManager, self.dataset, expressions)
		dialog.show()
		self.jobsManager.execute()
		return dialog

	def plotxyz_old(self, xname, yname, zname):
		dialog = vp.PlotDialog3d(self, self.dataset, xname, yname, zname)
		dialog.show()

	def histogram(self, xname, **kwargs):
		dialog = vp.HistogramPlotDialog(self, self.jobsManager, self.dataset, **kwargs)
		dialog.add_layer([xname], **kwargs)
		dialog.show()
		self.plot_dialogs.append(dialog)
		self.jobsManager.execute()
		self.signal_open_plot.emit(dialog)
		return dialog

	def onOpenRank(self):
		if self.dataset is not None:
			self.ranking()

	def onOpenTable(self):
		if self.dataset is not None:
			self.tableview()

	def onFractionSet(self):
		index = self.fractionSlider.value()
		fraction = possibleFractions[index]
		if self.dataset:
			self.dataset.setFraction(fraction)
			self.numberLabel.setText("{:,}".format(len(self.dataset)))
			self.jobsManager.execute()

	def onValueChanged(self, index):
		fraction = possibleFractions[index]
		text = 'Fraction used: %9.4f%%' % (fraction*100)
		self.fractionLabel.setText(text)

	def onDataSelected(self, data_item, previous):
		if data_item is not None:
			data = data_item.data(QtCore.Qt.UserRole)
			if hasattr(data, "toPyObject"):
				data = data.toPyObject()
			self.dataset = data
			self.dataset = data
			self.name.setText(data.name)
			self.columns.setText(str(len(data.columns)))
			self.length.setText("{:,}".format(self.dataset.full_length()))
			self.numberLabel.setText("{:,}".format(len(self.dataset)))
			fraction = self.dataset.fraction
			distances = np.abs(np.array(possibleFractions) - fraction)
			index = np.argsort(distances)[0]
			self.fractionSlider.setValue(index) # this will fire an event and execute the above event code
			self.scatterButton.setEnabled(len(self.dataset.columns) > 0)
			#self.scatter2dSeries.setEnabled(len(self.dataset.rank1s) >= 2)
			#self.scatter3dButton.setEnabled(False)
			#self.scatter1dSeries.setEnabled(len(self.dataset.rank1s) >= 1)
			#self.serieSlice.setEnabled(len(self.dataset.rank1s) >= 2)

			self.histogramMenu = QtGui.QMenu(self)
			for column_name in self.dataset.get_column_names():
				#action = QtGui.QAction
				#QtGui.QAction(QtGui.QIcon(iconfile('glue_cross')), '&Pick', self)
				action = QtGui.QAction(column_name, self)
				action.triggered.connect(functools.partial(self.histogram, xname=column_name))
				self.histogramMenu.addAction(action)
			self.histogramButton.setMenu(self.histogramMenu)

			self.scatterMenu = QtGui.QMenu(self)
			for column_name1 in self.dataset.get_column_names():
				#action1 = QtGui.QAction(column_name, self)
				submenu = self.scatterMenu.addMenu(column_name1)
				for column_name2 in self.dataset.get_column_names():
					action = QtGui.QAction(column_name2, self)
					action.triggered.connect(functools.partial(self.plotxy, xname=column_name1, yname=column_name2))
					submenu.addAction(action)
			self.scatterButton.setMenu(self.scatterMenu)

			self.scatterMenu3d = QtGui.QMenu(self)
			for column_name1 in self.dataset.get_column_names():
				#action1 = QtGui.QAction(column_name, self)
				submenu = self.scatterMenu3d.addMenu(column_name1)
				for column_name2 in self.dataset.get_column_names():
					subsubmenu = submenu.addMenu(column_name2)
					for column_name3 in self.dataset.get_column_names():
						action = QtGui.QAction(column_name3, self)
						action.triggered.connect(functools.partial(self.plotxyz, xname=column_name1, yname=column_name2, zname=column_name3))
						subsubmenu.addAction(action)
			self.scatter3dButton.setMenu(self.scatterMenu3d)

			if 0:
				self.serieSliceMenu = QtGui.QMenu(self)
				for column_name1 in self.dataset.rank1names:
					#action1 = QtGui.QAction(column_name, self)
					submenu = self.serieSliceMenu.addMenu(column_name1)
					for column_name2 in self.dataset.rank1names:
						action = QtGui.QAction(column_name2, self)
						action.triggered.connect(functools.partial(self.plotseriexy, xname=column_name1, yname=column_name2))
						submenu.addAction(action)
				self.serieSlice.setMenu(self.serieSliceMenu)

	def plotseriexy(self, xname, yname):
		if self.dataset is not None:
			dialog = vp.Rank1ScatterPlotDialog(self, self.jobsManager, self.dataset, xname+"[index]", yname+"[index]")
			self.plot_dialogs.append(dialog)
			self.jobsManager.execute()
			dialog.show()

	def tableview(self):
		dialog = TableDialog(self.dataset, self)
		dialog.show()
		return dialog

	def ranking(self):
		dialog = RankDialog(self.dataset, self, self)
		dialog.show()
		return dialog


from numba import jit
import numba
#print numba.__version__
import math
#@jit('(f8[:],f8[:], i4[:,:], f8, f8, f8, f8)')
@jit(nopython=True)
def histo2d(x, y, counts, dataminx, datamaxx, dataminy, datamaxy):
	length = len(x)
	#counts = np.zeros((bincountx, bincounty), dtype=np.int32)
	bincountx, bincounty = counts.shape
	#print length
	#return bindata#
	for i in range(length):
		binNox = int(math.floor( ((float(x[i]) - dataminx) / (float(datamaxx) - dataminx)) * float(bincountx)))
		binNoy = int(math.floor( ((float(y[i]) - dataminy) / (float(datamaxy) - dataminy)) * float(bincounty)))
		if binNox >= 0 and binNox < bincountx and binNoy >= 0 and binNoy < bincounty:
			counts[binNox, binNoy] += 1
	#step = float(datamax-datamin)/bincount
	#return numpy.arange(datamin, datamax+step/2, step), binData
	return counts
	#for i in range(N):
	#	offset = data[

@jit(nopython=True)
def find_nearest_index(datax, datay, x, y, wx, wy):
	N = len(datax)
	index = 0
	mindistance = math.sqrt((datax[0]-x)**2/wx**2 + (datay[0]-y)**2/wy**2)
	for i in range(1,N):
		distance = math.sqrt((datax[i]-x)**2/wx**2 + (datay[i]-y)**2/wy**2)
		if distance < mindistance:
			mindistance = distance
			index = i
	return index

@jit(nopython=True)
def find_nearest_index1d(datax, x):
	N = len(datax)
	index = 0
	mindistance = math.sqrt((datax[0]-x)**2)
	for i in range(1,N):
		distance = math.sqrt((datax[i]-x)**2)
		if distance < mindistance:
			mindistance = distance
			index = i
	return index



#import mab.utils.numpy

import psutil

class WidgetUsage(QtGui.QWidget):
	def __init__(self, parent):
		super(WidgetUsage, self).__init__(parent)
		self.setMinimumHeight(16)
		self.setMinimumWidth(100)
		self.timer = QtCore.QTimer(self)
		self.timer.timeout.connect(self.update)
		self.timer.start(500)
		self.t_prev = time.time()
		self.bytes_read_prev = psutil.disk_io_counters().read_bytes

	def paintEvent(self, event):
		painter = QtGui.QPainter()
		painter.begin(self)
		painter.fillRect(event.rect(), QtGui.QBrush(QtCore.Qt.white))
		size = self.size()
		width, height = size.width(), size.height()
		#print height
		self.tool_lines = []
		#self.tool_text = ""
		try:
			def drawbar(index, count, fraction, color=QtCore.Qt.red):
				if fraction == fraction: # check nan
					#print "bar", index, count, height * (index)/ count, height * (index+1)/ count
					rect = QtCore.QRect(0, height * (index)/ count, int(width*fraction+0.5), height/count)
					#painter.setBrush(QtGui.QBrush(QtCore.Qt.blue))
					painter.fillRect(rect, QtGui.QBrush(color))

			cpu_fraction = psutil.cpu_percent()/100.
			#print cpu_fraction
			drawbar(0, 4, cpu_fraction, QtCore.Qt.green)
			self.tool_lines.append("Cpu usage: %.1f%%" % (cpu_fraction*100,))


			vmem = psutil.virtual_memory()
			mem_fraction = (vmem.total-vmem.available) * 1./vmem.total
			self.tool_lines.append("Virtual memory: %s used of %s (=%.1f%%)%%" % (gavi.utils.filesize_format(vmem.total-vmem.available), gavi.utils.filesize_format(vmem.total), mem_fraction*100.))
			drawbar(1, 4, mem_fraction, QtCore.Qt.red)

			swapmem = psutil.swap_memory()
			swap_fraction = swapmem.used * 1./swapmem.total
			drawbar(2, 4, swap_fraction, QtCore.Qt.blue)
			self.tool_lines.append("Swap memory: %s used of %s (=%.1f%%)" % (gavi.utils.filesize_format(swapmem.used), gavi.utils.filesize_format(swapmem.total), swap_fraction*100.))

			self.t_now = time.time()
			self.bytes_read_new = psutil.disk_io_counters().read_bytes
			bytes_per_second = (self.bytes_read_new - self.bytes_read_prev) / (self.t_now - self.t_prev)
			Mbytes_per_second = bytes_per_second/1024**2
			# go from 1 mb to 10*1024 mb/s in log spacing
			disk_fraction = np.clip(np.log2(Mbytes_per_second)/np.log2(10*1024), 0, 1)
			drawbar(3, 4, disk_fraction, QtCore.Qt.magenta)
			self.tool_lines.append("Reading at %.2f MiB/s" % (Mbytes_per_second,))



			self.t_prev = self.t_now
			self.bytes_read_prev = self.bytes_read_new



			self.tool_text = "\n".join(self.tool_lines)
			painter.end()
			self.setToolTip(self.tool_text)
		except:
			pass

class Vaex(QtGui.QMainWindow):

	signal_samp_notification = QtCore.pyqtSignal(str, str, str, dict, dict)
	signal_samp_call = QtCore.pyqtSignal(str, str, str, str, dict, dict)

	def __init__(self, argv):
		super(Vaex, self).__init__()

		self.initUI(argv)

	def initUI(self, argv):

		QtGui.QToolTip.setFont(QtGui.QFont('SansSerif', 10))

		#self.setToolTip('This is a <b>QWidget</b> widget')


		if 0:
			qbtn = QtGui.QPushButton('Quit', self)
			qbtn.clicked.connect(QtCore.QCoreApplication.instance().quit)
			qbtn.resize(qbtn.sizeHint())
			qbtn.move(150, 150)

			btn = QtGui.QPushButton('Button', self)
			btn.setToolTip('This is a <b>QPushButton</b> widget')
			btn.resize(btn.sizeHint())
			btn.move(50, 50)


		#self.setGeometry(300, 300, 250, 150)
		self.resize(700,500)
		#self.center()
		#self.setWindowTitle('Gavi samp test')
		self.setWindowTitle(u'V\xe6X v' + gavi.vaex.__version__)
		#self.statusBar().showMessage('Ready')

		self.toolbar = self.addToolBar('Main toolbar')


		self.left = QtGui.QFrame(self)
		self.left.setFrameShape(QtGui.QFrame.StyledPanel)

		self.list = DataList(self.left)
		self.list.setMinimumWidth(300)

		self.right = MainPanel(self, self.list.datasets) #QtGui.QFrame(self)
		self.right.setFrameShape(QtGui.QFrame.StyledPanel)
		self.main_panel = self.right

		self.splitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
		self.splitter.addWidget(self.left)
		self.splitter.addWidget(self.right)

		#self.hbox = QtGui.QHBoxLayout(self)
		#self.hbox.addWidget(self.splitter)
		self.setCentralWidget(self.splitter)
		#self.setLayout(self.hbox)


		# this widget uses a time which causes an fps drop for opengl
		#self.widget_usage = WidgetUsage(self.left)

		#self.list.resize(30

		self.boxlist = QtGui.QVBoxLayout(self.left)
		self.boxlist.addWidget(self.list)
		#self.boxlist.addWidget(self.widget_usage)
		self.left.setLayout(self.boxlist)

		#self.list.currentItemChanged.connect(self.infoPanel.onDataSelected)
		self.list.currentItemChanged.connect(self.right.onDataSelected)
		#self.list.testfill()

		self.show()
		self.raise_()

		#self.list.itemSelectionChanged.connect(self.right.onDataSelected)



		#self.action_open = QtGui.QAction(vp.iconfile('quickopen-file', '&Open', self)
		#self.action_open.
		self.action_open_hdf5_gadget = QtGui.QAction(QtGui.QIcon(vp.iconfile('table-import')), '&Open gadget hdf5', self)
		self.action_open_hdf5_gavi = QtGui.QAction(QtGui.QIcon(vp.iconfile('table-import')), '&Open GAIA hdf5', self)
		self.action_open_hdf5_amuse = QtGui.QAction(QtGui.QIcon(vp.iconfile('table-import')), '&Open Amuse hdf5', self)
		self.action_open_fits = QtGui.QAction(QtGui.QIcon(vp.iconfile('table-import')), '&Open FITS (binary table)', self)


		self.action_save_hdf5 = QtGui.QAction(QtGui.QIcon(vp.iconfile('table-export')), '&Export to hdf5', self)

		exitAction = QtGui.QAction(QtGui.QIcon('icons/png/24x24/actions/application-exit-2.png'), '&Exit', self)
		exitAction.setShortcut('Ctrl+Q')
		exitAction.setShortcut('Alt+Q')
		exitAction.setStatusTip('Exit application')
		exitAction.triggered.connect(QtGui.qApp.quit)
		self.samp = None


		menubar = self.menuBar()
		fileMenu = menubar.addMenu('&File')
		self.menu_open = fileMenu.addMenu("&Open")
		self.menu_open.addAction(self.action_open_hdf5_gadget)
		self.menu_open.addAction(self.action_open_hdf5_gavi)
		self.menu_open.addAction(self.action_open_hdf5_amuse)
		if (not frozen) or darwin:
			self.menu_open.addAction(self.action_open_fits)
		fileMenu.addAction(self.action_save_hdf5)
		#fileMenu.addAction(self.action_open)
		fileMenu.addAction(exitAction)


		self.menu_data = menubar.addMenu('&Data')
		def check_memory(bytes):
			if bytes > psutil.virtual_memory().available:
				if bytes < (psutil.virtual_memory().available +psutil.swap_memory().free):
					text = "Action requires %s, you have enough swap memory available but it will make your computer slower, do you want to continue?" % (gavi.utils.filesize_format(bytes),)
					return confirm(self, "Memory usage issue", text)
				else:
					text = "Action requires %s, you do not have enough swap memory available, do you want try anyway?" % (gavi.utils.filesize_format(bytes),)
					return confirm(self, "Memory usage issue", text)

			return True
		for level in [20, 25, 27, 29, 30, 31, 32]:
			N = 2**level
			action = QtGui.QAction('Generate Soneira Peebles fractal: N={:,}'.format(N), self)
			def do(ignore=None, level=level):
				if level < 29:
					if check_memory(4*8*2**level):
						sp = SoneiraPeebles(dimension=4, eta=2, max_level=level, L=[1.1, 1.3, 1.6, 2.])
						self.list.addDataset(sp)
				else:
					if check_memory(2*8*2**level):
						sp = SoneiraPeebles(dimension=2, eta=2, max_level=level, L=[1.6, 2.])
						self.list.addDataset(sp)
			action.triggered.connect(do)
			self.menu_data.addAction(action)

		for dim in [2,3]:
			if dim == 3:
				res = [128, 256, 512, 1024]
			if dim == 2:
				res = [512, 1024, 2048]
			for N in res:
				for power in [-1.5, -2.5]:
					count = N**dim
					name = 'Zeldovich d={dim} N={N:,}, count={count:,} powerspectrum={power:}'.format(**locals())
					action = QtGui.QAction('Generate '+name, self)
					def do(ignore=None, dim=dim, N=N, power=power, name=name):
						t = None
						z = gavi.dataset.Zeldovich(dim, N, power, t, name=name)
						self.list.addDataset(z)
					action.triggered.connect(do)
					self.menu_data.addAction(action)


		use_toolbar = "darwin" not in platform.system().lower()
		use_toolbar = True
		self.toolbar.setIconSize(QtCore.QSize(16, 16))
		#self.toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)

		#self.toolbar.addAction(exitAction)

		self.action_samp_connect = QtGui.QAction(QtGui.QIcon(vp.iconfile('plug-connect')), 'Connect to SAMP HUB', self)
		self.action_samp_connect.setShortcut('Alt+S')
		self.action_samp_connect.setCheckable(True)
		if use_toolbar:
			self.toolbar.addAction(self.action_samp_connect)
		self.action_samp_connect.triggered.connect(self.onSampConnect)

		if 1:
			self.action_samp_table_send = QtGui.QAction(QtGui.QIcon(vp.iconfile('table--arrow')), 'Send active dataset via SAMP', self)
			self.action_samp_table_send.setShortcut('Alt+T')
			if use_toolbar:
				self.toolbar.addAction(self.action_samp_table_send)
			self.action_samp_table_send.triggered.connect(self.onSampSend)

		self.action_samp_sand_table_select_row_list = QtGui.QAction(QtGui.QIcon(vp.iconfile('block--arrow')), 'Send selection via SAMP(table.select.rowlist)', self)
		self.action_samp_sand_table_select_row_list.setShortcut('Alt+R')
		if use_toolbar:
			self.toolbar.addAction(self.action_samp_sand_table_select_row_list)
		self.action_samp_sand_table_select_row_list.triggered.connect(self.on_samp_send_table_select_rowlist)

		self.toolbar.addSeparator()

		self.action_save_hdf5.triggered.connect(self.onSaveTable)

		self.sampMenu = menubar.addMenu('&Samp')
		self.sampMenu.addAction(self.action_samp_connect)
		#self.sampMenu.addAction(self.action_samp_table_send)
		self.sampMenu.addAction(self.action_samp_sand_table_select_row_list)


		if use_toolbar:
			#self.toolbar.addAction(self.action_open_hdf5_gadget)
			#self.toolbar.addAction(self.action_open_hdf5_gavi)
			#if (not frozen) or darwin:
			#	self.toolbar.addAction(self.action_open_fits)
			self.toolbar.addAction(self.action_save_hdf5)

		if len(argv) == 0:
			if custom is not None:
				custom.loadDatasets(self.list)
				custom.openPlots(self.right)
			elif 1:#frozen:
				for index, name in list(enumerate("gas halo disk stars sat".split()))[::-1]:
					self.list.addGadgetHdf5(os.path.join(application_path, 'data/disk-galaxy.hdf5'), name, index)
				f = gavi.utils.get_data_file("data/helmi-dezeeuw-2000-10p.hdf5")
				print "datafile", f
				self.list.addHdf5(f)
				self.list.addHdf5(os.path.join(application_path, "data/Aq-A-2-999-shuffled-fraction.hdf5"))
		for pluginpath in [os.path.expanduser('~/.vaex/plugin')]:
			logger.debug("pluginpath: %s" % pluginpath)
			if os.path.exists(pluginpath):
				import glob
				paths = glob.glob(pluginpath + "/*.py")
				for path in paths:
					logger.debug("plugin file: %s" % path)
					filename = os.path.basename(path)
					name = os.path.splitext(filename)[0]
					imp.load_source('vaexuser.plugin.' + name, path)

		self.open_generators = [] # for reference counts
		self.action_open_hdf5_gadget.triggered.connect(self.openGenerator(self.gadgethdf5, "Gadget HDF5 file", "*.hdf5"))
		self.action_open_hdf5_gavi.triggered.connect(self.openGenerator(self.gaia_hdf5, "Gaia HDF5 file", "*.hdf5"))
		self.action_open_hdf5_amuse.triggered.connect(self.openGenerator(self.amuse_hdf5, "Amuse HDF5 file", "*.hdf5"))
		if (not frozen) or darwin:
			self.action_open_fits.triggered.connect(self.openGenerator(self.open_fits, "FITS file", "*.fits"))
		self.help_menu = menubar.addMenu('&Help')

		self.action_help = QtGui.QAction("Help", self)
		self.action_credits = QtGui.QAction("Credits", self)
		self.help_menu.addAction(self.action_help)
		self.help_menu.addAction(self.action_credits)

		self.action_help.triggered.connect(self.onActionHelp)
		self.action_credits.triggered.connect(self.onActionCredits)


		self.signal_samp_notification.connect(self.on_samp_notification)
		self.signal_samp_call.connect(self.on_samp_call)

		QtCore.QCoreApplication.instance().aboutToQuit.connect(self.clean_up)
		self.action_samp_connect.setChecked(True)
		self.onSampConnect(ignore_error=True)
		self.list.signal_pick.connect(self.on_pick)

		self.samp_ping_timer = QtCore.QTimer()
		self.samp_ping_timer.timeout.connect(self.on_samp_ping_timer)
		#self.samp_ping_timer.start(1000)

		self.highlighed_row_from_samp = False

		def on_open_plot(plot_dialog):
			plot_dialog.signal_samp_send_selection.connect(lambda dataset: self.on_samp_send_table_select_rowlist(dataset=dataset))
		self.right.signal_open_plot.connect(on_open_plot)

		self.parse_args(argv)

	def parse_args(self, args):
		#args = sys.argv[1:]
		index = 0
		def error(msg):
			print >>sys.stderr, msg
			sys.exit(1)
		hold_plot = False
		plot = None
		while index < len(args):
			filename = args[index]
			if filename[0] == ":": # not a filename, but a classname
				classname = filename.split(":")[1]
				if classname not in gavi.dataset.dataset_type_map:
					print classname, "does not exist, options are", sorted(gavi.dataset.dataset_type_map.keys())
					sys.exit(-1)
				class_ = gavi.dataset.dataset_type_map[classname]
				clsargs = [eval(value) for value in filename.split(":")[2:]]
				dataset = class_(*clsargs)
			else:
				options = filename.split(":")
				clsargs = [eval(value) for value in options[1:]]
				filename = options[0]
				dataset = gavi.dataset.load_file(filename, *clsargs)
			if dataset is None:
				error("cannot open file {filename}".format(**locals()))
			index += 1
			self.list.addDataset(dataset)

			# for this dataset, keep opening plots (seperated by -) or add layers (seperated by +)
			plot = plot if hold_plot else None
			options = {}
			while index < len(args) and args[index] != "--":
				columns = []
				while  index < len(args) and args[index] not in ["+", "-", "--", "++"]:
					if "=" in args[index]:
						key, value = args[index].split("=",1)
						if ":" in key:
							type, key = key.split(":", 1)
							if type == "vcol":
								dataset.virtual_columns[key] = value
							elif type == "var":
								dataset.variables[key] = value
							else:
								error("unknown expression, %s, type %s not recognized" % (type + ":" + key, type))
						else:
							options[key] = value
					else:
						columns.append(args[index])
					index += 1
				if plot is None:
					if len(columns) == 1:
						plot = self.right.histogram(columns[0], **options)
					elif len(columns) == 2:
						plot = self.right.plotxy(columns[0], columns[1], **options)
					elif len(columns) == 3:
						plot = self.right.plotxyz(columns[0], columns[1], columns[2], **options)
					else:
						error("cannot plot more than 3 columns yet: %r" % columns)
				else:
					layer = plot.add_layer(columns, dataset=dataset, **options)
					layer.jobs_manager.execute()
				options = {}
				if index < len(args) and args[index] == "-":
					plot = None # set to None to create a new plot, + will do a new layer
				if index < len(args) and args[index] == "--":
					hold_plot = False
					break # break out for the next dataset
				if index < len(args) and args[index] == "++":
					hold_plot = True
					break # break out for the next dataset, but keep the same plot
				index += 1
			if index < len(args):
				pass
			index += 1

	def on_samp_ping_timer(self):
		if self.samp:
			connected = self.samp.client.isConnected()
			#print "samp is", "connected" if connected else "disconnected!"
			if not connected:
				self.samp = None
		if self.samp:
			try:
				self.samp.client.ping()
			except:
				print "oops, ping went wrong, disconnect detected"
				try:
					self.samp.disconnect()
				except:
					pass
				self.samp = None
		self.action_samp_connect.setChecked(self.samp is not None)


	def on_pick(self, dataset, row):
		logger.debug("samp pick event")
		# avoid sending an event if this was caused by a samp event
		if self.samp and not self.highlighed_row_from_samp: # TODO: check if connected,
			kwargs = {"row": str(row)}
			if dataset.samp_id:
				kwargs["table-id"] = dataset.samp_id
				#kwargs["url"] = "file:" + dataset.filename
				kwargs["url"] = dataset.samp_id #
			else:
				if dataset.path:
					kwargs["table-id"] = "file:" + dataset.path
					kwargs["url"] = "file:" + dataset.path
				else:
					kwargs["table-id"] = "file:" + dataset.name
					kwargs["url"] = "file:" + dataset.name
			self.samp.client.enotifyAll("table.highlight.row", **kwargs)

	def on_samp_send_table_select_rowlist(self, ignore=None, dataset=None):
		if self.samp: # TODO: check if connected
			dataset = dataset or self.right.dataset
			rows = []
			if dataset.mask is not None:
				rows = np.arange(len(dataset))[dataset.mask]
			rowlist = map(str, rows)

			kwargs = {"row-list": rowlist}
			if dataset.samp_id:
				kwargs["table-id"] = dataset.samp_id
				#kwargs["url"] = "file:" + dataset.filename
				kwargs["url"] = "file:" +dataset.samp_id #
			else:
				kwargs["table-id"] = "file:" + dataset.path
			self.samp.client.enotifyAll("table.select.rowList", **kwargs)


	def onActionHelp(self):
		filename = gavi.utils.get_data_file("doc/index.html")
		url = "file://" + filename
		gavi.utils.os_open(url)
		#self.webDialog("doc/index.html")

	def onActionCredits(self):
		filename = gavi.utils.get_data_file("doc/credits.html")
		url = "file://" + filename
		gavi.utils.os_open(url)
		#gavi.utils.os_open("doc/credits.html")
		#self.webDialog("html/credits.html")

	def _webDialog(self, url):
		view = QWebView()
		view.load(QtCore.QUrl(url))
		dialog = QtGui.QDialog(self)
		layout = QtGui.QVBoxLayout()
		dialog.setLayout(layout)
		#text = file("html/credits.html").read()
		#print text
		#label = QtGui.QLabel(text, dialog)
		#layout.addWidget(label)
		layout.addWidget(view)
		dialog.resize(300, 300)
		dialog.show()

	def onSaveTable(self):
		dataset = self.right.dataset
		name = dataset.name + "-mysubset.hdf5"
		options = ["All: %r records, filesize: %r" % (len(dataset), gavi.utils.filesize_format(dataset.byte_size())) ]
		options += ["Selection: %r records, filesize: %r" % (dataset.length(selection=True), gavi.utils.filesize_format(dataset.byte_size(selection=True))) ]

		index = choose(self, "What do you want to export?", "Choose what to export:", options)
		if index is None:
			return
		export_selection = index == 1


		#select_many(None, "lala", ["aap", "noot"] + ["item-%d-%s" % (k, "-" * k) for k in range(30)])
		ok, columns_mask = select_many(self, "Select columns", dataset.get_column_names())
		if not ok: # cancel
			return

		selected_column_names = [column_name for column_name, selected in zip(dataset.get_column_names(), columns_mask) if selected]

		shuffle = dialog_confirm(self, "Shuffle?", "Do you want the dataset to be shuffled (output the rows in random order)")
		if shuffle and dataset.full_length() != len(dataset):
			dialog_info(self, "Shuffle", "You selected shuffling while not exporting the full dataset, will select random rows from the full dataset")
			partial_shuffle = True
		else:
			partial_shuffle = False

		if export_selection and shuffle:
			dialog_info(self, "Shuffle", "Shuffling with selection not supported")
			return





		filename = QtGui.QFileDialog.getSaveFileName(self, "Save to HDF5", name, "HDF5 *.hdf5")
		if isinstance(filename, tuple):
			filename = str(filename[0])#]
		#print args
		filename = str(filename)
		if filename:
				print filename

				# first open file using h5py api
				h5file_output = h5py.File(filename, "w")

				h5data_output = h5file_output.require_group("data")
				i1, i2 = dataset.current_slice
				N = dataset.length(selection=export_selection)
				print "N", N
				for column_name in selected_column_names:
					column = dataset.columns[column_name]
					#dataset_output.add_column(column_name, length=len(column), dtype=column.dtype)
					#assert N == len(column)
					print column_name, column.shape, column.strides
					#array = h5file_output.require_dataset("/data/%s" % column_name, shape=column.shape, dtype=column.dtype)
					print column_name, column.dtype, column.dtype.type
					array = h5file_output.require_dataset("/data/%s" % column_name, shape=(N,), dtype=column.dtype.type)
					array[0] = array[0] # make sure the array really exists
				if shuffle:
					shuffle_array = h5file_output.require_dataset("/data/random_index", shape=(N,), dtype=np.int64)
					shuffle_array[0] = shuffle_array[0]

				# close file, and reopen it using out class
				h5file_output.close()
				dataset_output = gavi.dataset.Hdf5MemoryMapped(filename, write=True)

				if shuffle:
					shuffle_array = dataset_output.columns["random_index"]
				if partial_shuffle:
					# if we only export a portion, we need to create the full length random_index array, and
					shuffle_array_full = np.zeros(dataset.full_length(), dtype=np.int64)
					gavifast.shuffled_sequence(shuffle_array_full)
					# then take a section of it
					shuffle_array[:] = shuffle_array_full[:len(dataset)]
					del shuffle_array_full
				elif shuffle:
					gavifast.shuffled_sequence(shuffle_array)

				#print "creating shuffled array"
				progress_total = len(selected_column_names)
				progress_value = 0
				progress_dialog = QtGui.QProgressDialog("Copying data...", "Abort export", 0, progress_total+1, self);
				try:

					progress_dialog.setWindowModality(QtCore.Qt.WindowModal);
					progress_dialog.setMinimumDuration(0)
					progress_dialog.setAutoClose(False)
					progress_dialog.setAutoReset(False)
					progress_dialog.show()
					QtCore.QCoreApplication.instance().processEvents()

					for column_name in selected_column_names:
						print column_name
						with gavi.utils.Timer("copying: %s" % column_name):
							from_array = dataset.columns[column_name]
							to_array = dataset_output.columns[column_name]
							#np.take(from_array, random_index, out=to_array)
							#print [(k.shape, k.dtype) for k in [from_array, to_array, random_index]]
							if export_selection:
								if dataset.mask is not None:
									to_array[:] = from_array[i1:i2][dataset.mask]
							else:
								if shuffle:
									#to_array[:] = from_array[i1:i2][shuffle_array]
									#to_array[:] = from_array[shuffle_array]
									#print [k.dtype for k in [from_array, to_array, shuffle_array]]
									#copy(from_array, to_array, shuffle_array)
									batch_copy_index(from_array, to_array, shuffle_array)
									#np.take(from_array, indices=shuffle_array, out=to_array)
									pass
								else:
									to_array[:] = from_array[i1:i2]
							#copy(, to_array, random_index)
						progress_value += 1
						progress_dialog.setValue(progress_value)
						QtCore.QCoreApplication.instance().processEvents()
						if progress_dialog.wasCanceled():
							dialog_info(self, "Cancel", "Export cancelled")
							break

				finally:
					progress_dialog.hide()

	def gadgethdf5(self, filename):
		print "filename", filename, repr(filename)
		for index, name in list(enumerate("gas halo disk bulge stars sat".split()))[::-1]:
			self.list.addGadgetHdf5(str(filename), name, index)

	def gaia_hdf5(self, filename):
		self.list.addHdf5(str(filename))

	def amuse_hdf5(self, filename):
		self.list.addAmuse(str(filename))

	def open_fits(self, filename):
		self.list.addFits(str(filename))


	def openGenerator(self, callback_, description, filemask):
		print repr(callback_)
		def open(arg=None, callback_=callback_, filemask=filemask):
			print repr(callback_), repr(filemask)
			filename = QtGui.QFileDialog.getOpenFileName(self, description, "", filemask)
			if isinstance(filename, tuple):
				filename = str(filename[0])#]
			print repr(callback_)
			callback_(filename)
		self.open_generators.append(open)
		return open

	def onSampConnect(self, ignore_error=False):
		if self.action_samp_connect.isChecked():
			print "connect"
			if self.samp is None:
					self.samp = Samp(daemon=True, name="vaex")
					#self.samp.tableLoadCallbacks.append(self.onLoadTable)
					connected = self.samp.client.isConnected()
					#print "samp is connected:", connected
					if connected:
						self.samp.client.bindReceiveNotification("table.highlight.row", self._on_samp_notification)
						self.samp.client.bindReceiveCall("table.select.rowList", self._on_samp_call)
						self.samp.client.bindReceiveNotification("table.load.votable", self._on_samp_notification)
						self.samp.client.bindReceiveCall("table.load.votable", self._on_samp_call)
						self.samp.client.bindReceiveNotification("table.load.fits", self._on_samp_notification)
						self.samp.client.bindReceiveCall("table.load.fits", self._on_samp_call)
					else:
						if not ignore_error:
							dialog_error(self, "Connecting to SAMP server", "Could not connect, make sure a SAMP HUB is running (for instance TOPCAT)")
						self.samp = None
						self.action_samp_connect.setChecked(False)
		else:
			print "disconnect"
			#try:
			self.samp.client.disconnect()
			self.samp = None
		#self.action_samp_connect.setText("disconnect from SAMP HUB" if self.samp else "conncet to SAMP HUB")
			#except:
			#	dialog_exception(self, "Connecting to SAMP server", "Could not connect, make sure a SAMP HUB is running (for instance TOPCAT)")



	def _on_samp_notification(self, private_key, sender_id, mtype, params, extra):
		# this callback will be in a different thread, so we use pyqt's signal mechanism to
		# push an event in the main thread's event loop
		print private_key, sender_id, mtype, params, extra
		self.signal_samp_notification.emit(private_key, sender_id, mtype, params, extra)

	def _on_samp_call(self, private_key, sender_id, msg_id, mtype, params, extra):
		# same as _on_samp_notification
		#print private_key, sender_id, msg_id, mtype, params, extra
		self.signal_samp_call.emit(private_key, sender_id, msg_id, mtype, params, extra)
		self.samp.client.ereply(msg_id, sampy.SAMP_STATUS_OK, result = {"txt": "printed"})

	def on_samp_notification(self, private_key, sender_id, mtype, params, extra):
		# and this should execute in the main thread
		logger.debug("samp notification: %r" % ((private_key, sender_id, mtype),))
		assert QtCore.QThread.currentThread() == main_thread
		def dash_to_underscore(hashmap):
			hashmap = dict(hashmap) # copy
			for key, value in hashmap.items():
				del hashmap[key]
				hashmap[key.replace("-", "_")] = value
			return hashmap
		params = dash_to_underscore(params)
		if mtype == "table.highlight.row":
			self.samp_table_highlight_row(**params)
		if mtype == "table.select.rowList":
			self.samp_table_select_rowlist(**params)
		if mtype == "table.load.votable":
			self.samp_table_load_votable(**params)


	def on_samp_call(self, private_key, sender_id, msg_id, mtype, params, extra):
		# and this should execute in the main thread
		assert QtCore.QThread.currentThread() == main_thread
		# we simply see a call as a notification
		self.on_samp_notification(private_key, sender_id, mtype, params, extra)

	def samp_table_highlight_row(self, row, url=None, table_id=None):
		logger.debug("highlight row: {url}:{row}".format(**locals()))
		print ("highlight row: {url}:{row}".format(**locals()))
		row = long(row)
		# only supports url for the moment
		for id in (url, table_id):
			if id != None:
				for dataset in self._samp_find_datasets(id):
					# avoid triggering another samp event and an infinite loop
					self.highlighed_row_from_samp = True
					try:
						dataset.selectRow(row)
					finally:
						self.highlighed_row_from_samp = False



	def samp_table_select_rowlist(self, row_list, url=None, table_id=None):
		print "----"
		logger.debug("select rowlist: {url}".format(**locals()))
		print ("select rowlist: {url}".format(**locals()))
		row_list = np.array([long(k) for k in row_list])
		did_select = False
		datasets_updated = [] # keep a list to avoid multiple 'setMask' calls (which would do an update twice)
		for id in (url, table_id):
			if id != None:
				for dataset in self._samp_find_datasets(id):
					if dataset not in datasets_updated:
						mask = np.zeros(len(dataset), dtype=np.bool)
						mask[row_list] = True
						print "match dataset", dataset
						dataset.selectMask(mask)
						did_select = True
					datasets_updated.append(dataset)
		if did_select:
			self.main_panel.jobsManager.execute()


	def samp_table_load_votable(self, url=None, table_id=None, name=None):
		filenames = []
		if table_id is not None:
			filename = table_id
			if filename.startswith("file:/"):
				filename = filename[5:]

			basename, ext = os.path.splitext(filename)
			if os.path.exists(filename):
				filenames.append(filename)
			for other_ext in [".hdf5", ".fits"]:
				filename = basename + other_ext
				print filename
				if os.path.exists(filename) and filename not in filenames:
					filenames.append(filename)
			filenames = filter(gavi.dataset.can_open, filenames)
		options = []
		for filename in filenames:
			options.append(filename + " | read directly from file (faster)")
		options.append(url + " | load as VOTable (slower)")
		#options.append("link to existing opened dataset")
		for dataset in self.list.datasets:
			options.append("link to existing open dataset: " + dataset.name)
		index = choose(self, "SAMP: load table", "Choose how to load table", options)
		if index is not None:
			if index < len(filenames):
				print "open file", filenames[index]
				self.load_file(filenames[index], table_id)
			elif index  == len(filenames):
				self.load_votable(url, table_id)
				print "load votable", url
			else:
				self.list.datasets[index-len(filenames)-1].samp_id = table_id

	def load_file(self, path, samp_id=None):
		dataset_class = None
		for name, class_ in gavi.dataset.dataset_type_map.items():
			if class_.can_open(path):
				dataset_class = class_
				break
		if dataset_class:
			dataset = dataset_class(path)
			dataset.samp_id = samp_id
			self.list.addDataset(dataset)

	def load_votable(self, url, table_id):
		table = astropy.io.votable.parse_single_table(url)
		print "done parsing table"
		names = table.array.dtype.names
		dataset = MemoryMapped(table_id, nommap=True)

		data = table.array.data
		for i in range(len(data.dtype)):
			name = data.dtype.names[i]
			type = data.dtype[i]
			if type.kind in ["f", "i"]: # only store float
				#datagroup.create_dataset(name, data=table.array[name].astype(np.float64))
				#dataset.addMemoryColumn(name, table.array[name].astype(np.float64))
				dataset.addColumn(name, array=table.array[name])
		dataset.samp_id = table_id
		self.list.addDataset(dataset)
		return dataset




	def message(self, text, index=0):
		print text
		self.messages[index] = text
		text = ""
		keys = self.messages.keys()
		keys.sort()
		text_parts = [self.messages[key] for key in keys]
		self.statusBar().showMessage(" | ".join(text_parts))

	def _samp_find_datasets(self, id):
		print self.list.datasets
		try:
			for dataset in self.list.datasets:
				if dataset.matches_url(id) or (dataset.samp_id == id):
					yield dataset
		except:
			logger.exception("problem")


	def onSampSend(self):
		if self.samp is None:
			self.onSampConnect()
		dataset = self.right.dataset
		params = {"rows":str(dataset._length), "columns":{}}
		params['id'] = dataset.filename
		type_map = {np.float64:"F8_LE", np.float32:"F4_LE", np.int64:"I8_LE", np.int32:"I4_LE", np.uint64:"U8_LE", np.uint32:"U4_LE"}
		print type_map
		for column_name in dataset.column_names:
			type = dataset.dtypes[column_name]
			if hasattr(type, "type"):
				type = type.type # TODO: why is this needed?
			bytes_type = np.zeros(1, dtype=type).dtype.itemsize
			column = {
					"filename":dataset.filenames[column_name],
					"type": type_map[type],
					"byte_offset": str(dataset.offsets[column_name]),
					"type_stride": str(dataset.strides[column_name]),
					"byte_stride": str(dataset.strides[column_name]*bytes_type),
					"bytes_type": str(bytes_type),
					}
			params["columns"][column_name] = column
		self.samp.client.callAll("send_mmap_"+dataset.name,
					{"samp.mtype": "table.load.memory_mapped_columns",
						"samp.params": params})

	def onLoadTable(self, url, table_id, name):
		# this is called from a different thread!
		print "loading table", url, table_id, name
		try:
			self.load(url, table_id, name)
		except:
			logger.exception("load table")
		return


	def load(self, url, table_id, name):
		print "parsing table..."
		table = astropy.io.votable.parse_single_table(url)
		print "done parsing table"
		names = table.array.dtype.names
		dataset = MemoryMapped(table_id, nommap=True)

		data = table.array.data
		for i in range(len(data.dtype)):
			name = data.dtype.names[i]
			type = data.dtype[i]
			if type.kind  == "f": # only store float
				#datagroup.create_dataset(name, data=table.array[name].astype(np.float64))
				dataset.addMemoryColumn(name, table.array[name].astype(np.float64))
		self.list.addDataset(dataset)
		if 0:
			h5file = h5py.File(hdf5filename, "w", driver="core")
			datagroup = h5file.create_group("data")
			#import pdb
			#pdb.set_trace()
			print "storing data..."

			for i in range(len(data.dtype)):
				name = data.dtype.names[i]
				type = data.dtype[i]
				if type.kind  == "f": # only store float
					datagroup.create_dataset(name, data=table.array[name].astype(np.float64))
			print "storing data done"
		#thread.interrupt_main()
		#sys.exit(0)
		#h5file.close()
		#semaphore.release()
		##samp.client.disconnect()


	def center(self):

		qr = self.frameGeometry()
		cp = QtGui.QDesktopWidget().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())

	def closeEvent(self, event):
		print "close event"
		return
		reply = QtGui.QMessageBox.question(self, 'Message',
			"Are you sure to quit?", QtGui.QMessageBox.Yes |
			QtGui.QMessageBox.No, QtGui.QMessageBox.No)

		if reply == QtGui.QMessageBox.Yes:
			event.accept()
		else:
			event.ignore()

	def clean_up(self):
		print "clean up"
		if self.samp is not None:
			print "disconnect samp"
			try:
				self.samp.client.disconnect()
			except:
				logger.exception("error disconnecting from SAMP hub")
		#event.accept()
		return


app = None
def main(argv=sys.argv[1:]):
	global main_thread
	global vaex
	global app
	if app is None:
		app = QtGui.QApplication(argv)
		if not (frozen and darwin): # osx app has its own icon file
			import gavi.icons
			icon = QtGui.QIcon(gavi.icons.iconfile('vaex32'))
			app.setWindowIcon(icon)
	#import gavi.vaex.ipkernel_qtapp
	#ipython_window = gavi.vaex.ipkernel_qtapp.SimpleWindow(app)
	main_thread = QtCore.QThread.currentThread()


	#print select_many(None, "lala", ["aap", "noot"] + ["item-%d-%s" % (k, "-" * k) for k in range(30)])
	#sys.exit(0)


	#sys._excepthook = sys.excepthook
	def qt_exception_hook(exctype, value, traceback):
		sys.__excepthook__(exctype, value, traceback)
		qt_exception(None, exctype, value, traceback)
		#sys._excepthook(exctype, value, traceback)
		#sys.exit(1)
	sys.excepthook = qt_exception_hook
	#raise RuntimeError, "blaat"


	vaex = Vaex(argv)


	#w = QtGui.QWidget()
	#w.resize(250, 150)
	#w.move(300, 300)
	#w.setWindowTitle('Simple')
	#w.show()
	#ipython_window.show()
	#ipython_window.ipkernel.start()
	sys.exit(app.exec_())

@jit(nopython=True)
def copy(from_array, to_array, indices):
	length = len(indices)
	for i in range(length):
		index = indices[i]
		to_array[i] = from_array[index]

def batch_copy_index(from_array, to_array, shuffle_array):
	N_per_batch = int(1e7)
	length = len(from_array)
	batches = long(math.ceil(float(length)/N_per_batch))
	print np.sum(from_array)
	for i in range(batches):
		#print "batch", i, "out of", batches, ""
		sys.stdout.flush()
		i1 = i * N_per_batch
		i2 = min(length, (i+1)*N_per_batch)
		#print "reading...", i1, i2
		sys.stdout.flush()
		data = from_array[shuffle_array[i1:i2]]
		#print "writing..."
		sys.stdout.flush()
		to_array[i1:i2] = data