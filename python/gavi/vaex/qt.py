# -*- coding: utf-8 -*-
try:
	from PyQt4 import QtGui, QtCore, QtNetwork
	from PyQt4.QtWebKit import QWebView

	import sip
	sip.setapi('QVariant', 1)
except ImportError, e1:
	try:
		from PySide import QtGui, QtCore, QtNetwork
		from PySide.QtWebKit import QWebView
		QtCore.pyqtSignal= QtCore.Signal 
		#QtCore.Slot = QtCore.pyqtSlot
	except ImportError, e2:
		print >>sys.stderr, "could not import PyQt4 or PySide, please install"
		print >>sys.stderr, "errors: ", repr(e1), repr(e2)
		sys.exit(1)



def getdir(parent, title, start_directory=""):
	return QtGui.QFileDialog.getExistingDirectory(parent, title, "",  QtGui.QFileDialog.ShowDirsOnly | QtGui.QFileDialog.DontResolveSymlinks)	
	
def gettext(parent, title, label, default=""):
	text, ok = QtGui.QInputDialog.getText(parent, title, label, QtGui.QLineEdit.Normal, default)
	return text if ok else None


def choose(parent, title, label, options, index=0):
	text, ok = QtGui.QInputDialog.getItem(parent, title, label, options, index, False)
	return options.index(text) if ok else None

def dialog_error(parent, title, msg):	
	QtGui.QMessageBox.warning(parent, title, msg)
	
def dialog_info(parent, title, msg):	
	QtGui.QMessageBox.information(parent, title, msg)
	
import traceback as tb
def qt_exception(parent, exctype, value, traceback):
	trace_lines = tb.format_exception(exctype, value, traceback)
	trace = "".join(trace_lines)
	text = """An unexpected error occured, you may press ok and continue, but the program might be unstable.
	
	""" + trace
	dialog = QtGui.QMessageBox(parent)
	dialog.setText("Unexpected error: %s" % (exctype, ))
	dialog.setInformativeText(text)
	QtGui.QMessageBox.critical(parent, "Unexpected error: %r" % (value, ), text)
	
	