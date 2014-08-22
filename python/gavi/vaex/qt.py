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
	except ImportError, e2:
		print >>sys.stderr, "could not import PyQt4 or PySide, please install"
		print >>sys.stderr, "errors: ", repr(e1), repr(e2)
		sys.exit(1)



def getdir(parent, title, start_directory=""):
	return QtGui.QFileDialog.getExistingDirectory(parent, title, "",  QtGui.QFileDialog.ShowDirsOnly | QtGui.QFileDialog.DontResolveSymlinks)	
	
def gettext(parent, title, label, default=""):
	text, ok = QtGui.QInputDialog.getText(parent, title, label, QtGui.QLineEdit.Normal, default)
	return text if ok else None