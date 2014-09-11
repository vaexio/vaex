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
import sys
import gavi.vaex
import smtplib
from email.mime.text import MIMEText

def email(text):
	# Open a plain text file for reading.  For this example, assume that
	msg = MIMEText(text)

	msg['Subject'] = 'Error report for: ' +gavi.vaex.__version_text__
	email_from = "vaex@astro.rug.nl"
	email_to = "breddels@astro.rug.nl"
	msg['From'] = email_to
	msg['To'] = email_to

	# Send the message via our own SMTP server, but don't include the
	# envelope header.
	s = smtplib.SMTP('mailhost.astro.rug.nl')
	s.helo('fw1.astro.rug.nl')
	s.sendmail(email_to, [email_to], msg.as_string())
	s.quit()
	
def qt_exception(parent, exctype, value, traceback):
	trace_lines = tb.format_exception(exctype, value, traceback)
	trace = "".join(trace_lines)
	text = """An unexpected error occured, you may press ok and continue, but the program might be unstable.
	
	""" + trace
	report = gavi.vaex.__version_text__ + "\n" + text 
	dialog = QtGui.QMessageBox(parent)
	dialog.setText("Unexpected error: %s\nDo you want to continue" % (exctype, ))
	#dialog.setInformativeText(text)
	dialog.setDetailedText(text)
	buttonSend = QtGui.QPushButton("Email report", dialog) 
	buttonQuit = QtGui.QPushButton("Quit program", dialog) 
	buttonContinue = QtGui.QPushButton("Continue", dialog) 
	def exit(ignore=None):
		print "exit"
		sys.exit(1)
	def _email(ignore=None):
		email(report)
	buttonQuit.clicked.connect(exit)
	buttonSend.clicked.connect(_email)
	dialog.addButton(buttonSend, QtGui.QMessageBox.YesRole)
	dialog.addButton(buttonQuit, QtGui.QMessageBox.NoRole)
	dialog.addButton(buttonContinue, QtGui.QMessageBox.YesRole)
	dialog.setDefaultButton(buttonSend)
	dialog.setEscapeButton(buttonContinue)
	dialog.raise_()
	dialog.exec_()
	#QtGui.QMessageBox.critical(parent, "Unexpected error: %r" % (value, ), text)
	
	