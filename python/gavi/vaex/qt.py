# -*- coding: utf-8 -*-
import math

try:
	from PyQt4 import QtGui, QtCore#, QtNetwork
	#from PyQt4.QtWebKit import QWebView
	qt_version = QtCore.PYQT_VERSION_STR
	import sip
	sip.setapi('QVariant', 1)
except ImportError, e1:
	try:
		from PySide import QtGui, QtCore#, QtNetwork
		#from PySide.QtWebKit import QWebView
		QtCore.pyqtSignal= QtCore.Signal 
		qt_version = QtCore.__version__
		#QtCore.Slot = QtCore.pyqtSlot
	except ImportError, e2:
		print >>sys.stderr, "could not import PyQt4 or PySide, please install"
		print >>sys.stderr, "errors: ", repr(e1), repr(e2)
		sys.exit(1)

def attrsetter(object, attr_name):
	def setter(value):
		setattr(object, attr_name, value)
	return setter

def attrgetter(object, attr_name):
	def getter():
		return getattr(object, attr_name)
	return getter


class Option(object):
	def __init__(self, parent, label, options, getter, setter, update=lambda: None):
		self.update = update
		self.options = options
		self.label = QtGui.QLabel(label, parent)
		self.combobox = QtGui.QComboBox(parent)
		self.combobox.addItems(options)
		def wrap_setter(value, update=True):
			self.combobox.setCurrentIndex(options.index(getter()))
			setter(value)
			if update:
				self.update()
		# auto getter and setter
		setattr(self, "get_value", getter)
		setattr(self, "set_value", wrap_setter)

		def on_change(index):
			setter(self.options[index])
			update()
		self.combobox.currentIndexChanged.connect(on_change)
		self.combobox.setCurrentIndex(options.index(getter()))

	def add_to_grid_layout(self, row, grid_layout):
		grid_layout.addWidget(self.label, row, 0)
		grid_layout.addWidget(self.combobox, row, 1)
		return row + 1

class Checkbox(object):
	def __init__(self, parent, label_text, getter, setter, update=lambda: None):
		self.update = update
		self.label = QtGui.QLabel(label_text, parent)
		self.checkbox = QtGui.QCheckBox(parent)
		def wrap_setter(value, update=True):
			self.checkbox.setChecked(value)
			setter(value)
			if update:
				self.update()
		# auto getter and setter
		setattr(self, "get_value", getter)
		setattr(self, "set_value", wrap_setter)

		def on_change(state):
			value = state == QtCore.Qt.Checked
			print label_text, "set to", value
			setter(value)
			self.update()
		self.checkbox.setChecked(getter())
		self.checkbox.stateChanged.connect(on_change)

	def add_to_grid_layout(self, row, grid_layout):
		grid_layout.addWidget(self.label, row, 0)
		grid_layout.addWidget(self.checkbox, row, 1)
		return row + 1

class Slider(object):
	def __init__(self, parent, label_text, value_min, value_max, value_steps, getter, setter, name=None, format="{0:<0.3f}", transform=lambda x: x, inverse=lambda x: x, update=lambda: None, uselog=False, numeric_type=float):
		if name is None:
			name = label_text
		if uselog:
			transform = lambda x: 10**x
			inverse = lambda x: math.log10(x)
		#self.properties.append(name)
		self.update = update
		self.label = QtGui.QLabel(label_text, parent)
		self.label_value = QtGui.QLabel(label_text, parent)
		self.slider = QtGui.QSlider(parent)
		self.slider.setOrientation(QtCore.Qt.Horizontal)
		self.slider.setRange(0, value_steps)

		def wrap_setter(value, update=True):
			self.slider.setValue((inverse(value) - inverse(value_min))/(inverse(value_max) - inverse(value_min)) * value_steps)
			setter(value)
			if update:
				self.update()
		# auto getter and setter
		setattr(self, "get_value", getter)
		setattr(self, "set_value", wrap_setter)

		def update_text():
			#label.setText("mean/sigma: {0:<0.3f}/{1:.3g} opacity: {2:.3g}".format(self.tool.function_means[i], self.tool.function_sigmas[i], self.tool.function_opacities[i]))
			self.label_value.setText(format.format(getter()))
		def on_change(index, slider=self.slider):
			value = numeric_type(index/float(value_steps) * (inverse(value_max) - inverse(value_min)) + inverse(value_min))
			print label_text, "set to", value
			setter(transform(value))
			self.update()
			update_text()
		self.slider.setValue((inverse(getter()) - inverse(value_min)) * value_steps/(inverse(value_max) - inverse(value_min)))
		update_text()
		self.slider.valueChanged.connect(on_change)
		#return label, slider, label_value

	def add_to_grid_layout(self, row, grid_layout):
		grid_layout.addWidget(self.label, row, 0)
		grid_layout.addWidget(self.slider, row, 1)
		grid_layout.addWidget(self.label_value, row, 2)
		return row + 1



def get_path_save(parent, title="Save file", path="", file_mask="HDF5 *.hdf5"):
	path = QtGui.QFileDialog.getSaveFileName(parent, title, path, file_mask)
	if isinstance(path, tuple):
		filename = str(path[0])#]
	return str(path)

def get_path_open(parent, title="Select file", path="", file_mask="HDF5 *.hdf5"):
	path = QtGui.QFileDialog.getOpenFileName(parent, title, path, file_mask)
	if isinstance(path, tuple):
		path = str(path[0])#]
	return str(path)
def getdir(parent, title, start_directory=""):
	result = QtGui.QFileDialog.getExistingDirectory(parent, title, "",  QtGui.QFileDialog.ShowDirsOnly | QtGui.QFileDialog.DontResolveSymlinks)
	return None if result is None else str(result)

def gettext(parent, title, label, default=""):
	text, ok = QtGui.QInputDialog.getText(parent, title, label, QtGui.QLineEdit.Normal, default)
	return str(text) if ok else None


def choose(parent, title, label, options, index=0, editable=False):
	text, ok = QtGui.QInputDialog.getItem(parent, title, label, options, index, editable)
	if editable:
		return text if ok else None
	else:
		return options.index(text) if ok else None

def select_many(parent, title, options):
	dialog = QtGui.QDialog(parent)
	dialog.setWindowTitle(title)
	dialog.setModal(True)
	layout = QtGui.QGridLayout(dialog)
	dialog.setLayout(layout)

	scroll_area = QtGui.QScrollArea(dialog)
	scroll_area.setWidgetResizable(True)
	frame = QtGui.QWidget(scroll_area)
	layout_frame = QtGui.QVBoxLayout(frame)
	#frame.setMinimumSize(100,400)
	frame.setLayout(layout_frame)
	checkboxes = [QtGui.QCheckBox(option, frame) for option in options]
	scroll_area.setWidget(frame)
	row = 0
	for checkbox in checkboxes:
		checkbox.setCheckState(QtCore.Qt.Checked)
		layout_frame.addWidget(checkbox) #, row, 0)
		row += 1

	buttonLayout = QtGui.QHBoxLayout()
	button_ok = QtGui.QPushButton("Ok", dialog)
	button_cancel = QtGui.QPushButton("Cancel", dialog)
	button_cancel.clicked.connect(dialog.reject)
	button_ok.clicked.connect(dialog.accept)

	buttonLayout.addWidget(button_cancel)
	buttonLayout.addWidget(button_ok)
	layout.addWidget(scroll_area)
	layout.addLayout(buttonLayout, row, 0)
	#options_selected = []
	value = dialog.exec_()
	mask =[checkbox.checkState() == QtCore.Qt.Checked for checkbox in checkboxes]
	return value == QtGui.QDialog.Accepted, mask



def dialog_error(parent, title, msg):	
	QtGui.QMessageBox.warning(parent, title, msg)
	
def dialog_info(parent, title, msg):	
	QtGui.QMessageBox.information(parent, title, msg)
	
def dialog_confirm(parent, title, msg, to_all=False):
	#return QtGui.QMessageBox.information(parent, title, msg, QtGui.QMessageBox.Yes|QtGui.QMessageBox.No) == QtGui.QMessageBox.Yes
	msgbox = QtGui.QMessageBox(parent)
	msgbox.setText(msg)
	msgbox.setWindowTitle(title)
	#, title, msg, QtGui.QMessageBox.Yes|QtGui.QMessageBox.No) == QtGui.QMessageBox.Yes
	msgbox.addButton(QtGui.QMessageBox.Yes)
	if to_all:
		msgbox.addButton(QtGui.QMessageBox.YesToAll)
		msgbox.setDefaultButton(QtGui.QMessageBox.YesToAll)
	else:
		msgbox.setDefaultButton(QtGui.QMessageBox.Yes)
	msgbox.addButton(QtGui.QMessageBox.No)
	result = msgbox.exec_()
	if to_all:
		return result in [QtGui.QMessageBox.Yes, QtGui.QMessageBox.YesToAll], result == QtGui.QMessageBox.YesToAll
	else:
		return result in [QtGui.QMessageBox.Yes]

confirm = dialog_confirm

import traceback as tb
import sys
import gavi.vaex
import smtplib
import platform
import getpass
import sys
import os
import urllib
import gavi.utils
#from email.mime.text import MIMEText

def email(text):
	osname = platform.system().lower()
	if osname == "linux":
		text = text.replace("#", "%23") # for some reason, # needs to be double quoted on linux, otherwise it is interpreted as comment symbol
	
	body = urllib.quote(text)
		
	subject = urllib.quote('Error report for: ' +gavi.vaex.__full_name__)
	mailto = "mailto:maartenbreddels@gmail.com?subject={subject}&body={body}".format(**locals())
	print "open:", mailto
	gavi.utils.os_open(mailto)
		


def old_email(text):
	# Open a plain text file for reading.  For this example, assume that
	msg = MIMEText(text)

	msg['Subject'] = 'Error report for: ' +gavi.vaex.__full_name__
	email_from = "vaex@astro.rug.nl"
	#email_to = "breddels@astro.rug.nl"
	email_to = "maartenbreddels@gmail.com"
	msg['From'] = email_to
	msg['To'] = email_to

	# Send the message via our own SMTP server, but don't include the
	# envelope header.
	#s = smtplib.SMTP('mailhost.astro.rug.nl')
	s = smtplib.SMTP('smtp.gmail.com')
	s.helo('fw1.astro.rug.nl')
	s.sendmail(email_to, [email_to], msg.as_string())
	s.quit()
	
def qt_exception(parent, exctype, value, traceback):
	trace_lines = tb.format_exception(exctype, value, traceback)
	trace = "".join(trace_lines)
	print trace
	info = "username: %r\n" % (getpass.getuser(),)
	info += "program: %r\n" % gavi.vaex.__program_name__
	info += "version: %r\n" % gavi.vaex.__version__
	info += "full name: %r\n" % gavi.vaex.__full_name__
	info += "arguments: %r\n" % sys.argv
	info += "Qt version: %r\n" % qt_version
	
	attrs = sorted(dir(platform))
	for attr in attrs:
		if not attr.startswith("_") and attr not in ["popen", "system_alias"]:
			f = getattr(platform, attr)
			if callable(f):
				try:
					info += "%s: %r\n" % (attr, f())
				except:
					pass
	#, platform.architecture(), platform.dist(), platform.linux_distribution(), 
	
	report = info + "\n" + trace
	text = """An unexpected error occured, you may press ok and continue, but the program might be unstable.
	
""" + report
	
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
		if QtGui.QMessageBox.information(dialog, "Send report", "Confirm that you want to send a report", QtGui.QMessageBox.Abort|QtGui.QMessageBox.Yes) == QtGui.QMessageBox.Yes:
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



	