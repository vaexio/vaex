from __future__ import print_function
# -*- coding: utf-8 -*-
import math
import collections
import time
import sys
import logging

logger = logging.getLogger("vaex.ui.qt")

try:
	from PySide import QtGui, QtCore, QtTest  # , QtNetwork

	# from PySide.QtWebKit import QWebView
	QtCore.pyqtSignal = QtCore.Signal
	qt_version = QtCore.__version__
	qt_mayor = 4
	print("using pyside")
except ImportError as e1:
	try:
		import sip
		sip.setapi('QVariant', 2)
		sip.setapi('QString', 2)
		from PyQt4 import QtGui, QtCore, QtTest #, QtNetwork
		sip.setapi('QVariant', 2)
		#from PyQt4.QtWebKit import QWebView
		qt_version = QtCore.PYQT_VERSION_STR
		qt_mayor = int(qt_version[0])
	except ImportError as e1b:
		try:
			from PyQt5 import QtGui, QtCore, QtTest, QtWidgets  # , QtNetwork

			for name in dir(QtWidgets):
				if name[0].lower() == "q":
					setattr(QtGui, name, getattr(QtWidgets, name))
			qt_version = QtCore.PYQT_VERSION_STR
			qt_mayor = int(qt_version[0])
			QtGui.QStringListModel = QtCore.QStringListModel
		#QtCore.Slot = QtCore.pyqtSlot
		except ImportError as e2:
			print("could not import PyQt4 or PySide, please install", file=sys.stderr)
			print("errors: ", repr(e1), repr(e2), file=sys.stderr)
			sys.exit(1)
print(qt_version)
def attrsetter(object, attr_name):
	def setter(value):
		setattr(object, attr_name, value)
	return setter

def attrgetter(object, attr_name):
	def getter():
		return getattr(object, attr_name)
	return getter

class ProgressExecution(object):
	def __init__(self, parent, title, cancel_text="Cancel", executor=None):
		self.parent = parent
		self.title = title
		self.cancel_text = cancel_text
		self.executor = executor
		if self.executor:
			def begin():
				self.time_begin = time.time()
				#self.progress_bar.setValue(0)
				self.cancelled = False
				#self.button_cancel.setEnabled(True)
			def end():
				#self.progress_bar.setValue(1000)
				self.cancelled = False
				#self.button_cancel.setEnabled(False)
				time_total = time.time() - self.time_begin
				#self.label_time.setText("%.2fs" % time_total)
			def progress(fraction):
				#self.progress_bar.setValue(fraction*1000)
				#QtCore.QCoreApplication.instance().processEvents()
				#logger.debug("queue: %r %r", self.queue_update.counter, self.queue_update.counter_processed)
				#return (not self.cancelled) and (not self.queue_update.in_queue(2))
				return self.progress(fraction*100)
			def cancel():
				#self.progress_bar.setValue(0)
				#self.button_cancel.setEnabled(False)
				#self.label_time.setText("cancelled")
				pass

			#self._begin_signal = self.executor.signal_begin.connect(begin)
			#self._progress_signal = self.executor.signal_progress.connect(progress)
			#self._end_signal = self.executor.signal_end.connect(end)
			#self._cancel_signal = self.executor.signal_cancel.connect(cancel)
		self.tasks = []
		self._task_signals = []

	def execute(self):
		logger.debug("show dialog")
		if isinstance(self.executor, vaex.remote.ServerExecutor):
			self.dialog.exec_()
			logger.debug("dialog stopped")
			# important possible deadlock possible:
			# in the case where fulfill or reject is indirectly called, the call to that promise is put in a qtevent
			# so while they are pending, keep processing events until they are handled in this thread
			while any(task.isPending for task in self.tasks):
				QtCore.QCoreApplication.instance().processEvents()
				time.sleep(0.01)
			self.finished_tasks()
		else:
			self.dialog.show()
			self.executor.execute()
			self.dialog.hide()
		logger.debug("self.dialog.wasCanceled() = %r", self.dialog.wasCanceled())
		return not self.dialog.wasCanceled()

	def add_task(self, task):
		self._task_signals.append(task.signal_progress.connect(self._on_progress))
		self.tasks.append(task)
		return task

	def _on_progress(self, fraction):
		total = self.get_progress_fraction()
		self.progress(total*100)
		QtCore.QCoreApplication.instance().processEvents()
		ok = not self.dialog.wasCanceled()
		if total == 1:
			self.dialog.hide()
		return ok

	def get_progress_fraction(self):
		total_fraction = 0
		for task in self.tasks:
			total_fraction += task.progress_fraction
		fraction = total_fraction / len(self.tasks)
		return fraction

	def finished_tasks(self):
		for task, signal in zip(self.tasks, self._task_signals):
			task.signal_progress.disconnect(signal)
		self.tasks = []
		self._task_signals = []

	def __enter__(self):
		self.dialog = QtGui.QProgressDialog(self.title, self.cancel_text, 0, 1000, self.parent)
		#self.dialog.show()
		self.dialog.setWindowModality(QtCore.Qt.WindowModal)
		self.dialog.setMinimumDuration(0)
		self.dialog.setAutoClose(True)
		self.dialog.setAutoReset(True)
		#QtCore.QCoreApplication.instance().processEvents()
		return self

	def progress(self, percentage):
		self.dialog.setValue(int(percentage*10))
		QtCore.QCoreApplication.instance().processEvents()
		#self.dialog.setValue(300)
		#self.dialog.update()
		#self.dialog.repaint()
		#print "progress", `percentage`, type(percentage), int(percentage*10)
		#QtCore.QCoreApplication.instance().processEvents()
		return not self.dialog.wasCanceled()
		#return False
			#raise RuntimeError("progress cancelled")

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.dialog.hide()
		if 0: #self.executor:
			self.executor.signal_begin.disconnect(self._begin_signal)
			self.executor.signal_progress.disconnect(self._progress_signal)
			self.executor.signal_end.disconnect(self._end_signal)
			self.executor.signal_cancel.disconnect(self._cancel_signal)

class assertError(object):
	def __init__(self, calls_expected=1):
		self.calls_expected = calls_expected

	def wrapper(self, *args, **kwargs):
		self.called += 1

	def __enter__(self):
		global dialog_error
		self.remember = dialog_error
		self.called = 0
		dialog_error = self.wrapper
		logger.debug("wrapped dialog_error")
	def __exit__(self, exc_type, exc_val, exc_tb):
		global dialog_error
		assert self.called == self.calls_expected, "expected the error dialog to be invoked %i time(s), was called %i times(s)" % (self.calls_expected, self.called)
		dialog_error = self.remember
		logger.debug("unwrapped dialog_error")

class settext(object):
	def __init__(self, value, calls_expected=1):
		self.value = value
		self.calls_expected = calls_expected

	def wrapper(self, *args, **kwargs):
		self.called += 1
		return self.value

	def __enter__(self):
		global gettext
		self.remember = gettext
		self.called = 0
		gettext = self.wrapper
		logger.debug("wrapped gettext")

	def __exit__(self, exc_type, exc_val, exc_tb):
		global gettext
		assert self.called == self.calls_expected, "expected the error dialog to be invoked %i time(s), was called %i times(s)" % (self.calls_expected, self.called)
		gettext = self.remember
		logger.debug("unwrapped gettext")

class setchoose(object):
	def __init__(self, value, calls_expected=1):
		self.value = value
		self.calls_expected = calls_expected

	def wrapper(self, *args, **kwargs):
		self.called += 1
		return self.value

	def __enter__(self):
		global choose
		self.remember = choose
		self.called = 0
		choose = self.wrapper
		logger.debug("wrapped choose")

	def __exit__(self, exc_type, exc_val, exc_tb):
		global choose
		assert self.called == self.calls_expected, "expected the error dialog to be invoked %i time(s), was called %i times(s)" % (self.calls_expected, self.called)
		choose = self.remember
		logger.debug("unwrapped choose")


class FakeProgressExecution(object):
	def __init__(self, *args):
		pass
	def __enter__(self):
		return self
	def __exit__(self, exc_type, exc_val, exc_tb):
		pass
	def progress(self, percentage):
		return True

class Codeline(object):
	def __init__(self, parent, label, options, getter, setter, update=lambda: None):
		self.update = update
		self.options = options
		self.label = QtGui.QLabel(label, parent)
		self.combobox = QtGui.QComboBox(parent)
		self.combobox.addItems(options)
		self.combobox.setEditable(True)
		def wrap_setter(value, update=True):
			self.combobox.lineEdit().setText(value)
			setter(value)
			if update:
				self.update()
		# auto getter and setter
		setattr(self, "get_value", getter)
		setattr(self, "set_value", wrap_setter)

		def on_change(index):
			on_edit_finished()
		def on_edit_finished():
			new_value = text = str(self.combobox.lineEdit().text())
			if new_value != self.current_value:
				self.current_value = new_value
				setter(self.current_value)
				update()
		self.combobox.currentIndexChanged.connect(on_change)
		self.combobox.lineEdit().editingFinished.connect(on_edit_finished)
		#self.combobox.setCurrentIndex(options.index(getter()))
		self.current_value = getter()
		self.combobox.lineEdit().setText(self.current_value)

	def add_to_grid_layout(self, row, grid_layout):
		grid_layout.addWidget(self.label, row, 0)
		grid_layout.addWidget(self.combobox, row, 1)
		return row + 1



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

class TextOption(object):
	def __init__(self, parent, label, value, placeholder, getter, setter, update=lambda: None):
		self.update = update
		self.value = value
		self.placeholder = placeholder
		self.label = QtGui.QLabel(label, parent)
		self.textfield = QtGui.QLineEdit(parent)
		self.textfield.setPlaceholderText(self.get_placeholder())
		#self.combobox.addItems(options)
		def wrap_setter(value, update=True):
			self.textfield.setText(value)
			setter(value)
			if update:
				self.update()
		# auto getter and setter
		setattr(self, "get_value", getter)
		setattr(self, "set_value", wrap_setter)

		def on_change(*ignore):
			setter(self.textfield.text())
			update()
		#self.combobox.currentIndexChanged.connect(on_change)
		#self.combobox.setCurrentIndex(options.index(getter()))
		self.textfield.returnPressed.connect(on_change)

		if 1:
			from vaex.ui.icons import iconfile
			self.tool_button = QtGui.QToolButton(parent)
			self.tool_button .setIcon(QtGui.QIcon(iconfile('gear')))
			self.tool_menu = QtGui.QMenu()
			self.tool_button.setMenu(self.tool_menu)
			self.tool_button.setPopupMode(QtGui.QToolButton.InstantPopup)

			if self.placeholder:
				def fill(_=None):
					value = self.get_placeholder()
					if value:
						setter(value)
						self.set_value(value)
				self.action_fill = QtGui.QAction("Fill in default value", parent)
				self.action_fill.triggered.connect(fill)
				self.tool_menu.addAction(self.action_fill)

			def copy(_=None):
				value = self.textfield.text()
				if value:
					clipboard = QtGui.QApplication.clipboard()
					text = str(value)
					clipboard.setText(text)
			self.action_copy = QtGui.QAction("Copy", parent)
			self.action_copy.triggered.connect(copy)
			self.tool_menu.addAction(self.action_copy)

			def paste(_=None):
				clipboard = QtGui.QApplication.clipboard()
				text = clipboard.text()
				setter(text)
				self.set_value(text)
			self.action_paste = QtGui.QAction("Paste", parent)
			self.action_paste.triggered.connect(paste)
			self.tool_menu.addAction(self.action_paste)


	def set_unit_completer(self):
		self.completer = vaex.ui.completer.UnitCompleter(self.textfield)
		self.textfield.setCompleter(self.completer)

	def get_placeholder(self):
		if callable(self.placeholder):
			return self.placeholder()
		else:
			return self.placeholder

	def add_to_grid_layout(self, row, grid_layout):
		grid_layout.addWidget(self.label, row, 0)
		grid_layout.addWidget(self.textfield, row, 1)
		grid_layout.addWidget(self.tool_button, row, 2)
		return row + 1

class RangeOption(object):
	def __init__(self, parent, label, values, getter, setter, update=lambda: None):
		self.update = update
		self.values = [str(k) for k in values]
		self.label = QtGui.QLabel(label, parent)
		self.combobox_min = QtGui.QComboBox(parent)
		self.combobox_max = QtGui.QComboBox(parent)
		self.combobox_min.setEditable(True)
		self.combobox_max.setEditable(True)
		self.vmin = None
		self.vmax = None
		#self.combobox_min.addItems(values)
		#self.combobox_max.addItems(values)
		def wrap_setter(value, update=True):
			if value is None:
				vmin, vmax = None, None
			else:
				vmin, vmax = value
			self.combobox_min.blockSignals(True)
			self.combobox_max.blockSignals(True)
			changed = False
			if vmin != self.vmin:
				#print(("setting vmin to", vmin))
				self.vmin = vmin
				self.combobox_min.lineEdit().setText(str(self.vmin) if self.vmin is not None else "")
				changed = True
			if vmax != self.vmax:
				#print(( "setting vmax to", vmax))
				self.vmax = vmax
				self.combobox_max.lineEdit().setText(str(self.vmax) if self.vmax is not None else "")
				changed = True
			self.combobox_min.blockSignals(False)
			self.combobox_max.blockSignals(False)
			#self.combobox.setCurrentIndex(options.index(getter()))
			#setter(value)
			if update and changed:
				self.update()
		# auto getter and setter
		setattr(self, "get_value", getter)
		setattr(self, "set_value", wrap_setter)

		def get():
			#vmin, vmax = None, None
			if self.combobox_min.lineEdit().text().strip():
				try:
					self.vmin = float(self.combobox_min.lineEdit().text())
				except:
					logger.exception("parsing vmin")
					dialog_error(self.combobox_min, "Error parsing number", "Cannot parse number: %s" % self.combobox_min.lineEdit().text())

			if self.combobox_max.lineEdit().text().strip():
				try:
					self.vmax = float(self.combobox_max.lineEdit().text())
				except:
					logger.exception("parsing vmax")
					dialog_error(self.combobox_max, "Error parsing number", "Cannot parse number: %s" % self.combobox_max.lineEdit().text())
			return (self.vmin, self.vmax) if self.vmin is not None and self.vmax is not None else None


		def on_change(_ignore=None):
			#setter(self.options[index])
			value = get()
			if value:
				vmin, vmax = value
				if setter((vmin, vmax)):
					update()
				#self.weight_x_box.lineEdit().editingFinished.connect(lambda _=None: self.onWeightXExpr())

		self.combobox_min.lineEdit().returnPressed.connect(on_change)
		self.combobox_max.lineEdit().returnPressed.connect(on_change)
		#self.combobox_min.currentIndexChanged.connect(on_change)
		#self.combobox_max.currentIndexChanged.connect(on_change)
		self.combobox_layout = QtGui.QHBoxLayout(parent)
		self.combobox_layout.addWidget(self.combobox_min)
		self.combobox_layout.addWidget(self.combobox_max)
		#self.combobox_min.currentIndexChanged.connect(on_change_min)
		#self.combobox_max.currentIndexChanged.connect(on_change_max)
		#self.combobox.setCurrentIndex(options.index(getter()))

		if 1:
			from vaex.ui.icons import iconfile
			self.tool_button = QtGui.QToolButton(parent)
			self.tool_button .setIcon(QtGui.QIcon(iconfile('gear')))
			self.tool_menu = QtGui.QMenu()
			self.tool_button.setMenu(self.tool_menu)
			self.tool_button.setPopupMode(QtGui.QToolButton.InstantPopup)

			def flip(_=None):
				value = get()
				if value:
					vmin, vmax = value
					setter((vmax, vmin))
					self.set_value((vmax, vmin))
			self.action_flip = QtGui.QAction("Flip axis", parent)
			self.action_flip.triggered.connect(flip)
			self.tool_menu.addAction(self.action_flip)

			def copy(_=None):
				value = get()
				if value:
					clipboard = QtGui.QApplication.clipboard()
					text = str(value)
					clipboard.setText(text)
			self.action_copy = QtGui.QAction("Copy", parent)
			self.action_copy.triggered.connect(copy)
			self.tool_menu.addAction(self.action_copy)

			def paste(_=None):
				clipboard = QtGui.QApplication.clipboard()
				text = clipboard.text()
				try:
					vmin, vmax = eval(text)
					setter((vmin, vmax))
					self.set_value((vmin, vmax))
				except Exception as e:
					dialog_error(parent, "Could not parse min/max values", "Could not parse min/max values: %r" % e)
			self.action_paste = QtGui.QAction("Paste", parent)
			self.action_paste.triggered.connect(paste)
			self.tool_menu.addAction(self.action_paste)



	def add_to_grid_layout(self, row, grid_layout):
		#grid_layout.addLayout(self.combobox_layout, row, 1)
		grid_layout.addWidget(self.label, row, 0)
		grid_layout.addWidget(self.combobox_min, row, 1)
		grid_layout.addWidget(self.tool_button, row, 2)
		row += 1
		#grid_layout.addWidget(self.label, row, 0)
		grid_layout.addWidget(self.combobox_max, row, 1)
		return row + 1

# list of rgb values from tableau20
color_list = [(255, 187, 120),
 (255, 127, 14),
 (174, 199, 232),
 (44, 160, 44),
 (31, 119, 180),
 (255, 152, 150),
 (214, 39, 40),
 (197, 176, 213),
 (152, 223, 138),
 (148, 103, 189),
 (247, 182, 210),
 (227, 119, 194),
 (196, 156, 148),
 (140, 86, 75),
 (127, 127, 127),
 (219, 219, 141),
 (199, 199, 199),
 (188, 189, 34),
 (158, 218, 229),
 (23, 190, 207)]

class ColorOption(object):
	def __init__(self, parent, label, getter, setter, update=lambda: None):
		self.update = update
		#self.options = options
		self.label = QtGui.QLabel(label, parent)
		self.combobox = QtGui.QComboBox(parent)
		index = 0
		self.qt_colors = []
		for color_tuple in color_list:
			#self.combobox.addItems(options)
			self.combobox.addItem(",".join(map(str, color_tuple)))
			model = self.combobox.model().index(index, 0)
			color = QtGui.QColor(*color_tuple)
			self.combobox.model().setData(model, color, QtCore.Qt.BackgroundColorRole)
			index += 1
			self.qt_colors.append(color)
		def wrap_setter(value, update=True):
			index = color_list.index(getter())
			self.combobox.setCurrentIndex(index)
			self.combobox.palette().setColor(QtGui.QPalette.Background, self.qt_colors[index]);
			self.combobox.palette().setColor(QtGui.QPalette.Highlight, self.qt_colors[index]);
			print("SETTING"*100, repr(value))
			setter([c/255. for c in value])
			if update:
				self.update()
		# auto getter and setter
		setattr(self, "get_value", getter)
		setattr(self, "set_value", wrap_setter)

		def on_change(index):
			print("setter", index, color_list[index])
			self.set_value(color_list[index])
			update()
		self.combobox.currentIndexChanged.connect(on_change)
		self.combobox.setCurrentIndex(color_list.index(getter()))

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
			print(label_text, "set to", value)
			setter(value)
			self.update()
		self.checkbox.setChecked(getter())
		self.checkbox.stateChanged.connect(on_change)

	def add_to_grid_layout(self, row, grid_layout, column_start=0):
		grid_layout.addWidget(self.label, row,  column_start+0)
		grid_layout.addWidget(self.checkbox, row,  column_start+1)
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
			print(label_text, "set to", value)
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

class QuickDialog(QtGui.QDialog):
	def __init__(self, parent, title, validate=None):
		QtGui.QDialog.__init__(self, parent)
		self.setWindowTitle(title)
		self.layout = QtGui.QFormLayout(self)
		self.layout.setFieldGrowthPolicy(QtGui.QFormLayout.AllNonFixedFieldsGrow)
		self.values = {}
		self.widgets = {}
		self.validate = validate
		self.button_box = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok|QtGui.QDialogButtonBox.Cancel, QtCore.Qt.Horizontal, self);
		self.button_box.accepted.connect(self.check_accept)
		self.button_box.rejected.connect(self.reject)
		self.setLayout(self.layout)

	def get(self):
		self.layout.addWidget(self.button_box)
		if self.exec_() == QtGui.QDialog.Accepted:
			return self.values
		else:
			return None

	def check_accept(self):
		logger.debug("on accepted")
		for name, widget in self.widgets.items():
			if isinstance(widget, QtGui.QLabel):
				pass#self.values[name] = None
			elif isinstance(widget, QtGui.QLineEdit):
				self.values[name] = widget.text()
			elif isinstance(widget, QtGui.QComboBox):
				self.values[name] = widget.currentText() #lineEdit().text()
			else:
				raise NotImplementedError
		if self.validate is None or self.validate(self, self.values):
			self.accept()


	def accept(self):#(self, *args):
		return QtGui.QDialog.accept(self)

	def add_label(self, name, label=""):
		self.widgets[name] = widget = QtGui.QLabel(label)
		self.layout.addRow("", widget)

	def add_text(self, name, label="", value="", placeholder=None):
		self.widgets[name] = widget = QtGui.QLineEdit(value, self)
		if placeholder:
			widget.setPlaceholderText(placeholder)
		self.layout.addRow(label, widget)

	def add_password(self, name, label="", value="", placeholder=None):
		self.widgets[name] = widget = QtGui.QLineEdit(value, self)
		widget.setEchoMode(QtGui.QLineEdit.Password)
		if placeholder:
			widget.setPlaceholderText(placeholder)
		self.layout.addRow(label, widget)

	def add_ucd(self, name, label="", value=""):
		self.widgets[name] = widget = QtGui.QLineEdit(value, self)
		widget.setCompleter(vaex.ui.completer.UCDCompleter(widget))
		self.layout.addRow(label, widget)

	def add_combo_edit(self, name, label="", value="", values=[]):
		self.widgets[name] = widget = QtGui.QComboBox(self)
		widget.addItems([value] + values)
		widget.setEditable(True)
		self.layout.addRow(label, widget)

	def add_expression(self, name, label, value, dataset):
		import vaex.ui.completer
		self.widgets[name] = widget = vaex.ui.completer.ExpressionCombobox(self, dataset)
		if value is not None:
			widget.lineEdit().setText(value)
		self.layout.addRow(label, widget)

	def add_variable_expression(self, name, label, value, dataset):
		import vaex.ui.completer
		self.widgets[name] = widget = vaex.ui.completer.ExpressionCombobox(self, dataset, variables=True)
		if value is not None:
			widget.lineEdit().setText(value)
		self.layout.addRow(label, widget)

	def add_combo(self, name, label="", values=[]):
		self.widgets[name] = widget = QtGui.QComboBox(self)
		widget.addItems(values)
		self.layout.addRow(label, widget)


def get_path_save(parent, title="Save file", path="", file_mask="HDF5 *.hdf5"):
	path = QtGui.QFileDialog.getSaveFileName(parent, title, path, file_mask)
	if isinstance(path, tuple):
		path = str(path[0])#]
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

class Thenner(object):
	def __init__(self, callback):
		self.callback = callback
		self.thennable = False
	def then(self, *args, **kwargs):
		self.thennable = True
		self.args = args
		self.kwargs = kwargs

	def do(self):
		if self.thennable:
			self.callback(*self.args, **self.kwargs)

def set_choose(value, ok=None):
	global choose
	thenner = Thenner(set_choose)
	def wrapper(*args):
		thenner.do()
		return value# if ok is None else (value, ok)
	choose = wrapper
	return thenner
QtGui.QInputDialog_getItem = QtGui.QInputDialog.getItem
def choose(parent, title, label, options, index=0, editable=False):
	text, ok = QtGui.QInputDialog_getItem(parent, title, label, options, index, editable)
	if editable:
		return text if ok else None
	else:
		return options.index(text) if ok else None

def set_select_many(ok, mask):
	global select_many
	def wrapper(*args):
		return ok, mask
	select_many = wrapper
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


import psutil
def memory_check_ok(parent, bytes_needed):
	bytes_available = psutil.virtual_memory().available
	alot = bytes_needed / bytes_available > 0.5
	required = vaex.utils.filesize_format(bytes_needed)
	available = vaex.utils.filesize_format(bytes_available)
	msg = "This action required {required} of memory, while you have {available}. Are you sure you want to continue?".format(**locals())
	return not alot or dialog_confirm(parent, "A lot of memory requested", msg)


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
import vaex
import smtplib
import platform
import getpass
import sys
import os
#import urllib.request, urllib.parse, urllib.error
import vaex.utils
#from email.mime.text import MIMEText

try:
	from urllib.request import urlopen
	from urllib.parse import urlparse, urlencode
except ImportError:
	from urlparse import urlparse
	from urllib import urlopen, urlencode, quote as urlquote


def email(text):
	osname = platform.system().lower()
	if osname == "linux":
		text = text.replace("#", "%23") # for some reason, # needs to be double quoted on linux, otherwise it is interpreted as comment symbol
	
	body = urlquote(text)
		
	subject = urlquote('Error report for: ' +vaex.__full_name__)
	mailto = "mailto:maartenbreddels@gmail.com?subject={subject}&body={body}".format(**locals())
	print("open:", mailto)
	vaex.utils.os_open(mailto)
		


def old_email(text):
	# Open a plain text file for reading.  For this example, assume that
	msg = MIMEText(text)

	msg['Subject'] = 'Error report for: ' +vaex.__full_name__
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
	print(trace)
	info = "username: %r\n" % (getpass.getuser(),)
	info += "program: %r\n" % vaex.__program_name__
	info += "version: %r\n" % vaex.__version__
	info += "full name: %r\n" % vaex.__full_name__
	info += "arguments: %r\n" % sys.argv
	info += "Qt version: %r\n" % qt_version
	
	attrs = sorted(dir(platform))
	for attr in attrs:
		if not attr.startswith("_") and attr not in ["popen", "system_alias"]:
			f = getattr(platform, attr)
			if isinstance(f, collections.Callable):
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
		print("exit")
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



	