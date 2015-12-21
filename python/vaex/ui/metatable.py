__author__ = 'breddels'

import sys
import vaex
from vaex.ui.qt import *
import vaex.ui.qt as dialogs
import astropy.units
import astropy.io.votable.ucd
import logging
from vaex.ui.icons import iconfile

logger = logging.getLogger("vaex.ui.metatable")

completerContents = "blaat schaap aap koe".split()
words = astropy.io.votable.ucd.UCDWords()
ucd_words =  list(words._primary.union(words._secondary))
ucd_words.sort()

# from http://stackoverflow.com/questions/5816383/model-view-qcompleter-in-a-qlineedit
# and http://stackoverflow.com/questions/24687620/looking-for-example-for-qcompleter-with-segmented-completion-tree-models

class UCDTreeModel(QtCore.QAbstractItemModel):
	def __init__(self):
		super(UCDTreeModel, self).__init__()
		self.rootlist = []
		for word in words._primary:
			self.rootlist.append(word)
		#for
		self.primary = list(words._primary)
		self.primary.sort()
		#self.primary = self.primary[:3]
		self.secondary = list(words._secondary)
		self.secondary.sort()
		#self.secondary = self.secondary[:3]
		self.refs = []

	def parent(self, index):
		#print "parent", index, index.row(), index.column()
		if not index.isValid():
			return QtCore.QModelIndex()
		parts = index.internalPointer()
		if len(parts) == 1:
			return QtCore.QModelIndex()
		else:
			parts = parts[:-1]
			self.refs.append(parts)
			if len(parts) == 1:
				return self.createIndex(self.primary.index(parts[0]), 0, parts)
			else:
				return self.createIndex(self.secondary.index(parts[-1]), 0, parts)
		#parts = ";".join(data.split(";")[:-1])
		#return


	def index(self, row, column, parent):
		#print "index", row, column, parent, parent.internalPointer()
		if not self.hasIndex(row, column, parent):
			return QtCore.QModelIndex()

		if not parent.isValid():
			parts = [self.primary[row]]
			self.refs.append(parts)
			return self.createIndex(row, column, parts)
		else:
			parts = parent.internalPointer() + [self.secondary[row]]
			self.refs.append(parts)
			return self.createIndex(row, column, parts)

	def columnCount(self, parent):
		return 1

	def rowCount(self, parent):
		if parent.isValid():
			return len(self.secondary)
		else:
			return len(self.primary)

	def data(self, index, role):
		#print "data", index, role, index.isValid(), index.column(), index.row()
		if role == QtCore.Qt.DisplayRole:
			if index.parent().isValid():
				#return self.data(index.parent(), role) + ";" + self.secondary[index.row()]
				return self.secondary[index.row()]
			else:
				return self.primary[index.row()]
		if role == QtCore.Qt.EditRole:
			if index.parent().isValid():
				return self.secondary[index.row()]
			else:
				return self.primary[index.row()]
		return None

class UCDCompleter(QtGui.QCompleter):
	def __init__(self, parent):
		QtGui.QCompleter.__init__(self, parent)
		self._model = UCDTreeModel()
		self.setModel(self._model)

	def splitPath(self, text):
		return text.split(";")

	def pathFromIndex(self, index):
		result = []
		while index.isValid():
			result = [self.model().data(index, QtCore.Qt.DisplayRole)] + result
			index = index.parent()
		r = ';'.join(result)
		return r

class LineEdit(QtGui.QLineEdit):
	def __init__(self, parent, word_list):
		super(LineEdit, self).__init__(parent)

		self.full_word_list = word_list

		#self.completerList = QtCore.QStringList()
		#for content in completerContents:
		#	self.completerList.append(QtCore.QString(content))
		#self.completer = UCDCompleter(self)
		self.completer = QtGui.QCompleter(self.full_word_list, self)
		#self.completer.setFilterMode(QtCore.Qt.MatchContains)
		self.completer.setCompletionMode(QtGui.QCompleter.UnfilteredPopupCompletion)
		self.completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
		self.setCompleter(self.completer)
		self.completer.activated.connect(self.onActivated)

		self.textChanged.connect(self.OnTextChanged)

	def onActivated(self, text):
		print text

	def OnTextChanged(self, text):
		#print self.completer.completionPrefix()
		#return
		right = left = pos = self.cursorPosition()
		print "pos", pos, repr(text)
		# find left index
		done = False
		while not done:
			if left == len(text):
				left -= 1
			elif left == 0:
				done = True
			elif text[left-1] == ";":
				done = True
			else:
				left -= 1
			#print "left", left

		done = False
		while not done:
			if right == len(text):
				done = True
			elif right == 0:
				right += 1
			elif text[right] == ";":
				done = True
			else:
				right -= 1
			#print "right", left

		part = text[left:right]
		if left == 0:
			full_word_list = words._primary
		else:
			full_word_list = words._secondary
		#print "part", part
		logger.debug("completion part: " +part)
		suggestions = []
		for word in full_word_list:
			if part in word:
				suggestions.append((text[:left] if text[:left] else "") + word + (text[right:] if text[right:] else ""))
		#matching_words = [word for word in full_word_list if part in word]
		model = QtGui.QStringListModel(suggestions)
		#model = QtGui.QStandardItemModel()
		#for word in full_word_list:
		#	item = QtGui.QStandardItem(word)
		#	item.setData("bla", QtCore.Qt.EditRole)
		#	item.setData("idps", QtCore.Qt.DisplayRole)
		#	model.appendRow(item)
		self.completer.setModel(model)


# from https://gist.github.com/Riateche/5984815
class ComboDelegate(QtGui.QItemDelegate):
	"""
	A delegate that places a fully functioning QComboBox in every
	cell of the column to which it's applied
	"""
	def __init__(self, parent):

		QtGui.QItemDelegate.__init__(self, parent)

	def createEditor(self, parent, option, index):
		combo = LineEdit(parent, list(ucd_words))
		#self.connect(combo, QtCore.SIGNAL("currentIndexChanged(int)"), self, QtCore.SLOT("currentIndexChanged()"))
		return combo

	def setEditorData(self, editor, index):
		editor.blockSignals(True)
		#editor.setCurrentIndex(int(index.model().data(index)))
		editor.setText(index.model().data(index))
		editor.blockSignals(False)

	def setModelData(self, editor, model, index):
		print model
		model.setData(index, editor.text())

	@QtCore.pyqtSlot()
	def currentIndexChanged(self):
		self.commitData.emit(self.sender())

class MetaTableModel(QtCore.QAbstractTableModel):
	def __init__(self, dataset, parent=None, *args):
		"""
		:type dataset: Dataset
		"""
		QtCore.QAbstractTableModel.__init__(self, parent, *args)
		self.dataset = dataset
		self.row_count_start = 1
		self.table_column_names = ["Visible", "Name", "Type", "Units", "UCD", "Description", "Expression"]

		self.show_virtual = True

	def get_dataset_column_names(self):
		return self.dataset.get_column_names(virtual=self.show_virtual)

	def rowCount(self, parent):
		column_names = self.get_dataset_column_names()
		return len(column_names)

	def columnCount(self, parent):
		return len(self.table_column_names) + 1

	def setData(self, index, value, role=QtCore.Qt.EditRole):
		row = index.row()
		column_index = index.column()-1
		column_name = self.get_dataset_column_names()[row]
		property = self.table_column_names[column_index]
		#print index, value, role
		if property == "Visible":
			logger.debug("set visibility to: %s", value == QtCore.Qt.Checked)
		if property == "Description":
			self.dataset.descriptions[column_name] = value
		if property == "UCD":
			self.dataset.ucds[column_name] = value
		if property == "Units":
			if value:
				logger.debug("setting unit to: %s (%s)" % (value, astropy.units.Unit(value)))
				self.dataset.units[column_name] = str(astropy.units.Unit(value))
			else:
				if column_name in self.dataset.units:
					del self.dataset.units[column_name]
		if property == "Expression":
			self.dataset.virtual_columns[column_name] = value
		self.dataset.write_meta()
		return True

	def data(self, index, role=QtCore.Qt.DisplayRole):
		#row_offset = self.get_row_offset()
		#print index, role
		if not index.isValid():
			return None
		if role == QtCore.Qt.CheckStateRole and index.column() == 1:
			return QtCore.Qt.Checked

		elif role not in [QtCore.Qt.DisplayRole, QtCore.Qt.EditRole]:
			return None
		if index.column() == 0:
			#return "{:,}".format(index.row()+self.row_count_start + row_offset)
			return str(index.row()+self.row_count_start)
		else:
			row = index.row()
			column_index = index.column()-1
			column_name = self.get_dataset_column_names()[row]
			property = self.table_column_names[column_index]
			column = None
			if column_name in self.dataset.columns:
				column = self.dataset.columns[column_name]

			#if property == "Visible":
			#	return QtCore.Qt.Checked
			if property == "Name":
				return column_name
			elif property == "Type":
				if column is not None:
					return str(column.dtype)
				else:
					return "virtual column"
			elif property == "Units":
				return str(self.dataset.units.get(column_name, ""))
			elif property == "UCD":
				return self.dataset.ucds.get(column_name, "")
			elif property == "Description":
				return self.dataset.descriptions.get(column_name, "")
			elif property == "Expression":
				return self.dataset.virtual_columns.get(column_name, "")

	def flags(self, index):
		row = index.row()
		column_index = index.column()-1
		if column_index == 0:
			return QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
		column_name = self.get_dataset_column_names()[row]
		property = self.table_column_names[column_index]
		column = None
		flags = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
		if column_name in self.dataset.columns:
			column = self.dataset.columns[column_name]
		if property in ["Description", "Units", "UCD"]:
			flags |= QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsSelectable
		if column_name in self.dataset.virtual_columns:
			flags |= QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsSelectable
		return flags


	def headerData(self, index, orientation, role):
		#row_offset = self.get_row_offset()
		if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
			if index == 0:
				return ""
			else:
				return self.table_column_names[index-1]
		#if orientation == QtCore.Qt.Vertical and role == QtCore.Qt.DisplayRole:
		#	return str(index+self.row_count_start + row_offset)
		return None

	def insertRows(self, *args):
		return True

class MetaTable(QtGui.QWidget):

	def set_dataset(self, dataset):
		if self.event_handler:
			self.dataset.signal_column_changed.disconnect(self.event_handler)
		self.dataset = dataset
		self.tableModel = MetaTableModel(self.dataset, self)
		self.tableView.setModel(self.tableModel)
		self.tableView.selectionModel().currentChanged.connect(self.onCurrentChanged)
		self.tableView.resizeColumnsToContents()
		#self.tableView.horizontalHeader().setResizeMode(QtGui.QHeaderView.Stretch);
		self.tableView.horizontalHeader().setStretchLastSection(True)
		self.event_handler = self.dataset.signal_column_changed.connect(self.on_column_change)

	def on_column_change(self, *args):
		self.reset()

	def __init__(self, parent):
		super(MetaTable, self).__init__(parent)
		#dataset.add_virtual_column("xp", "x")
		self.event_handler = None
		self.resize(700, 500)
		self.tableView = QtGui.QTableView()
		#self.tableView.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows);
		#self.header = self.dataset.column_names
		#self.tableView.pressed.connect(self.onSelectRow)

		self.tableView.verticalHeader().setResizeMode(QtGui.QHeaderView.Interactive)
		self.tableView.setItemDelegateForColumn(5, ComboDelegate(self))

		self.toolbar = QtGui.QToolBar(self)
		#self.description = QtGui.QTextEdit(self.dataset.description, self)
		#self.description.setFixedHeight(100)
		#self.description.textChanged.connect(self.onTextChanged)

		self.action_add = QtGui.QAction(QtGui.QIcon(iconfile('table-insert-column')), 'Add virtual column', self)
		self.action_remove = QtGui.QAction(QtGui.QIcon(iconfile('table-delete-column')), 'Remove virtual column', self)
		self.action_remove.setEnabled(False)
		self.action_add.setShortcut("Ctrl++")
		self.action_remove.setShortcut("Ctrl+-")

		self.toolbar.addAction(self.action_add)
		self.toolbar.addAction(self.action_remove)

		self.action_add.triggered.connect(self.onAdd)
		self.action_remove.triggered.connect(self.onRemove)
		#self.tableView.pressed.connect(self.onSelectRow)
		#self.tableView.activated.connect(self.onActivateRow)
		#self.tableView.selectionModel().currentChanged.connect(self.onCurrentChanged)



		self.boxlayout = QtGui.QVBoxLayout(self)
		self.boxlayout.addWidget(self.toolbar, 0)
		#self.boxlayout.addWidget(self.description, 0)
		self.boxlayout.addWidget(self.tableView, 1)
		self.setLayout(self.boxlayout)
		self.tableView.resizeColumnsToContents()
		#self.tableView.horizontalHeader().setResizeMode(QtGui.QHeaderView.Stretch);
		self.tableView.horizontalHeader().setStretchLastSection(True)

	def onTextChanged(self, *args):
		self.dataset.description = self.description.toPlainText()
		logger.debug("setting description to: %s", self.dataset.description)
		self.dataset.write_meta()

	def onSelectRow(self, model):
		row_index = model.row()
		logger.debug("row index selected %d" % row_index)

	def onCurrentChanged(self, model, previous):
		#row_index = model.row()
		#logger.debug("row index activated %d" % row_index)
		self.check_remove()

	def check_remove(self):
		model = self.tableView.selectionModel().currentIndex()
		column_names = self.tableModel.get_dataset_column_names()
		column_name = column_names[model.row()]
		self.action_remove.setEnabled(column_name in self.dataset.virtual_columns)


	def onRemove(self, _=None):
		model = self.tableView.selectionModel().currentIndex()
		column_names = self.tableModel.get_dataset_column_names()
		column_name = column_names[model.row()]
		logger.debug("removing %s", column_name)
		#del self.dataset.virtual_columns[column_name]
		self.dataset.delete_virtual_column(column_name)
		#self.reset()
		self.check_remove()

	def reset(self):
		self.tableModel.beginResetModel()
		self.tableView.reset()
		self.tableModel.endResetModel()

	def onAdd(self, _=None):
		name = dialogs.gettext(self, "Give a name for the virtual column", "Give a name for the virtual column", "r")
		if name:
			expression = dialogs.gettext(self, "Enter an expression", "Enter an expression", "sqrt(x**2+y**2+z**2)")
			if expression:
				self.dataset.add_virtual_column(name, expression)
				#begin = self.tableModel.index(0, 0)
				#end = self.tableModel.index(50, 50)
				#self.tableModel.dataChanged.emit(begin, end)
				#self.tableModel.insertRows(0, 1)
				#self.update()
				#self.tableView.hide()
				#self.tableView.show()
				#self.tableView.setModel(self.tableModel)



def main(argv=sys.argv):
	dataset = vaex.open(argv[1])
	app = QtGui.QApplication(argv)
	table = MetaTable(None)
	self.set_dataset(dataset)
	table.show()
	table.raise_()
	sys.exit(app.exec_())

if __name__ == "__main__":
	vaex.set_log_level_debug()
	main()
	for i in range(3):
		for j in range(3):
			dataset.add_virtual_column("bla_%s%s" % (i, j), expr_matrix[i,j])

	dataset.add_virtual_columns_matrix3d("vx", "vy", "vz", "mu_alpha", "mu_delta", "vr", "bla")