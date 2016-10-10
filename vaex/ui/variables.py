import sys
import vaex
import vaex.ui.qt as dialogs
from vaex.ui.qt import *
import logging
from vaex.ui.icons import iconfile

logger = logging.getLogger("vaex.ui.variables")

class VariablesTableModel(QtCore.QAbstractTableModel):
	def __init__(self, dataset, parent=None, *args):
		"""
		:type dataset: Dataset
		"""
		QtCore.QAbstractTableModel.__init__(self, parent, *args)
		self.dataset = dataset
		self.row_count_start = 1
		#self.table_column_names = ["Type", "Name", "Value"]
		self.table_column_names = ["Name", "Expression", "Value"]
		#self.show_virtual = True

	def get_dataset_column_names(self):
		return list(self.dataset.variables.keys())

	def rowCount(self, parent):
		column_names = self.get_dataset_column_names()
		return len(column_names)

	def columnCount(self, parent):
		return len(self.table_column_names) + 1

	def setData(self, index, value, role=QtCore.Qt.EditRole):
		row = index.row()
		column_index = index.column()-1
		variable_name = self.get_dataset_column_names()[row]
		property = self.table_column_names[column_index]
		#print index, value, role
		#if property == "Visible":
		#	logger.debug("set visibility to: %s", value == QtCore.Qt.Checked)
		#if property == "Description":
		#	self.dataset.descriptions[column_name] = value
		#if property == "UCD":
		#	self.dataset.ucds[column_name] = value
		#	# TODO: move to dataset class
		#	self.dataset.signal_column_changed.emit(self.dataset, column_name, "change")
		#if property == "Units":
		#	if value:
		#		try:
		#			unit = astropy.units.Unit(value)
		#			logger.debug("setting unit to: %s (%s)" % (value, unit))
		#			self.dataset.units[column_name] = unit
		#			# TODO: move to dataset class
		#			self.dataset.signal_column_changed.emit(self.dataset, column_name, "change")
		#		except Exception, e:
		#			dialogs.dialog_error(None, "Cannot parse unit", "Cannot parse unit:\n %s" % e)
		#	else:
		#		if column_name in self.dataset.units:
		#			del self.dataset.units[column_name]
		if property == "Expression":
			try:
				test = eval(value, vaex.dataset.expression_namespace, self.dataset.variables)
				self.dataset.add_variable(variable_name, value)
			except Exception as e:
				dialogs.dialog_error(None, "Invalid expression", "Invalid expression: %s" % e)
			# although it may not be a valid expression, still set it to the user can edit it
			#self.dataset.virtual_columns[column_name] = value

		self.dataset.write_meta()
		return True

	def data(self, index, role=QtCore.Qt.DisplayRole):
		#row_offset = self.get_row_offset()
		#print index, role
		if not index.isValid():
			return None
		if 0: #role == QtCore.Qt.CheckStateRole and index.column() == 0:
			return QtCore.Qt.Checked

		elif role not in [QtCore.Qt.DisplayRole, QtCore.Qt.EditRole]:
			return None
		if index.column() == 0:
			#return "{:,}".format(index.row()+self.row_count_start + row_offset)
			return str(index.row()+self.row_count_start)
		else:
			row = index.row()
			column_index = index.column()-1
			variable_name = self.get_dataset_column_names()[row]
			property = self.table_column_names[column_index]
			#column = None
			#if column_name in self.dataset.get_column_names():
			#	column = self.dataset.columns[column_name]

			#if property == "Visible":
			#	return QtCore.Qt.Checked
			if property == "Name":
				return variable_name
			elif property == "Type":
				if column_name in self.dataset.get_column_names():
					return str(self.dataset.dtype(column_name))
				else:
					return "virtual column"
			elif property == "Units":
				unit = self.dataset.unit(column_name)
				return str(unit) if unit else ""
			elif property == "UCD":
				return self.dataset.ucds.get(column_name, "")
			elif property == "Description":
				return self.dataset.descriptions.get(column_name, "")
			elif property == "Value":
				#return str(self.dataset.variables[variable_name])
				try:
					return str(self.dataset.evaluate_variable(variable_name))
				except Exception as e:
					#dialogs.dialog_error(None, "Invalid expression", "Invalid expression: %s" % e)
					return "Error in expression: %s" % e
			elif property == "Expression":
				return str(self.dataset.variables[variable_name])

	def flags(self, index):
		row = index.row()
		column_index = index.column()-1
		if column_index == 0:
			return QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
		column_name = self.get_dataset_column_names()[row]
		property = self.table_column_names[column_index]
		column = None
		flags = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
		if property in ["Expression", "Description", "Units", "UCD"]:
			flags |= QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsSelectable
		#if column_name in self.dataset.virtual_columns:
		else:
			flags |= QtCore.Qt.ItemIsSelectable
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

class VariablesTable(QtGui.QWidget):

	def set_dataset(self, dataset):
		if self.event_handler:
			self.dataset.signal_variable_changed.disconnect(self.event_handler)
		self.dataset = dataset
		self.tableModel = VariablesTableModel(self.dataset, self)
		self.tableView.setModel(self.tableModel)
		self.tableView.selectionModel().currentChanged.connect(self.onCurrentChanged)
		self.tableView.resizeColumnsToContents()
		#self.tableView.horizontalHeader().setResizeMode(QtGui.QHeaderView.Stretch);
		self.tableView.horizontalHeader().setStretchLastSection(True)
		self.event_handler = self.dataset.signal_variable_changed.connect(self.on_variable_change)

	def on_variable_change(self, *args):
		self.reset()
		pass

	def __init__(self, parent, menu=None):
		super(VariablesTable, self).__init__(parent)
		#dataset.add_virtual_column("xp", "x")
		self.event_handler = None
		self.resize(700, 500)
		self.tableView = QtGui.QTableView()
		#self.tableView.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows);
		#self.header = self.dataset.column_names
		#self.tableView.pressed.connect(self.onSelectRow)

		if qt_mayor == 5:
			self.tableView.verticalHeader().setSectionResizeMode(QtGui.QHeaderView.Interactive)
		else:
			self.tableView.verticalHeader().setResizeMode(QtGui.QHeaderView.Interactive)
		#self.unit_delegate = vaex.ui.completer.UnitDelegate(self.tableView)
		#self.ucd_delegate = vaex.ui.completer.UCDDelegate(self.tableView)
		#self.tableView.setItemDelegateForColumn(4, self.unit_delegate)
		#self.tableView.setItemDelegateForColumn(5, self.ucd_delegate)

		self.toolbar = QtGui.QToolBar(self)
		#self.description = QtGui.QTextEdit(self.dataset.description, self)
		#self.description.setFixedHeight(100)
		#self.description.textChanged.connect(self.onTextChanged)

		#self.action_group_add = QtGui.QActionGroup(self)


		self.action_add = QtGui.QAction(QtGui.QIcon(iconfile('table-insert-column')), 'Add variable', self)
		self.action_remove = QtGui.QAction(QtGui.QIcon(iconfile('table-delete-column')), 'Remove variable', self)
		self.action_remove.setEnabled(False)
		self.action_add.setShortcut("Ctrl+Alt++")
		self.action_remove.setShortcut("Ctrl+Alt+-")

		self.toolbar.addAction(self.action_add)
		self.toolbar.addAction(self.action_remove)

		if 0:
			self.action_add_menu = QtGui.QMenu()
			self.action_add.setMenu(self.action_add_menu)

			self.action_celestial = QtGui.QAction(QtGui.QIcon(iconfile('table-insert-column')), 'Equatorial to galactic', self)
			self.action_celestial.setShortcut("Ctrl+G")
			self.action_add.menu().addAction(self.action_celestial)
			self.action_celestial.triggered.connect(lambda *args: add_celestial(self, self.dataset))

			self.action_car_to_gal = QtGui.QAction(QtGui.QIcon(iconfile('table-insert-column')), 'Cartesian to galactic', self)
			self.action_car_to_gal.setShortcut("Ctrl+S")
			self.action_add.menu().addAction(self.action_car_to_gal)
			self.action_car_to_gal.triggered.connect(lambda *args: add_sky(self, self.dataset, True))

			self.action_par_to_dis = QtGui.QAction(QtGui.QIcon(iconfile('table-insert-column')), 'Parallax to distance', self)
			self.action_par_to_dis.setShortcut("Ctrl+D")
			self.action_add.menu().addAction(self.action_par_to_dis)
			self.action_par_to_dis.triggered.connect(lambda *args: add_distance(self, self.dataset))

			self.action_gal_to_car = QtGui.QAction(QtGui.QIcon(iconfile('table-insert-column')), 'Galactic to cartesian', self)
			self.action_gal_to_car.setShortcut("Ctrl+C")
			self.action_add.menu().addAction(self.action_gal_to_car)
			self.action_gal_to_car.triggered.connect(lambda *args: add_cartesian(self, self.dataset, True))

			self.action_gal_to_aitoff = QtGui.QAction(QtGui.QIcon(iconfile('table-insert-column')), 'Galactic to Aitoff projection', self)
			self.action_gal_to_aitoff.setShortcut("Ctrl+A")
			self.action_add.menu().addAction(self.action_gal_to_aitoff)
			self.action_gal_to_aitoff.triggered.connect(lambda *args: add_aitoff(self, self.dataset, True))

		#action_group_add.add(self.action_add)

		self.action_add.triggered.connect(self.onAdd)
		self.action_remove.triggered.connect(self.onRemove)

		if menu:
			menu.addAction(self.action_add)
			menu.addAction(self.action_remove)
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
		self.action_remove.setEnabled(column_name not in ["e", "pi"])


	def onRemove(self, _=None):
		model = self.tableView.selectionModel().currentIndex()
		column_names = self.tableModel.get_dataset_column_names()
		column_name = column_names[model.row()]
		logger.debug("removing %s", column_name)
		#del self.dataset.virtual_columns[column_name]
		self.dataset.delete_variable(column_name)
		#self.reset()
		self.check_remove()

	def reset(self):
		self.tableModel.beginResetModel()
		self.tableView.reset()
		self.tableModel.endResetModel()

	def onAdd(self, _=None):
		dialog = dialogs.QuickDialog(self, title="Add variable")
		dialog.add_text("name", "Variable name", make_unique("var", self.dataset))
		dialog.add_variable_expression("expression", "Expression", "e**-1+sin(pi)", self.dataset)
		#dialog.add_unit("unit", "Expression", "sqrt(%s)" % self.dataset.get_column_names()[0], self.dataset)
		#dialog.add_ucd("ucd", "UCD", "")
		#dialog.add_text("description", "Description", placeholder="Enter a description")
		values = dialog.get()
		if values:
			#if values["description"]:
			#	self.dataset.descriptions[values["name"]] = values["description"]
			#if values["ucd"]:
			#	self.dataset.ucds[values["name"]] = values["ucd"]
			self.dataset.add_variable(values["name"], values["expression"])

def make_unique(name, dataset):
	postfix = ""
	number = 2
	original_name = name
	while name in dataset.get_column_names(virtual=True):
		name = original_name + "_" + str(number)
		number += 1
	return name

def main(argv=sys.argv):
	dataset = vaex.open(argv[1])
	app = QtGui.QApplication(argv)
	table = VariablesTable(None)
	table.set_dataset(dataset)
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