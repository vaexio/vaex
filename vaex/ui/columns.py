# -*- coding: utf-8 -*-
__author__ = 'breddels'
import sys
import vaex
from vaex.ui.qt import *
import vaex.ui.qt as dialogs
import astropy.units
import astropy.io.votable.ucd
import logging
from vaex.ui.icons import iconfile

logger = logging.getLogger("vaex.ui.columns")

completerContents = "blaat schaap aap koe".split()
words = astropy.io.votable.ucd.UCDWords()
ucd_words =  list(words._primary.union(words._secondary))
ucd_words.sort()
import vaex.ui.completer

class ColumnsTableModel(QtCore.QAbstractTableModel):
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
		return self.dataset.get_column_names(virtual=self.show_virtual, hidden=True, strings=True)

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
			# TODO: move to dataset class
			self.dataset.signal_column_changed.emit(self.dataset, column_name, "change")
		if property == "Units":
			if value:
				try:
					unit = astropy.units.Unit(value)
					logger.debug("setting unit to: %s (%s)" % (value, unit))
					self.dataset.units[column_name] = unit
					# TODO: move to dataset class
					self.dataset.signal_column_changed.emit(self.dataset, column_name, "change")
				except Exception as e:
					dialogs.dialog_error(None, "Cannot parse unit", "Cannot parse unit:\n %s" % e)
			else:
				if column_name in self.dataset.units:
					del self.dataset.units[column_name]
		if property == "Expression":
			try:
				self.dataset.validate_expression(value)
			except Exception as e:
				dialogs.dialog_error(None, "Invalid expression", "Invalid expression: %s" % e)
			# although it may not be a valid expression, still set it to the user can edit it
			self.dataset.virtual_columns[column_name] = value

		self.dataset.write_meta()
		return True

	def data(self, index, role=QtCore.Qt.DisplayRole):
		#row_offset = self.get_row_offset()
		#print index, role
		if not index.isValid():
			return None
		if role == QtCore.Qt.CheckStateRole and index.column() == 1:
			row = index.row()
			column_name = self.get_dataset_column_names()[row]
			return QtCore.Qt.Checked if not column_name.startswith("__") else QtCore.Qt.Unchecked

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
			#if column_name in self.dataset.get_column_names():
			#	column = self.dataset.columns[column_name]

			#if property == "Visible":
			#	return QtCore.Qt.Checked
			if property == "Name":
				return column_name
			elif property == "Type":
				if column_name in self.dataset.get_column_names(strings=True):
					dtype = self.dataset.dtype(column_name)
					return dtype.name
					#return str(self.dataset.dtype(column_name))
				else:
					return "virtual column"
			elif property == "Units":
				unit = self.dataset.unit(column_name)
				return str(unit) if unit else ""
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

class ColumnsTable(QtGui.QWidget):

	def set_dataset(self, dataset):
		if self.event_handler:
			self.dataset.signal_column_changed.disconnect(self.event_handler)
		self.dataset = dataset
		self.tableModel = ColumnsTableModel(self.dataset, self)
		self.tableView.setModel(self.tableModel)
		self.tableView.selectionModel().currentChanged.connect(self.onCurrentChanged)
		self.tableView.resizeColumnsToContents()
		#self.tableView.horizontalHeader().setResizeMode(QtGui.QHeaderView.Stretch);
		self.tableView.horizontalHeader().setStretchLastSection(True)
		self.event_handler = self.dataset.signal_column_changed.connect(self.on_column_change)

	def on_column_change(self, *args):
		self.reset()
		pass

	def __init__(self, parent, menu=None):
		super(ColumnsTable, self).__init__(parent)
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
		self.unit_delegate = vaex.ui.completer.UnitDelegate(self.tableView)
		self.ucd_delegate = vaex.ui.completer.UCDDelegate(self.tableView)
		self.tableView.setItemDelegateForColumn(4, self.unit_delegate)
		self.tableView.setItemDelegateForColumn(5, self.ucd_delegate)

		self.toolbar = QtGui.QToolBar(self)
		#self.description = QtGui.QTextEdit(self.dataset.description, self)
		#self.description.setFixedHeight(100)
		#self.description.textChanged.connect(self.onTextChanged)

		#self.action_group_add = QtGui.QActionGroup(self)

		self.action_add = QtGui.QAction(QtGui.QIcon(iconfile('table-insert-column')), 'Add virtual column', self)
		self.action_remove = QtGui.QAction(QtGui.QIcon(iconfile('table-delete-column')), 'Remove virtual column', self)
		self.action_remove.setEnabled(False)
		#self.action_add.setShortcut("Ctrl++")
		self.action_remove.setShortcut("Ctrl+-")

		self.toolbar.addAction(self.action_add)
		self.toolbar.addAction(self.action_remove)

		self.action_add_menu = QtGui.QMenu()
		self.action_add.setMenu(self.action_add_menu)

		self.action_normal = QtGui.QAction(QtGui.QIcon(iconfile('table-insert-column')), 'Add virtual column', self)
		self.action_normal.setShortcut("Ctrl++")
		self.action_add.menu().addAction(self.action_normal)
		self.action_normal.triggered.connect(self.onAdd)

		self.action_celestial = QtGui.QAction(QtGui.QIcon(iconfile('table-insert-column')), 'Equatorial to galactic', self)
		self.action_celestial.setShortcut("Ctrl+G")
		self.action_add.menu().addAction(self.action_celestial)
		self.action_celestial.triggered.connect(lambda *args: add_celestial(self, self.dataset))

		self.action_eq2ecl = QtGui.QAction(QtGui.QIcon(iconfile('table-insert-column')), 'Equatorial to ecliptic', self)
		#self.action_eq2ecl.setShortcut("Ctrl+G")
		self.action_add.menu().addAction(self.action_eq2ecl)
		self.action_eq2ecl.triggered.connect(lambda *args: add_celestial_eq2ecl(self, self.dataset))

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

		self.action_eq2gal_pm = QtGui.QAction(QtGui.QIcon(iconfile('table-insert-column')), 'Equatorial to galactic proper motion', self)
		#self.action_gal_to_aitoff.setShortcut("Ctrl+A")
		self.action_add.menu().addAction(self.action_eq2gal_pm)
		self.action_eq2gal_pm.triggered.connect(lambda *args: add_proper_motion_eq2gal(self, self.dataset))




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
		dialog = QuickDialog(self, title="Add virtual column")
		dialog.add_text("name", "Column name", make_unique("user", self.dataset))
		dialog.add_expression("expression", "Expression", "sqrt(%s)" % self.dataset.get_column_names()[0], self.dataset)
		#dialog.add_unit("unit", "Expression", "sqrt(%s)" % self.dataset.get_column_names()[0], self.dataset)
		dialog.add_ucd("ucd", "UCD", "")
		dialog.add_text("description", "Description", placeholder="Enter a description")
		values = dialog.get()
		if values:
			if values["description"]:
				self.dataset.descriptions[values["name"]] = values["description"]
			if values["ucd"]:
				self.dataset.ucds[values["name"]] = values["ucd"]
			self.dataset.add_virtual_column(values["name"], values["expression"])

	def onAddCelestial(self, *args):
		add_celestial(self, self.dataset)

def add_celestial_eq2ecl(parent, dataset):
	add_celestial(parent, dataset, type="ecliptic")

def add_celestial(parent, dataset, type="galactic"):
	result = dataset.ucd_find(["^pos.eq.ra", "^pos.eq.dec"])
	column_names = dataset.get_column_names(virtual=True)
	if result is None:
		result = ["", ""]

	if QtGui.QApplication.keyboardModifiers()  == QtCore.Qt.ShiftModifier and result is not None:
			values = dict(ra=result[0], dec=result[1], l="l", b="b", degrees="degrees")
	else:
		dialog = QuickDialog(parent, title="Celestial transform: equatorial to %s" % type)
		#dialog.add_combo("degrees", "Input in", ["degrees", "radians"])

		logger.debug("unit = %s", dataset.unit(column_names[0], default=astropy.units.deg))
		logger.debug("unit = %s", dataset.unit(column_names[0], default=astropy.units.deg) == astropy.units.rad)
		radians = (dataset.unit(result[0], default=astropy.units.deg) == astropy.units.rad)
		if radians:
			dialog.add_combo("degrees", "Input in", ["degrees", "radians"][::-1])
		else:
			dialog.add_combo("degrees", "Input in", ["degrees", "radians"])



		dialog.add_expression("ra", "Right ascension", result[0], dataset)
		dialog.add_expression("dec", "Declination", result[1], dataset)
		if type == "galactic":
			dialog.add_text("l", "Galactic l", "l")
			dialog.add_text("b", "Galactic b", "b")
		else:
			dialog.add_text("l", "Ecliptic ra", "lambda_")
			dialog.add_text("b", "Ecliptic dec", "beta")
		values = dialog.get()
	if values:
		dataset.ucds[values["l"]] = "pos.%s.lon" % type
		dataset.ucds[values["b"]] = "pos.%s.lat" % type
		dataset.units[values["l"]] = astropy.units.deg if values["degrees"] == "degrees" else astropy.units.rad
		dataset.units[values["b"]] = astropy.units.deg if values["degrees"] == "degrees" else astropy.units.rad
		if type == "galactic":
			dataset.add_virtual_columns_celestial(long_in=values["ra"], lat_in=values["dec"],
												   long_out=values["l"], lat_out=values["b"],
												   radians=values["degrees"] == "radians")
		else:
			dataset.add_virtual_columns_eq2ecl(long_in=values["ra"], lat_in=values["dec"],
												   long_out=values["l"], lat_out=values["b"],
												   radians=values["degrees"] == "radians")

def add_distance(parent, dataset):
	parallax = dataset.ucd_find(["pos.parallax"])
	column_names = dataset.get_column_names(virtual=True)
	if parallax is None:
		parallax = ""

	unit = dataset.unit(parallax)
	distance_name = make_unique("distance", dataset)
	if unit:
		convert = unit.to(astropy.units.mas)
		distance_expression = "%f/(%s)" % (convert, parallax)
	else:
		distance_expression = "1/(%s)" % (parallax)


	if QtGui.QApplication.keyboardModifiers()  == QtCore.Qt.ShiftModifier and parallax is not None:
		values = dict(distance=distance_name, parallax=parallax)
	else:
		dialog = QuickDialog(parent, title="Parallax to distance transform")
		#dialog.add_combo("parallax", "Input in", ["degrees", "radians"])
		dialog.add_expression("parallax", "Parallax", parallax, dataset)
		dialog.add_text("distance", "Distance name", distance_name)
		values = dialog.get()
	if values:
		dataset.ucds[values["distance"]] = "pos.distance"
		if unit:
			if unit == astropy.units.milliarcsecond:
				dataset.units["distance"] = astropy.units.kpc
			if unit == astropy.units.arcsecond:
				dataset.units["distance"] = astropy.units.parsec
		dataset.add_virtual_column(values["distance"], distance_expression)


def add_cartesian(parent, dataset, galactic=True):
	if galactic:
		spherical = [dataset.ucd_find(["pos.distance"]), dataset.ucd_find(["pos.galactic.lon"]), dataset.ucd_find(["pos.galactic.lat"])]
	else:
		spherical = [dataset.ucd_find(["pos.distance"]), dataset.ucd_find(["pos.eq.ra"]), dataset.ucd_find(["pos.eq.dec"])]
	column_names = dataset.get_column_names(virtual=True)

	if QtGui.QApplication.keyboardModifiers()  == QtCore.Qt.ShiftModifier and None not in spherical:
			values = dict(alpha=spherical[1], delta=spherical[2], distance=spherical[0], x="x", y="y", z="z",
						  degrees="degrees", solar_pos=repr(default_solar_position)
						  )
	else:
		dialog = QuickDialog(parent, title="Spherical to cartesian transform")
		if spherical[1]:
			radians = dataset.unit(spherical[1], default=astropy.units.deg) == astropy.units.rad
			if radians:
				dialog.add_combo("degrees", "Input in", ["degrees", "radians"][::-1])
			else:
				dialog.add_combo("degrees", "Input in", ["degrees", "radians"])
		else:
			dialog.add_combo("degrees", "Input in", ["degrees", "radians"])
		dialog.add_expression("distance", "Distance", spherical[0], dataset)
		dialog.add_expression("alpha", "Alpha", spherical[1], dataset)
		dialog.add_expression("delta", "Delta", spherical[2], dataset)
		# TODO: 8 should be in proper units
		dialog.add_combo_edit("solar_pos", "Solar position (x,y,z)", repr(default_solar_position), column_names)

		dialog.add_text("x", "x", make_unique("x", dataset))
		dialog.add_text("y", "y", make_unique("y", dataset))
		dialog.add_text("z", "z", make_unique("z", dataset))
		values = dialog.get()
	if values:
		pos = "pos.galactocentric" if galactic else "pos.heliocentric"
		if 0:
			units = dataset.unit(values["distance"])
			if units:
				dataset.units[values["x"]] = units
				dataset.units[values["y"]] = units
				dataset.units[values["z"]] = units
		dataset.ucds[values["x"]] = "pos.cartesian.x;%s" % pos
		dataset.ucds[values["y"]] = "pos.cartesian.y;%s" % pos
		dataset.ucds[values["z"]] = "pos.cartesian.z;%s" % pos
		solar_position = eval(values["solar_pos"])
		dataset.add_virtual_columns_spherical_to_cartesian(values["alpha"], values["delta"], values["distance"],
														   values["x"], values["y"], values["z"],
														   center=solar_position,
														   radians=values["degrees"] == "radians")
def add_cartesian_velocities(parent, dataset, galactic=True):
	if galactic:
		ucds = ["pos.distance", "^pos.galactic.lon", "^pos.galactic.lat", "pos.pm;pos.galactic.lon", "pos.pm;pos.galactic.lat", "spect.dopplerVeloc"]
	else:
		raise NotImplementedError("is this useful?")
	spherical = [dataset.ucd_find([ucd]) for ucd in ucds]
	column_names = dataset.get_column_names(virtual=True)

	if QtGui.QApplication.keyboardModifiers()  == QtCore.Qt.ShiftModifier and None not in spherical:
			values = dict(alpha=spherical[1], delta=spherical[2], distance=spherical[0], pm_alpha=spherical[3], pm_delta=spherical[4], vr=spherical[5],
						  degrees="degrees", solar_pos=repr(default_solar_position)
						  )
	else:
		dialog = QuickDialog(parent, title="Spherical motion to cartesian velocity")
		if spherical[1]:
			radians = dataset.unit(spherical[1], default=astropy.units.deg) == astropy.units.rad
			if radians:
				dialog.add_combo("degrees", "Input in", ["degrees", "radians"][::-1])
			else:
				dialog.add_combo("degrees", "Input in", ["degrees", "radians"])
		else:
			dialog.add_combo("degrees", "Input in", ["degrees", "radians"])
		dialog.add_expression("distance", "Distance", spherical[0], dataset)
		dialog.add_expression("alpha", "Alpha", spherical[1], dataset)
		dialog.add_expression("delta", "Delta", spherical[2], dataset)
		dialog.add_expression("pm_alpha", "pm_Alpha*", spherical[3], dataset)
		dialog.add_expression("pm_delta", "pm_Delta", spherical[4], dataset)
		dialog.add_expression("vr", "radial velocity", spherical[5], dataset)
		# TODO: 8 should be in proper units
		dialog.add_combo_edit("solar_velocity", "Solar velocity (vx,vy,vz)", default_solar_velocity, column_names)

		dialog.add_text("vx", "vx_gal", make_unique("vx_gal", dataset))
		dialog.add_text("vy", "vy_gal", make_unique("vy_gal", dataset))
		dialog.add_text("vz", "vz_gal", make_unique("vz_gal", dataset))
		values = dialog.get()
	if values:
		pos = "pos.galactocentric" if galactic else "pos.heliocentric"
		if 0:
			units = dataset.unit(values["distance"])
			if units:
				dataset.units[values["x"]] = units
				dataset.units[values["y"]] = units
				dataset.units[values["z"]] = units
		dataset.ucds[values["vx"]] = "phys.veloc;pos.cartesian.x;%s" % pos
		dataset.ucds[values["vy"]] = "phys.veloc;pos.cartesian.y;%s" % pos
		dataset.ucds[values["vz"]] = "phys.veloc;pos.cartesian.z;%s" % pos
		solar_velocity = eval(values["solar_velocity"])
		dataset.add_virtual_columns_lbrvr_proper_motion2vcartesian(values["alpha"], values["delta"], values["distance"],
														   values["pm_alpha"], values["pm_delta"], values["vr"],
															values["vx"], values["vy"], values["vz"],
														   center_v=solar_velocity,
														   radians=values["degrees"] == "radians")
def make_unique(name, dataset):
	postfix = ""
	number = 2
	original_name = name
	while name in dataset.get_column_names(virtual=True):
		name = original_name + "_" + str(number)
		number += 1
	return name

default_solar_position = (-8, 0, 0)
default_solar_velocity = "(10., 220+5.2, 7.2)"
def add_sky(parent, dataset, galactic=True):
	if galactic:
		pos = "pos.galactocentric"
	else:
		pos = "pos.heliocentric"
	cartesian = [dataset.ucd_find(["pos.cartesian.x;%s" % pos]), dataset.ucd_find(["pos.cartesian.y;%s" % pos]),
				 dataset.ucd_find(["pos.cartesian.z;%s" % pos])]
	column_names = dataset.get_column_names(virtual=True)

	if QtGui.QApplication.keyboardModifiers()  == QtCore.Qt.ShiftModifier and None not in cartesian:
			values = dict(x=cartesian[0], y=cartesian[1], z=cartesian[2],
							alpha=make_unique("l" if galactic else "alpha", dataset),
							delta=make_unique("b" if galactic else "delta", dataset),
							distance=make_unique("distance", dataset),
							solar_pos=repr(default_solar_position),
							degrees="degrees")
	else:
		dialog = QuickDialog(parent, title="Cartesian to spherical transform")
		dialog.add_expression("x", "x", cartesian[0], dataset)
		dialog.add_expression("y", "y", cartesian[1], dataset)
		dialog.add_expression("z", "z", cartesian[2], dataset)
		# TODO: 8 should be in proper units
		dialog.add_combo_edit("solar_pos", "Solar position (x,y,z)", repr(default_solar_position), [])
		dialog.add_combo("degrees", "Output in", ["degrees", "radians"])

		dialog.add_text("distance", "Distance", make_unique("distance", dataset))
		dialog.add_text("alpha", "Alpha", make_unique("l" if galactic else "alpha", dataset))
		dialog.add_text("delta", "Delta", make_unique("b" if galactic else "delta", dataset))
		values = dialog.get()
	if values:
		units = dataset.unit(values["x"])
		pos = "pos.galactocentric" if galactic else "pos.heliocentric"
		dataset.units[values["alpha"]] = astropy.units.deg if values["degrees"] == "degrees" else astropy.units.rad
		dataset.units[values["delta"]] = astropy.units.deg if values["degrees"] == "degrees" else astropy.units.rad
		if units:
			dataset.units[values["distance"]] = units
		dataset.ucds[values["distance"]] = "pos.distance;%s" % pos
		dataset.ucds[values["alpha"]] = "pos.galactic.lon" if galactic else "pos.eq.ra"
		dataset.ucds[values["delta"]] = "pos.galactic.lat" if galactic else "pos.eq.dec"
		solar_position = eval(values["solar_pos"])
		dataset.add_virtual_columns_cartesian_to_spherical(values["x"], values["y"], values["z"],
														   values["alpha"], values["delta"], values["distance"],
														   radians=values["degrees"] == "radians", center=solar_position)

def add_aitoff(parent, dataset, galactic=True):
	if galactic:
		spherical = [dataset.ucd_find(["pos.galactic.lon"]), dataset.ucd_find(["pos.galactic.lat"])]
	else:
		spherical = [dataset.ucd_find(["pos.eq.ra"]), dataset.ucd_find(["pos.eq.dec"])]
	column_names = dataset.get_column_names(virtual=True)

	if QtGui.QApplication.keyboardModifiers()  == QtCore.Qt.ShiftModifier and None not in spherical:
			values = dict(alpha=spherical[0], delta=spherical[1], x="x_aitoff", y="y_aitoff", degrees="degrees")
	else:
		dialog = QuickDialog(parent, title="Spherical to cartesian transform")
		if spherical[1]:
			radians = dataset.unit(spherical[1], default=astropy.units.deg) == astropy.units.rad
			if radians:
				dialog.add_combo("degrees", "Input in", ["degrees", "radians"][::-1])
			else:
				dialog.add_combo("degrees", "Input in", ["degrees", "radians"])
		else:
			dialog.add_combo("degrees", "Input in", ["degrees", "radians"])
		dialog.add_expression("alpha", "Alpha", spherical[0], dataset)
		dialog.add_expression("delta", "Delta", spherical[1], dataset)

		dialog.add_text("x", "x", make_unique("x_aitoff", dataset))
		dialog.add_text("y", "y", make_unique("y_aitoff", dataset))
		values = dialog.get()
	if values:
		#pos = "pos.galactic" if galactic else "pos.eq"
		#dataset.ucds[values["x"]] = "pos.cartesian.x;%s" % pos
		#dataset.ucds[values["y"]] = "pos.cartesian.y;%s" % pos
		#dataset.ucds[values["z"]] = "pos.cartesian.z;%s" % pos
		alpha = values["alpha"]
		if galactic: # go from 0-360 to -180-180
			if values["degrees"] == "radians":
				alpha = "((%s+pi) %% (2*pi) - pi)" % values["alpha"]
			else:
				alpha = "((%s+180) %% 360 - 180)" % values["alpha"]

		dataset.add_virtual_columns_aitoff(alpha, values["delta"],
										   values["x"], values["y"], radians=values["degrees"] == "radians")


def add_proper_motion_eq2gal(parent, dataset, type="galactic"):
	assert type == "galactic"
	default_columns = dataset.ucd_find(["^pos.eq.ra", "^pos.eq.dec", "pos.pm;pos.eq.ra", "pos.pm;pos.eq.dec"])
	column_names = dataset.get_column_names(virtual=True)
	if default_columns is None:
		default_columns = ["", "", "", ""]

	if QtGui.QApplication.keyboardModifiers()  == QtCore.Qt.ShiftModifier and default_columns is not None:
		values = dict(alpha=default_columns[0], delta=default_columns[1], pm_alpha=default_columns[2], pm_delta=default_columns[3], pm_alpha_out="pm_l", pm_delta_out="pm_b", degrees="degrees")
	else:
		dialog = QuickDialog(parent, title="Proper motion transform: equatorial to %s" % type)
		#dialog.add_combo("degrees", "Input in", ["degrees", "radians"])

		#logger.debug("unit = %s", dataset.unit(column_names[0], default=astropy.units.deg))
		#logger.debug("unit = %s", dataset.unit(column_names[0], default=astropy.units.deg) == astropy.units.rad)
		radians = (dataset.unit(default_columns[0], default=astropy.units.deg) == astropy.units.rad)
		if radians:
			dialog.add_combo("degrees", "Input in", ["degrees", "radians"][::-1])
		else:
			dialog.add_combo("degrees", "Input in", ["degrees", "radians"])



		dialog.add_expression("alpha", "Right ascension", default_columns[0], dataset)
		dialog.add_expression("delta", "Declination", default_columns[1], dataset)
		if type == "galactic":
			dialog.add_expression("pm_alpha", "pm_ra", default_columns[2], dataset)
			dialog.add_expression("pm_delta", "pm_dec", default_columns[3], dataset)
			dialog.add_text("pm_alpha_out", "pm_long name", "pm_l")
			dialog.add_text("pm_delta_out", "pm_lat name", "pm_b")
		else:
			#dialog.add_text("l", "Ecliptic ra", "ra_lambda")
			#dialog.add_text("b", "Ecliptic dec", "dec_beta")
			pass
		values = dialog.get()
	if values:
		dataset.ucds[values["pm_alpha_out"]] = "pos.pm;pos.galactic.lon"# % type
		dataset.ucds[values["pm_delta_out"]] = "pos.pm;pos.galactic.lat"# % type
		dataset.units[values["pm_alpha_out"]] = dataset.unit(values["pm_alpha"])
		dataset.units[values["pm_delta_out"]] = dataset.unit(values["pm_delta"])
		if type == "galactic":
			dataset.add_virtual_columns_proper_motion_eq2gal(long_in=values["alpha"], lat_in=values["delta"],
												   pm_long=values["pm_alpha"], pm_lat=values["pm_delta"],
												   pm_long_out=values["pm_alpha_out"], pm_lat_out=values["pm_delta_out"],
												   radians=values["degrees"] == "radians")
		else:
			pass
			#dataset.add_virtual_columns_eq2ecl(long_in=values["ra"], lat_in=values["dec"],
			#									   long_out=values["l"], lat_out=values["b"],
			#									   radians=values["degrees"] == "radians")

def main(argv=sys.argv):
	dataset = vaex.open(argv[1])
	app = QtGui.QApplication(argv)
	table = ColumnsTable(None)
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