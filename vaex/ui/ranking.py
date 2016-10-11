import itertools

import numpy as np
#from pyjavaproperties import Properties
from vaex.ext import jprops
import collections

from vaex.ui.qt import *
import vaex.dataset
import vaex.ui.plot_windows
import logging as logging
import vaex.ui.qt as dialogs
import vaex.execution
import vaex.kld
logger = logging.getLogger("vaex.ranking")


# since we do many columns at once, a smallar buffer will lead to more resposiveness in the gui
buffer_size = 1e6

def unique_column_names(dataset):
	#return list(set(dataset.column_names) | set(dataset.virtual_columns.keys()))
	return dataset.get_column_names(virtual=True)

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import scipy.stats

testing = False

class PlotDialog(QtGui.QDialog):
	def __init__(self, parent, width=5, height=4, dpi=100, **options):
		super(PlotDialog, self).__init__()
		self.parent_widget = parent
		self.figure = plt.figure()
		self.canvas = FigureCanvas(self.figure)
		self.toolbar = NavigationToolbar(self.canvas, self)

		self.layout = QtGui.QVBoxLayout()
		self.layout.addWidget(self.toolbar)
		self.layout.addWidget(self.canvas)
		self.setLayout(self.layout)
		self.figure.canvas.mpl_connect('motion_notify_event', self._on_mouse_motion)
		self.figure.canvas.mpl_connect('button_press_event', self._on_mouse_button)

		#self.plot()

	def _on_mouse_motion(self, event):
		if event.xdata != None and event.ydata != None:
			self.on_mouse_motion(event)

	def _on_mouse_button(self, event):
		if event.xdata != None and event.ydata != None:
			self.on_mouse_button(event)

	def on_mouse_button(self, event):
		pass
	def on_mouse_motion(self, event):
		label, x, y = self.get_tooltip(event)
		self.set_tooltip(x, y, label)

	def set_tooltip(self, x, y, label):
		y = self.canvas.height() - 1 - y
		print(("motion", label, x, y))
		if label:
			print(("self.canvas.x/y()", self.canvas.geometry().x(), self.canvas.geometry().y()))
			print(("self.pos.x/y()", self.canvas.pos().x() + self.pos().x(), self.canvas.pos().y() + self.pos().y()))
			point = QtCore.QPoint(x + self.canvas.x() + self.pos().x(), y + self.canvas.y() + self.pos().y())
			QtGui.QToolTip.showText(point, label)

	def plot(self):
		pass

class RankPlot(PlotDialog):
	def __init__(self, parent, table):
		super(RankPlot, self).__init__(parent)
		self.table = table

		self.button_update_data = QtGui.QPushButton('Update data')
		self.button_update_data.clicked.connect(self.update_data)
		self.layout.addWidget(self.button_update_data)

		grid_layout = QtGui.QGridLayout()
		grid_layout.setColumnStretch(2, 1)
		grid_layout.setAlignment(QtCore.Qt.AlignTop)
		grid_layout.setSpacing(0)
		grid_layout.setContentsMargins(0,0,0,0)
		self.layout.addLayout(grid_layout)

		row = 0
		self.options = ["mutual_information", "rank(mutual_information)", "correlation_coefficient", "rank(correlation_coefficient)", "abs(correlation_coefficient)", "rank(abs(correlation_coefficient))"]
		self.expression_x = "mutual_information"
		self.expression_y = "correlation_coefficient"
		self.codeline_x = Codeline(self, "x", self.options,
								   getter=attrgetter(self, "expression_x"),
								   setter=attrsetter(self, "expression_x"),
								   update=self.plot)
		self.codeline_y = Codeline(self, "y", self.options,
								   getter=attrgetter(self, "expression_y"),
								   setter=attrsetter(self, "expression_y"),
								   update=self.plot)
		row = self.codeline_x.add_to_grid_layout(row, grid_layout)
		row = self.codeline_y.add_to_grid_layout(row, grid_layout)


		self.axes = self.figure.add_subplot(111)
		if 0:
			def onpick(event):
				thisline = event.artist
				xdata = thisline.get_xdata()
				ydata = thisline.get_ydata()
				index = event.ind
				print((index, xdata[index], ydata[index], self.pairs[index]))
				self.table.select_pair(self.pairs[index])
			self.figure.canvas.mpl_connect('pick_event', onpick)

		self.update_data()
		self.plot()

	def on_mouse_button(self, event):
		min_distance_index, x, y = self.find_nearest(event)
		print(("click", x, y))
		self.table.select_pair(self.pairs[min_distance_index])
		def update_tooltip():
			label, x, y = self.get_tooltip(event)
			self.set_tooltip(x, y, label)
		QtCore.QTimer.singleShot(100, update_tooltip)

	def find_nearest(self, event):
		transform = event.inaxes.transData.transform
		xy = transform(list(zip(self.x, self.y)))
		#print xy
		x, y = xy.T
		print(("event.x/y", event.x, event.y))
		distances = np.sqrt((x-event.x)**2 + (y-event.y)**2)
		min_distance_index = np.argmin(distances)
		return min_distance_index, x[min_distance_index], y[min_distance_index]

	def get_tooltip(self, event):
		min_distance_index, x, y = self.find_nearest(event)
		label = "-".join(self.pairs[min_distance_index])
		return label, x, y

	def update_data(self):
		self.variables = {}
		self.pairs = pairs = self.table.getSelected()
		mi = [self.table.qualities[pair] for pair in pairs]
		corr = [self.table.correlation_map[pair] for pair in pairs]
		self.variables["mutual_information"] = np.array(mi)
		self.variables["correlation_coefficient"] = np.array(corr)


	def plot(self):
		scope = {}
		scope.update(np.__dict__)
		scope.update(self.variables)
		scope["rank"] = scipy.stats.rankdata
		self.x = x = eval(self.expression_x, scope)
		self.y = y = eval(self.expression_y, scope)
		self.axes.cla()
		self.axes.plot(x, y, '.', picker=5)
		self.canvas.draw()



class ____RankingTableModel(QtCore.QAbstractTableModel):
	def __init__(self, dataset, dim=1, parent=None, *args): 
		QtCore.QAbstractTableModel.__init__(self, parent, *args) 
		self.dataset = dataset


		self.pairs = list(itertools.combinations(unique_column_names(self.dataset), dim))
		self.ranking = [None for pair in self.pairs]
		self.headers = ["subspace", "Mutual information", "MI ranking", "correlation", "MI ranking", 'selected']
		self.indices = list(range(len(self.pairs)))
	
	def rowCount(self, parent): 
		return len(self.pairs)

	def columnCount(self, parent): 
		return len(self.headers)

	def data(self, index, role): 
		if not index.isValid(): 
			return None
		elif role != QtCore.Qt.DisplayRole: 
			return None
		column = index.column()
		index = self.indices[index.row()] # use sorted index
		if column == 0:
			return "-vs".join(self.pairs[index])
		if column == 1:
			rank = self.ranking[index]
			return "" if rank is None else str(rank)
		if column == 2:
			rank = self.ranking[index]
			return False if random.random() < 0.5 else True

	def headerData(self, index, orientation, role):
		if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
			return self.headers[index]
		if orientation == QtCore.Qt.Vertical and role == QtCore.Qt.DisplayRole:
			return str(index+1)
		return None

	def sort(self, Ncol, order):
		"""Sort table by given column number.
		"""
		self.emit(QtCore.SIGNAL("layoutAboutToBeChanged()"))
		if Ncol == 0:
			print("by name")
			# get indices, sorted by pair name
			sortlist = list(zip(self.pairs, list(range(len(self.pairs)))))
			print(sortlist)
			sortlist.sort(key=operator.itemgetter(0))
			print(sortlist)
			self.indices = list(map(operator.itemgetter(1), sortlist))
			print((self.indices))
		if Ncol == 1:
			# get indices, sorted by ranking, or no sorting
			if None not in self.ranking:
				sortlist = list(zip(self.ranking, list(range(len(self.pairs)))))
				sortlist.sort(key=operator.itemgetter(0))
				self.indices = list(map(operator.itemgetter(1), sortlist))
			else:
				self.indices = list(range(len(self.pairs)))
			print((self.indices))
		if order == QtCore.Qt.DescendingOrder:
			self.indices.reverse()
		print((self.indices))
		self.emit(QtCore.SIGNAL("layoutChanged()"))

class SubspaceTable(QtGui.QTableWidget):
	def __init__(self, dialog, parent, mainPanel, dataset, pairs, dim, properties):
		self.dialog = dialog
		self.dim = dim
		if dim == 1:
			self.headers = ['', 'space', "min", "max", 'plot']
		else:
			self.headers = ['', 'space', "Mutual information", "Correlation", 'plot']
		self.properties = properties
		self.qualities = {}
		self.correlation_map = {}
		if testing:
			self.qualities = {key: np.random.random() for key in pairs}
			self.correlation_map = {key: np.random.normal() for key in pairs}

		#print ", ".join([""+("-".join(pair))+"" for pair in pairs])
		self.dataset = dataset
		self.filter_terms = []
		self.mainPanel = mainPanel
		self.pairs = list(pairs) #list(itertools.combinations(self.dataset.column_names, dim))
		QtGui.QTableWidget.__init__(self, len(self.pairs), len(self.headers), parent)
		self.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows);
		self.filter_mask = np.array([True for pair in pairs])
		self.selected_dict = {pair:self.properties.get(".".join(pair) + ".use", "True") == "True" for pair in pairs}
		#self.tableModel = RankingTableModel(self.dataset, dim, parent)
		#self.setModel(self.tableModel)
		#self.sortByColumn(0, QtCore.Qt.AscendingOrder)
		#self.setSortingEnabled(True)
		#self.pair_to_item = {}
		self.defaultFlags = QtCore.Qt.ItemIsSelectable|QtCore.Qt.ItemIsEnabled|QtCore.Qt.ItemIsEditable
		#print self.properties._props
		if 1:
			#self.ranking = [None for pair in self.pairs]
			self.dim = dim
			self.setHorizontalHeaderLabels(self.headers)
			#self.setVerticalHeaderLabels(map(str, range(len(self.pairs))))
			self.fill_table()
			self.setSortingEnabled(True)
		self.queue_fill_table = vaex.ui.plot_windows.Queue("fill table", 200, self.fill_table)

	def pair_to_text(self, pair):
		return " ".join(map(str, pair))

	def select_pair(self, pair):
		#self.setSortingEnabled(False)
		index = self.pairs.index(pair)
		for i in range(self.rowCount()):
			item = self.item(i, 1)
			print((item.text(), self.pair_to_text(pair)))
			if item.text() == self.pair_to_text(pair):
				self.selectRow(i)
		#print index, self.visualRow(index)
		#self.selectRow(self.visualRow(index))
		#self.setSortingEnabled(True)


	def fill_table(self):
		# bug in qt? http://stackoverflow.com/questions/7960505/strange-qtablewidget-behavior-not-all-cells-populated-after-sorting-followed-b
		# fix: disable sorting, then enable again
		self.setSortingEnabled(False)
		self.checkboxes = []
		self.buttons = []
		pairs = [pair for pair, display in zip(self.pairs, self.filter_mask) if display]
		self.setRowCount(len(pairs))
		self.setVerticalHeaderLabels(list(map(str, list(range(len(pairs))))))
		for i in range(len(pairs)):
			pair = pairs[i]
			text = self.pair_to_text(pair)
			item = QtGui.QTableWidgetItem(text)
			self.setItem(i, 1, item)
			item.setFlags(self.defaultFlags)
			#item = QtGui.QTableWidgetItem()
			#item.setData(QtCore.Qt.DisplayRole, QtCore.QVariant(True))
			#item.setFlags(QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsUserCheckable)
			checkbox = QtGui.QCheckBox(self)
			use_key = ".".join(map(str, pair)) + ".use"
			#if self.dim == 1 and use_key in self.properties._props:
			#	#print use_key, eval(self.properties[use_key])
			#	checkbox.setCheckState(QtCore.Qt.Checked if eval(self.properties[use_key]) else QtCore.Qt.Unchecked)
			#else:
			#	checkbox.setCheckState(QtCore.Qt.Checked)
			print(("fill", pair, self.selected_dict[pair]))
			checkbox.setCheckState(QtCore.Qt.Checked if self.selected_dict[pair] else QtCore.Qt.Unchecked)
			self.checkboxes.append(checkbox)
			self.setCellWidget(i, 0, checkbox)
			def stateChanged(state, pair=pair):
				self.selected_dict[pair] = state == QtCore.Qt.Checked
				print(("set", pair, "to", self.selected_dict[pair]))
			checkbox.stateChanged.connect(stateChanged)

			if self.dim == 1:
				button = QtGui.QPushButton("plot: " + text, self)
				def plot(_ignore=None, pair=pair):
					ranges = [self.dialog.range_map[k] for k in pair]
					self.mainPanel.histogram(*pair, ranges=ranges)
				button.clicked.connect(plot)
				self.setCellWidget(i, 4, button)

				min_key = pair[0]+".min"
				max_key = pair[0]+".max"
				if 1:
					#print "test", min_key
					if min_key in self.properties:
						item = QtGui.QTableWidgetItem()#"%s"  % quality)
						value = self.properties[min_key]
						#print "it is in... and value =", value
						item.setText("%s"  % value)
						item.setData(QtCore.Qt.DisplayRole, float(value))
						item.setFlags(self.defaultFlags)
						self.setItem(i, 2, item)

					if max_key in self.properties:
						value = self.properties[max_key]
						item = QtGui.QTableWidgetItem()#"%s"  % quality)
						item.setText("%s"  % value)
						item.setData(QtCore.Qt.DisplayRole, float(value))
						item.setFlags(self.defaultFlags)
						self.setItem(i, 3, item)
			else:
				#print "quality", quality, qualities
				#row = self.pairs.index(pair)
				quality = self.qualities.get(pair)
				if quality is not None:
					item = QtGui.QTableWidgetItem()#"%s"  % quality)
					item.setText("%s"  % quality)
					item.setData(QtCore.Qt.DisplayRole, float(quality))
					item.setFlags(self.defaultFlags)
					self.setItem(i, 2, item)
				correlation = self.correlation_map.get(pair)
				if correlation is not None:
					item = QtGui.QTableWidgetItem()#"%s"  % quality)
					item.setText("%s"  % correlation)
					item.setData(QtCore.Qt.DisplayRole, float(correlation))
					item.setFlags(self.defaultFlags)
					self.setItem(i, 3, item)


			if self.dim == 2:
				button = QtGui.QPushButton("plot: " + text, self)
				def plot(_ignore=None, pair=pair):
					ranges = [self.dialog.range_map[k] for k in pair]
					self.mainPanel.plotxy(*pair, ranges=ranges)
				button.clicked.connect(
					plot)
				self.setCellWidget(i, 4, button)
				self.buttons.append(button) # keep ref count
			if self.dim == 3:
				button = QtGui.QPushButton("plot: " + text, self)
				def plot(_ignore=None, pair=pair):
					ranges = [self.dialog.range_map[k] for k in pair]
					self.mainPanel.plotxyz(*pair, ranges=ranges)
				button.clicked.connect(plot)
				self.setCellWidget(i, 4, button)
				self.buttons.append(button) # keep ref count
			#self.setItem(i, 1, item)
		self.setSortingEnabled(True)

	def getSelected(self):
		selection = [checkbox.checkState() == QtCore.Qt.Checked for checkbox in self.checkboxes]
		selected_pairs = [pair for pair, selected in zip(self.pairs, selection) if selected]
		return selected_pairs
		
	def setQualities(self, pairs, qualities):
		self.qualities = {}
		for quality, pair in zip(qualities, pairs):
			self.qualities[pair] = quality
			#item = self.pair_to_item[pair]
			#print "quality", quality, qualities
			#row = self.pairs.index(pair)
			#item = QtGui.QTableWidgetItem()#"%s"  % quality)
			#item.setText("%s"  % quality)
			#item.setData(QtCore.Qt.DisplayRole, float(quality))
			#item.setFlags(self.defaultFlags)
			#self.setItem(row, 2, item)
		self.fill_table()

	def set_correlations(self, correlation_map):
		self.correlation_map = dict(correlation_map)
		self.fill_table()

	def get_range(self, pair):
		index = self.pairs.index(pair)
		mi = self.item(index, 2)
		ma = self.item(index, 3)
		if mi is None or ma is None:
			return None, None
		#print pair, mi, ma
		#print mi.data(QtCore.Qt.DisplayRole)
		mi = None if mi is None else float(mi.data(QtCore.Qt.DisplayRole))
		ma = None if ma is None else float(ma.data(QtCore.Qt.DisplayRole))
		#print "->", pair, mi, ma
		return mi, ma

	def setRanges(self, pairs, ranges):
		for (mi, ma), pair in zip(ranges, pairs):
			#item = self.pair_to_item[pair]
			row = self.pairs.index(pair)
			item = QtGui.QTableWidgetItem()#"%s"  % quality)
			item.setText("%s"  % mi)
			item.setData(QtCore.Qt.DisplayRole, float(mi))
			item.setFlags(self.defaultFlags)
			self.setItem(row, 2, item)
			item = QtGui.QTableWidgetItem()#"%s"  % quality)
			item.setText("%s"  % ma)
			item.setData(QtCore.Qt.DisplayRole, float(ma))
			item.setFlags(self.defaultFlags)
			self.setItem(row, 3, item)
			

	def deselect(self, pair):
		index = self.pairs.index(pair)
		print(("deselect", pair, index))
		checkbox = self.checkboxes[index]
		checkbox.setCheckState(QtCore.Qt.Unchecked)
		
	def select(self, pair):
		index = self.pairs.index(pair)
		print(("deselect", pair, index))
		checkbox = self.checkboxes[index]
		checkbox.setCheckState(QtCore.Qt.Checked)

	def setPairs(self, pairs):
		#selection = [checkbox.checkState() == QtCore.Qt.Checked for checkbox in self.checkboxes]
		#non_selected_pairs = [pair for pair, selected in zip(self.pairs, selection) if not selected]

		self.pairs = list(pairs)
		for pair in self.pairs:
			if pair not in self.selected_dict:
				self.selected_dict[pair] = self.properties.get(".".join(pair) + ".use", "True")
		self.filter_mask = np.array([True for pair in pairs])
		self.fill_table()
		#self.checkboxes = []
		#self.setRowCount(len(self.pairs))
		#self.setVerticalHeaderLabels(map(str, range(len(self.pairs))))
		#for i in range(len(self.pairs)):
			#text = " ".join(map(str, self.pairs[i]))
			#print text
			#item = QtGui.QTableWidgetItem(text)
			#item.setFlags(self.defaultFlags)
			#self.setItem(i, 1, item)
			#checkbox = self.checkboxes[i] #QtGui.QCheckBox(self)
			#if not (self.pairs[i] in non_selected_pairs):
			#	checkbox.setCheckState(QtCore.Qt.Checked)
			#self.checkboxes.append(checkbox)
			#self.setCellWidget(i, 0, checkbox)
		#print self.checkboxes

	def set_filter_terms(self, filter_terms):
		def filter(pair):
			found = True
			for filter_term in filter_terms:
				found_term = False
				for expression in pair:
					found_term = found_term or filter_term.lower() in expression.lower()
				found = found and found_term
			return found

		self.filter_terms = filter_terms
		#print list, filter, self.pairs
		self.filter_mask = np.array([filter(pair) for pair in self.pairs])
		self.queue_fill_table()
		#self.fill_table()

import functools

def joinpairs(pairs1d, pairsNd):
	previous = []
	for pair1d in pairs1d:
		subspacename = pair1d[0] # tuple only has one element
		for pairNd in pairsNd:
			if subspacename not in pairNd:
				pair = pair1d + pairNd
				if sorted(pair) not in previous:
					previous.append(sorted(pair))
					#print previous
					yield pair

class RankDialog(QtGui.QDialog):
	def __init__(self, dataset, parent, mainPanel, **options):
		super(RankDialog, self).__init__(parent)
		self.dataset = dataset
		self.mainPanel = mainPanel
		self.range_map = {}
		self.grid_size = int(options.get("grid_size", "32"))


		#print "options", options
		self.properties = collections.OrderedDict()# Properties()
		if self.dataset.is_local():
			self.properties_path = os.path.splitext(self.dataset.path)[0] + ".properties"
		else:
			dir = os.path.join(vaex.utils.get_private_dir(), "ranking")
			if not os.path.exists(dir):
				os.makedirs(dir)
			server = self.dataset.server
			name = "%s_%s_%s_%s" % (server.hostname, server.port, server.base_path.replace("/", "_"), self.dataset.name)
			self.properties_path = os.path.join(dir, name+".properties")

		self.properties_path = options.get("file", self.properties_path)
		if os.path.exists(self.properties_path):
			self.load_properties()
		else:
			pass
			#if not os.access(properties_path, os.W_OK):
			#	dialog_error(self, "File access", "Cannot write to %r, so cannot save options" % properties_path)



		
		self.tabs = QtGui.QTabWidget(self)
		
		self.tab1d = QtGui.QWidget(self.tabs)
		self.table1d = SubspaceTable(self, self.tab1d, mainPanel, self.dataset,  list(itertools.combinations(unique_column_names(self.dataset), 1)),  1, self.properties)
		
		self.subspaceTables = {}
		self.subspaceTabs = {}
		self.subspaceTables[1] = self.table1d
		self.subspaceTabs[1] = self.tab1d
		
		def onclick(dim=2):
			self.open(dim=dim)
		self.subspace2d = QtGui.QPushButton("create 2d subspaces", self.tab1d)
		self.subspace2d.clicked.connect(functools.partial(onclick, dim=2))

		self.get_ranges_menu = QtGui.QMenu()
		self.button_get_ranges = QtGui.QToolButton()
		self.button_get_ranges.setText("calculate min/max")
		self.button_get_ranges.setPopupMode(QtGui.QToolButton.InstantPopup)
		#self.button_get_ranges.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
		self.get_ranges_menu = QtGui.QMenu()
		self.button_get_ranges.setMenu(self.get_ranges_menu)

		self.action_ranges_minmax = QtGui.QAction("absolute min/max", self)
		self.action_ranges_minmax_3sigma = QtGui.QAction("3 sigma clipping", self)
		self.get_ranges_menu.addAction(self.action_ranges_minmax)
		self.get_ranges_menu.addAction(self.action_ranges_minmax_3sigma)

		#self.button_get_ranges = QtGui.QToolButton(self.tab1d)
		#self.button_get_ranges.setText("calculate min/max")
		#self.button_get_ranges.setM
		self.action_ranges_minmax.triggered.connect(self.onCalculateMinMax)
		self.action_ranges_minmax_3sigma.triggered.connect(self.onCalculateMinMax3Sigma)
		
		self.button_store = QtGui.QToolButton(self.tab1d)
		self.button_store.setText("store")
		self.button_store.clicked.connect(self.onStore)


		self.actions_menu = QtGui.QMenu()
		self.button_actions = QtGui.QToolButton()
		self.button_actions.setText("Extra")
		self.button_actions.setPopupMode(QtGui.QToolButton.InstantPopup)
		self.button_actions.setMenu(self.actions_menu)

		self.action_select_all = QtGui.QAction("Select all", self)
		self.action_select_none = QtGui.QAction("Select none", self)
		self.action_remove_empty = QtGui.QAction("Remove empty columns", self)
		self.action_pca = QtGui.QAction("PCA transformation", self)
		self.action_select_all.triggered.connect(self.onSelectAll)
		self.action_select_none.triggered.connect(self.onSelectNone)
		self.action_remove_empty.triggered.connect(self.onRemoveEmpty)
		self.action_pca.triggered.connect(self.onPca)
		self.actions_menu.addAction(self.action_select_all)
		self.actions_menu.addAction(self.action_select_none)
		self.actions_menu.addAction(self.action_remove_empty)
		self.actions_menu.addSeparator()
		self.actions_menu.addAction(self.action_pca)

		self.tab1dlayout = QtGui.QVBoxLayout(self)
		self.tab1d_button_layout = QtGui.QHBoxLayout(self)
		self.tab1dlayout.addLayout(self.tab1d_button_layout)
		self.tab1d_button_layout.addWidget(self.subspace2d)
		self.tab1d_button_layout.addWidget(self.button_get_ranges)
		self.tab1d_button_layout.addWidget(self.button_store)
		self.tab1d_button_layout.addWidget(self.button_actions)


		self.filter_line_edit = QtGui.QLineEdit(self)
		self.filter_line_edit.setPlaceholderText("Enter space seperated search terms")
		self.filter_line_edit.textEdited.connect(functools.partial(self.onFilter, table=self.table1d))

		self.tab1dlayout.addWidget(self.filter_line_edit)
		self.tab1dlayout.addWidget(self.table1d)
		#self.tab1dlayout.addWidget(self.rankButton)
		#self.setCentralWidget(self.splitter)
		self.tab1d.setLayout(self.tab1dlayout)
		
		self.tabs.addTab(self.tab1d, "1d")
		
		self.resize(700,500)
		
		if 0:
			for name in self.dataset.column_names:
				item = QtGui.QListWidgetItem(self.list1d)
				item.setText(name)
				item.setCheckState(False)
				#self.list1d.


		self.boxlayout = QtGui.QVBoxLayout(self)

		self.gridlayout = QtGui.QGridLayout()
		self.gridlayout.setColumnStretch(1, 1)
		self.gridlayout.setSpacing(0)
		self.gridlayout.setContentsMargins(2,1,2,1)
		self.gridlayout.setAlignment(QtCore.Qt.AlignTop)
		row = 1

		self.selection_label = QtGui.QLabel("Use for computations:", self)
		self.gridlayout.addWidget(self.selection_label, row, 0);
		self.radio_button_all = QtGui.QRadioButton("Complete dataset", self)
		self.radio_button_selection = QtGui.QRadioButton("Selection", self)
		self.radio_button_all.setChecked(True)
		self.gridlayout.addWidget(self.radio_button_all, row, 1); row += 1
		self.gridlayout.addWidget(self.radio_button_selection, row, 1); row += 1

		def get():
			return str(self.grid_size)
		def set(value):
			self.grid_size = int(value)
		self.option_grid_size = Option(self, "grid size (for mutual info)", "32 64 128 256".split(), get, set)
		row = self.option_grid_size.add_to_grid_layout(row, self.gridlayout)
		#self.gridlayout.addWidget(self.option_grid_size.combobox); row += 1
		#self.gridlayout.addWidget(self.option_grid_size.combobox); row += 1

		self.boxlayout.addLayout(self.gridlayout)
		self.boxlayout.addWidget(self.tabs)
		#self.boxlayout.addWidget(self.rankButton)
		#self.setCentralWidget(self.splitter)
		self.setLayout(self.boxlayout)


		if "2" in options.get("open", ""):
			self.open(dim=2)
		if "3" in options.get("open", ""):
			self.open(dim=3)
		if "4" in options.get("open", ""):
			self.open(dim=4)

		self.fill_range_map()

	def open(self, dim=2):
		pairs1d = self.subspaceTables[1].getSelected()
		pairsprevd = self.subspaceTables[dim-1].getSelected()
		#print pairs1d
		#print pairsprevd
		newpairs = list(joinpairs(pairs1d, pairsprevd))
		print(("newpairs", newpairs))
		if dim not in self.subspaceTables:
			self.tabNd = QtGui.QWidget(self.tabs)
			self.tableNd = SubspaceTable(self, self.tabNd, self.mainPanel, self.dataset, newpairs, dim, self.properties)
			self.tabNdlayout = QtGui.QVBoxLayout(self)
			self.tabNdButtonLayout = QtGui.QHBoxLayout(self)
			self.subspaceNd = QtGui.QPushButton("Create %dd subspaces" % (dim+1), self.tab1d)
			self.plotNd = QtGui.QPushButton("Rank plot")
			self.exportNd = QtGui.QPushButton("Export ranking")
			if dim == len(self.dataset.column_names):
				self.subspaceNd.setDisabled(True)

			self.menu_calculate = QtGui.QMenu()
			self.button_calculate = QtGui.QToolButton()
			self.button_calculate.setText("Calculate")
			self.button_calculate.setPopupMode(QtGui.QToolButton.InstantPopup)
			#self.button_get_ranges.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
			self.button_calculate.setMenu(self.menu_calculate)

			self.tabNdButtonLayout.addWidget(self.subspaceNd)
			self.tabNdButtonLayout.addWidget(self.button_calculate)
			#self.tabNdButtonLayout.addWidget(self.miNd)
			#self.tabNdButtonLayout.addWidget(self.correlationNd)
			self.tabNdButtonLayout.addWidget(self.exportNd)
			self.tabNdButtonLayout.addWidget(self.plotNd)
			self.tabNdlayout.addLayout(self.tabNdButtonLayout)
			self.subspaceNd.clicked.connect(functools.partial(self.open, dim=dim+1))
			self.exportNd.clicked.connect(functools.partial(self.export, table=self.tableNd))
			self.plotNd.clicked.connect(functools.partial(self.rank_plot, table=self.tableNd))


			self.action_calculate_mi = QtGui.QAction("Calculate mutual information", self)
			self.action_calculate_correlation = QtGui.QAction("Calculate correlation", self)
			self.action_calculate_mi.triggered.connect(functools.partial(self.rankSubspaces, table=self.tableNd))
			self.action_calculate_correlation.triggered.connect(functools.partial(self.calculate_correlation, table=self.tableNd))
			self.menu_calculate.addAction(self.action_calculate_mi)
			self.menu_calculate.addAction(self.action_calculate_correlation)


			self.action_calculate_rank_correlation_MI_corr_kendall = QtGui.QAction("MI - correlation", self)
			self.action_calculate_rank_correlation_MI_abs_corr_kendall = QtGui.QAction("MI - abs(correlation)", self)

			self.action_calculate_rank_correlation_MI_corr_spearman = QtGui.QAction("MI - correlation", self)
			self.action_calculate_rank_correlation_MI_abs_corr_spearman = QtGui.QAction("MI - abs(correlation)", self)

			self.menu_correlation_kendall = self.menu_calculate.addMenu("Kendall's rank correlation")
			self.menu_correlation_spearman = self.menu_calculate.addMenu("Spearman's rank correlation")
			self.menu_correlation_kendall.setEnabled(False)
			self.menu_correlation_spearman.setEnabled(False)

			self.menu_correlation_kendall.addAction(self.action_calculate_rank_correlation_MI_corr_kendall)
			self.menu_correlation_kendall.addAction(self.action_calculate_rank_correlation_MI_abs_corr_kendall)

			self.menu_correlation_spearman.addAction(self.action_calculate_rank_correlation_MI_corr_spearman)
			self.menu_correlation_spearman.addAction(self.action_calculate_rank_correlation_MI_abs_corr_spearman)

			self.action_calculate_rank_correlation_MI_corr_kendall.triggered.connect(functools.partial(self.calculate_rank_correlation_kendall, table=self.tableNd))
			self.action_calculate_rank_correlation_MI_abs_corr_kendall.triggered.connect(functools.partial(self.calculate_rank_correlation_kendall, table=self.tableNd, absolute=True))

			self.action_calculate_rank_correlation_MI_corr_spearman.triggered.connect(functools.partial(self.calculate_rank_correlation_spearman, table=self.tableNd))
			self.action_calculate_rank_correlation_MI_abs_corr_spearman.triggered.connect(functools.partial(self.calculate_rank_correlation_spearman, table=self.tableNd, absolute=True))



			def func(index, name=""):
				print((name, index.row(), index.column()))
			self.tableNd.pressed.connect(functools.partial(func, name="pressed"))
			self.tableNd.entered.connect(functools.partial(func, name="entered"))
			self.tableNd.clicked.connect(functools.partial(func, name="clicked"))
			self.tableNd.activated.connect(functools.partial(func, name="activated"))
			def func(index, previous, name=""):
				print((name, index.row(), index.column(), previous.row(), previous.column()))
			self.selectionModel = self.tableNd.selectionModel()
			self.selectionModel.currentChanged.connect(functools.partial(func, name="currentChanged"))

			self.filter_Nd_line_edit = QtGui.QLineEdit(self)
			self.filter_Nd_line_edit.setPlaceholderText("Enter space seperated search terms")
			self.filter_Nd_line_edit.textEdited.connect(functools.partial(self.onFilter, table=self.tableNd))
			self.tabNdlayout.addWidget(self.filter_Nd_line_edit)

			self.tabNdlayout.addWidget(self.tableNd)
			#self.tab1dlayout.addWidget(self.rankButton)
			#self.setCentralWidget(self.splitter)
			self.tabNd.setLayout(self.tabNdlayout)
			self.subspaceTables[dim] = self.tableNd
			self.subspaceTabs[dim] = self.tabNd

			self.tabs.addTab(self.tabNd, "%dd" % dim)
			self.tabs.setCurrentWidget(self.tabNd)
		else:
			self.subspaceTables[dim].setPairs(newpairs)
			self.tabs.setCurrentWidget(self.subspaceTabs[dim])


	def onPca(self):
		#vaex.pca.
		pass

	def onFilter(self, text, table):
		table.set_filter_terms(text.split())

	def onStore(self):
		selected_pairs = self.table1d.getSelected()
		#error = False
		for pair in self.table1d.pairs:
			key = str(pair[0])
			#print repr(key+".use"), repr(pair in selected_pairs)
			self.properties[key+".use"] = repr(pair in selected_pairs)
			if key in self.range_map:
				mi, ma = self.range_map[key]
				self.properties[key+".min"] = repr(mi)
				self.properties[key+".max"] = repr(ma)
			else:
				print(("min/max not present", key))
		print(("save to", self.properties_path))
		self.store_properties()
		dialog_info(self, "Stored", "Stored configuration to: %r" % self.properties_path)

	def load_properties(self):
		with open(self.properties_path, "rb") as f:
			self.properties = jprops.load_properties(f, collections.OrderedDict)

	def store_properties(self):
		with open(self.properties_path, "wb") as f:
			jprops.store_properties(f, self.properties)

	def fill_range_map(self):
		pairs = self.table1d.getSelected()
		for pair in pairs:
			mi, ma = self.table1d.get_range(pair)
			if mi is not None and ma is not None:
				self.range_map[pair[0]] = mi, ma

	def onSelectAll(self):
		pairs = self.table1d.pairs
		for pair in pairs:
			self.table1d.select(pair)

	def onSelectNone(self):
		pairs = self.table1d.pairs
		for pair in pairs:
			self.table1d.deselect(pair)

	def onRemoveEmpty(self):
		self.fill_range_map()
		pairs = self.table1d.getSelected()
		error = False
		for pair in pairs:
			print(pair)
			if pair[0] in self.range_map:
				min, max = self.range_map[pair[0]]
				if min == max:
					print((pair, "empty", min, max))
					self.table1d.deselect(pair)
			else:
				if not error: # only give a warning once
					dialog_error(self, "Min/max missing", "Min max missing for %s" % pair)
					error = True


	def onCalculateMinMax(self):
		pairs = self.table1d.getSelected()
		logger.debug("estimate min/max for %r" % pairs)
		if self.dataset.is_local():
			executor = vaex.execution.Executor(buffer_size=buffer_size)
		else:
			executor = vaex.remote.ServerExecutor()

		expressions = [pair[0] for pair in pairs]
		assert len(pairs[0]) == 1
		self.range_map = {}
		try:
			with dialogs.ProgressExecution(self, "Calculating min/max", executor=executor) as progress:
				subspace = self.dataset.subspace(*expressions, executor=executor, async=True)
				minmax = subspace.minmax()
				progress.add_task(minmax).end()
				progress.execute()
			ranges = minmax.get()
			self.table1d.setRanges(pairs, ranges)
			self.fill_range_map()
		except:
			logger.exception("Error in min/max or cancelled")
		#dialog.hide()

	def onCalculateMinMax3Sigma(self):
		pairs = self.table1d.getSelected()

		expressions = [pair[0] for pair in pairs]
		if self.dataset.is_local():
			executor = vaex.execution.Executor(buffer_size=buffer_size)
		else:
			executor = vaex.remote.ServerExecutor()

		if self.dataset.is_local():
			executor = vaex.execution.Executor()
		else:
			executor = vaex.remote.ServerExecutor()
		subspace = self.dataset.subspace(*expressions, executor=executor, async=True)
		means = subspace.mean()
		with dialogs.ProgressExecution(self, "Calculating mean", executor=executor) as progress:
			progress.add_task(means).end()
			progress.execute()
		logger.debug("get means")
		means = means.get()
		logger.debug("got means")

		vars = subspace.var(means=means)
		with dialogs.ProgressExecution(self, "Calculating variance", executor=executor) as progress:
			progress.add_task(vars).end()
			progress.execute()
		#limits  = limits.get()
		vars = vars.get()
		stds = vars**0.5
		sigmas = 3
		ranges = list(zip(means-sigmas*stds, means+sigmas*stds))
		self.table1d.setRanges(pairs, ranges)
		self.fill_range_map()


	def calculate_rank_correlation_kendall(self, table, absolute=False):
		print(("kendall", table, absolute))
		pairs = table.getSelected()
		mi_values = [table.qualities[pair] for pair in pairs]
		correlation_values = [table.correlation_map[pair] for pair in pairs]
		ranking_mi = np.argsort(mi_values)
		if absolute:
			ranking_correlation = np.argsort(np.abs(correlation_values))
		else:
			ranking_correlation = np.argsort(correlation_values)
		N = len(pairs)
		A = np.zeros((N, N))
		B = np.zeros((N, N))
		for i in range(N):
			for j in range(N):
				A[i,j] = np.sign(ranking_mi[i] - ranking_mi[j])
				B[i,j] = np.sign(ranking_correlation[i] - ranking_correlation[j])
		AB = 0
		AA = 0
		BB = 0
		for i in range(N):
			for j in range(N):
				AB += A[i,j] * B[i,j]
				AA += A[i,j]**2
				BB += B[i,j]**2


	def calculate_rank_correlation_spearman(self, table, absolute=False):
		print(("spearman", table, absolute))

	def calculate_correlation(self, table):
		print(("calculate correlation for ", table))
		pairs = table.getSelected()

		expressions = set()
		for pair in pairs:
			for expression in pair:
				expressions.add(expression)
		expressions = list(expressions)
		if self.dataset.is_local():
			executor = vaex.execution.Executor(buffer_size=buffer_size)
		else:
			executor = vaex.remote.ServerExecutor()

		def on_error(exc):
			raise exc
		if 1:
			#subspace = self.dataset(*expressions, executor=executor, async=True)
			subspaces = self.dataset.subspaces(pairs, executor=executor, async=True)
			means_promise = subspaces.mean()
			#print means_promise, type(means_promise), subs
			with dialogs.ProgressExecution(self, "Calculating means", executor=executor) as progress:
				progress.add_task(means_promise)
				progress.execute()
			means  = means_promise.get()

			variances_promise = subspaces.var(means=means)
			with dialogs.ProgressExecution(self, "Calculating variances", executor=executor) as progress:
				progress.add_task(variances_promise)
				progress.execute()
			vars = variances_promise.get()

			#means = subspaces._unpack(means_packed)
			#vars = subspaces._unpack(vars_packed)
			tasks = []
			with dialogs.ProgressExecution(self, "Calculating correlation", executor=executor) as progress:
				for subspace, mean, var in zip(subspaces.subspaces, means, vars):
					task = subspace.correlation(means=mean, vars=var)
					progress.add_task(task).end()
					tasks.append(task)
				progress.execute()
			correlations = [task.get() for task in tasks]

			correlation_map = dict(zip(pairs, correlations))
			table.set_correlations(correlation_map)
			return
			#mean_map = dict(zip(expressions, means))
			#var_map = dict(zip(expressions, variances))
		else:
			mean_map = {}
			def on_error(exc):
				raise exc
			for expression in expressions:
				subspace = self.dataset(expression, executor=executor, async=True)
				def assign(mean_list, expression=expression):
					logger.debug("assigning %r to %s", mean_list, expression)
					mean_map[expression] = mean_list
				subspace.mean().then(assign, on_error).end()
			with dialogs.ProgressExecution(self, "Calculating means", executor=executor):
				executor.execute()

			var_map = {}
			for expression in expressions:
				subspace = self.dataset(expression, executor=executor, async=True)
				def assign(mean_list, expression=expression):
					logger.debug("assigning %r to %s", mean_list, expression)
					var_map[expression] = mean_list[0].tolist()
				subspace.var(means=mean_map[expression]).then(assign, on_error).end()
			with dialogs.ProgressExecution(self, "Calculating variances", executor=executor):
				executor.execute()

			means = [mean_map[expressions[0]] for expressions in pairs]
			variances = [var_map[expressions[0]] for expressions in pairs]

		correlation_map = {}
		for pair in pairs:
			means = [mean_map[expression] for expression in pair]
			vars = [var_map[expression] for expression in pair]
			subspace = self.dataset(*pair, executor=executor, async=True)
			def assign(correlation, pair=pair):
				logger.debug("assigning %r to %s", correlation, pair)
				correlation_map[pair] = correlation
			subspace.correlation(means, vars).then(assign, on_error).end()

		with dialogs.ProgressExecution(self, "Calculating correlation", executor=executor):
			executor.execute()

		table.set_correlations(correlation_map)
		return

		jobsManager = vaex.dataset.JobsManager()
		expressions = set()
		for pair in pairs:
			for expression in pair:
				expressions.add(expression)
		expressions = list(expressions)
		print("means")
		with ProgressExecution(self, "Calculating means") as progress:
			means = jobsManager.calculate_mean(self.dataset, use_mask=self.radio_button_selection.isChecked(), expressions=expressions, feedback=progress.progress)
		mean_map = dict(list(zip(expressions, means)))
		centered_expressions_map = {expression: "(%s - %.20e)" % (expression, mean) for (expression, mean) in list(mean_map.items())}
		variances_expressions_map = {expression: "%s**2" % centered_expressions for expression, centered_expressions in list(centered_expressions_map.items())}
		with ProgressExecution(self, "Calculating variances") as progress:
			variances = jobsManager.calculate_mean(self.dataset, use_mask=self.radio_button_selection.isChecked(), expressions=list(variances_expressions_map.values()), feedback=progress.progress)
		variances_map = dict(list(zip(list(variances_expressions_map.keys()), variances)))

		covariances_expressions = []
		for pair in pairs:
			centered_expressions = [centered_expressions_map[expression] for expression in pair]
			covariance_expression = "*".join(centered_expressions)
			covariances_expressions.append(covariance_expression)

		print(covariances_expressions)
		with ProgressExecution(self, "Calculating covariances") as progress:
			#progress.progress(20)
			covariances = jobsManager.calculate_mean(self.dataset, use_mask=self.radio_button_selection.isChecked(), expressions=covariances_expressions, feedback=progress.progress)
			#progress.progress(20)
		print(variances)
		print(covariances)

		correlation_map = {}
		for pair, covariance in zip(pairs, covariances):
			normalization = 1
			for expression in pair:
				normalization *= np.sqrt(variances_map[expression])
			correlation_map[pair] = covariance / normalization
		table.set_correlations(correlation_map)

		return



	def export(self, table):
		print(("export", table))
		basename, ext = os.path.splitext(self.dataset.path)
		path = basename + ("-ranking-%dd" % table.dim) + ".properties"
		filename = get_path_save(self, path=path, title="Export ranking", file_mask="properties file *.properties")
		expressions = set()
		if filename:
			counts = 0
			with open(filename, "w") as file:
				for pair in table.pairs:
					if pair in table.qualities:
						file.write("%s.mutual_information=%f\n" % (".".join(pair),  table.qualities[pair]))
						counts += 1
					if pair in table.correlation_map:
						file.write("%s.correlation_coefficient=%f\n" % (".".join(pair),  table.correlation_map[pair]))
						counts += 1
					expressions.update(pair)
				for expression in expressions:
					mi, ma = self.range_map[expression]
					file.write("%s.min=%f\n" % (expression,  mi))
					file.write("%s.max=%f\n" % (expression,  ma))
				dialog_info(self, "Wrote ranking file", "wrote %d lines" % counts)




	def rankSubspaces(self, table):
		self.fill_range_map()

		pairs = table.getSelected()
		error = False
		ranges = []

		for pair in pairs:
			for expression in pair:
				if expression not in self.range_map:
					error = True
					print(("missing", expression))
		if error:
			dialog_error(self, "Missing min/max", "Please calculate the minimum and maximum for the dimensions")
			return


		#expressions = [pair[0] for pair in pairs]
		#executor = vaex.execution.Executor(buffer_size=buffer_size)
		if self.dataset.is_local():
			executor = vaex.execution.Executor(buffer_size=buffer_size)
		else:
			executor = vaex.remote.ServerExecutor()


		tasks = []
		with dialogs.ProgressExecution(self, "Calculating mutual information", executor=executor) as progress:
			for pair in pairs:
				limits = [self.range_map[expr] for expr in pair]
				task = self.dataset(*pair, executor=executor, async=True).mutual_information(limits=limits, size=self.grid_size)
				progress.add_task(task).end()
				tasks.append(task)
			if not progress.execute():
				return
		logger.debug("get means")
		mutual_information = [task.get() for task in tasks]

		#mutual_information_list = [MI_map[pair] for pair in pairs]
		table.setQualities(pairs, mutual_information)
		return

		print(table)
		qualities = []
		pairs = table.getSelected()

		if 0:
			for pair in pairs:
				dim = len(pair)
				#if dim == 2:
				columns = [self.dataset.columns[name] for name in pair]
				print(pair)
				information = vaex.kld.kld_shuffled(columns, mask=mask)
				qualities.append(information)
				#print pair
		if 0:
			dialog = QtGui.QProgressDialog("Calculating Mutual information", "Abort", 0, 1000, self)
			dialog.show()
			def feedback(percentage):
				print(percentage)
				dialog.setValue(int(percentage*10))
				QtCore.QCoreApplication.instance().processEvents()
				if dialog.wasCanceled():
					return True
		with ProgressExecution(self, "Calculating Mutual information") as progress:
			qualities = vaex.kld.kld_shuffled_grouped(self.dataset, self.range_map, pairs, feedback=progress.progress, use_mask=self.radio_button_selection.isChecked())
			#dialog.hide()
		if qualities is not None:
			print(qualities)
			table.setQualities(pairs, qualities)
		

	def rank_plot(self, table):
		plot = RankPlot(self, table)
		plot.show()