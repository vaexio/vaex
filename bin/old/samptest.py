# -*- coding: utf-8 -*-
import SocketServer
import sys
import os
import sampy
from gavi.samp import Samp
import astropy.io.votable
import itertools
import gavi.wxui
import gavi.kde
import wx
from wx.lib.mixins.listctrl import CheckListCtrlMixin
import  wx.lib.newevent

SocketServer.ThreadingMixIn.daemon_threads = True
sampy.ThreadingXMLRPCServer.daemon_threads = True

title = "Density estimation"


#print samp.client
#print samp.client.client
#print samp.client.client.client
#print samp.client.client.client
#samp.client.enotifyAll("samp.app.echo", txt = "Hello world!")
#import pdb
#samp.client.client.client.process_request_thread.setDaemon(True)
#samp.client.client.client.process_request_thread.join()

#pdb.set_trace()
#samp.connect()



# for frozen (standalone) app, we prefer the stdout to be redirected to a file
isfrozen = hasattr(sys, "frozen")
if isfrozen:
	app = wx.App(True, filename="output.txt")
else:
	app = wx.App(False, filename=None)

def i18(text):
	return text

class CheckListCtrl(wx.ListCtrl, CheckListCtrlMixin):
	def __init__(self, parent, style):
		wx.ListCtrl.__init__(self, parent, -1, style=style|wx.LC_REPORT)
		CheckListCtrlMixin.__init__(self)
		#self.log = log
		#self.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self.OnItemActivated)
        
        
PreviewKdeEvent, EVT_PREVIEW_KDE_EVENT = wx.lib.newevent.NewEvent()
VotableLoadEvent, EVT_VOTABLE_LOAD_EVENT = wx.lib.newevent.NewEvent()


class CombinationSelectionPanelNd(wx.Panel):
	def __init__(self, parent, dimension):
		wx.Panel.__init__(self, parent)
		self.dimension = dimension
		self.mainSizer = wx.BoxSizer(wx.VERTICAL)
		self.buttonSizer = wx.BoxSizer(wx.HORIZONTAL)

		self.buttonAll = wx.Button(self, label=i18("Select all"))
		self.Bind(wx.EVT_BUTTON, self.OnSelectAll, self.buttonAll)
		self.buttonNone = wx.Button(self, label=i18("Deselect"))
		self.Bind(wx.EVT_BUTTON, self.OnSelectNone, self.buttonNone)
		
		self.buttonSizer.Add(self.buttonAll, 0)
		self.buttonSizer.Add(self.buttonNone, 0)

		self.checkboxList = CheckListCtrl(self, style=wx.SUNKEN_BORDER|wx.LIST_STATE_SELECTED)
		self.checkboxList.InsertColumn(0, "dimension", 400)
		self.mainSizer.Add(self.buttonSizer, 0)
		self.mainSizer.Add(self.checkboxList,  1, wx.GROW)
		self.SetSizer(self.mainSizer)
		self.items = []
		self.lookup = {} # from item to combination tuple
		
		# for wxMSW
		self.checkboxList.Bind(wx.EVT_COMMAND_RIGHT_CLICK, self.OnRightClick)

		# for wxGTK
		self.checkboxList.Bind(wx.EVT_RIGHT_UP, self.OnRightClick)
		
	def listCombinations(self, names):
		index = 0
		self.items = []
		self.checkboxList.DeleteAllItems()
		for combination in itertools.combinations(names, self.dimension):
			name = " / ".join(combination)
			self.items.append(name)
			item = self.checkboxList.InsertStringItem(index, name)
			#print item
			self.lookup[item] = combination
			self.checkboxList.CheckItem(index)
			index += 1
		#self.checkboxList.Set(self.items)
		#self.checkboxList.SetChecked(range(len(self.items)))
		
	def OnSelectAll(self, event):
		for i in range(len(self.items)):
			self.checkboxList.CheckItem( i, True)

	def OnSelectNone(self, event):
		for i in range(len(self.items)):
			self.checkboxList.CheckItem( i, False)

	def OnRightClick(self, event):
		x = event.GetX()
		y = event.GetY()
		#self.log.WriteText("x, y = %s\n" % str((x, y)))
		item, flags = self.checkboxList.HitTest((x, y))
		print item, flags
		
		self.currentCombination = None
		if item in self.lookup:
			self.currentCombination = self.lookup[item]
			# only do this part the first time so the events are only bound once
			if not hasattr(self, "popupID"):
				self.popupID = wx.NewId()

				self.Bind(wx.EVT_MENU, self.OnPopup, id=self.popupID)

			# make a menu
			menu = wx.Menu()
			# add some items
			menu.Append(self.popupID, "Preview")
			#menu.Append(self.popupID2, "Iterate Selected")
			#menu.Append(self.popupID3, "ClearAll and repopulate")
			#menu.Append(self.popupID4, "DeleteAllItems")
			#menu.Append(self.popupID5, "GetItem")
			#menu.Append(self.popupID6, "Edit")

			# Popup the menu.  If an item is selected then its handler
			# will be called before PopupMenu returns.
			self.PopupMenu(menu)
			menu.Destroy()
		
	def OnPopup(self, event):
		print self.currentCombination
		event = PreviewKdeEvent(combination=self.currentCombination)
		wx.PostEvent(self, event)
		

class CombinationSelectionPanel(wx.Panel):
	def __init__(self, parent):
		wx.Panel.__init__(self, parent)
		
		self.tab = wx.Notebook(self)
		self.mainSizer = wx.BoxSizer(wx.VERTICAL)
		self.mainSizer.Add(self.tab,  1, wx.LEFT | wx.TOP | wx.GROW)
		
		self.selection1d = CombinationSelectionPanelNd(self.tab, 1)
		self.selection2d = CombinationSelectionPanelNd(self.tab, 2)
		self.selection3d = CombinationSelectionPanelNd(self.tab, 3)
		self.tab.AddPage(self.selection1d, "1d")
		self.tab.AddPage(self.selection2d, "2d")
		self.tab.AddPage(self.selection3d, "3d")
		
		self.selection1d.Bind(EVT_PREVIEW_KDE_EVENT, self.OnPreview1or2d)
		self.selection2d.Bind(EVT_PREVIEW_KDE_EVENT, self.OnPreview1or2d)
		
		self.SetSizer(self.mainSizer)
		self.currentTable = None
		
	def OnPreview1or2d(self, event):
		print event, event.combination
		combination = event.combination
		dialog = wx.Dialog(self, -1, size=(400, 400))
		plotWindow = gavi.wxui.PlotWindow1d(dialog)
		
		# only 1 and 2d supported
		assert len(combination) in [1,2]
		if len(combination) == 1:
			column1 = combination[0]
			data = self.currentTable.array[column1]
			Nkde1d = 128 # TODO: make this configurable
			density = gavi.kde.kde1d(data, min(data), max(data), Nkde1d)
			plotWindow.plot1d(density, min(data), max(data), xlabel=column1)
		else:
			column1 = combination[0]
			column2 = combination[1]
			x = self.currentTable.array[column1]
			y = self.currentTable.array[column2]
			Nkde2d = 128 # TODO: make this configurable
			xmin, xmax = min(x), max(x)
			ymin, ymax = min(y), max(y)
			print xmin, xmax, ymin, ymax
			density = gavi.kde.kde2d(x, y, xmin, xmax, ymin, ymax, (Nkde2d, Nkde2d))
			plotWindow.plot2d(density, xmin, xmax, ymin, ymax, xlabel=column1, ylabel=column2)
		
		sizer = wx.BoxSizer(wx.VERTICAL)
		sizer.Add(plotWindow)
		dialog.SetSizer(sizer)
		dialog.CenterOnScreen()
		dialog.ShowModal()
		
	def selectTable(self, table):
		self.currentTable = table
		names = table.array.dtype.names
		for s in [self.selection1d, self.selection2d, self.selection3d]:
			s.listCombinations(names)
		
		
		
# contains a list of all tables and on the right all information and actions etc
class TablesPanel(wx.Panel):
	def __init__(self, parent):
		wx.Panel.__init__(self, parent, -1)
		self.mainSizer = wx.BoxSizer(wx.HORIZONTAL)

		self.splitter = wx.SplitterWindow(self)
		
		#self.leftSizer = wx.BoxSizer(wx.HORIZONTAL)
		#self.rightSizer = wx.BoxSizer(wx.HORIZONTAL)
	
		self.tableList = wx.ListCtrl(self.splitter, -1, style=wx.LC_REPORT)
		self.tableList.InsertColumn( 0, "Name", width=-1)
		self.tableList.InsertColumn( 1, "Columns", width=-1)
		self.tableList.InsertColumn( 2, "Rows", width=-1)
		
		self.infoPanel = wx.Panel(self.splitter, -1)
		self.infoName = wx.StaticText(self.infoPanel, label="Name: ")
		self.infoSize = wx.StaticText(self.infoPanel, label="Size: ")
		self.combinationSelection = CombinationSelectionPanel(self.infoPanel)
		
		self.infoSizer = wx.BoxSizer(wx.VERTICAL)
		self.infoSizer.Add(self.infoName)
		self.infoSizer.Add(self.infoSize)
		self.infoSizer.Add(self.combinationSelection, 1, wx.GROW)
		self.infoPanel.SetSizer(self.infoSizer)
		
		self.splitter.SplitVertically(self.tableList, self.infoPanel, 300)
		self.splitter.SetSashGravity(0.3)
		self.mainSizer.Add(self.splitter, 1, wx.EXPAND)
		#self.leftSizer.Add(self.tableList
		self.SetSizer(self.mainSizer)
		self.lookup = {} # maps from integer to python data
		self.tableList.Bind(wx.EVT_LIST_ITEM_SELECTED, self.OnSelected)
		
	def OnSelected(self, event):
		print event
		print event.GetData(), event.GetIndex()
		id = self.tableList.GetItemData(event.GetIndex())
		votable = self.lookup[id]
		#self.selectTable(votable)
		
	def selectTable(self, table):
		self.combinationSelection.selectTable(table)
		
	def addVotable(self, votable, name):
		item = self.tableList.InsertStringItem(0, name)
		self.tableList.SetStringItem(item, 1, str(len(votable.array.dtype.names)))
		self.tableList.SetStringItem(item, 2, str(len(votable.array)))
		id = wx.NewId() 
		self.tableList.SetItemData(item, id)
		self.lookup[id] = votable
		print "id", id
		self.selectTable(votable)

class MainFrame(wx.Frame):
	def __init__(self, parent, title):
		self.dirname = os.path.abspath(".")
		wx.Frame.__init__(self, parent, title=title, size=(1200,800), pos=(-1,-1))
		#self.control = wx.TextCtrl(self, style=wx.TE_MULTILINE)
		self.CreateStatusBar()
		
		filemenu= wx.Menu()
		menuOpen = filemenu.Append(wx.ID_OPEN, "&Open"," Open a file to edit")
		menuAbout= filemenu.Append(wx.ID_ABOUT, "&About"," Information about this program")
		menuExit = filemenu.Append(wx.ID_EXIT,"E&xit"," Terminate the program")
		menuBar = wx.MenuBar()
		menuBar.Append(filemenu,"&File") # Adding the "filemenu" to the MenuBar
		self.SetMenuBar(menuBar)  # Adding the MenuBar to the Frame content.
		
		self.Bind(wx.EVT_MENU, self.OnOpen, menuOpen)
		self.Bind(wx.EVT_MENU, self.OnExit, menuExit)
		self.Bind(wx.EVT_MENU, self.OnAbout, menuAbout)

		self.samp = Samp()

		self.samp.tableLoadCallbacks.append(self.tableLoad)
		self.mainSizer = wx.BoxSizer(wx.HORIZONTAL)
		
		self.tablesPanel = TablesPanel(self)
		
	
		self.mainSizer.Add(self.tablesPanel, 1, wx.EXPAND)
		self.SetSizer(self.mainSizer)
		self.SetAutoLayout(1)
		kwargs = {'url': 'http://127.0.0.1:2525/dynamic/114/t2.vot', 'table_id': 'topcatd9ac-2', 'name': 'vlos_helio_scl_gius2011.csv'}
		event = VotableLoadEvent(**kwargs)
		#self.OnTableLoad(event)
		self.Bind(EVT_VOTABLE_LOAD_EVENT, self.OnTableLoad)


	def OnTableLoad(self, event):
		print "loading table"
		table = astropy.io.votable.parse_single_table(event.url)
		self.tablesPanel.addVotable(table, event.name)
		
	def tableLoad(self, url, table_id, name):
		print "load table event"
		#array = table.array
		#print array.dtype.names
		#print array.shape
		event = VotableLoadEvent(url=url, table_id=table_id, name=name)
		wx.PostEvent(self, event)
			
	def OnAbout(self, e):
			# Create a message dialog box
			dlg = wx.MessageDialog(self, " A density esitmator \n for GAIA data", "About: " +title, wx.OK)
			dlg.ShowModal() # Shows it
			dlg.Destroy() # finally destroy it when finished.		

	def OnExit(self,e):
		self.samp.client.disconnect()
		self.Close(True)  # Close the frame.

	def OnOpen(self,e):
		""" Open a file"""
		dlg = wx.FileDialog(self, "Choose a file", self.dirname, "", "*.hdf5", wx.OPEN)
		if dlg.ShowModal() == wx.ID_OK:
			self.filename = dlg.GetFilename()
			self.dirname = dlg.GetDirectory()
			filename = os.path.join(self.dirname, self.filename)
			self.addH5File(filename)
			#f = open(os.path.join(self.dirname, self.filename), 'r')
			#self.control.SetValue(f.read())
			#f.close()
		dlg.Destroy()
		
	def addH5File(self, filename):
		print "opening hdf5 file", filename




frame = MainFrame(None, title) # A Frame is a top-level window.
frame.Show(True)     # Show the frame.
app.MainLoop()
if 0:
	try:
		while True:
			thread.join(10000)
			if not thread.isAlive():
				break
	except KeyboardInterrupt, e:
		print "error", e
#samp.client.client._thread.join()