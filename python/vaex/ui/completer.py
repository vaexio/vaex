__author__ = 'maartenbreddels'
import string
import astropy.io.votable.ucd
import astropy.units
import logging
from vaex.ui.qt import *

logger = logging.getLogger("vaex.ui.completer")

# based on http://stackoverflow.com/a/26065682
class Completer(QtGui.QCompleter):
	def __init__(self, line_edit, allowed_chars=string.ascii_letters + string.digits +"_", separators=None, match_contains=True, match_case=False):
		QtGui.QCompleter.__init__(self, [], line_edit)
		self.line_edit = line_edit
		self.match_contains = match_contains
		self.match_case = match_case
		self.allowed_chars = allowed_chars
		self.separators = separators
		self.line_edit.cursorPositionChanged.connect(self.onCursorPositionChanged)
		self.last_editing_cursor_position = None
		self.activated.connect(self.onActivated)
		self.setWrapAround(False)
		self.setCompletionMode(QtGui.QCompleter.UnfilteredPopupCompletion)
		self.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
		#self.setCompletionMode(QtGui.QCompleter.InlineCompletion)

		self.model = QtGui.QStringListModel([], self)
		self.setModel(self.model)

	def pathFromIndex(self, index):
		#return QtGui.QCompleter.pathFromIndex(self, index)
		suggested_word = QtGui.QCompleter.pathFromIndex(self, index)

		full_text = self.line_edit.text()

		index = self.line_edit.cursorPosition()
		left, right = self.find_word_bounds(full_text, index, self.allowed_chars)

		suggested_text = full_text[:left] + suggested_word + full_text[right:]
		new_cursor_pos = len(full_text[:left] + suggested_word)

		logger.debug("cursor should go to: %d", new_cursor_pos)
		def fixcursor():
			logger.debug("cursor set to: %d", new_cursor_pos)
			self.line_edit.setCursorPosition(new_cursor_pos)
		#print(("pathFromIndex", index, repr(full_text), repr(suggested_text), repr(suggested_text), self.last_editing_cursor_position))

		# after the text is set by completer, the cursor is set to the end of the lineedit, we correct is by fixcursor to set it at
		# the end of the word
		QtCore.QTimer.singleShot(0, fixcursor);
		return suggested_text

	def splitPath(self, path):
		index = self.line_edit.cursorPosition()
		left, right = self.find_word_bounds(path, index, self.allowed_chars)
		part = path[left:right]

		result = QtGui.QCompleter.splitPath(self, path)
		#print "splitPath", path, result, part

		suggestions = self.get_suggestions(part, path[:left], path[right:])
		self.suggestions = suggestions
		self.model.setStringList(suggestions)
		#self.model = QtGui.QStringListModel(suggestions, self.parent())
		#self.setModel(self.model)
		#QtCore.QTimer.singleShot(0, lambda: self.setModel(self.model));
		self.parts = [part]
		return self.parts

	def onActivated(self, text):
		pass#print "activated", text

	def onCursorPositionChanged(self, old_pos, new_pos):
		#print "cursor", old_pos, new_pos
		# this trick didn't work, as suggested by the SO anwser
		self.last_editing_cursor_position = None if old_pos == new_pos else new_pos

	def word_boundary_char(self, char):
		return (self.allowed_chars is not None and char not in self.allowed_chars) or\
				(self.separators is not None and char in self.separators)

	def find_word_bounds(self, text, index, allowed_chars):
		right = left = index
		done = False
		while not done:
			if left == 0:
				done = True
			elif not self.word_boundary_char(text[left-1]):
				left -= 1
			else:
				done = True
		done = False
		while not done:
			if right == len(text):
				done = True
			elif not self.word_boundary_char(text[right]):
				right += 1
			else:
				done = True
		return left, right

	def get_word_list(self, word, text_left, text_right):
		return "aap aardappel schaap koe blaat".split()

	def get_suggestions(self, typed_word, text_left, text_right):
		def case(word):
			return word if self.match_case else word.lower()

		suggestions = []
		if typed_word:
			word_list = self.get_word_list(typed_word, text_left, text_right)
			for word in word_list:

				if (self.match_contains and case(typed_word) in case(word)) or (not self.match_contains and case(word).startswith(case(typed_word))):
					suggestions.append(word)
		return suggestions

ucd_words = astropy.io.votable.ucd.UCDWords()
primary_list = list(sorted(ucd_words._primary))
secondary_list = list(sorted(ucd_words._secondary))
#from astropy.utils import data
import astropy.utils.data
astropy.utils.data

class UCDCompleter(Completer):
	"""
	UCDs have primary words (that come first), and secondary, that come after the second
	UCD words are seperated by a ; char
	"""
	def __init__(self, line_edit):
		Completer.__init__(self, line_edit, allowed_chars=None, separators=";")

	def get_word_list(self, word, text_left, text_right):
		if text_left.strip():
			return secondary_list
		else:
			return primary_list

	def get_suggestions(self, typed_word, text_left, text_right):
		typed_word = typed_word.lower()
		if text_left.strip():
			word_list = secondary_list
		else:
			word_list = primary_list
		descriptions = {key:desc.lower() for key, desc in ucd_words._descriptions.items() if key in word_list}
		suggestions = []
		if typed_word:
			for word in word_list:
				#if any([typed_word in word for word in word_list]) or any([typed_word in desc for desc in descriptions]):
				#print(typed_word)
				#print(list([typed_word in desc for desc in descriptions]))
				#print(any([typed_word in desc for desc in descriptions]))
				#print([desc for desc in descriptions if typed_word in desc])
				if (typed_word in word) or typed_word in descriptions[word]:
					suggestions.append(ucd_words._capitalization[word])
		return suggestions

class IdentifierCompleter(Completer):
	"""Completes variables and functions"""
	def __init__(self, line_edit, variables=[]):
		self.variables = variables
		Completer.__init__(self, line_edit)

	def get_word_list(self, word, text_left, text_right):
		return self.variables

unit_list = [name for name, unit in vars(astropy.units).items() if isinstance(unit, astropy.units.UnitBase)]
class UnitCompleter(Completer):
	"""Completes units found in astropy"""
	def __init__(self, line_edit, unit_list=unit_list):
		self.unit_list = unit_list
		Completer.__init__(self, line_edit, match_contains=False)

	def get_word_list(self, word, text_left, text_right):
		return self.unit_list


# based on from https://gist.github.com/Riateche/5984815
class UCDDelegate(QtGui.QItemDelegate):
	def __init__(self, parent):
		QtGui.QItemDelegate.__init__(self, parent)

	def createEditor(self, parent, option, index):
		editor = QtGui.QLineEdit(parent)
		completer = vaex.ui.completer.UCDCompleter(editor)
		editor.setCompleter(completer)
		#self.connect(combo, QtCore.SIGNAL("currentIndexChanged(int)"), self, QtCore.SLOT("currentIndexChanged()"))
		return editor

	def setEditorData(self, editor, index):
		editor.blockSignals(True)
		#editor.setCurrentIndex(int(index.model().data(index)))
		editor.setText(index.model().data(index))
		editor.blockSignals(False)

	def setModelData(self, editor, model, index):
		model.setData(index, editor.text())

	@QtCore.pyqtSlot()
	def currentIndexChanged(self):
		self.commitData.emit(self.sender())

class UnitDelegate(QtGui.QStyledItemDelegate):
	def __init__(self, parent):
		QtGui.QItemDelegate.__init__(self, parent)
		self.lastEditor = None

	def createEditor(self, parent, option, index):
		self.lastEditor = editor = QtGui.QLineEdit(parent)
		self.completer = vaex.ui.completer.UnitCompleter(editor)
		editor.setCompleter(self.completer)
		#self.connect(combo, QtCore.SIGNAL("currentIndexChanged(int)"), self, QtCore.SLOT("currentIndexChanged()"))

		return editor

	def setEditorData(self, editor, index):
		editor.blockSignals(True)
		#editor.setCurrentIndex(int(index.model().data(index)))
		editor.setText(index.model().data(index))
		editor.blockSignals(False)

	def setModelData(self, editor, model, index):
		model.setData(index, editor.text())

	@QtCore.pyqtSlot()
	def currentIndexChanged(self):
		self.commitData.emit(self.sender())



import numpy as np
vars = dir(np)
#vars.append("hoeba")

class LineCompleter(QtGui.QLineEdit):
	def __init__(self, parent):
		QtGui.QLineEdit.__init__(self, parent)
		#completer = UCDCompleter(self)
		#assert completer.word_boundary_char(";")
		completer = UnitCompleter(self)
		self.setCompleter(completer)
from vaex.dataset import Dataset
class ExpressionCombobox(QtGui.QComboBox):
	def __init__(self, parent, dataset, variables=False):
		"""

		:param parent:
		:param Dataset dataset:
		"""
		QtGui.QComboBox.__init__(self, parent)
		self.identifiers = []
		self.columns = dataset.get_column_names(virtual=True)
		self.identifiers.extend(vaex.dataset.expression_namespace.keys())
		self.identifiers.extend(self.columns)
		if variables:
			self.identifiers.extend(list(dataset.variables.keys()))
		self.addItems([""] + self.columns)
		#	= list(dataset.variables.keys()) + list(vaex.dataset.expression_namespace.keys())
		#	self.addItems(list(dataset.variables.keys()))
		#else:
		#	self.identifiers = list(self.columns) + list(vaex.dataset.expression_namespace.keys())
		self.setEditable(True)
		lineEdit = self.lineEdit()
		self.completer = IdentifierCompleter(lineEdit, self.identifiers)
		lineEdit.setCompleter(self.completer)

if __name__ == "__main__":
	app = QtGui.QApplication([])
	dialog = QtGui.QDialog()
	dialog.resize(400, 100)
	if 1:
		lineEdit = LineCompleter(dialog)
		#combo = QtGui.QComboBox(dialog)
		#combo.setEditable(True)
		#combo.addItems("hoeba blaat schaap".split())
		#lineEdit = combo.lineEdit()
		#completer = MathCompleter(lineEdit, vars)
		completer = UnitCompleter(lineEdit)
		lineEdit.setCompleter(completer)
	else:
		dataset = vaex.open(sys.argv[1])
		box = ExpressionCombobox(dialog, dataset)
	dialog.show()
	dialog.raise_()
	dialog.exec_()
	#app.exec_()