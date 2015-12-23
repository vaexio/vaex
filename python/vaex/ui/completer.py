__author__ = 'maartenbreddels'
from vaex.ui.qt import *
import string

def get_string_part(s, index, allowed_chars):
	right = left = index
	done = False
	while not done:
		if left == 0:
			done = True
		elif s[left-1] in allowed_chars:
			left -= 1
		else:
			done = True
	done = False
	while not done:
		if right == len(s):
			done = True
		elif s[right] in allowed_chars:
			right += 1
		else:
			done = True
	return left, right

class Completer(QtGui.QCompleter):
	def __init__(self, line_edit):
		self.line_edit = line_edit
		words = "aap aardappel schaap koe blaat".split()
		QtGui.QCompleter.__init__(self, words)
		self.allowed_chars = string.ascii_letters + string.digits +"_"
		self.line_edit.cursorPositionChanged.connect(self.onCursorPositionChanged)
		self.last_editing_cursor_position = None
		self.activated.connect(self.onActivated)
		self.setWrapAround(False)
		#self.setCompletionMode(QtGui.QCompleter.InlineCompletion)

	def pathFromIndex(self, index):
		suggested_word = QtGui.QCompleter.pathFromIndex(self, index)

		full_text = self.line_edit.text()

		index = self.line_edit.cursorPosition() #this won't work
		left, right = get_string_part(full_text, index, self.allowed_chars)
		#part = path[left:right]
		suggested_text = full_text[:left] + suggested_word + full_text[right:]
		new_cursor_pos = len(full_text[:left] + suggested_word)

		#left, right = get_string_part(path, index, self.allowed_chars)
		print "cursor should go to", new_cursor_pos
		def fixcursor():
			print "fix cursor to pos", new_cursor_pos
			self.line_edit.setCursorPosition(new_cursor_pos)
		print "pathFromIndex", index, repr(full_text), repr(suggested_text), repr(suggested_text), self.last_editing_cursor_position

		QtCore.QTimer.singleShot(0, fixcursor);
		return suggested_text

	def splitPath(self, path):
		index = self.line_edit.cursorPosition()
		left, right = get_string_part(path, index, self.allowed_chars)
		part = path[left:right]

		result = QtGui.QCompleter.splitPath(self, path)
		print "splitPath", path, result, part
		return [part]

	def onActivated(self, text):
		print "activated", text
	def onCursorPositionChanged(self, old_pos, new_pos):
		print "cursor", old_pos, new_pos
		self.last_editing_cursor_position = None if old_pos == new_pos else new_pos



class LineCompleter(QtGui.QLineEdit):
	def __init__(self, parent):
		QtGui.QLineEdit.__init__(self, parent)
		completer = Completer(self)
		self.setCompleter(completer)

if __name__ == "__main__":
	app = QtGui.QApplication([])
	dialog = QtGui.QDialog()
	dialog.resize(400, 100)
	lineEdit = LineCompleter(dialog)
	dialog.show()
	dialog.raise_()
	dialog.exec_()
	#app.exec_()