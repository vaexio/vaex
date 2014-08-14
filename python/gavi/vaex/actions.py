import gavi.logger
import numpy as np

logger = gavi.logger.getLogger("gavi.undo")

class UndoManager(object):
	def __init__(self, max_bytes=1024**3):
		self.action_history = []
		self.undo_count = 0# number of times undo is pressed
		
	def undo(self):
		action = self.action_history[-(1+self.undo_count)]
		logger.debug("undoing: %r" % action)
		action.undo()
	
	def add_action(self, action):
		# cut off any remaining 'redo' action, and add action to the list
		logger.debug("adding action: %r" % action)
		logger.debug("history was %r" % self.action_history)
		self.action_history = self.action_history[:-(1+self.undo_count)]
		self.action_history.append(action)
		logger.debug("history is now %r" % self.action_history)
		self.undo_count = 0 # this will reset the redo possibility

	def redo(self):
		logger.debug("redoing")
		
class Action(object):
	"""
	action should support
	 - byteSize() # nr of bytes the action occupies
	 - do() - does the operation, used at moment of the actual action, and during redo
	 - redo() - redo the operation
	 - description() - gui friendly desciption
	"""
	pass

class ActionMask(Action):
	def __init__(self, description, dataset, previous_mask):
		""" Assuming mask is a bool array"""
		# store the 1 byte mask as a 1 bit mask to save memory
		data = np.packbits(mask.astype(np.uint8))
		
	def do(
					 
		
