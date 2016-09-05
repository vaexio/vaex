import logging
import numpy as np
import copy

logger = logging.getLogger("vaex.ui.undo")

class UndoManager(object):
	def __init__(self, max_bytes=1024**3):
		self.actions_undo = []
		self.actions_redo = []
		self.undo_count = 0# number of times undo is pressed

	def undo(self):
		logger.debug("history was %r-%r" % (self.actions_undo, self.actions_redo))
		action = self.actions_undo.pop()
		logger.debug("undoing: %r" % action)
		self.actions_redo.insert(0, action)
		try:
			action.undo()
		except:
			logger.exception("error executing action")
		logger.debug("history is  %r-%r" % (self.actions_undo, self.actions_redo))
	
	def add_action(self, action):
		# cut off any remaining 'redo' action, and add action to the list
		logger.debug("history was %r-%r" % (self.actions_undo, self.actions_redo))
		logger.debug("adding action: %r" % action)
		self.actions_redo = []
		self.actions_undo.append(action)
		logger.debug("history is  %r-%r" % (self.actions_undo, self.actions_redo))

	def redo(self):
		logger.debug("history was %r-%r" % (self.actions_undo, self.actions_redo))
		logger.debug("redoing")
		action = self.actions_redo.pop(0)
		try:
			action.do()
		except:
			logger.exception("error executing action")
		self.actions_undo.append(action)
		logger.debug("history is  %r-%r" % (self.actions_undo, self.actions_redo))
		
	def can_undo(self):
		return len(self.actions_undo) > 0

	def can_redo(self):
		return len(self.actions_redo) > 0


class Action(object):
	"""
	action should support
	 - byteSize() # nr of bytes the action occupies
	 - do() - does the operation, used at moment of the actual action, and during redo
	 - redo() - redo the operation
	 - description() - gui friendly desciption
	 
	 and actions should add itself to it's UndoManager
	"""
	pass

class ActionMask(Action):
	def __init__(self, undo_manager, description, mask, apply_mask):
		""" Assuming mask is a bool array"""
		# store the 1 byte mask as a 1 bit mask to save memory
		self.undo_manager = undo_manager
		self.data = None if mask is None else np.packbits(mask.astype(np.uint8))
		self.length = 0 if mask is None else len(mask)
		self._description = description
		self.mask = mask
		self.apply_mask = apply_mask
		self.undo_manager.add_action(self)
		
	def description(self):
		return self._description
		
	def do(self):
		mask = None if self.data is None else np.unpackbits(self.data).astype(np.bool)[:self.length]
		self.apply_mask(mask)
		
	def undo(self):
		# find a previous ActionMask, and execute it, but just keep it in the history
		for action in self.undo_manager.actions_undo[::-1]: # traverse from most recent to last
			if isinstance(action, ActionMask):
				action.do()
				break
		else: # if nothing was selected before, select None
			self.apply_mask(None)
		

class ActionZoom(Action):		
	def __init__(self, undo_manager, description, apply_ranges, all_axis_indices,
	             previous_ranges_viewport, previous_range_level_show, axis_indices, ranges_viewport=None, range_level_show=None):
		self.undo_manager = undo_manager
		self.apply_ranges = apply_ranges
		
		self.all_axis_indices = all_axis_indices
		#self.previous_ranges = list(previous_ranges)
		self.previous_ranges_viewport = copy.deepcopy(previous_ranges_viewport)
		self.previous_range_level_show = None if previous_range_level_show is None else copy.deepcopy(previous_range_level_show)
		
		self.axis_indices = axis_indices
		#self.ranges = ranges
		self.ranges_viewport = copy.deepcopy(ranges_viewport)
		self.range_level_show = copy.deepcopy(range_level_show)
		self._description = description
		self.undo_manager.add_action(self)
		
	def description(self):
		return self._description
		
	def do(self):
		self.apply_ranges(self.axis_indices, self.ranges_viewport, self.range_level_show)
		
	def undo(self):
		self.apply_ranges(self.all_axis_indices, self.previous_ranges_viewport, self.previous_range_level_show)
