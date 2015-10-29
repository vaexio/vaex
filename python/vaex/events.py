import traceback
import logging
logger = logging.getLogger("vaex.events")

class Signal(object):
	def __init__(self, name=None):
		"""

		:type name: str
		:return:
		"""
		self.name = name or repr(self)
		self.callbacks = []
		self.extra_args = {}
		
	def connect(self, callback, prepend=False, *args, **kwargs):
		if prepend:
			self.callbacks.insert(0, callback)
		else:
			self.callbacks.append(callback)
		self.extra_args[callback] = (args, kwargs)
		return callback
						   
	def emit(self, *args, **kwargs):
		results = []
		for callback in self.callbacks:
			extra_args, extra_kwargs = self.extra_args[callback]
			final_args = args + extra_args
			final_kwargs = {}
			final_kwargs.update(extra_kwargs)
			final_kwargs.update(kwargs)
			try:
				value = callback(*final_args, **final_kwargs)
				results.append(value)
			except Exception:
				logger.error("error in handling callback %r with arguments %r and kwargs %r" % (callback, final_args, final_kwargs))
				raise
				#tb = traceback.format_exc()
				#raise Exception("error while calling callback: %r with arguments %r and kwargs %r" % (callback, final_args, final_kwargs), tb)

		return results
			
	def disconnect(self, callback):
		self.callbacks.remove(callback)
		del self.extra_args[callback]
		