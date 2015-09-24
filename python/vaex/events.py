

class Signal(object):
	def __init__(self, name=None):
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
			results.append(callback(*final_args, **final_kwargs))
		return results
			
	def disconnect(self, callback):
		self.callbacks.remove(callback)
		del self.extra_args[callback]
		