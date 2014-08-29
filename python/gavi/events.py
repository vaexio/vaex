

class Signal(object):
	def __init__(self, name=None):
		self.name = name or repr(self)
		self.callbacks = []
		self.extra_args = {}
		
	def connect(self, callback, *args, **kwargs):
		self.callbacks.append(callback)
		self.extra_args[callback] = (args, kwargs)
		return callback
						   
	def emit(self, *args, **kwargs):
		print "emit", self.name, self.callbacks, args, kwargs
		for callback in self.callbacks:
			extra_args, extra_kwargs = self.extra_args[callback]
			final_args = args + extra_args
			final_kwargs = {}
			final_kwargs.update(extra_kwargs)
			final_kwargs.update(kwargs)
			callback(*final_args, **final_kwargs)
			
	def disconnect(self, callback):
		self.callbacks.remove(callback)
		del self.extra_args[callback]
		