
order_toolbar_navigate = 1.

def pluginclass(cls):

	cls.__bases__[0].registry.add(cls)
	return cls

def pluginbase(cls):
	if not hasattr(cls, 'registry'):
		cls.registry = set()
	return cls


class RegisterPlugins(type):
	def __init__(cls, name, bases, nmspc):
		super(RegisterPlugins, cls).__init__(name, bases, nmspc)
		if not hasattr(cls, 'registry'):
			cls.registry = set()
		cls.registry.add(cls)
		cls.registry -= set(bases) # Remove base classes

	def __iter__(cls):
		return iter(cls.registry)

	def __str__(cls):
		if cls in cls.registry:
			return cls.__name__
		return cls.__name__ + ": " + ", ".join([sc.__name__ for sc in cls])


@pluginbase
class PluginPlot(object):
	#__metaclass__ = RegisterPlugins
	def __init__(self, dialog):
		self.dialog = dialog

	def clean_up(self):
		pass

	def apply_options(self, options):
		pass

	def get_options(self):
		pass

	def use_layer(self, layer):
		pass

	@staticmethod
	def useon(dialog_class):
		return True

	def syncToolbar(self):
		pass

	def setMode(self, action):
		pass


@pluginbase
class PluginLayer(object):
	#__metaclass__ = RegisterPlugins
	def __init__(self, parent, layer):
		self.parent = parent
		self.layer = layer
		self.dataset = layer.dataset

	def clean_up(self):
		pass
		
	@staticmethod
	def useon(layer_class):
		return True
	
	def get_options(self):
		return {}

	def apply_options(self, options):
		pass

	def start_animation(self, widget):
		pass

	def stop_animation(self):
		pass

@pluginbase
class PluginDataset(object):
	#__metaclass__ = RegisterPlugins
	def __init__(self, dataset, widget):
		self.dataset = dataset
		self.widget = widget

	@staticmethod
	def useon(dataset):
		return True
	
