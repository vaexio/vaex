import vaex.iniscope

class Settings(object):
	def __init__(self, inifilename):
		self.inifilename = inifilename
		
	def load(self):
		self.scope = vaex.iniscope.IniScope(self.inifilename, load=True)
		#self.scope.init()
		self.files = self.scope["files"]
		
		
class Files(object):
	def __init__(self, open, recent):
		self.open = open
		self.recent = recent