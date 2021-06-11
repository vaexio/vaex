import astropy.io.votable

import vaex
from vaex.dataset import DatasetFile
from vaex.dataset_misc import _try_unit

class VOTable(DatasetFile):
	snake_name = "votable"
	def __init__(self, filename, fs_options={}, fs=None):
		super().__init__(filename)
		self.ucds = {}
		self.units = {}
		self.filename = filename
		self.path = filename
		with vaex.file.open(filename, fs_options=fs_options, fs=fs) as f:
			votable = astropy.io.votable.parse(f)

		self.first_table = votable.get_first_table()
		self.description = self.first_table.description

		for field in self.first_table.fields:
			name = field.name
			data = self.first_table.array[name]
			type = self.first_table.array[name].dtype
			clean_name = name
			if field.ucd:
				self.ucds[clean_name] = field.ucd
			if field.unit:
				unit = _try_unit(field.unit)
				if unit:
					self.units[clean_name] = unit
			if field.description:
				self.descriptions[clean_name] = field.description
			if type.kind in "fiubSU": # only store float and int and boolean
				self.add_column(clean_name, data) #self.first_table.array[name].data)
			if type.kind == "O":
				print("column %r is of unsupported object type , will try to convert it to string" % (name,))
				try:
					data = data.astype("S")
					self.add_column(name, data)
				except Exception as e:
					print("Giving up column %s, error: %r" %(name, e))
			#if type.kind in ["S"]:
			#	self.add_column(clean_name, self.first_table.array[name].data)
		self._freeze()

	@classmethod
	def can_open(cls, path, *args, **kwargs):
		can_open = vaex.file.stringyfy(path).endswith(".vot")
		return can_open

	def close(self):
		pass
