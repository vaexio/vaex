
class DatasetAstropyTable(DatasetArrays):
	def __init__(self, filename=None, format=None, table=None, **kwargs):
		self.ucds = {}
		self.units = {}
		columns = {}
		if table is None:
			self.filename = filename
			self.format = format
			self.read_table()
		else:
			#print vars(table)
			#print dir(table)
			self.description = table.meta.get("description")
			self.table = table
			#self.name

		#data = table.array.data
		for i in range(len(self.table.dtype)):
			name = self.table.dtype.names[i]
			column = self.table[name]
			type = self.table.dtype[i]
			#clean_name = re.sub("[^a-zA-Z_]", "_", name)
			if type.kind in "fiuSU": # only store float and int
				#datagroup.create_dataset(name, data=table.array[name].astype(np.float64))
				#dataset.addMemoryColumn(name, table.array[name].astype(np.float64))
				masked_array = self.table[name].data
				if "ucd" in column._meta:
					self.ucds[name] = column._meta["ucd"]
				if column.unit:
					unit = _try_unit(column.unit)
					if unit:
						self.units[name] = unit
				if column.description:
					self.descriptions[name] = column.description
				if hasattr(masked_array, "mask"):
					if type.kind in ["f"]:
						masked_array.data[masked_array.mask] = np.nan
					if type.kind in ["i"]:
						masked_array.data[masked_array.mask] = 0
				columns[name] = self.table[name].data
			if type.kind in ["SU"]:
				columns[name] = self.table[name].data

		super().__init__(columns)

		#dataset.samp_id = table_id
		#self.list.addDataset(dataset)
		#return dataset

	def read_table(self):
		self.table = astropy.table.Table.read(self.filename, format=self.format, **kwargs)
