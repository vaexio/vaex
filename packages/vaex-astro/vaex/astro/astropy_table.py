import numpy as np

from vaex.dataset import DatasetArrays
from vaex.dataset_misc import _try_unit

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
            self.description = table.meta.get("description")
            self.table = table

        for i in range(len(self.table.dtype)):
            name = self.table.dtype.names[i]
            column = self.table[name]
            type = self.table.dtype[i]
            if type.kind in "fiuSU": # only store float and int
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

    def read_table(self):
        self.table = astropy.table.Table.read(self.filename, format=self.format, **kwargs)
