__author__ = 'maartenbreddels'
import logging

import numpy as np
import pyarrow as pa

import vaex.dataset
import vaex.file.other
from .convert import numpy_array_from_arrow_array
logger = logging.getLogger("vaex_arrow")


class DatasetArrow(vaex.dataset.DatasetLocal):
    """Implements storage using arrow"""

    def __init__(self, filename=None, table=None, write=False):
        super(DatasetArrow, self).__init__(name=filename, path=filename, column_names=[])
        self._write = write
        if table is None:
            self._load()
        else:
            self._load_table(table)

    def _load(self):
        source = pa.memory_map(self.path)
        reader = pa.open_stream(source)
        table = pa.Table.from_batches([b for b in reader])
        self._load_table(table)
    
    def _load_table(self, table):
        self._length_unfiltered =  self._length_original = table.num_rows
        for col in table.columns:
            name = col.name
            # TODO: keep the arrow columns, and support and test chunks
            arrow_array = col.data.chunks[0]
            array = numpy_array_from_arrow_array(arrow_array)

            self.columns[name] = array
            self.column_names.append(name)
            self._save_assign_expression(name, vaex.expression.Expression(self, name))


    @classmethod
    def can_open(cls, path, *args, **kwargs):
        return path.rpartition('.')[2] == 'arrow'

    @classmethod
    def get_options(cls, path):
        return []

    @classmethod
    def option_to_args(cls, option):
        return []

vaex.file.other.dataset_type_map["arrow"] = DatasetArrow

