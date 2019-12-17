__author__ = 'maartenbreddels'
import logging

import pyarrow as pa
import pyarrow.parquet as pq

import vaex.dataset
import vaex.file.other
from .convert import column_from_arrow_array
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
        try:
            # first we try if it opens as stream
            reader = pa.ipc.open_stream(source)
        except pa.lib.ArrowInvalid:
            # if not, we open as file
            reader = pa.ipc.open_file(source)
            # for some reason this reader is not iterable
            batches = [reader.get_batch(i) for i in range(reader.num_record_batches)]
        else:
            # if a stream, we're good
            batches = reader  # this reader is iterable
        table = pa.Table.from_batches(batches)
        self._load_table(table)
    
    def _load_table(self, table):
        self._length_unfiltered = self._length_original = table.num_rows
        self._index_end = self._length_original = table.num_rows
        for col, name in zip(table.columns, table.schema.names):
            # TODO: keep the arrow columns, and support and test chunks
            arrow_array = col.chunk(0)
            if isinstance(arrow_array.type, pa.DictionaryType):
                column = column_from_arrow_array(arrow_array.indices)
                labels = column_from_arrow_array(arrow_array.dictionary).tolist()
                self._categories[name] = dict(labels=labels, N=len(labels))
            else:
                column = column_from_arrow_array(arrow_array)
            self.columns[name] = column
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

class DatasetParquet(DatasetArrow):
    def _load(self):
        # might not be optimal, but it works, we can always see if we can
        # do mmapping later on
        table = pq.read_table(self.path)
        self._load_table(table)

vaex.file.other.dataset_type_map["arrow"] = DatasetArrow
vaex.file.other.dataset_type_map["parquet"] = DatasetParquet

