__author__ = 'maartenbreddels'
import logging

import numpy as np
import pyarrow as pa

import vaex.dataset
import vaex.file.other
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
            arrow_type = arrow_array.type
            buffers = arrow_array.buffers()
            assert len(buffers) == 2
            mask = None
            bitmap_buffer = buffers[0]
            data_buffer = buffers[1]
            if bitmap_buffer is not None:
                # arrow uses a bitmap https://github.com/apache/arrow/blob/master/format/Layout.md
                bitmap = np.frombuffer(bitmap_buffer, np.uint8, len(bitmap_buffer))
                # we do have to change the ordering of the bits
                mask = 1-np.unpackbits(bitmap).reshape((len(bitmap),8))[:,::-1].reshape(-1)[:len(arrow_array)]
            if isinstance(arrow_type, type(pa.binary(1))):
                # mimics python/pyarrow/array.pxi::Array::to_numpy
                # print(name, "seems to be a bytes type")
                buffers = arrow_array.buffers()
                assert len(buffers) == 2
                dtype = "S" + str(arrow_type.byte_width)
                # arrow seems to do padding, check if it is all ok
                expected_length = arrow_type.byte_width * len(arrow_array)
                actual_length = len(buffers[-1])
                if actual_length < expected_length:
                    raise ValueError('buffer is smaller (%d) than expected (%d)' % (actual_length, expected_length))
                array = np.frombuffer(buffers[-1], dtype, len(arrow_array))# TODO: deal with offset ? [arrow_array.offset:arrow_array.offset + len(arrow_array)]
            else:
                dtype = arrow_array.type.to_pandas_dtype()
            array = np.frombuffer(data_buffer, dtype, len(arrow_array))
            if mask is not None:
                array = np.ma.MaskedArray(array, mask=mask)
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

