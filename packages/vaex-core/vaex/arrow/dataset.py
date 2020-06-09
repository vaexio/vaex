__author__ = 'maartenbreddels'
import logging

import pyarrow as pa
import pyarrow.parquet as pq

import vaex.dataset
import vaex.file.other
from .convert import column_from_arrow_array
logger = logging.getLogger("vaex.arrow")


def from_table(table, as_numpy=True):
    df = vaex.dataframe.DataFrameLocal(None, None, [])
    df._length_unfiltered = df._length_original = table.num_rows
    df._index_end = df._length_original = table.num_rows
    for col, name in zip(table.columns, table.schema.names):
        df.add_column(name, col)
    return df.as_numpy() if as_numpy else df


def open(filename, as_numpy=True):
    source = pa.memory_map(filename)
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
    return from_table(table, as_numpy=as_numpy)


def open_parquet(filename, as_numpy=True):
    table = pq.read_table(filename)
    return from_table(table, as_numpy=as_numpy)

# vaex.file.other.dataset_type_map["arrow"] = DatasetArrow
# vaex.file.other.dataset_type_map["parquet"] = DatasetParquet

