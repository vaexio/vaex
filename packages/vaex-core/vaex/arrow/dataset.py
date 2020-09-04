__author__ = 'maartenbreddels'
import logging

import pyarrow as pa
import pyarrow.parquet as pq

import vaex.dataset
import vaex.file.other
from .convert import column_from_arrow_array
logger = logging.getLogger("vaex.arrow")


def from_table(table, as_numpy=True):
    columns = dict(zip(table.schema.names, table.columns))
    # TODO: this should be an DatasetArrow and/or DatasetParquet
    dataset = vaex.dataset.DatasetArrays(columns)
    df = vaex.dataframe.DataFrameLocal(dataset)
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

