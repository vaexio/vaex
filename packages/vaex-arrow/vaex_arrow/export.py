__author__ = 'maartenbreddels'
import os
import sys
import warnings
import collections
import logging

import numpy as np
import vaex

from .convert import arrow_array_from_numpy_array

max_length = int(1e5)

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except:
    if not on_rtd:
        raise

logger = logging.getLogger("vaex_arrow.export")

def export(dataset, path, column_names=None, byteorder="=", shuffle=False, selection=False, progress=None, virtual=True, sort=None, ascending=True):
    table = _export_table(dataset, column_names, byteorder, shuffle, selection, progress, virtual, sort, ascending)
    b = table.to_batches()
    with pa.OSFile(path, 'wb') as sink:
        writer = pa.RecordBatchStreamWriter(sink, b[0].schema)
        writer.write_table(table)

def export_parquet(dataset, path, column_names=None, byteorder="=", shuffle=False, selection=False, progress=None, virtual=True, sort=None, ascending=True):
    table = _export_table(dataset, column_names, byteorder, shuffle, selection, progress, virtual, sort, ascending)
    pq.write_table(table, path)

def _export_table(dataset, column_names=None, byteorder="=", shuffle=False, selection=False, progress=None, virtual=True, sort=None, ascending=True):
    """
    :param DatasetLocal dataset: dataset to export
    :param str path: path for file
    :param lis[str] column_names: list of column names to export or None for all columns
    :param str byteorder: = for native, < for little endian and > for big endian
    :param bool shuffle: export rows in random order
    :param bool selection: export selection or not
    :param progress: progress callback that gets a progress fraction as argument and should return True to continue,
            or a default progress bar when progress=True
    :param: bool virtual: When True, export virtual columns
    :return:
    """
    column_names = column_names or dataset.get_column_names(virtual=virtual, strings=True)
    for name in column_names:
        if name not in dataset.columns:
            warnings.warn('Exporting to arrow with virtual columns is not efficient')
    N = len(dataset) if not selection else dataset.selected_length(selection)
    if N == 0:
        raise ValueError("Cannot export empty table")

    if shuffle and sort:
        raise ValueError("Cannot shuffle and sort at the same time")

    if shuffle:
        random_index_column = "random_index"
        while random_index_column in dataset.get_column_names():
            random_index_column += "_new"
    partial_shuffle = shuffle and len(dataset) != N

    order_array = None
    if partial_shuffle:
        # if we only export a portion, we need to create the full length random_index array, and
        shuffle_array_full = np.random.choice(len(dataset), len(dataset), replace=False)
        # then take a section of it
        # shuffle_array[:] = shuffle_array_full[:N]
        shuffle_array = shuffle_array_full[shuffle_array_full < N]
        del shuffle_array_full
        order_array = shuffle_array
    elif shuffle:
        shuffle_array = np.random.choice(N, N, replace=False)
        order_array = shuffle_array

    if sort:
        if selection:
            raise ValueError("sorting selections not yet supported")
        logger.info("sorting...")
        indices = np.argsort(dataset.evaluate(sort))
        order_array = indices if ascending else indices[::-1]
        logger.info("sorting done")

    if selection:
        full_mask = dataset.evaluate_selection_mask(selection)
    else:
        full_mask = None

    arrow_arrays = []
    for column_name in column_names:
        mask = full_mask
        if selection:
            values = dataset.evaluate(column_name, filtered=False)
            values = values[mask]
        else:
            values = dataset.evaluate(column_name)
            if shuffle or sort:
                indices = order_array
                values = values[indices]
        arrow_arrays.append(arrow_array_from_numpy_array(values))
    if shuffle:
        arrow_arrays.append(arrow_array_from_numpy_array(order_array))
        column_names = column_names + [random_index_column]
    table = pa.Table.from_arrays(arrow_arrays, column_names)
    return table

