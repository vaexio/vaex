__author__ = 'maartenbreddels'
import os
import sys
import collections
import logging
import concurrent.futures
import threading

import numpy as np
import pyarrow as pa

import vaex
import vaex.utils
import vaex.execution
from vaex.column import ColumnStringArrow, _to_string_sequence


max_length = int(1e5)

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
try:
    import h5py
except:
    if not on_rtd:
        raise
# from vaex.dataset import DatasetLocal

logger = logging.getLogger("vaex.export")
progress_lock = threading.Lock()

class ProgressStatus(object):
    pass

def _export(dataset_input, dataset_output, random_index_column, path, column_names=None, byteorder="=", shuffle=False, selection=False, progress=None, virtual=True, sort=None, ascending=True, parallel=True):
    """
    :param DatasetLocal dataset: dataset to export
    :param str path: path for file
    :param lis[str] column_names: list of column names to export or None for all columns
    :param str byteorder: = for native, < for little endian and > for big endian
    :param bool shuffle: export rows in random order
    :param bool selection: export selection or not
    :param progress: progress callback that gets a progress fraction as argument and should return True to continue
    :return:
    """

    if selection:
        if selection == True:  # easier to work with the name
            selection = "default"

    N = len(dataset_input) if not selection else dataset_input.selected_length(selection)
    if N == 0:
        raise ValueError("Cannot export empty table")

    if shuffle and sort:
        raise ValueError("Cannot shuffle and sort at the same time")

    if shuffle:
        shuffle_array = dataset_output.columns[random_index_column]

    partial_shuffle = shuffle and len(dataset_input) != N

    order_array = None
    order_array_inverse = None

    # for strings we also need the inverse order_array, keep track of that
    has_strings = any([dataset_input.is_string(k) for k in column_names])

    if partial_shuffle:
        # if we only export a portion, we need to create the full length random_index array, and
        shuffle_array_full = np.random.choice(len(dataset_input), len(dataset_input), replace=False)
        # then take a section of it
        shuffle_array[:] = shuffle_array_full[shuffle_array_full < N]
        del shuffle_array_full
        order_array = shuffle_array
    elif shuffle:
        # better to do this in memory
        shuffle_array_memory = np.random.choice(N, N, replace=False)
        shuffle_array[:] = shuffle_array_memory
        order_array = shuffle_array
    if order_array is not None:
        indices_r = np.zeros_like(order_array)
        indices_r[order_array] = np.arange(len(order_array))
        order_array_inverse = indices_r
        del indices_r

    if sort:
        if selection:
            raise ValueError("sorting selections not yet supported")
        # these indices sort the input array, but we evaluate the input in sequential order and write it out in sorted order
        # e.g., not b[:] = a[indices]
        # but b[indices_r] = a
        logger.info("sorting...")
        indices = np.argsort(dataset_input.evaluate(sort))
        indices_r = np.zeros_like(indices)
        indices_r[indices] = np.arange(len(indices))
        if has_strings:
            # in this case we already have the inverse ready
            order_array_inverse = indices if ascending else indices[:--1]
        else:
            del indices
        order_array = indices_r if ascending else indices_r[::-1]
        logger.info("sorting done")

    if progress == True:
        progress = vaex.utils.progressbar_callable(title="exporting")
    progress = progress or (lambda value: True)
    progress_total = len(column_names) * len(dataset_input)
    progress_status = ProgressStatus()
    progress_status.cancelled = False
    progress_status.value = 0
    if selection:
        dataset_input.count(selection=selection)  # fill cache for filter and selection
    else:
        len(dataset_input)  # fill filter cache

    sparse_groups = collections.defaultdict(list)
    sparse_matrices = {}  # alternative to a set of matrices, since they are not hashable
    string_columns = []
    futures = []

    thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1) 
    if True:
        for column_name in column_names:
            sparse_matrix = dataset_output._sparse_matrix(column_name)
            if sparse_matrix is not None:
                # sparse columns are written differently
                sparse_groups[id(sparse_matrix)].append(column_name)
                sparse_matrices[id(sparse_matrix)] = sparse_matrix
                continue
            logger.debug("  exporting column: %s " % column_name)
            future = thread_pool.submit(_export_column, dataset_input, dataset_output, column_name,
                shuffle, sort, selection, N, order_array, order_array_inverse, progress_status, parallel=parallel)
            futures.append(future)

    done = False
    progress(0)
    while not done:
        done = True
        for future in futures:
            try:
                future.result(0.1/4)
            except concurrent.futures.TimeoutError:
                done = False
                break
        if not done:
            if not progress(progress_status.value / float(progress_total)):
                progress_status.cancelled = True
    if not progress_status.cancelled:
        progress(1)

    for sparse_matrix_id, column_names in sparse_groups.items():
        sparse_matrix = sparse_matrices[sparse_matrix_id]
        for column_name in column_names:
            assert not shuffle
            assert selection in [None, False]
            column = dataset_output.columns[column_name]
            column.matrix.data[:] = dataset_input.columns[column_name].matrix.data
            column.matrix.indptr[:] = dataset_input.columns[column_name].matrix.indptr
            column.matrix.indices[:] = dataset_input.columns[column_name].matrix.indices
    return column_names

def _export_column(dataset_input, dataset_output, column_name, shuffle, sort, selection, N,
    order_array, order_array_inverse, progress_status, parallel=True):

        if 1:
            to_array = dataset_output.columns[column_name]
            dtype = dataset_input.data_type(column_name)
            is_string = dtype.is_string
            if is_string:
                # assert isinstance(to_array, pa.Array)  # we don't support chunked arrays here
                # TODO legacy: we still use ColumnStringArrow to write, find a way to do this with arrow
                # this is the case with hdf5 and remote storage
                if not isinstance(to_array, ColumnStringArrow):
                    to_array = ColumnStringArrow.from_arrow(to_array)
            if shuffle or sort:  # we need to create a in memory copy, otherwise we will do random writes which is VERY inefficient
                to_array_disk = to_array
                if np.ma.isMaskedArray(to_array):
                    to_array = np.empty_like(to_array_disk)
                else:
                    if vaex.array_types.is_string_type(dtype):
                        # we create an empty column copy
                        to_array = to_array._zeros_like()
                    else:
                        to_array = np.zeros_like(to_array_disk)
            to_offset = 0  # we need this for selections
            to_offset_unselected = 0 # we need this for filtering
            count = len(dataset_input)# if not selection else dataset_input.length_unfiltered()
            # TODO: if no filter, selection or mask, we can choose the quick path for str
            string_byte_offset = 0

            for i1, i2, values in dataset_input.evaluate(column_name, chunk_size=max_length, filtered=True, parallel=parallel, selection=selection, array_type='numpy-arrow'):
                logger.debug("from %d to %d (total length: %d, output length: %d)", i1, i2, len(dataset_input), N)
                no_values = len(values)
                if no_values:
                    if is_string:
                        # for strings, we don't take sorting/shuffling into account when building the structure
                        to_column = to_array
                        from_sequence = _to_string_sequence(values)
                        to_sequence = to_column.string_sequence.slice(to_offset, to_offset+no_values, string_byte_offset)
                        string_byte_offset += to_sequence.fill_from(from_sequence)
                        to_offset += no_values
                    else:
                        fill_value = np.nan if dtype.kind == "f" else None
                        # assert np.ma.isMaskedArray(to_array) == np.ma.isMaskedArray(values), "to (%s) and from (%s) array are not of both masked or unmasked (%s)" %\
                        # (np.ma.isMaskedArray(to_array), np.ma.isMaskedArray(values), column_name)
                        if shuffle or sort:
                            target_set_item = order_array[i1:i2]
                        else:
                            target_set_item = slice(to_offset, to_offset + no_values)
                        if dtype.is_datetime:
                            values = values.view(np.int64)
                        if np.ma.isMaskedArray(to_array) and np.ma.isMaskedArray(values):
                            to_array.data[target_set_item] = values.filled(fill_value)
                            to_array.mask[target_set_item] = values.mask
                        elif not np.ma.isMaskedArray(to_array) and np.ma.isMaskedArray(values):
                            to_array[target_set_item] = values.filled(fill_value)
                        else:
                            to_array[target_set_item] = values
                        to_offset += no_values

                with progress_lock:
                    progress_status.value += i2 - i1
                if progress_status.cancelled:
                    break
                #if not progress(progress_value / float(progress_total)):
                #    break
            if is_string:  # write out the last index
                to_column = to_array
                if selection:
                    to_column.indices[to_offset] = string_byte_offset
                else:
                    to_column.indices[count] = string_byte_offset
            if shuffle or sort:  # write to disk in one go
                if is_string:  # strings are sorted afterwards
                    view = to_array.string_sequence.lazy_index(order_array_inverse)
                    to_array_disk.string_sequence.fill_from(view)
                else:
                    if np.ma.isMaskedArray(to_array) and np.ma.isMaskedArray(to_array_disk):
                        to_array_disk.data[:] = to_array.data
                        to_array_disk.mask[:] = to_array.mask
                    else:
                        to_array_disk[:] = to_array


def export_hdf5_v1(dataset, path, column_names=None, byteorder="=", shuffle=False, selection=False, progress=None, virtual=True):
    kwargs = locals()
    import vaex.hdf5.export
    vaex.hdf5.export.export_hdf5_v1(**kwargs)


def export_hdf5(dataset, path, column_names=None, byteorder="=", shuffle=False, selection=False, progress=None, virtual=True, sort=None, ascending=True, parallel=True):
    kwargs = locals()
    import vaex.hdf5.export
    vaex.hdf5.export.export_hdf5(**kwargs)


if __name__ == "__main__":
    sys.exit(main(sys.argv))


