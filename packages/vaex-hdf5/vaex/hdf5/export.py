from __future__ import division, print_function
__author__ = 'maartenbreddels'
import os
import sys
import collections
import numpy as np
import logging
import vaex
import vaex.utils
import vaex.execution
import vaex.export
import vaex.hdf5.dataset
from vaex.column import ColumnStringArrow

max_length = int(1e5)

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
try:
    import h5py
except:
    if not on_rtd:
        raise

logger = logging.getLogger("vaex.hdf5.export")

max_int32 = 2**31-1

def export_hdf5_v1(dataset, path, column_names=None, byteorder="=", shuffle=False, selection=False, progress=None, virtual=True):
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

    if selection:
        if selection == True:  # easier to work with the name
            selection = "default"
    # first open file using h5py api
    with h5py.File(path, "w") as h5file_output:

        h5data_output = h5file_output.require_group("data")
        # i1, i2 = dataset.current_slice
        N = len(dataset) if not selection else dataset.selected_length(selection)
        if N == 0:
            raise ValueError("Cannot export empty table")
        logger.debug("virtual=%r", virtual)
        logger.debug("exporting %d rows to file %s" % (N, path))
        # column_names = column_names or (dataset.get_column_names() + (list(dataset.virtual_columns.keys()) if virtual else []))
        column_names = column_names or dataset.get_column_names(virtual=virtual, strings=True)

        logger.debug("exporting columns(hdf5): %r" % column_names)
        for column_name in column_names:
            dtype = dataset.data_type(column_name)
            if column_name in dataset.get_column_names(strings=True):
                column = dataset.columns[column_name]
                shape = (N,) + column.shape[1:]
            else:
                shape = (N,)
            if dtype.type == np.datetime64:
                array = h5file_output.require_dataset("/data/%s" % column_name, shape=shape, dtype=np.int64)
                array.attrs["dtype"] = dtype.name
            else:
                try:
                    array = h5file_output.require_dataset("/data/%s" % column_name, shape=shape, dtype=dtype.newbyteorder(byteorder))
                except:
                    logging.exception("error creating dataset for %r, with type %r " % (column_name, dtype))
            array[0] = array[0]  # make sure the array really exists
        random_index_name = None
        column_order = list(column_names)  # copy
        if shuffle:
            random_index_name = "random_index"
            while random_index_name in dataset.get_column_names():
                random_index_name += "_new"
            shuffle_array = h5file_output.require_dataset("/data/" + random_index_name, shape=(N,), dtype=byteorder + "i8")
            shuffle_array[0] = shuffle_array[0]
            column_order.append(random_index_name)  # last item
        h5data_output.attrs["column_order"] = ",".join(column_order)  # keep track or the ordering of columns

    # after this the file is closed,, and reopen it using out class
    dataset_output = vaex.hdf5.dataset.Hdf5MemoryMapped(path, write=True)

    column_names = vaex.export._export(dataset_input=dataset, dataset_output=dataset_output, path=path, random_index_column=random_index_name,
                                       column_names=column_names, selection=selection, shuffle=shuffle, byteorder=byteorder,
                                       progress=progress)
    import getpass
    import datetime
    user = getpass.getuser()
    date = str(datetime.datetime.now())
    source = dataset.path
    description = "file exported by vaex, by user %s, on date %s, from source %s" % (user, date, source)
    if dataset.description:
        description += "previous description:\n" + dataset.description
    dataset_output.copy_metadata(dataset)
    dataset_output.description = description
    logger.debug("writing meta information")
    dataset_output.write_meta()
    dataset_output.close()
    return


def export_hdf5(dataset, path, column_names=None, byteorder="=", shuffle=False, selection=False, progress=None, virtual=True, sort=None, ascending=True, parallel=True):
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

    if selection:
        if selection == True:  # easier to work with the name
            selection = "default"
    # first open file using h5py api
    with h5py.File(path, "w") as h5file_output:

        h5table_output = h5file_output.require_group("/table")
        h5table_output.attrs["type"] = "table"
        h5columns_output = h5file_output.require_group("/table/columns")
        # i1, i2 = dataset.current_slice
        N = len(dataset) if not selection else dataset.selected_length(selection)
        if N == 0:
            raise ValueError("Cannot export empty table")
        logger.debug("virtual=%r", virtual)
        logger.debug("exporting %d rows to file %s" % (N, path))
        # column_names = column_names or (dataset.get_column_names() + (list(dataset.virtual_columns.keys()) if virtual else []))
        column_names = column_names or dataset.get_column_names(virtual=virtual, strings=True)

        logger.debug("exporting columns(hdf5): %r" % column_names)
        sparse_groups = collections.defaultdict(list)
        sparse_matrices = {}  # alternative to a set of matrices, since they are not hashable
        for column_name in list(column_names):
            sparse_matrix = dataset._sparse_matrix(column_name)
            if sparse_matrix is not None:
                # sparse columns are stored differently
                sparse_groups[id(sparse_matrix)].append(column_name)
                sparse_matrices[id(sparse_matrix)] = sparse_matrix
                continue
            dtype = dataset.data_type(column_name)
            shape = (N, ) + dataset._shape_of(column_name)[1:]
            h5column_output = h5columns_output.require_group(column_name)
            if vaex.array_types.is_string_type(dtype):
                # TODO: if no selection or filter, we could do this
                # if isinstance(column, ColumnStringArrow):
                #     data_shape = column.bytes.shape
                #     indices_shape = column.indices.shape
                # else:

                byte_length = dataset[column_name].str.byte_length().sum(selection=selection)
                if byte_length > max_int32:
                    dtype_indices = 'i8'
                else:
                    dtype_indices = 'i4'

                data_shape = (byte_length, )
                indices_shape = (N+1, )

                array = h5column_output.require_dataset('data', shape=data_shape, dtype='S1')
                if byte_length > 0:
                    array[0] = array[0]  # make sure the array really exists

                index_array = h5column_output.require_dataset('indices', shape=indices_shape, dtype=dtype_indices)
                index_array[0] = index_array[0]  # make sure the array really exists

                null_value_count = N - dataset.count(dataset[column_name], selection=selection)
                if null_value_count > 0:
                    null_shape = ((N + 7) // 8, )  # TODO: arrow requires padding right?
                    null_bitmap_array = h5column_output.require_dataset('null_bitmap', shape=null_shape, dtype='u1')
                    null_bitmap_array[0] = null_bitmap_array[0]  # make sure the array really exists

                array.attrs["dtype"] = 'str'
                # TODO: masked support ala arrow?
            else:
                if dtype.kind in 'mM':
                    array = h5column_output.require_dataset('data', shape=shape, dtype=np.int64)
                    array.attrs["dtype"] = dtype.name
                elif dtype.kind == 'U':
                    # numpy uses utf32 for unicode
                    char_length = dtype.itemsize // 4
                    shape = (N, char_length)
                    array = h5column_output.require_dataset('data', shape=shape, dtype=np.uint8)
                    array.attrs["dtype"] = 'utf32'
                    array.attrs["dlength"] = char_length
                else:
                    try:
                        array = h5column_output.require_dataset('data', shape=shape, dtype=dtype.numpy.newbyteorder(byteorder))
                    except:
                        logging.exception("error creating dataset for %r, with type %r " % (column_name, dtype))
                        del h5columns_output[column_name]
                        column_names.remove(column_name)
                        continue
                array[0] = array[0]  # make sure the array really exists

                data = dataset.evaluate(column_name, 0, 1, parallel=False)
                if dataset.is_masked(column_name):
                    mask = h5column_output.require_dataset('mask', shape=shape, dtype=bool)
                    mask[0] = mask[0]  # make sure the array really exists
        random_index_name = None
        column_order = list(column_names)  # copy
        if shuffle:
            random_index_name = "random_index"
            while random_index_name in dataset.get_column_names():
                random_index_name += "_new"
            shuffle_array = h5columns_output.require_dataset(random_index_name + "/data", shape=(N,), dtype=byteorder + "i8")
            shuffle_array[0] = shuffle_array[0]
            column_order.append(random_index_name)  # last item
        h5columns_output.attrs["column_order"] = ",".join(column_order)  # keep track or the ordering of columns

        sparse_index = 0
        for sparse_matrix in sparse_matrices.values():
            columns = sorted(sparse_groups[id(sparse_matrix)], key=lambda col: dataset.columns[col].column_index)
            name = "sparse" + str(sparse_index)
            sparse_index += 1
            # TODO: slice columns
            # sparse_matrix = sparse_matrix[:,]
            sparse_group = h5columns_output.require_group(name)
            sparse_group.attrs['type'] = 'csr_matrix'
            ar = sparse_group.require_dataset('data', shape=(len(sparse_matrix.data), ), dtype=sparse_matrix.dtype)
            ar[0] = ar[0]
            ar = sparse_group.require_dataset('indptr', shape=(len(sparse_matrix.indptr), ), dtype=sparse_matrix.indptr.dtype)
            ar[0] = ar[0]
            ar = sparse_group.require_dataset('indices', shape=(len(sparse_matrix.indices), ), dtype=sparse_matrix.indices.dtype)
            ar[0] = ar[0]
            for i, column_name in enumerate(columns):
                h5column = sparse_group.require_group(column_name)
                h5column.attrs['column_index'] = i

    # after this the file is closed,, and reopen it using out class
    dataset_output = vaex.hdf5.dataset.Hdf5MemoryMapped(path, write=True)
    df_output = vaex.dataframe.DataFrameLocal(dataset_output)

    column_names = vaex.export._export(dataset_input=dataset, dataset_output=df_output, path=path, random_index_column=random_index_name,
                                       column_names=column_names, selection=selection, shuffle=shuffle, byteorder=byteorder,
                                       progress=progress, sort=sort, ascending=ascending, parallel=parallel)
    dataset_output._freeze()
    # We aren't really making use of the metadata, we could put this back in some form in the future
    # import getpass
    # import datetime
    # user = getpass.getuser()
    # date = str(datetime.datetime.now())
    # source = dataset.path
    # description = "file exported by vaex, by user %s, on date %s, from source %s" % (user, date, source)
    # if dataset.description:
    #     description += "previous description:\n" + dataset.description
    # dataset_output.copy_metadata(dataset)
    # dataset_output.description = description
    # logger.debug("writing meta information")
    # dataset_output.write_meta()
    dataset_output.close()
    return
