__author__ = 'maartenbreddels'
import os
import numpy as np
import logging
import vaex
import vaex.utils
import vaex.execution
import vaex.export
import vaex.hdf5.dataset

max_length = int(1e5)

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
try:
    import h5py
except:
    if not on_rtd:
        raise

logger = logging.getLogger("vaex.hdf5.export")


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
            if column_name in dataset.get_column_names(strings=True):
                column = dataset.columns[column_name]
                shape = (N,) + column.shape[1:]
                dtype = column.dtype
            else:
                dtype = np.float64().dtype
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

    column_names = vaex.export._export(dataset_input=dataset,
                                       dataset_output=dataset_output,
                                       path=path,
                                       random_index_column=random_index_name,
                                       column_names=column_names,
                                       selection=selection,
                                       shuffle=shuffle,
                                       byteorder=byteorder,
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
    dataset_output.close_files()
    return


def export_hdf5(dataset, path, column_names=None, byteorder="=", shuffle=False, selection=False, progress=None, virtual=True, sort=None, ascending=True):
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
        for column_name in column_names:
            if column_name in dataset.get_column_names(strings=True):
                column = dataset.columns[column_name]
                shape = (N,) + column.shape[1:]
                dtype = column.dtype
            else:
                dtype = np.float64().dtype
                shape = (N,)
            h5column_output = h5columns_output.require_group(column_name)
            if dtype.type == np.datetime64:
                array = h5column_output.require_dataset('data', shape=shape, dtype=np.int64)
                array.attrs["dtype"] = dtype.name
            else:
                try:
                    array = h5column_output.require_dataset('data', shape=shape, dtype=dtype.newbyteorder(byteorder))
                except:
                    logging.exception("error creating dataset for %r, with type %r " % (column_name, dtype))
            array[0] = array[0]  # make sure the array really exists

            data = dataset.evaluate(column_name, 0, 1)
            if np.ma.isMaskedArray(data):
                mask = h5column_output.require_dataset('mask', shape=shape, dtype=np.bool)
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

    # after this the file is closed,, and reopen it using out class
    dataset_output = vaex.hdf5.dataset.Hdf5MemoryMapped(path, write=True)

    column_names = vaex.export._export(dataset_input=dataset,
                                       dataset_output=dataset_output,
                                       path=path,
                                       random_index_column=random_index_name,
                                       column_names=column_names,
                                       selection=selection,
                                       shuffle=shuffle,
                                       byteorder=byteorder,
                                       progress=progress,
                                       sort=sort,
                                       ascending=ascending)
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
    dataset_output.close_files()
    return
