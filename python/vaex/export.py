__author__ = 'maartenbreddels'
import numpy as np
import h5py
from . import logging
import vaex
import vaex.utils
import vaex.execution
import vaex.io.colfits

#from vaex.dataset import DatasetLocal

logger = logging.getLogger("vaex.export")

def _export(dataset_input, dataset_output, random_index_column, path, column_names=None, byteorder="=", shuffle=False, selection=False, progress=None, virtual=True):
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
		assert dataset_input.has_selection(), "cannot export selection is there is none"

	N = len(dataset_input) if not selection else dataset_input.selected_length()
	if N == 0:
		raise ValueError("Cannot export empty table")

	if shuffle:
		shuffle_array = dataset_output.columns[random_index_column]


	partial_shuffle = shuffle and dataset_input.full_length() != len(dataset_input)

	if partial_shuffle:
		# if we only export a portion, we need to create the full length random_index array, and
		shuffle_array_full = np.zeros(dataset_input.full_length(), dtype=byteorder+"i8")
		vaex.vaexfast.shuffled_sequence(shuffle_array_full)
		# then take a section of it
		#shuffle_array[:] = shuffle_array_full[:N]
		shuffle_array[:] = shuffle_array_full[shuffle_array_full<N]
		del shuffle_array_full
	elif shuffle:
		# better to do this in memory
		shuffle_array_memory = np.zeros_like(shuffle_array)
		vaex.vaexfast.shuffled_sequence(shuffle_array_memory)
		shuffle_array[:] = shuffle_array_memory

	i1, i2 = 0, N #len(dataset)
	#print "creating shuffled array"
	progress_total = len(column_names) * N
	progress_value = 0
	for column_name in column_names:
		logger.debug("  exporting column: %s " % column_name)
		with vaex.utils.Timer("copying column %s" % column_name, logger):
			block_scope = dataset_input._block_scope(0, vaex.execution.buffer_size)
			to_array = dataset_output.columns[column_name]
			if shuffle: # we need to create a in memory copy, otherwise we will do random writes which is VERY inefficient
				to_array_disk = to_array
				to_array = np.zeros_like(to_array_disk)
			to_offset = 0 # we need this for selections
			for i1, i2 in vaex.utils.subdivide(len(dataset_input), max_length=vaex.execution.buffer_size):
				block_scope.move(i1, i2)

				values = block_scope.evaluate(column_name)

				if selection:
					selection_block_length = np.sum(dataset_input.mask[i1:i2])
					to_array[to_offset:to_offset+selection_block_length] = values[dataset_input.mask[i1:i2]]
					to_offset += selection_block_length
				else:
					if shuffle:
						indices = shuffle_array[i1:i2]
						to_array[indices] = values
					else:
						to_array[i1:i2] = values

				progress_value += i2-i1
				if not progress(progress_value/float(progress_total)):
					break
			if shuffle: # write to disk in one go
				to_array_disk[:] = to_array

def export_fits(dataset, path, column_names=None, shuffle=False, selection=False, progress=None, virtual=True):
	"""
	:param DatasetLocal dataset: dataset to export
	:param str path: path for file
	:param lis[str] column_names: list of column names to export or None for all columns
	:param bool shuffle: export rows in random order
	:param bool selection: export selection or not
	:param progress: progress callback that gets a progress fraction as argument and should return True to continue
	:return:
	"""
	if shuffle:
		random_index_name = "random_index"
		while random_index_name in dataset.get_column_names():
			random_index_name += "_new"

	column_names = column_names or (dataset.get_column_names() + (list(dataset.virtual_columns.keys()) if virtual else []))
	N = len(dataset) if not selection else dataset.selected_length()
	data_types = []
	data_shapes = []
	for column_name in column_names:
		if column_name in dataset.get_column_names():
			column = dataset.columns[column_name]
			shape = (N,) + column.shape[1:]
			dtype = column.dtype
		else:
			dtype = np.float64().dtype
			shape = (N,)
		data_types.append(dtype)
		data_shapes.append(shape)

	if shuffle:
		column_names.append(random_index_name)
		data_types.append(np.int64().dtype)
		data_shapes.append((N,))
	else:
		random_index_name = None

	vaex.io.colfits.empty(path, N, column_names, data_types, data_shapes)
	if shuffle:
		del column_names[-1]
		del data_types[-1]
		del data_shapes[-1]
	dataset_output = vaex.dataset.FitsBinTable(path, write=True)
	progress = progress or (lambda value: True)
	_export(dataset_input=dataset, dataset_output=dataset_output, path=path, random_index_column=random_index_name,
			column_names=column_names, selection=selection, shuffle=shuffle,
			progress=progress)

def export_hdf5(dataset, path, column_names=None, byteorder="=", shuffle=False, selection=False, progress=None, virtual=True):
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
		assert dataset.has_selection(), "cannot export selection is there is none"
	# first open file using h5py api
	with h5py.File(path, "w") as h5file_output:

		progress = progress or (lambda value: True)

		h5data_output = h5file_output.require_group("data")
		#i1, i2 = dataset.current_slice
		N = len(dataset) if not selection else dataset.selected_length()
		if N == 0:
			raise ValueError("Cannot export empty table")
		logger.debug("exporting %d rows to file %s" % (N, path))
		column_names = column_names or (dataset.get_column_names() + (list(dataset.virtual_columns.keys()) if virtual else []))
		logger.debug(" exporting columns: %r" % column_names)
		for column_name in column_names:
			if column_name in dataset.get_column_names():
				column = dataset.columns[column_name]
				shape = (N,) + column.shape[1:]
				dtype = column.dtype
			else:
				dtype = np.float64().dtype
				shape = (N,)
			array = h5file_output.require_dataset("/data/%s" % column_name, shape=shape, dtype=dtype.newbyteorder(byteorder))
			array[0] = array[0] # make sure the array really exists
		random_index_name = None
		if shuffle:
			random_index_name = "random_index"
			while random_index_name in dataset.get_column_names():
				random_index_name += "_new"
			shuffle_array = h5file_output.require_dataset("/data/" + random_index_name, shape=(N,), dtype=byteorder+"i8")
			shuffle_array[0] = shuffle_array[0]

	# after this the file is closed,, and reopen it using out class
	dataset_output = vaex.dataset.Hdf5MemoryMapped(path, write=True)

	_export(dataset_input=dataset, dataset_output=dataset_output, path=path, random_index_column=random_index_name,
			column_names=column_names, selection=selection, shuffle=shuffle, byteorder=byteorder,
			progress=progress)
	return
	if shuffle:
		shuffle_array = dataset_output.columns[random_index_name]


	partial_shuffle = shuffle and dataset.full_length() != len(dataset)

	if partial_shuffle:
		# if we only export a portion, we need to create the full length random_index array, and
		shuffle_array_full = np.zeros(dataset.full_length(), dtype=byteorder+"i8")
		vaex.vaexfast.shuffled_sequence(shuffle_array_full)
		# then take a section of it
		#shuffle_array[:] = shuffle_array_full[:N]
		shuffle_array[:] = shuffle_array_full[shuffle_array_full<N]
		del shuffle_array_full
	elif shuffle:
		# better to do this in memory
		shuffle_array_memory = np.zeros_like(shuffle_array)
		vaex.vaexfast.shuffled_sequence(shuffle_array_memory)
		shuffle_array[:] = shuffle_array_memory

	i1, i2 = 0, N #len(dataset)
	#print "creating shuffled array"
	progress_total = len(column_names) * N
	progress_value = 0
	for column_name in column_names:
		logger.debug("  exporting column: %s " % column_name)
		with vaex.utils.Timer("copying column %s" % column_name, logger):
			block_scope = dataset._block_scope(0, vaex.execution.buffer_size)
			to_array = dataset_output.columns[column_name]
			if shuffle: # we need to create a in memory copy, otherwise we will do random writes which is VERY inefficient
				to_array_disk = to_array
				to_array = np.zeros_like(to_array_disk)
			to_offset = 0 # we need this for selections
			for i1, i2 in vaex.utils.subdivide(len(dataset), max_length=vaex.execution.buffer_size):
				block_scope.move(i1, i2)

				#from_array = dataset.columns[column_name][i:] # TODO: horribly inefficient for concatenated tables
				#np.take(from_array, random_index, out=to_array)
				#print [(k.shape, k.dtype) for k in [from_array, to_array, random_index]]
				values = block_scope.evaluate(column_name)
				#print i1, i2, N
				if selection:
					selection_block_length = np.sum(dataset.mask[i1:i2])
					to_array[to_offset:to_offset+selection_block_length] = values[dataset.mask[i1:i2]]
					to_offset += selection_block_length
				else:
					if shuffle:
						indices = shuffle_array[i1:i2]
						to_array[indices] = values
					else:
						to_array[i1:i2] = values

				if False:
					if shuffle:
						#to_array[:] = from_array[i1:i2][shuffle_array]
						#to_array[:] = from_array[shuffle_array]
						#print [k.dtype for k in [from_array, to_array, shuffle_array]]
						#copy(from_array, to_array, shuffle_array)
						batch_copy_index(from_array, to_array, shuffle_array)
						#np.take(from_array, indices=shuffle_array, out=to_array)
						pass
					else:
						to_array[:] = from_array[i1:i2]
				progress_value += i2-i1
				if not progress(progress_value/float(progress_total)):
					break
			if shuffle: # write to disk in one go
				to_array_disk[:] = to_array

import math
import sys
def batch_copy_index(from_array, to_array, shuffle_array):
	N_per_batch = int(1e7)
	length = len(from_array)
	batches = int(math.ceil(float(length)/N_per_batch))
	#print np.sum(from_array)
	for i in range(batches):
		#print "batch", i, "out of", batches, ""
		sys.stdout.flush()
		i1 = i * N_per_batch
		i2 = min(length, (i+1)*N_per_batch)
		#print "reading...", i1, i2
		sys.stdout.flush()
		data = from_array[shuffle_array[i1:i2]]
		#print "writing..."
		sys.stdout.flush()
		to_array[i1:i2] = data

