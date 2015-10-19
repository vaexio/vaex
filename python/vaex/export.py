__author__ = 'maartenbreddels'
import numpy as np
import h5py
import logging
import vaex
import vaex.utils
#from vaex.dataset import DatasetLocal

logger = logging.getLogger("vaex.export")

def export_hdf5(dataset, path, column_names=None, byteorder="=", shuffle=False, selection=False, progress=None):
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
	h5file_output = h5py.File(path, "w")

	progress = progress or (lambda value: True)

	h5data_output = h5file_output.require_group("data")
	#i1, i2 = dataset.current_slice
	N = len(dataset) if not selection else dataset.selected_length()
	if N == 0:
		raise ValueError, "Cannot export empty table"
	logger.debug("exporting %d rows to file %s" % (N, path))
	column_names = column_names or dataset.get_column_names()
	logger.debug(" exporting columns: %r" % column_names)
	for column_name in column_names:
		column = dataset.columns[column_name]
		#print column_name, column.shape, column.strides
		#print column_name, column.dtype, column.dtype.type
		shape = (N,) + column.shape[1:]
		array = h5file_output.require_dataset("/data/%s" % column_name, shape=shape, dtype=column.dtype.newbyteorder(byteorder))
		array[0] = array[0] # make sure the array really exists
	if shuffle:
		random_index_name = "random_index"
		while random_index_name in dataset.get_column_names():
			random_index_name += "_new"
		shuffle_array = h5file_output.require_dataset("/data/" + random_index_name, shape=(N,), dtype=byteorder+"i8")
		shuffle_array[0] = shuffle_array[0]

	# close file, and reopen it using out class
	h5file_output.close()
	dataset_output = vaex.dataset.Hdf5MemoryMapped(path, write=True)

	if shuffle:
		shuffle_array = dataset_output.columns["random_index"]


	partial_shuffle = shuffle and dataset.full_length() != len(dataset)

	if partial_shuffle:
		# if we only export a portion, we need to create the full length random_index array, and
		shuffle_array_full = np.zeros(dataset.full_length(), dtype=byteorder+"i8")
		vaex.vaexfast.shuffled_sequence(shuffle_array_full)
		# then take a section of it
		shuffle_array[:] = shuffle_array_full[:N]
		del shuffle_array_full
	elif shuffle:
		vaex.vaexfast.shuffled_sequence(shuffle_array)

	i1, i2 = 0, N #len(dataset)
	#print "creating shuffled array"
	progress_total = len(column_names)
	progress_value = 0
	for column_name in column_names:
		logger.debug("  exporting column: %s " % column_name)
		with vaex.utils.Timer("copying column %s" % column_name, logger):
			from_array = dataset.columns[column_name][:] # TODO: horribly inefficient for concatenated tables
			to_array = dataset_output.columns[column_name]
			#np.take(from_array, random_index, out=to_array)
			#print [(k.shape, k.dtype) for k in [from_array, to_array, random_index]]
			if selection:
				if dataset.mask is not None:
					data = from_array[0:len(dataset)][dataset.mask]
					to_array[:] = data
			else:
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
			#copy(, to_array, random_index)
		progress_value += 1
		if not progress(progress_value/float(progress_total)):
			break

import math
import sys
def batch_copy_index(from_array, to_array, shuffle_array):
	N_per_batch = int(1e7)
	length = len(from_array)
	batches = long(math.ceil(float(length)/N_per_batch))
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

