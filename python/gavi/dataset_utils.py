# -*- coding: utf-8 -*-
#from gavi import dataset
import gavi.dataset
from optparse import OptionParser
from mab.utils.progressbar import ProgressBar
import sys
import h5py
import numpy as np


def merge(output_filename, datasets, sort_property=None, order_column_name=None, ascending=True):
	datasets = list(datasets)
	if sort_property:
		datasets.sort(key=lambda dataset: dataset.variables[sort_property], reverse=not ascending)
	h5output = h5py.File(output_filename, "w")
	dataset = datasets[0]
	max_length = max(len(dataset) for dataset in datasets)
	shape = (len(datasets), max_length)
	print "shape of new arrays will be", shape
	if 0:
		for dataset1 in datasets:
			for dataset2 in datasets:
				if dataset1 != dataset2:
					if len(dataset1) != len(dataset2):
						print dataset1.name, "is of length", len(dataset1), "but", dataset2.name, "is of length", len(dataset2)
						sys.exit(1)
		
	for column_name in dataset.column_names:
		d = h5output.require_dataset("/columns/"+column_name, shape=shape, dtype=dataset.columns[column_name].dtype, exact=True)
		d[0,0] = dataset.columns[column_name][0] # ensure the array exists
	# each float propery will be a new axis in the merged file (TODO: int and other types?)
	for property_name in dataset.variables.keys():
		property = dataset.variables[property_name]
		if isinstance(property, (float,)):
			d = h5output.require_dataset("/axes/"+property_name, shape=(len(datasets),), dtype=np.float64, exact=True)
			d[0] = 0. # make sure it exists
	# close file and open it again with our interface
	h5output.close()
	dataset_output = gavi.dataset.Hdf5MemoryMapped(output_filename, write=True)
		
	progressBar = ProgressBar(0, len(datasets)-1)

	if 0:
		idmap = {}
		for index, dataset in enumerate(datasets):
			ids = dataset.columns["ParticleIDs"]
			for id in ids:
				idmap[id] = None
		used_ids = idmap.keys()
		print sorted(used_ids)
	
	for index, dataset in enumerate(datasets):
		for column_name in dataset.column_names:
			column_output = dataset_output.rank1s[column_name]
			column_input = dataset.columns[column_name]
			if order_column_name:
				order_column = dataset.columns[order_column_name]
			else:
				order_column = None
			#print dataset.name, order_column, order_column-order_column.min()
			if order_column is not None:
				column_output[index,order_column-order_column.min()] = column_input[:]
			else:
				column_output[index,:] = np.nan
				column_output[index,:len(dataset)] = column_input[:]
		for property_name in dataset.variables.keys():
			property = dataset.variables[property_name]
			if isinstance(property, (float,)):
				#print "propery ignored: %r" % property
				#print "propery set: %s %r" % (property_name, property)
				dataset_output.axes[property_name][index] = property
			else:
				#print "propery ignored: %s %r" % (property_name, property)
				pass
		#print "one file"
		progressBar.update(index)
		
	

	
	
if __name__ == "__main__":
	usage = "use the source luke!"
	parser = OptionParser(usage=usage)

	#parser.add_option("-n", "--name",
	#                 help="dataset name [default=%default]", default="data", type=str)
	parser.add_option("-o", "--order", default=None, help="rows in the input file are ordered by this column (For gadget: ParticleID)")
	parser.add_option("-f", "--format", default=None, help="file format")
	#parser.add_option("-i", "--ignore", default=None, help="ignore errors while loading files")
	parser.add_option("-r", "--reverse", action="store_true", default=False, help="reverse sorting")
	parser.add_option("-s", "--sort",
	                 help="sort datasets by propery [by default it will be the file order]", default=None, type=str)
	(options, args) = parser.parse_args()
	inputs = args[:-1]
	output = args[-1]
	print "merging:", "\n\t".join(inputs)
	print "to:", output
	if options.format is None:
		print "specify format"
		parser.print_help()
		sys.exit(1)
	dataset_type_and_options = options.format.split(":")
	dataset_type, dataset_options = dataset_type_and_options[0], dataset_type_and_options[1:]
	if dataset_type not in gavi.dataset.dataset_type_map:
		print "unknown type", dataset_type
		print "possible options are:\n\t", "\n\t".join(gavi.dataset.dataset_type_map.keys())
		sys.exit(1)
	evaluated_options = []
	for option in dataset_options:
		evaluated_options.append(eval(option))
		
	class_ = gavi.dataset.dataset_type_map[dataset_type]
	#@datasets = [class_(filename, *options) for filename in inputs]
	datasets = []
	for filename in inputs:
		print "loading file", filename
		datasets.append(class_(filename, *evaluated_options))

	merge(output, datasets, options.sort, ascending=not options.reverse, order_column_name=options.order)
	
	