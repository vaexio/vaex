__author__ = 'breddels'
# due to 32 bit limitations in numpy, we cannot use astropy's fits module for writing colfits
import sys
import math

import vaex.dataset
import astropy.io.fits
import numpy as np
np.linspace()

def write_colfits(dataset, path, selection=False):
	f = open(path, "wb")
	class Scope(object):
		pass

	vars = Scope()
	#vars.
	def write(key, value, comment=""):
		f.write("{key:8}= {value:20} / {comment:47}".format(key=key, value=value, comment=comment))
		print "at pos", f.tell()
	def finish_header():
		print f.write("{end:80}".format(end="END"))
		offset = f.tell()
		bytes_over_padding = offset % 2880
		print "bytes_over_padding", bytes_over_padding
		if bytes_over_padding > 0:
			padding = 2880 - bytes_over_padding
			f.write(" "*padding)
	def finish_data():
		offset = f.tell()
		bytes_over_padding = offset % 2880
		if bytes_over_padding > 0:
			padding = 2880 - bytes_over_padding
			f.write("\0"*padding)


	write("SIMPLE", "T", "file conforms to FITS standard")
	write("BITPIX", 8, "number of bits per data pixel")
	write("NAXIS", 0, "number of array dimensions")
	finish_header()
	write("XTENSION", repr("BINTABLE"), "binary table extension")

	write("BITPIX", 8, "number of bits per data pixel")
	write("NAXIS", 2, "number of array dimensions")
	write("NAXIS1", dataset.byte_size(selection=selection), "length of dim 1")
	write("NAXIS2", 1, "length of dim 2")

	write("TFIELDS", len(dataset.column_names), "number of columns")
	for i, column_name in enumerate(dataset.column_names):
		i += 1 # 1 based index
		column = dataset.columns[column_name]
		write("TTYPE%d" % i, repr(str(column_name)), "column name %i" % (i))
		numpy_type_name = column.dtype.descr[0][1][1:] # i4, f8 etc
		fits_type = astropy.io.fits.column.NUMPY2FITS[numpy_type_name]
		# TODO: support rank1 arrays
		write("TFORM%d" % i , repr("{length}{type}".format(length=len(dataset), type=fits_type)), "")
		write("TDIM%d" % i, repr("(1,{length})".format(length=len(dataset))), "")

	finish_header()

	for i, column_name in enumerate(dataset.column_names):
		column = dataset.columns[column_name]
		numpy_type_name = column.dtype.descr[0][1][1:] # i4, f8 etc
		fits_type = astropy.io.fits.column.NUMPY2FITS[numpy_type_name]
		chunk_size = 1024**2 # 1 meg at a time
		chunks = int(math.ceil(len(dataset)/float(chunk_size)))
		for i in range(chunks):
			i1 = i * chunk_size
			i2 = min(len(dataset), (i+1) * chunk_size)
			data_big_endian = column[i1:i2].astype(">" + numpy_type_name)
			f.write(data_big_endian)
		print f.tell(), f.tell() / 1024**2, "mb", len(dataset)
		assert i2 == len(dataset)
	finish_data()

	f.close()



if __name__ == "__main__":
	input = sys.argv[1]
	output = sys.argv[2]
	dataset_in = vaex.dataset.load_file(input)
	write_colfits(dataset_in, output)