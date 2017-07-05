__author__ = 'breddels'
# due to 32 bit limitations in numpy, we cannot use astropy's fits module for writing colfits
import sys
import math

import vaex.dataset
import astropy.io.fits
import numpy as np
import logging

logger = logging.getLogger("vaex.file.colfits")

def empty(filename, length, column_names, data_types, data_shapes, ucds, units, null_values={}):
	with open(filename, "wb") as f:
		logger.debug("preparing empty fits file: %s", filename)
		class Scope(object):
			pass

		def write(key, value, comment=""):
			first_part = "{key:8}= {value:20} / ".format(key=key, value=value)
			f.write(first_part.encode("ascii"))
			leftover = 80 - len(first_part)
			f.write(("{comment:"+str(leftover) +"}").format(comment=comment).encode("ascii"))
			logger.debug("at pos: %s", f.tell())
		def finish_header():
			f.write("{end:80}".format(end="END").encode("ascii"))
			offset = f.tell()
			bytes_over_padding = offset % 2880
			logger.debug(("bytes_over_padding: %s", bytes_over_padding))
			if bytes_over_padding > 0:
				padding = 2880 - bytes_over_padding
				f.write((" "*padding).encode("ascii"))
		def finish_data():
			offset = f.tell()
			bytes_over_padding = offset % 2880
			if bytes_over_padding > 0:
				padding = 2880 - bytes_over_padding
				f.write(("\0"*padding).encode("ascii"))

		byte_size = sum([length * type.itemsize for type in data_types])

		write("SIMPLE", "T", "file conforms to FITS standard")
		write("BITPIX", 8, "number of bits per data pixel")
		write("NAXIS", 0, "number of array dimensions")
		finish_header()
		write("XTENSION", repr("BINTABLE"), "binary table extension")

		write("BITPIX", 8, "number of bits per data pixel")
		write("NAXIS", 2, "number of array dimensions")
		write("NAXIS1", byte_size, "length of dim 1")
		write("NAXIS2", 1, "length of dim 2")
		write("PCOUNT", 0, "number of group parameters")
		write("GCOUNT", 1, "number of groups")

		write("TFIELDS", len(column_names), "number of columns")
		for i, (column_name, type, shape) in enumerate(zip(column_names, data_types, data_shapes)):
			i += 1 # 1 based index
			#column = dataset.columns[column_name]
			write("TTYPE%d" % i, repr(str(column_name)), "column name %i" % (i))
			numpy_type_name = type.descr[0][1][1:] # i4, f8 etc
			if numpy_type_name[0] == 'S':
				string_length = numpy_type_name[1:]
				fits_type = str(int(string_length)*length)+"A"
				logger.debug("type for %s: numpy=%r, fits=%r, string_length=%r length=%r", column_name, numpy_type_name, fits_type, string_length, length)
				# TODO: support rank1 arrays
				write("TFORM%d" % i , repr("{type}".format(type=fits_type)), "")
				write("TDIM%d" % i, repr("({string_length},{length})".format(string_length=string_length, length=length)), "")
			else:
				fits_type = astropy.io.fits.column.NUMPY2FITS[numpy_type_name]
				logger.debug("type for %s: numpy=%r, fits=%r", column_name, numpy_type_name, fits_type)
				# TODO: support rank1 arrays
				write("TFORM%d" % i , repr("{length}{type}".format(length=length, type=fits_type)), "")
				write("TDIM%d" % i, repr("(1,{length})".format(length=length)), "")
			ucd = ucds[i-1]
			if ucd:
				write("TUCD%d" % i, repr(str(ucd)))
			unit = units[i-1]
			if unit:
				write("TUNIT%d" % i, repr(str(unit)))
			if column_name in null_values:
				write("TNULL%d" % i, str(null_values[column_name]))

		finish_header()

		for i, (column_name, type, shape) in enumerate(zip(column_names, data_types, data_shapes)):
			byte_size = length * type.itemsize
			f.seek(f.tell() + byte_size)
		finish_data()

def write_colfits(dataset, path, selection=False):
	with open(path, "wb") as f:
		class Scope(object):
			pass

		vars = Scope()
		#vars.
		def write(key, value, comment=""):
			f.write("{key:8}= {value:20} / {comment:47}".format(key=key, value=value, comment=comment))
			print(("at pos", f.tell()))
		def finish_header():
			print((f.write("{end:80}".format(end="END"))))
			offset = f.tell()
			bytes_over_padding = offset % 2880
			print(("bytes_over_padding", bytes_over_padding))
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
			print((f.tell(), f.tell() / 1024**2, "mb", len(dataset)))
			assert i2 == len(dataset)
		finish_data()



if __name__ == "__main__":
	input = sys.argv[1]
	output = sys.argv[2]
	dataset_in = vaex.dataset.load_file(input)
	write_colfits(dataset_in, output)