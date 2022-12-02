import logging
import math
import os
import sys

import astropy.io.fits
import astropy.io.fits as fits
import numpy as np

import vaex.dataset
from vaex.dataset_misc import _try_unit
from vaex.dataset_mmap import DatasetMemoryMapped
import vaex.export
from vaex.utils import _python_save_name


logger = logging.getLogger("vaex.astro.fits")


class FitsBinTable(DatasetMemoryMapped):
    snake_name='fits'
    def __init__(self, filename, write=False, fs_options={}, fs=None):
        super(FitsBinTable, self).__init__(filename, write=write)
        self.ucds = {}
        self.units = {}
        with fits.open(filename) as fitsfile:
            for table in fitsfile:
                if isinstance(table, fits.BinTableHDU):
                    table_offset = table._data_offset
                    #import pdb
                    #pdb.set_trace()
                    if table.columns[0].dim is not None: # for sure not a colfits
                        dim = eval(table.columns[0].dim) # TODO: can we not do an eval here? not so safe
                        if len(dim) == 2 and dim[0] <= dim[1]: # we have colfits format
                            logger.debug("colfits file!")
                            offset = table_offset
                            for i in range(len(table.columns)):
                                column = table.columns[i]
                                cannot_handle = False
                                column_name = column.name.strip()
                                self._get_column_meta_data(table, column_name, column, i)


                                # flatlength == length * arraylength
                                flatlength, fitstype = int(column.format[:-1]),column.format[-1]
                                arraylength, length = arrayshape = eval(column.dim)

                                # numpy dtype code, like f8, i4
                                dtypecode = astropy.io.fits.column.FITS2NUMPY[fitstype]


                                dtype = np.dtype((">" +dtypecode, arraylength))
                                if 0:
                                    if arraylength > 1:
                                        dtype = np.dtype((">" +dtypecode, arraylength))
                                    else:
                                        if dtypecode == "a": # I think numpy needs by default a length 1
                                            dtype = np.dtype(dtypecode + "1")
                                        else:
                                            dtype = np.dtype(">" +dtypecode)
                                    #	bytessize = 8

                                bytessize = dtype.itemsize
                                logger.debug("%r", (column.name, dtype, column.format, column.dim, length, bytessize, arraylength))
                                if (flatlength > 0): # and dtypecode != "a": # TODO: support strings
                                    if dtypecode == "a": # for ascii, we need to add the length again..
                                        dtypecode += str(arraylength)
                                    logger.debug("column type: %r", (column.name, offset, dtype, length, column.format, column.dim))
                                    if arraylength == 1 or dtypecode[0] == "a":
                                        ar = self._map_array(offset=offset, dtype=dtype, shape=(length,))
                                        self.add_column(column_name, ar)
                                    else:
                                        for i in range(arraylength):
                                            name = column_name+"_" +str(i)
                                            self.add_column(name, offset=offset+bytessize*i//arraylength, dtype=">" +dtypecode, length=length, stride=arraylength)
                                if flatlength > 0: # flatlength can be
                                    offset += bytessize * length
                                self._check_null(table, column_name, column, i)

                    else:
                        logger.debug("adding table: %r" % table)
                        for i, column in enumerate(table.columns):
                            array = column.array[:]
                            array = column.array[:] # 2nd time it will be a real np array
                            #import pdb
                            #pdb.set_trace()
                            if array.dtype.kind in "fiubSU":
                                column_name = _python_save_name(column.name, used=self._columns.keys())
                                self.add_column(column_name, data=array)
                                self._get_column_meta_data(table, column_name, column, i)
                                self._check_null(table, column_name, column, i)
            self._try_votable(fitsfile[0])

        self._freeze()
        # self.update_meta()
        # self.update_virtual_meta()

    def _check_null(self, table, column_name, column, i):
        null_name = "TNULL%d" % (i+1)
        if null_name in table.header:
            mask_value = table.header[null_name]
            array = self._columns[column_name]
            mask = array == mask_value
            self._columns[column_name] = np.ma.masked_array(array, mask)

    def _try_votable(self, table):
        try:
            from io import BytesIO as StringIO
        except:
            from StringIO import StringIO
        if table.data is None:
            return
        vodata = table.data.tostring()
        if vodata.startswith(b"<?xml"):
            f = StringIO()
            f.write(vodata)
            votable = astropy.io.votable.parse(f)
            first_table = votable.get_first_table()
            used_names = []
            for field in first_table.fields:
                name = field.name.strip()
                clean_name = _python_save_name(name, used=used_names)
                used_names.append(name)
                if field.ucd:
                    self.ucds[clean_name] = field.ucd
                unit = _try_unit(field.unit)
                if unit:
                    self.units[clean_name] = unit
                if unit is None and field.unit:
                    print("unit error for: %r", field.unit)
                self.descriptions[clean_name] = field.description
            self.description = first_table.description

    def _get_column_meta_data(self, table, column_name, column, i):
        ucd_header_name = "TUCD%d" % (i+1)
        if ucd_header_name in table.header:
            self.ucds[column_name] = table.header[ucd_header_name]
        if column.unit:
            try:
                unit = _try_unit(column.unit)
                if unit:
                    self.units[column_name] = unit
            except:
                logger.exception("could not understand unit: %s" % column.unit)
        else: # we may want to try ourselves
            unit_header_name = "TUNIT%d" % (i+1)
            if unit_header_name in table.header:
                unit_str = table.header[unit_header_name]
                unit = _try_unit(unit_str)
                if unit:
                    self.units[column_name] = unit
        #unit_header_name = "TUCD%d" % (i+1)
        #if ucd_header_name in table.header:
    @classmethod
    def can_open(cls, path, *args, **kwargs):
        return os.path.splitext(path)[1] == ".fits"

    @classmethod
    def get_options(cls, path):
        return [] # future: support multiple tables?

    @classmethod
    def option_to_args(cls, option):
        return []


def empty(filename, length, column_names, data_types, data_shapes, ucds, units, null_values={}):
	with open(filename, "wb") as f:
		logger.debug("preparing empty fits file: %s", filename)
		class Scope(object):
			pass

		def write(key, value, comment=""):
			first_part = "{key:8}= {value:>20} / ".format(key=key, value=value)
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


def export_fits(dataset, path, column_names=None, shuffle=False, selection=False, progress=None, virtual=True, sort=None, ascending=True):
    """
    :param DatasetLocal dataset: dataset to export
    :param str path: path for file
    :param lis[str] column_names: list of column names to export or None for all columns
    :param bool shuffle: export rows in random order
    :param bool selection: export selection or not
    :param progress: progress callback that gets a progress fraction as argument and should return True to continue,
            or a default progress bar when progress=True
    :param: bool virtual: When True, export virtual columns
    :return:
    """
    if shuffle:
        random_index_name = "random_index"
        while random_index_name in dataset.get_column_names():
            random_index_name += "_new"

    column_names = column_names or dataset.get_column_names(virtual=virtual, strings=True)
    logger.debug("exporting columns(fits): %r" % column_names)
    N = len(dataset) if not selection else dataset.selected_length(selection)
    data_types = []
    data_shapes = []
    ucds = []
    units = []
    for column_name in column_names:
        if column_name in dataset.get_column_names(strings=True, virtual=False):
            column = dataset.columns[column_name]
            shape = (N,) + column.shape[1:]
            dtype = column.dtype
            if dataset.is_string(column_name):
                max_length = dataset[column_name].apply(lambda x: len(x)).max(selection=selection)
                dtype = np.dtype('S'+str(int(max_length)))
        else:
            dtype = np.float64().dtype
            shape = (N,)
        ucds.append(dataset.ucds.get(column_name))
        units.append(dataset.units.get(column_name))
        data_types.append(dtype)
        data_shapes.append(shape)

    if shuffle:
        column_names.append(random_index_name)
        data_types.append(np.int64().dtype)
        data_shapes.append((N,))
        ucds.append(None)
        units.append(None)
    else:
        random_index_name = None

    # TODO: all expressions can have missing values.. how to support that?
    null_values = {key: dataset.columns[key].fill_value for key in dataset.get_column_names() if dataset.is_masked(key) and dataset.data_type(key).kind != "f"}
    empty(path, N, column_names, data_types, data_shapes, ucds, units, null_values=null_values)
    if shuffle:
        del column_names[-1]
        del data_types[-1]
        del data_shapes[-1]
    dataset_output = vaex.astro.fits.FitsBinTable(path, write=True)
    df_output = vaex.dataframe.DataFrameLocal(dataset_output)
    vaex.export._export(dataset_input=dataset, dataset_output=df_output, path=path, random_index_column=random_index_name,
            column_names=column_names, selection=selection, shuffle=shuffle,
            progress=progress, sort=sort, ascending=ascending)
    dataset_output.close()
