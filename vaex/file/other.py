__author__ = 'maartenbreddels'
import os
import mmap
import itertools
import functools
import collections
import logging
import numpy as np
import vaex
import astropy.table
import astropy.units
from vaex.utils import ensure_string
import astropy.io.fits as fits
import re

from vaex.dataset import DatasetLocal, DatasetArrays
logger = logging.getLogger("vaex.file")
import vaex.dataset
import vaex.file
dataset_type_map = {}

# h5py doesn't want to build at readthedocs
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
try:
	import h5py
except:
	if not on_rtd:
		raise


class DatasetMemoryMapped(DatasetLocal):
	"""Represents a dataset where the data is memory mapped for efficient reading"""


	# nommap is a hack to get in memory datasets working
	def __init__(self, filename, write=False, nommap=False, name=None):
		super(DatasetMemoryMapped, self).__init__(name=name or os.path.splitext(os.path.basename(filename))[0], path=os.path.abspath(filename) if filename is not None else None, column_names=[])
		self.filename = filename or "no file"
		self.write = write
		#self.name = name or os.path.splitext(os.path.basename(self.filename))[0]
		#self.path = os.path.abspath(filename) if filename is not None else None
		self.nommap = nommap
		if not nommap:
			self.file = open(self.filename, "r+" if write else "r")
			self.fileno = self.file.fileno()
			self.mapping = mmap.mmap(self.fileno, 0, prot=mmap.PROT_READ | 0 if not write else mmap.PROT_WRITE )
			self.file_map = {filename: self.file}
			self.fileno_map = {filename: self.fileno}
			self.mapping_map = {filename: self.mapping}
		else:
			self.file_map = {}
			self.fileno_map = {}
			self.mapping_map = {}
		self._length = None
		#self._fraction_length = None
		self.nColumns = 0
		self.column_names = []
		self.rank1s = {}
		self.rank1names = []
		self.virtual_columns = collections.OrderedDict()

		self.axes = {}
		self.axis_names = []

		# these are replaced by variables
		#self.properties = {}
		#self.property_names = []

		self.current_slice = None
		self.fraction = 1.0


		self.selected_row_index = None
		self.selected_serie_index = 0
		self.row_selection_listeners = []
		self.serie_index_selection_listeners = []
		#self.mask_listeners = []

		self.all_columns = {}
		self.all_column_names = []
		self.global_links = {}

		self.offsets = {}
		self.strides = {}
		self.filenames = {}
		self.dtypes = {}
		self.samp_id = None
		#self.variables = collections.OrderedDict()

		self.undo_manager = vaex.ui.undo.UndoManager()

	def close_files(self):
		for name, file in self.file_map.items():
			file.close()

	def has_snapshots(self):
		return len(self.rank1s) > 0

	def get_path(self):
		return self.path

	def addFile(self, filename, write=False):
		self.file_map[filename] = open(filename, "r+" if write else "r")
		self.fileno_map[filename] = self.file_map[filename].fileno()
		self.mapping_map[filename] = mmap.mmap(self.fileno_map[filename], 0, prot=mmap.PROT_READ | 0 if not write else mmap.PROT_WRITE )


	def selectSerieIndex(self, serie_index):
		self.selected_serie_index = serie_index
		for serie_index_selection_listener in self.serie_index_selection_listeners:
			serie_index_selection_listener(serie_index)
		self.signal_sequence_index_change.emit(self, serie_index)

	def matches_url(self, url):
		filename = url
		if filename.startswith("file:/"):
			filename = filename[5:]
		similar = os.path.splitext(os.path.abspath(self.filename))[0] == os.path.splitext(filename)[0]
		logger.info("matching urls: %r == %r == %r" % (os.path.splitext(self.filename)[0], os.path.splitext(filename)[0], similar) )
		return similar

	def close(self):
		self.file.close()
		self.mapping.close()

	def addAxis(self, name, offset=None, length=None, dtype=np.float64, stride=1, filename=None):
		if filename is None:
			filename = self.filename
		mapping = self.mapping_map[filename]
		mmapped_array = np.frombuffer(mapping, dtype=dtype, count=length if stride is None else length * stride, offset=offset)
		if stride:
			mmapped_array = mmapped_array[::stride]
		self.axes[name] = mmapped_array
		self.axis_names.append(name)


	def addColumn(self, name, offset=None, length=None, dtype=np.float64, stride=1, filename=None, array=None):
		if filename is None:
			filename = self.filename
		if not self.nommap:
			mapping = self.mapping_map[filename]

		if array is not None:
			length = len(array)

		if self._length is not None and length != self._length:
			logger.error("inconsistent length", "length of column %s is %d, while %d was expected" % (name, length, self._length))
		else:
			if self.current_slice is None:
				self.current_slice = (0, length)
				self.fraction = 1.
				self._full_length = length
				self._length = length
				self._index_end = self._full_length
			self._length = length
			#print self.mapping, dtype, length if stride is None else length * stride, offset
			if array is not None:
				length = len(array)
				mmapped_array = array
				stride = None
				offset = None
				dtype = array.dtype
			else:
				if offset is None:
					print("offset is None")
					sys.exit(0)
				mmapped_array = np.frombuffer(mapping, dtype=dtype, count=length if stride is None else length * stride, offset=offset)
				if stride:
					#import pdb
					#pdb.set_trace()
					mmapped_array = mmapped_array[::stride]
			self.columns[name] = mmapped_array
			self.column_names.append(name)
			self.all_columns[name] = mmapped_array
			self.all_column_names.append(name)
			#self.column_names.sort()
			self.nColumns += 1
			self.nRows = self._length
			self.offsets[name] = offset
			self.strides[name] = stride
			if filename is not None:
				self.filenames[name] = os.path.abspath(filename)
			self.dtypes[name] = dtype

	def addRank1(self, name, offset, length, length1, dtype=np.float64, stride=1, stride1=1, filename=None, transposed=False):
		if filename is None:
			filename = self.filename
		mapping = self.mapping_map[filename]
		if (not transposed and self._length is not None and length != self._length) or (transposed and self._length is not None and length1 != self._length):
			logger.error("inconsistent length", "length of column %s is %d, while %d was expected" % (name, length, self._length))
		else:
			if self.current_slice is None:
				self.current_slice = (0, length if not transposed else length1)
				self.fraction = 1.
				self._full_length = length if not transposed else length1
				self._length = self._full_length
				self._index_end = self._full_length
			self._length = length if not transposed else length1
			#print self.mapping, dtype, length if stride is None else length * stride, offset
			rawlength = length * length1
			rawlength *= stride
			rawlength *= stride1

			mmapped_array = np.frombuffer(mapping, dtype=dtype, count=rawlength, offset=offset)
			mmapped_array = mmapped_array.reshape((length1*stride1, length*stride))
			mmapped_array = mmapped_array[::stride1,::stride]

			self.rank1s[name] = mmapped_array
			self.rank1names.append(name)
			self.all_columns[name] = mmapped_array
			self.all_column_names.append(name)



import struct
class HansMemoryMapped(DatasetMemoryMapped):
	def __init__(self, filename, filename_extra=None):
		super(HansMemoryMapped, self).__init__(filename)
		self.pageSize, \
		self.formatSize, \
		self.numberParticles, \
		self.numberTimes, \
		self.numberParameters, \
		self.numberCompute, \
		self.dataOffset, \
		self.dataHeaderSize = struct.unpack("Q"*8, self.mapping[:8*8])
		zerooffset = offset = self.dataOffset
		length = self.numberParticles+1
		stride = self.formatSize/8 # stride in units of the size of the element (float64)

		# TODO: ask Hans for the self.numberTimes-2
		lastoffset = offset + (self.numberParticles+1)*(self.numberTimes-2)*self.formatSize
		t_index = 3
		names = "x y z vx vy vz".split()
		midoffset = offset + (self.numberParticles+1)*self.formatSize*t_index
		names = "x y z vx vy vz".split()

		for i, name in enumerate(names):
			self.addColumn(name+"_0", offset+8*i, length, dtype=np.float64, stride=stride)

		for i, name in enumerate(names):
			self.addColumn(name+"_last", lastoffset+8*i, length, dtype=np.float64, stride=stride)


		names = "x y z vx vy vz".split()

		if 1:
			stride = self.formatSize/8
			#stride1 = self.numberTimes #*self.formatSize/8
			for i, name in enumerate(names):
				# TODO: ask Hans for the self.numberTimes-1
				self.addRank1(name, offset+8*i, length=self.numberParticles+1, length1=self.numberTimes-1, dtype=np.float64, stride=stride, stride1=1)

		if filename_extra is None:
			basename = os.path.basename(filename)
			if os.path.exists(basename + ".omega2"):
				filename_extra = basename + ".omega2"

		if filename_extra is not None:
			self.addFile(filename_extra)
			mapping = self.mapping_map[filename_extra]
			names = "J_r J_theta J_phi Theta_r Theta_theta Theta_phi Omega_r Omega_theta Omega_phi r_apo r_peri".split()
			offset = 0
			stride = 11
			#import pdb
			#pdb.set_trace()
			for i, name in enumerate(names):
				# TODO: ask Hans for the self.numberTimes-1
				self.addRank1(name, offset+8*i, length=self.numberParticles+1, length1=self.numberTimes-1, dtype=np.float64, stride=stride, stride1=1, filename=filename_extra)

				self.addColumn(name+"_0", offset+8*i, length, dtype=np.float64, stride=stride, filename=filename_extra)
				self.addColumn(name+"_last", offset+8*i + (self.numberParticles+1)*(self.numberTimes-2)*11*8, length, dtype=np.float64, stride=stride, filename=filename_extra)


	@classmethod
	def can_open(cls, path, *args, **kwargs):
		return os.path.splitext(path)[-1] == ".bin"
		basename, ext = os.path.splitext(path)
		#if os.path.exists(basename + ".omega2"):
		#	return True
		#return True

	@classmethod
	def get_options(cls, path):
		return []

	@classmethod
	def option_to_args(cls, option):
		return []
dataset_type_map["buist"] = HansMemoryMapped

def _python_save_name(name, used=[]):
	first, rest = name[0], name[1:]
	name = re.sub("[^a-zA-Z_]", "_", first) +  re.sub("[^a-zA-Z_0-9]", "_", rest)
	if name in used:
		nr = 1
		while name + ("_%d" % nr) in used:
			nr += 1
		name = name + ("_%d" % nr)
	return name

class FitsBinTable(DatasetMemoryMapped):
	def __init__(self, filename, write=False):
		super(FitsBinTable, self).__init__(filename, write=write)
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
								column_name = _python_save_name(column.name, used=self.columns.keys())

								ucd_header_name = "TUCD%d" % (i+1)
								if ucd_header_name in table.header:
									self.ucds[column_name] = table.header[ucd_header_name]
								def _try_unit(unit):
									try:
										return astropy.units.Unit(unit)
									except:
										#logger.exception("could not parse unit: %r", unit)
										pass
									try:
										unit = re.match(".*\[(.*)\]", unit).groups()[0]
										return astropy.units.Unit(unit)
									except:
										pass#logger.exception("could not parse unit: %r", unit)
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
											self.unit[column_name] = unit
								#unit_header_name = "TUCD%d" % (i+1)
								#if ucd_header_name in table.header:

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
										self.addColumn(column_name, offset=offset, dtype=dtype, length=length)
									else:
										for i in range(arraylength):
											name = column_name+"_" +str(i)
											self.addColumn(name, offset=offset+bytessize*i/arraylength, dtype=">" +dtypecode, length=length, stride=arraylength)
								if flatlength > 0: # flatlength can be
									offset += bytessize * length

					else:
						logger.debug("adding table: %r" % table)
						for column in table.columns:
							array = column.array[:]
							array = column.array[:] # 2nd time it will be a real np array
							#import pdb
							#pdb.set_trace()
							if array.dtype.kind in "fi":
								self.addColumn(column.name, array=array)
		self.update_meta()
		self.update_virtual_meta()
		self.selections_favorite_load()

	@classmethod
	def can_open(cls, path, *args, **kwargs):
		return os.path.splitext(path)[1] == ".fits"

	@classmethod
	def get_options(cls, path):
		return [] # future: support multiple tables?

	@classmethod
	def option_to_args(cls, option):
		return []

dataset_type_map["fits"] = FitsBinTable

class Hdf5MemoryMapped(DatasetMemoryMapped):
	"""Implements the vaex hdf5 file format"""
	def __init__(self, filename, write=False):
		super(Hdf5MemoryMapped, self).__init__(filename, write=write)
		self.h5file = h5py.File(self.filename, "r+" if write else "r")
		self.h5table_root_name = None
		try:
			self._load()
		finally:
			self.h5file.close()

	def write_meta(self):
		"""ucds, descriptions and units are written as attributes in the hdf5 file, instead of a seperate file as
		 the default :func:`Dataset.write_meta`.
		 """
		with h5py.File(self.filename, "r+") as h5file_output:
			h5table_root = h5file_output[self.h5table_root_name]
			if self.description is not None:
				h5table_root.attrs["description"] = self.description
			for column_name in self.columns.keys():
				h5dataset = h5table_root[column_name]
				for name, values in [("ucd", self.ucds), ("unit", self.units), ("description", self.descriptions)]:
					if column_name in values:
						value = str(values[column_name])
						h5dataset.attrs[name] = value
	@classmethod
	def create(cls, path, N, column_names, dtypes=None, write=True):
		"""Create a new (empty) hdf5 file with columns given by column names, of length N

		Optionally, numpy dtypes can be passed, default is floats
		"""

		dtypes = dtypes or [np.float] * len(column_names)

		if N == 0:
			raise ValueError("Cannot export empty table")
		with h5py.File(path, "w") as h5file_output:
			h5data_output = h5file_output.require_group("data")
			for column_name, dtype in zip(column_names, dtypes):
				shape = (N,)
				print(dtype)
				if dtype.type == np.datetime64:
					array = h5file_output.require_dataset("/data/%s" % column_name, shape=shape, dtype=np.int64)
					array.attrs["dtype"] = dtype.name
				else:
					array = h5file_output.require_dataset("/data/%s" % column_name, shape=shape, dtype=dtype)
				array[0] = array[0] # make sure the array really exists
		return Hdf5MemoryMapped(path, write=write)

	@classmethod
	def can_open(cls, path, *args, **kwargs):
		h5file = None
		try:
			with open(path, "rb") as f:
				signature = f.read(4)
				hdf5file = signature == b"\x89\x48\x44\x46"
		except:
			logger.error("could not read 4 bytes from %r", path)
			return
		if hdf5file:
			try:
				h5file = h5py.File(path, "r")
			except:
				logger.exception("could not open file as hdf5")
				return False
			if h5file is not None:
				with h5file:
					return ("data" in h5file) or ("columns" in h5file)
			else:
				logger.debug("file %s has no data or columns group" % path)
		return False


	@classmethod
	def get_options(cls, path):
		return []

	@classmethod
	def option_to_args(cls, option):
		return []

	def _load(self):
		if "data" in self.h5file:
			self._load_columns(self.h5file["/data"])
			self.h5table_root_name = "/data"
		# TODO: shall we rename it vaex... ?
		# if "vaex" in self.h5file:
		#	self.load_columns(self.h5file["/vaex"])
		#	h5table_root = "/vaex"
		if "columns" in self.h5file:
			self._load_columns(self.h5file["/columns"])
			self.h5table_root_name = "/columns"
		if "properties" in self.h5file:
			self._load_variables(self.h5file["/properties"]) # old name, kept for portability
		if "variables" in self.h5file:
			self._load_variables(self.h5file["/variables"])
		if "axes" in self.h5file:
			self._load_axes(self.h5file["/axes"])
		self.update_meta()
		self.update_virtual_meta()
		self.selections_favorite_load()

	#def
	def _load_axes(self, axes_data):
		for name in axes_data:
			axis = axes_data[name]
			logger.debug("loading axis %r" % name)
			offset = axis.id.get_offset()
			shape = axis.shape
			assert len(shape) == 1 # ony 1d axes
			#print name, offset, len(axis), axis.dtype
			self.addAxis(name, offset=offset, length=len(axis), dtype=axis.dtype)
			#self.axis_names.append(axes_data)
			#self.axes[name] = np.array(axes_data[name])

	def _load_variables(self, h5variables):
		for key, value in list(h5variables.attrs.items()):
			self.variables[key] = value


	def _load_columns(self, h5data):
		#print h5data
		# make sure x y x etc are first
		first = "x y z vx vy vz".split()
		finished = set()
		if "description" in h5data.attrs:
			self.description = ensure_string(h5data.attrs["description"])
		for column_name in first + list(h5data):
			if column_name in h5data and column_name not in finished:
				#print type(column_name)
				column = h5data[column_name]
				if "ucd" in column.attrs:
					self.ucds[column_name] = ensure_string(column.attrs["ucd"])
				if "description" in column.attrs:
					self.descriptions[column_name] = ensure_string(column.attrs["description"])
				if "unit" in column.attrs:
					try:
						unitname = ensure_string(column.attrs["unit"])
						if unitname and unitname != "None":
							self.units[column_name] = astropy.units.Unit(unitname)
					except:
						logger.exception("error parsing unit: %s", column.attrs["unit"])
				if "units" in column.attrs: # Amuse case
					unitname = ensure_string(column.attrs["units"])
					logger.debug("amuse unit: %s", unitname)
					if unitname == "(0.01 * system.get('S.I.').base('length'))":
						self.units[column_name] = astropy.units.Unit("cm")
					if unitname == "((0.01 * system.get('S.I.').base('length')) * (system.get('S.I.').base('time')**-1))":
						self.units[column_name] = astropy.units.Unit("cm/s")
					if unitname == "(0.001 * system.get('S.I.').base('mass'))":
						self.units[column_name] = astropy.units.Unit("gram")

					if unitname == "system.get('S.I.').base('length')":
						self.units[column_name] = astropy.units.Unit("m")
					if unitname == "(system.get('S.I.').base('length') * (system.get('S.I.').base('time')**-1))":
						self.units[column_name] = astropy.units.Unit("m/s")
					if unitname == "system.get('S.I.').base('mass')":
						self.units[column_name] = astropy.units.Unit("kg")

				if hasattr(column, "dtype"):
					#print column, column.shape
					offset = column.id.get_offset()
					if offset is None:
						raise Exception("columns doesn't really exist in hdf5 file")
					shape = column.shape
					if True: #len(shape) == 1:
						dtype = column.dtype
						if "dtype" in column.attrs:
							dtype = column.attrs["dtype"]
						logger.debug("adding column %r with dtype %r", column_name, dtype)
						self.addColumn(column_name, offset, len(column), dtype=dtype)
					else:

						#transposed = self._length is None or shape[0] == self._length
						transposed = shape[1] < shape[0]
						self.addRank1(column_name, offset, shape[1], length1=shape[0], dtype=column.dtype, stride=1, stride1=1, transposed=transposed)
						#if len(shape[0]) == self._length:
						#	self.addRank1(column_name, offset, shape[1], length1=shape[0], dtype=column.dtype, stride=1, stride1=1)
						#self.addColumn(column_name+"_0", offset, shape[1], dtype=column.dtype)
						#self.addColumn(column_name+"_last", offset+(shape[0]-1)*shape[1]*column.dtype.itemsize, shape[1], dtype=column.dtype)
						#self.addRank1(name, offset+8*i, length=self.numberParticles+1, length1=self.numberTimes-1, dtype=np.float64, stride=stride, stride1=1, filename=filename_extra)
			finished.add(column_name)

	def close(self):
		super(Hdf5MemoryMapped, self).close()
		self.h5file.close()

	def __expose_array(self, hdf5path, column_name):
		array = self.h5file[hdf5path]
		array[0] = array[0] # without this, get_offset returns None, probably the array isn't really created
		offset = array.id.get_offset()
		self.remap()
		self.addColumn(column_name, offset, len(array), dtype=array.dtype)

	def __add_column(self, column_name, dtype=np.float64, length=None):
		array = self.h5data.create_dataset(column_name, shape=(self._length if length is None else length,), dtype=dtype)
		array[0] = array[0] # see above
		offset = array.id.get_offset()
		self.h5file.flush()
		self.remap()
		self.addColumn(column_name, offset, len(array), dtype=array.dtype)

dataset_type_map["h5vaex"] = Hdf5MemoryMapped

class AmuseHdf5MemoryMapped(Hdf5MemoryMapped):
	"""Implements reading Amuse hdf5 files `amusecode.org <http://amusecode.org/>`_"""
	def __init__(self, filename, write=False):
		super(AmuseHdf5MemoryMapped, self).__init__(filename, write=write)

	@classmethod
	def can_open(cls, path, *args, **kwargs):
		h5file = None
		try:
			h5file = h5py.File(path, "r")
		except:
			return False
		if h5file is not None:
			with h5file:
				return ("particles" in h5file)# or ("columns" in h5file)
		return False

	def _load(self):
		particles = self.h5file["/particles"]
		for group_name in particles:
			#import pdb
			#pdb.set_trace()
			group = particles[group_name]
			self._load_columns(group["attributes"])

			column_name = "keys"
			column = group[column_name]
			offset = column.id.get_offset()
			self.addColumn(column_name, offset, len(column), dtype=column.dtype)
		self.update_meta()
		self.update_virtual_meta()
		self.selections_favorite_load()

dataset_type_map["amuse"] = AmuseHdf5MemoryMapped


gadget_particle_names = "gas halo disk bulge stars dm".split()

class Hdf5MemoryMappedGadget(DatasetMemoryMapped):
	"""Implements reading `Gadget2 <http://wwwmpa.mpa-garching.mpg.de/gadget/>`_ hdf5 files """
	def __init__(self, filename, particle_name=None, particle_type=None):
		if "#" in filename:
			filename, index = filename.split("#")
			index = int(index)
			particle_type = index
			particle_name = gadget_particle_names[particle_type]
		elif particle_type is not None:
			self.particle_name = gadget_particle_names[self.particle_type]
			self.particle_type = particle_type
		elif particle_name is not None:
			if particle_name.lower() in gadget_particle_names:
				self.particle_type = gadget_particle_names.index(particle_name.lower())
				self.particle_name = particle_name.lower()
			else:
				raise ValueError("particle name not supported: %r, expected one of %r" % (particle_name, " ".join(gadget_particle_names)))
		else:
			raise Exception("expected particle type or name as argument, or #<nr> behind filename")
		super(Hdf5MemoryMappedGadget, self).__init__(filename)
		self.particle_type = particle_type
		self.particle_name = particle_name
		self.name = self.name + "-" + self.particle_name
		h5file = h5py.File(self.filename, 'r')
		#for i in range(1,4):
		key = "/PartType%d" % self.particle_type
		if key not in h5file:
			raise KeyError("%s does not exist" % key)
		particles = h5file[key]
		for name in list(particles.keys()):
			#name = "/PartType%d/Coordinates" % i
			data = particles[name]
			if isinstance(data, h5py.highlevel.Dataset): #array.shape
				array = data
				shape = array.shape
				if len(shape) == 1:
					offset = array.id.get_offset()
					if offset is not None:
						self.addColumn(name, offset, data.shape[0], dtype=data.dtype)
				else:
					if name == "Coordinates":
						offset = data.id.get_offset()
						if offset is None:
							print((name, "is not of continuous layout?"))
							sys.exit(0)
						bytesize = data.dtype.itemsize
						self.addColumn("x", offset, data.shape[0], dtype=data.dtype, stride=3)
						self.addColumn("y", offset+bytesize, data.shape[0], dtype=data.dtype, stride=3)
						self.addColumn("z", offset+bytesize*2, data.shape[0], dtype=data.dtype, stride=3)
					elif name == "Velocity":
						offset = data.id.get_offset()
						self.addColumn("vx", offset, data.shape[0], dtype=data.dtype, stride=3)
						self.addColumn("vy", offset+bytesize, data.shape[0], dtype=data.dtype, stride=3)
						self.addColumn("vz", offset+bytesize*2, data.shape[0], dtype=data.dtype, stride=3)
					elif name == "Velocities":
						offset = data.id.get_offset()
						self.addColumn("vx", offset, data.shape[0], dtype=data.dtype, stride=3)
						self.addColumn("vy", offset+bytesize, data.shape[0], dtype=data.dtype, stride=3)
						self.addColumn("vz", offset+bytesize*2, data.shape[0], dtype=data.dtype, stride=3)
					else:
						logger.error("unsupported column: %r of shape %r" % (name, array.shape))
		if "Header" in h5file:
			for name in "Redshift Time_GYR".split():
				if name in h5file["Header"].attrs:
					value = h5file["Header"].attrs[name].decode("utf-8")
					logger.debug("property[{name!r}] = {value}".format(**locals()))
					self.variables[name] = value
					#self.property_names.append(name)

		name = "particle_type"
		value = particle_type
		logger.debug("property[{name}] = {value}".format(**locals()))
		self.variables[name] = value
		#self.property_names.append(name)

	@classmethod
	def can_open(cls, path, *args, **kwargs):
		if len(args) == 2:
			particleName = args[0]
			particleType = args[1]
		elif "particle_name" in kwargs:
			particle_type = gadget_particle_names.index(kwargs["particle_name"].lower())
		elif "particle_type" in kwargs:
			particle_type = kwargs["particle_type"]
		elif "#" in path:
			filename, index = path.split("#")
			particle_type = gadget_particle_names[index]
		else:
			return False
		h5file = None
		try:
			h5file = h5py.File(path, "r")
		except:
			return False
		has_particles = False
		#for i in range(1,6):
		key = "/PartType%d" % particle_type
		exists = key in h5file
		h5file.close()
		return exists

		#has_particles = has_particles or (key in h5file)
		#return has_particles


	@classmethod
	def get_options(cls, path):
		return []

	@classmethod
	def option_to_args(cls, option):
		return []


dataset_type_map["gadget-hdf5"] = Hdf5MemoryMappedGadget

class InMemory(DatasetMemoryMapped):
	def __init__(self, name):
		super(InMemory, self).__init__(filename=None, nommap=True, name=name)


class SoneiraPeebles(DatasetArrays):
	def __init__(self, dimension, eta, max_level, L):
		super(SoneiraPeebles, self).__init__(name="soneira-peebles")
		#InMemory.__init__(self)
		def todim(value):
			if isinstance(value, (tuple, list)):
				assert len(value) >= dimension, "either a scalar or sequence of length equal to or larger than the dimension"
				return value[:dimension]
			else:
				return [value] * dimension

		eta = eta
		max_level = max_level
		N = eta**(max_level)
		# array[-1] is used as a temp storage
		array = np.zeros((dimension+1, N), dtype=np.float64)
		L = todim(L)

		for d in range(dimension):
			vaex.vaexfast.soneira_peebles(array[d], 0, 1, L[d], eta, max_level)
		for d, name in zip(list(range(dimension)), "x y z w v u".split()):
			self.add_column(name, array[d])
		if 0:
			order = np.zeros(N, dtype=np.int64)
			vaex.vaexfast.shuffled_sequence(order);
			for i, name in zip(list(range(dimension)), "x y z w v u".split()):
				#np.take(array[i], order, out=array[i])
				reorder(array[i], array[-1], order)
				self.addColumn(name, array=array[i])

dataset_type_map["soneira-peebles"] = Hdf5MemoryMappedGadget


class Zeldovich(InMemory):
	def __init__(self, dim=2, N=256, n=-2.5, t=None, seed=None, scale=1, name="zeldovich approximation"):
		super(Zeldovich, self).__init__(name=name)

		if seed is not None:
			np.random.seed(seed)
		#sys.exit(0)
		shape = (N,) * dim
		A = np.random.normal(0.0, 1.0, shape)
		F = np.fft.fftn(A)
		K = np.fft.fftfreq(N, 1./(2*np.pi))[np.indices(shape)]
		k = (K**2).sum(axis=0)
		k_max = np.pi
		F *= np.where(np.sqrt(k) > k_max, 0, np.sqrt(k**n) * np.exp(-k*4.0))
		F.flat[0] = 0
		#pylab.imshow(np.where(sqrt(k) > k_max, 0, np.sqrt(k**-2)), interpolation='nearest')
		grf = np.fft.ifftn(F).real
		Q = np.indices(shape) / float(N-1) - 0.5
		s = np.array(np.gradient(grf)) / float(N)
		#pylab.imshow(s[1], interpolation='nearest')
		#pylab.show()
		s /= s.max() * 100.
		#X = np.zeros((4, 3, N, N, N))
		#for i in range(4):
		#if t is None:
		#	s = s/s.max()
		t = t or 1.
		X = Q + s * t

		for d, name in zip(list(range(dim)), "xyzw"):
			self.addColumn(name, array=X[d].reshape(-1) * scale)
		for d, name in zip(list(range(dim)), "xyzw"):
			self.addColumn("v"+name, array=s[d].reshape(-1) * scale)
		for d, name in zip(list(range(dim)), "xyzw"):
			self.addColumn(name+"0", array=Q[d].reshape(-1) * scale)
		return

dataset_type_map["zeldovich"] = Zeldovich


class AsciiTable(DatasetMemoryMapped):
	def __init__(self, filename):
		super(AsciiTable, self).__init__(filename, nommap=True)
		import asciitable
		table = asciitable.read(filename)
		logger.debug("done parsing ascii table")
		#import pdb
		#pdb.set_trace()
		#names = table.array.dtype.names
		names = table.dtype.names

		#data = table.array.data
		for i in range(len(table.dtype)):
			name = table.dtype.names[i]
			type = table.dtype[i]
			if type.kind in ["f", "i"]: # only store float and int
				#datagroup.create_dataset(name, data=table.array[name].astype(np.float64))
				#dataset.addMemoryColumn(name, table.array[name].astype(np.float64))
				self.addColumn(name, array=table[name])
		#dataset.samp_id = table_id
		#self.list.addDataset(dataset)
		#return dataset

	@classmethod
	def can_open(cls, path, *args, **kwargs):
		can_open = path.endswith(".asc")
		logger.debug("%r can open: %r"  %(cls.__name__, can_open))
		return can_open
dataset_type_map["ascii"] = AsciiTable

class MemoryMappedGadget(DatasetMemoryMapped):
	def __init__(self, filename):
		super(MemoryMappedGadget, self).__init__(filename)
		#h5file = h5py.File(self.filename)
		import vaex.file.gadget
		length, posoffset, veloffset, header = vaex.file.gadget.getinfo(filename)
		self.addColumn("x", posoffset, length, dtype=np.float32, stride=3)
		self.addColumn("y", posoffset+4, length, dtype=np.float32, stride=3)
		self.addColumn("z", posoffset+8, length, dtype=np.float32, stride=3)

		self.addColumn("vx", veloffset, length, dtype=np.float32, stride=3)
		self.addColumn("vy", veloffset+4, length, dtype=np.float32, stride=3)
		self.addColumn("vz", veloffset+8, length, dtype=np.float32, stride=3)
dataset_type_map["gadget-plain"] = MemoryMappedGadget

class DatasetAstropyTable(DatasetArrays):
	def __init__(self, filename=None, format=None, table=None, **kwargs):
		if table is None:
			self.filename = filename
			self.format = format
			DatasetArrays.__init__(self, filename)
			self.read_table()
		else:
			#print vars(table)
			#print dir(table)
			DatasetArrays.__init__(self, table.meta.get("name", "unknown-astropy"))
			self.table = table
			#self.name

		#data = table.array.data
		for i in range(len(self.table.dtype)):
			name = self.table.dtype.names[i]
			column = self.table[name]
			type = self.table.dtype[i]
			#clean_name = re.sub("[^a-zA-Z_]", "_", name)
			clean_name = _python_save_name(name, self.columns.keys())
			if type.kind in ["f", "i"]: # only store float and int
				#datagroup.create_dataset(name, data=table.array[name].astype(np.float64))
				#dataset.addMemoryColumn(name, table.array[name].astype(np.float64))
				masked_array = self.table[name].data
				if "ucd" in column._meta:
					self.ucds[clean_name] = column._meta["ucd"]
				if column.description:
					self.descriptions[clean_name] = column.description
				if hasattr(masked_array, "mask"):
					if type.kind in ["f"]:
						masked_array.data[masked_array.mask] = np.nan
					if type.kind in ["i"]:
						masked_array.data[masked_array.mask] = 0
				self.add_column(clean_name, self.table[name].data)
			if type.kind in ["S"]:
				self.add_column(clean_name, self.table[name].data)

		#dataset.samp_id = table_id
		#self.list.addDataset(dataset)
		#return dataset

	def read_table(self):
		self.table = astropy.table.Table.read(self.filename, format=self.format, **kwargs)

import astropy.io.votable
import string
class VOTable(DatasetArrays):
	def __init__(self, filename):
		DatasetArrays.__init__(self, filename)
		self.filename = filename
		self.path = filename
		votable = astropy.io.votable.parse(self.filename)

		self.first_table = votable.get_first_table()
		self.description = self.first_table.description

		for field in self.first_table.fields:
			name = field.name
			data = self.first_table.array[name].data
			type = self.first_table.array[name].dtype
			clean_name = re.sub("[^a-zA-Z_0-9]", "_", name)
			if clean_name in string.digits:
				clean_name = "_" + clean_name
			self.ucds[clean_name] = field.ucd
			self.units[clean_name] = field.unit
			self.descriptions[clean_name] = field.description
			if type.kind in ["f", "i"]: # only store float and int
				masked_array = self.first_table.array[name]
				if type.kind in ["f"]:
					masked_array.data[masked_array.mask] = np.nan
				if type.kind in ["i"]:
					masked_array.data[masked_array.mask] = 0
				self.add_column(clean_name, self.first_table.array[name].data)
			#if type.kind in ["S"]:
			#	self.add_column(clean_name, self.first_table.array[name].data)

	@classmethod
	def can_open(cls, path, *args, **kwargs):
		can_open = path.endswith(".vot")
		logger.debug("%r can open: %r"  %(cls.__name__, can_open))
		return can_open

dataset_type_map["votable"] = VOTable



class DatasetNed(DatasetAstropyTable):
	def __init__(self, code="2012AJ....144....4M"):
		url = "http://ned.ipac.caltech.edu/cgi-bin/objsearch?refcode={code}&hconst=73&omegam=0.27&omegav=0.73&corr_z=1&out_csys=Equatorial&out_equinox=J2000.0&obj_sort=RA+or+Longitude&of=xml_main&zv_breaker=30000.0&list_limit=5&img_stamp=YES&search_type=Search"\
			.format(code=code)
		super(DatasetNed, self).__init__(url, format="votable", use_names_over_ids=True)
		self.name = "ned:" + code

class DatasetTap(DatasetArrays):
	class TapColumn(object):
		def __init__(self, tap_dataset, column_name, column_type, ucd):
			self.tap_dataset = tap_dataset
			self.column_name = column_name
			self.column_type = column_type
			self.ucd = ucd
			self.alpha_min = 0
			length = len(tap_dataset)
			steps = length/1e6 # try to do it in chunks
			self.alpha_step = 360/steps
			self.alpha_max = self.alpha_min + self.alpha_step
			logger.debug("stepping in alpha %f" % self.alpha_step)
			self.data = []
			self.offset = 0
			self.shape = (length,)
			self.dtype = DatasetTap.type_map[self.column_type]().dtype
			self.left_over_chunk = None
			self.rows_left = length
			import tempfile
			self.download_file = tempfile.mktemp(".vot")

		def __getitem__(self, slice):
			start, stop, step = slice.start, slice.stop, slice.step
			required_length = stop - start
			assert start >= self.offset
			chunk_data = self.left_over_chunk
			enough = False if chunk_data is None else len(chunk_data) >= required_length
			if chunk_data is not None:
				logger.debug("start %s offset %s chunk length %s", start, self.offset, len(chunk_data))
				#assert len(chunk_data) == start - self.offset
			if enough:
				logger.debug("we can skip the query, already have results from previous query")
			while not enough:
				adql_query = "SELECT {column_name} FROM {table_name} WHERE alpha >= {alpha_min} AND alpha < {alpha_max} ORDER BY alpha ASC"\
					.format(column_name=self.column_name, table_name=self.tap_dataset.table_name, alpha_min=self.alpha_min, alpha_max=self.alpha_max)
				logger.debug("executing: %s" % adql_query)
				logger.debug("executing: %s" % adql_query.replace(" ", "+"))


				url = self.tap_dataset.tap_url + "/sync?REQUEST=doQuery&LANG=ADQL&MAXREC=10000000&FORMAT=votable&QUERY=" +adql_query.replace(" ", "+")
				import urllib2
				response = urllib2.urlopen(url)
				with open(self.download_file, "w") as f:
					f.write(response.read())
				votable = astropy.io.votable.parse(self.download_file)
				data = votable.get_first_table().array[self.column_name].data
				# TODO: respect masked array
				#table = astropy.table.Table.read(url, format="votable") #, show_progress=False)
				#data = table[self.column_name].data.data.data
				logger.debug("new chunk is of lenght %d", len(data))
				self.rows_left -= len(data)
				logger.debug("rows left %d", self.rows_left)
				if chunk_data is None:
					chunk_data = data
				else:
					chunk_data = np.concatenate([chunk_data, data])
				if len(chunk_data) >= required_length:
					enough = True
				logger.debug("total chunk is of lenght %d, enough: %s", len(chunk_data), enough)
				self.alpha_min += self.alpha_step
				self.alpha_max += self.alpha_step


			result, self.left_over_chunk = chunk_data[:required_length], chunk_data[required_length:]
			#print(result)
			logger.debug("left over is of length %d", len(self.left_over_chunk))
			return result #np.zeros(N, dtype=self.dtype)



	type_map = {
		'REAL':np.float32,
	    'SMALLINT':np.int32,
		'DOUBLE':np.float64,
		'BIGINT':np.int64,
		'INTEGER':np.int32,
		'BOOLEAN':np.bool8
	}
	#not supported types yet 'VARCHAR',', u'BOOLEAN', u'INTEGER', u'CHAR
	def __init__(self, tap_url="http://gaia.esac.esa.int/tap-server/tap/g10_smc", table_name=None):
		logger.debug("tap url: %r", tap_url)
		self.tap_url = tap_url
		self.table_name = table_name
		if table_name is None: # let us try to infer the table name
			if tap_url.endswith("tap") or tap_url.endswith("tap/"):
				pass # this mean we really didn't provide one
			else:
				index = tap_url.rfind("tap/")
				if index != -1:
					self.tap_url, self.table_name = tap_url[:index+4], self.tap_url[index+4:]
					logger.debug("inferred url is %s, and table name is %s", self.tap_url, self.table_name)

		if self.tap_url.startswith("tap+"): # remove tap+ part from tap+http(s), only keep http(s) part
			self.tap_url = self.tap_url[len("tap+"):]
		import requests
		super(DatasetTap, self).__init__(self.table_name)
		self.req = requests.request("get", self.tap_url+"/tables/")
		self.path = "tap+" +self.tap_url + "/" + table_name

		#print dir(self.req)
		from bs4 import BeautifulSoup
			#self.soup = BeautifulSoup(req.response)
		tables = BeautifulSoup(self.req.content, 'xml')
		self.tap_tables = collections.OrderedDict()
		for table in tables.find_all("table"):
			#print table.find("name").string, table.description.string, table["gaiatap:size"]
			table_name = unicode(table.find("name").string)
			table_size = int(table["esatapplus:size"])
			#print table_name, table_size
			logger.debug("tap table %r ", table_name)
			columns = []
			for column in table.find_all("column"):
				column_name = unicode(column.find("name").string)
				column_type = unicode(column.dataType.string)
				ucd = column.ucd.string if column.ucd else None
				unit = column.unit.string if column.unit else None
				description = column.description.string if column.description else None
				#print "\t", column_name, column_type, ucd
				#types.add()
				columns.append((column_name, column_type, ucd, unit, description))
			self.tap_tables[table_name] = (table_size, columns)
		if not self.tap_tables:
			raise ValueError("no tables or wrong url")
		for name, (table_size, columns) in self.tap_tables.items():
			logger.debug("table %s has length %d", name, table_size)
		self._full_length, self._tap_columns = self.tap_tables[self.table_name]
		self._length = self._full_length
		logger.debug("selected table table %s has length %d", self.table_name, self._full_length)
		#self.column_names = []
		#self.columns = collections.OrderedDict()
		for column_name, column_type, ucd, unit, description in self._tap_columns:
			logger.debug("  column %s has type %s and ucd %s, unit %s and description %s", column_name, column_type, ucd, unit, description)
			if column_type in self.type_map.keys():
				self.column_names.append(column_name)
				if ucd:
					self.ucds[column_name] = ucd
				if unit:
					self.units[column_name] = unit
				if description:
					self.descriptions[column_name] = description
				self.columns[column_name] = self.TapColumn(self, column_name, column_type, ucd)
			else:
				logger.warning("  type of column %s is not supported, it will be skipped", column_name)


	@classmethod
	def can_open(cls, path, *args, **kwargs):
		can_open = False
		url = None
		try:
			url = urlparse(path)
		except:
			return False
		if url.scheme:
			if url.scheme.startswith("tap+http"): # will also catch https
				can_open = True
		logger.debug("%r can open: %r"  %(cls.__name__, can_open))
		return can_open

dataset_type_map["tap"] = DatasetTap

