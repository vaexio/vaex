__author__ = 'maartenbreddels'
import os
import mmap
import itertools
import functools
import collections
import logging
import numpy as np
import numpy.ma
import vaex
import astropy.table
import astropy.units
from vaex.utils import ensure_string
import astropy.io.fits as fits
import re
import six

from vaex.dataset import DatasetLocal, DatasetArrays
logger = logging.getLogger("vaex.file")
import vaex.dataset
import vaex.file
dataset_type_map = {}

from vaex.expression import Expression

from vaex.dataset_mmap import DatasetMemoryMapped


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
		stride = self.formatSize//8 # stride in units of the size of the element (float64)

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
			stride = self.formatSize//8
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

def _try_unit(unit):
	try:
		unit = astropy.units.Unit(str(unit))
		if not isinstance(unit, astropy.units.UnrecognizedUnit):
			return unit
	except:
		#logger.exception("could not parse unit: %r", unit)
		pass
	try:
		unit_mangle = re.match(".*\[(.*)\]", str(unit)).groups()[0]
		unit = astropy.units.Unit(unit_mangle)
	except:
		pass#logger.exception("could not parse unit: %r", unit)
	if isinstance(unit, six.string_types):
		return None
	elif isinstance(unit, astropy.units.UnrecognizedUnit):
		return None
	else:
		return unit

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
								column_name = _python_save_name(column.name.strip(), used=self.columns.keys())
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
										self.addColumn(column_name, offset=offset, dtype=dtype, length=length)
									else:
										for i in range(arraylength):
											name = column_name+"_" +str(i)
											self.addColumn(name, offset=offset+bytessize*i//arraylength, dtype=">" +dtypecode, length=length, stride=arraylength)
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
								column_name = _python_save_name(column.name, used=self.columns.keys())
								self.addColumn(column_name, array=array)
								self._get_column_meta_data(table, column_name, column, i)
								self._check_null(table, column_name, column, i)
			self._try_votable(fitsfile[0])

		self.update_meta()
		self.update_virtual_meta()

	def _check_null(self, table, column_name, column, i):
		null_name = "TNULL%d" % (i+1)
		if null_name in table.header:
			mask_value = table.header[null_name]
			array = self.columns[column_name]
			mask = array == mask_value
			self.columns[column_name] = numpy.ma.masked_array(array, mask)

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
					self.unit[column_name] = unit
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

dataset_type_map["fits"] = FitsBinTable

# class InMemory(DatasetMemoryMapped):
# 	def __init__(self, name):
# 		super(InMemory, self).__init__(filename=None, nommap=True, name=name)


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

dataset_type_map["soneira-peebles"] = SoneiraPeebles


class Zeldovich(DatasetArrays):
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
			self.add_column(name, X[d].reshape(-1) * scale)
		for d, name in zip(list(range(dim)), "xyzw"):
			self.add_column("v"+name, s[d].reshape(-1) * scale)
		for d, name in zip(list(range(dim)), "xyzw"):
			self.add_column(name+"0", Q[d].reshape(-1) * scale)
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
	@classmethod
	def can_open(cls, path, *args, **kwargs):
		try:
			with open(path, 'rb') as f:
				first_words = struct.unpack('4I',f.read(4*4))
				if first_words[0] == 8 and first_words[2] == 8 and first_words[3] == 256:
					logg.debug('gadget file SnapFormat=2 detected')
					return True
				elif first_words[0] == 256:
					f.seek(256+4)
					if struct.unpack('I',f.read(4))[0] == 256:
						logger.debug('gadget file SnapFormat=1 detected')
						return True
		except:
			pass

		return False



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
			self.description = table.meta.get("description")
			self.table = table
			#self.name

		#data = table.array.data
		for i in range(len(self.table.dtype)):
			name = self.table.dtype.names[i]
			column = self.table[name]
			type = self.table.dtype[i]
			#clean_name = re.sub("[^a-zA-Z_]", "_", name)
			clean_name = _python_save_name(name, self.columns.keys())
			if type.kind in "fiuSU": # only store float and int
				#datagroup.create_dataset(name, data=table.array[name].astype(np.float64))
				#dataset.addMemoryColumn(name, table.array[name].astype(np.float64))
				masked_array = self.table[name].data
				if "ucd" in column._meta:
					self.ucds[clean_name] = column._meta["ucd"]
				if column.unit:
					unit = _try_unit(column.unit)
					if unit:
						self.units[clean_name] = unit
				if column.description:
					self.descriptions[clean_name] = column.description
				if hasattr(masked_array, "mask"):
					if type.kind in ["f"]:
						masked_array.data[masked_array.mask] = np.nan
					if type.kind in ["i"]:
						masked_array.data[masked_array.mask] = 0
				self.add_column(clean_name, self.table[name].data)
			if type.kind in ["SU"]:
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
			data = self.first_table.array[name]
			type = self.first_table.array[name].dtype
			clean_name = _python_save_name(name, self.columns.keys())
			if field.ucd:
				self.ucds[clean_name] = field.ucd
			if field.unit:
				unit = _try_unit(field.unit)
				if unit:
					self.units[clean_name] = unit
			if field.description:
				self.descriptions[clean_name] = field.description
			if type.kind in "fiubSU": # only store float and int and boolean
				self.add_column(clean_name, data) #self.first_table.array[name].data)
			if type.kind == "O":
				print("column %r is of unsupported object type , will try to convert it to string" % (name,))
				try:
					data = data.astype("S")
					self.add_column(name, data)
				except Exception as e:
					print("Giving up column %s, error: %r" (name, e))
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
