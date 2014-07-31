# -*- coding: utf-8 -*-
import h5py
import mmap
import numpy as np
import os

def error(title, msg):
	print "Error", title, msg


class MemoryMapped(object):
	def __init__(self, filename, write=False):
		self.filename = filename
		self.write = write
		self.name = os.path.basename(self.filename)
		self.file = file(self.filename, "r+" if write else "r")
		self.fileno = self.file.fileno()
		self.mapping = mmap.mmap(self.fileno, 0, prot=mmap.PROT_READ | 0 if not write else mmap.PROT_WRITE )
		self._length = None
		self.nColumns = 0
		self.columns = {}
		self.column_names = []
		self.current_slice = None
		self.fraction = 1.0
		self.rank1s = {}
		self.rank1names = []
		self.selected_row_index = None
		self.selected_serie_index = None
		self.row_selection_listeners = []
		self.serie_index_selection_listeners = []
		self.mask_listeners = []
		self.all_columns = {}
		self.all_column_names = []
		self.mask = None
		self.file_map = {filename: self.file}
		self.fileno_map = {filename: self.fileno}
		self.mapping_map = {filename: self.mapping}
		
	def addFile(self, filename, write=False):
		self.file_map[filename] = file(filename, "r+" if write else "r")
		self.fileno_map[filename] = self.file_map[filename].fileno()
		self.mapping_map[filename] = mmap.mmap(self.fileno_map[filename], 0, prot=mmap.PROT_READ | 0 if not write else mmap.PROT_WRITE )


	def selectMask(self, mask):
		self.mask = mask
		for mask_listener in self.mask_listeners:
			mask_listener(mask)
		
		
	def selectRow(self, index):
		self.selected_row_index = index
		for row_selection_listener in self.row_selection_listeners:
			row_selection_listener(index)
		
	def selectSerieIndex(self, serie_index):
		self.selected_serie_index = serie_index
		for serie_index_selection_listener in self.serie_index_selection_listeners:
			serie_index_selection_listener(serie_index)
		
		
	def close(self):
		self.file.close()
		self.mapping.close()
		
	def setFraction(self, fraction):
		self.fraction = fraction
		self.current_slice = (0, int(self._length * fraction))
		self.selectMask(None)
		self.selectRow(None)
		
	def addColumn(self, name, offset, length, dtype=np.float64, stride=None, filename=None):
		if filename is None:
			filename = self.filename
		mapping = self.mapping_map[filename]
		if self._length is not None and length != self._length:
			error("inconsistent length", "length of column %s is %d, while %d was expected" % (name, length, self._length))
		else:
			if self.current_slice is None:
				self.current_slice = (0, length)
				self.fraction = 1.
			self._length = length
			#print self.mapping, dtype, length if stride is None else length * stride, offset
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
			
	def addRank1(self, name, offset, length, length1, dtype=np.float64, stride=1, stride1=1, filename=None):
		if filename is None:
			filename = self.filename
		mapping = self.mapping_map[filename]
		if self._length is not None and length != self._length:
			error("inconsistent length", "length of column %s is %d, while %d was expected" % (name, length, self._length))
		else:
			if self.current_slice is None:
				self.current_slice = (0, length)
				self.fraction = 1.
			self._length = length
			#print self.mapping, dtype, length if stride is None else length * stride, offset
			rawlength = length * length1
			rawlength *= stride
			rawlength *= stride1
			#print rawlength, offset
			#print rawlength * 8, offset, self.mapping.size()
			#import pdb
			#pdb.set_trace()
			mmapped_array = np.frombuffer(mapping, dtype=dtype, count=rawlength, offset=offset)
			mmapped_array = mmapped_array.reshape((length1*stride1, length*stride))
			mmapped_array = mmapped_array[::stride1,::stride]
			self.rank1s[name] = mmapped_array
			self.rank1names.append(name)
			self.all_columns[name] = mmapped_array
			self.all_column_names.append(name)
			
			#self.column_names.sort()
			#self.nColumns += 1
			#self.nRows = self._length
			
import struct
class HansMemoryMapped(MemoryMapped):
	
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
		print "offset", self.dataOffset, self.formatSize, self.numberParticles, self.numberTimes
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
		
		#for i, name in enumerate(names):
		#	self.addColumn(name+"_mid", midoffset+8*i, length, dtype=np.float64, stride=stride)
		

		names = "x y z vx vy vz".split()
		#import pdb
		#pdb.set_trace()
		if 1:
			stride = self.formatSize/8 
			#stride1 = self.numberTimes #*self.formatSize/8 
			#print (self.numberParticles+1)
			#print stride, stride1
			for i, name in enumerate(names):
				#print name, offset
				# TODO: ask Hans for the self.numberTimes-1
				self.addRank1(name, offset+8*i, length=self.numberParticles+1, length1=self.numberTimes-1, dtype=np.float64, stride=stride, stride1=1)
				
		if filename_extra is not None:
			self.addFile(filename_extra)
			mapping = self.mapping_map[filename_extra]
			names = "J_r J_theta J_phi Theta_r Theta_theta Theta_phi Omega_r Omega_theta Omega_phi r_apo r_peri".split()
			offset = 0
			stride = 11
			#import pdb
			#pdb.set_trace()
			for i, name in enumerate(names):
				#print name, offset
				# TODO: ask Hans for the self.numberTimes-1
				self.addRank1(name, offset+8*i, length=self.numberParticles+1, length1=self.numberTimes-1, dtype=np.float64, stride=stride, stride1=1, filename=filename_extra)
				#print "min/max", np.min(self.rank1s[name]), np.max(self.rank1s[name]), offset+8*i, self.rank1s[name][0][0]
				#for i, name in enumerate(names):
				
				self.addColumn(name+"_0", offset+8*i, length, dtype=np.float64, stride=stride, filename=filename_extra)
				self.addColumn(name+"_last", offset+8*i + (self.numberParticles+1)*(self.numberTimes-2)*11*8, length, dtype=np.float64, stride=stride, filename=filename_extra)
			
			

		#for i, name in enumerate(names):
		#	self.addColumn(name+"_last", offset+8*i + (self.formatSize*(self.numberTimes-1)), length, dtype=np.float64, stride=stride)
		#for i, name in enumerate(names):
		#	self.addRank1(name, offset+8*i, (length, numberTimes), dtype=np.float64, stride=stride)
		
		
		
		
		#uint64 = np.frombuffer(self.mapping, dtype=dtype, count=length if stride is None else length * stride, offset=offset)
		

if __name__ == "__main__":
	path = "/Users/users/buist/research/2014 Simulation Data/12Orbits/Sigma/Orbitorb1.ac0.10000.100.5.orb.omega2"
	path = "/net/pannekoek/data/users/buist/Research/2014 Simulation Data/12Orbits/Integration/Orbitorb9.ac8.10000.100.5.orb.bin"

	hmm = HansMemoryMapped(path)

class Hdf5MemoryMapped(MemoryMapped):
	def __init__(self, filename, write=False):
		super(Hdf5MemoryMapped, self).__init__(filename, write=write)
		self.h5file = h5py.File(self.filename, "r+" if write else "r")
		if "data" in self.h5file:
			self.h5data = self.h5file["/data"]
			for column_name in self.h5data:
				column = self.h5data[column_name]
				offset = column.id.get_offset() 
				self.addColumn(column_name, offset, len(column), dtype=column.dtype)
			
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

class Hdf5MemoryMappedGadget(MemoryMapped):
	def __init__(self, filename, particleName, particleType):
		super(Hdf5MemoryMappedGadget, self).__init__(filename)
		self.particleType = particleType
		self.particleName = particleName
		self.name = self.name + "-" + self.particleName
		h5file = h5py.File(self.filename)
		#for i in range(1,4):
		key = "/PartType%d" % self.particleType
		if key not in h5file:
			raise KeyError, "%s does not exist" % key
		particles = h5file[key]
		for name in particles.keys():
			print name
			#name = "/PartType%d/Coordinates" % i
			data = particles[name]
			if isinstance(data, h5py.highlevel.Dataset): #array.shape
				array = data
				print array.shape, array.dtype
				shape = array.shape
				if len(shape) == 1:
					offset = array.id.get_offset() 
					self.addColumn(name, offset, data.shape[0], dtype=data.dtype)
				else:
					if name == "Coordinates":
						offset = data.id.get_offset() 
						self.addColumn("x", offset, data.shape[0], dtype=data.dtype, stride=3)
						self.addColumn("y", offset+4, data.shape[0], dtype=data.dtype, stride=3)
						self.addColumn("z", offset+8, data.shape[0], dtype=data.dtype, stride=3)
					elif name == "Velocity":
						offset = data.id.get_offset() 
						self.addColumn("vx", offset, data.shape[0], dtype=data.dtype, stride=3)
						self.addColumn("vy", offset+4, data.shape[0], dtype=data.dtype, stride=3)
						self.addColumn("vz", offset+8, data.shape[0], dtype=data.dtype, stride=3)
					else:
						print "unsupported column: %r of shape %r" % (name, array.shape)
		

class MemoryMappedGadget(MemoryMapped):
	def __init__(self, filename):
		super(MemoryMappedGadget, self).__init__(filename)
		#h5file = h5py.File(self.filename)
		import gavi.file.gadget
		length, posoffset, veloffset, header = gavi.file.gadget.getinfo(filename)
		print length, posoffset, posoffset
		print posoffset, hex(posoffset)
		self.addColumn("x", posoffset, length, dtype=np.float32, stride=3)
		self.addColumn("y", posoffset+4, length, dtype=np.float32, stride=3)
		self.addColumn("z", posoffset+8, length, dtype=np.float32, stride=3)
		
		self.addColumn("vx", veloffset, length, dtype=np.float32, stride=3)
		self.addColumn("vy", veloffset+4, length, dtype=np.float32, stride=3)
		self.addColumn("vz", veloffset+8, length, dtype=np.float32, stride=3)
		