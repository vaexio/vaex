import struct
import logging

import numpy as np

from vaex.dataset_mmap import DatasetMemoryMapped


logger = logging.getLogger("vaex.astro.gadget")


def getinfo(filename, seek=None):
	"""Read header data from Gadget data file 'filename' with Gadget file
	type 'gtype'. Returns offsets of positions and velocities."""
	DESC = '=I4sII'                                # struct formatting string
	HEAD = '=I6I6dddii6iiiddddii6ii60xI'        # struct formatting string
	keys = ('Npart', 'Massarr', 'Time', 'Redshift', 'FlagSfr', 'FlagFeedback', 'Nall', 'FlagCooling', 'NumFiles', 'BoxSize', 'Omega0', 'OmegaLambda', 'HubbleParam', 'FlagAge', 'FlagMetals', 'NallHW', 'flag_entr_ics', 'filename')
	f = open(filename, 'rb')
	
	"""Detects Gadget file type (type 1 or 2; resp. without or with the 16
	byte block headers)."""
	firstbytes = struct.unpack('I',f.read(4))
	if firstbytes[0] == 8:
		gtype = 2
	else:
		gtype = 1
	if gtype == 2:
		f.seek(16)
	else:
	 	f.seek(0)
	if seek is not None:
		f.seek(seek)
	raw = struct.unpack(HEAD,f.read(264))[1:-1]
	values = (raw[:6], raw[6:12]) + raw[12:16] + (raw[16:22],) + raw[22:30] + (raw[30:36], raw[36], filename)
	header = dict(list(zip(keys, values)))
	f.close()
	
	
	if gtype == 2:
		posoffset = (2*16 + (8 + 256))
	else:
		posoffset = (8 + 256)
	
	Npart = sum(header['Npart'])
	if gtype == 2:
		veloffset = 3*16 + (8 + 256) + (8 + 3*4*Npart)
	else:
		veloffset= (8 + 256) + (8 + 3*4*Npart)
	return Npart, posoffset+4, veloffset+4, header


class MemoryMappedGadget(DatasetMemoryMapped):
	snake_name = 'gadget'
	def __init__(self, filename, fs_options={}, fs=None):
		super(MemoryMappedGadget, self).__init__(filename)
		#h5file = h5py.File(self.filename)
		length, posoffset, veloffset, header = getinfo(filename)
		self._add("x", posoffset, length, dtype=np.float32, stride=3)
		self._add("y", posoffset+4, length, dtype=np.float32, stride=3)
		self._add("z", posoffset+8, length, dtype=np.float32, stride=3)

		self._add("vx", veloffset, length, dtype=np.float32, stride=3)
		self._add("vy", veloffset+4, length, dtype=np.float32, stride=3)
		self._add("vz", veloffset+8, length, dtype=np.float32, stride=3)
		self._freeze()

	def _add(self, name, offset, length, dtype, stride):
		ar = self._map_array(offset, length, dtype)
		ar = ar[::stride]
		self.add_column(name, ar)


	@classmethod
	def can_open(cls, path, *args, **kwargs):
		with open(path, 'rb') as f:
			try:
				first_words = struct.unpack('4I',f.read(4*4))
			except struct.error:
				return False
			if first_words[0] == 8 and first_words[2] == 8 and first_words[3] == 256:
				logger.debug('gadget file SnapFormat=2 detected')
				return True
			elif first_words[0] == 256:
				f.seek(256+4)
				if struct.unpack('I',f.read(4))[0] == 256:
					logger.debug('gadget file SnapFormat=1 detected')
					return True
		return False


