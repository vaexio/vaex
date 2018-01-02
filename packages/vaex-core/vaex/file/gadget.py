# -*- coding: utf-8 -*-

import struct

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

