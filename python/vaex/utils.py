# -*- coding: utf-8 -*-

import time
import platform
import os
import sys
import numpy as np
import math

is_frozen = getattr(sys, 'frozen', False)

def subdivide(length, parts=None, max_length=None):
	"""Generates a list with start end stop indices of length parts, [(0, length/parts), ..., (.., length)]"""
	if max_length:
		i1 = 0
		done = False
		while not done:
			i2 = min(length, i1 + max_length)
			#print i1, i2
			yield i1, i2
			i1 = i2
			if i1 == length:
				done = True
	else:
		part_length = int(math.ceil(float(length)/parts))
		#subblock_count = math.ceil(total_length/subblock_size)
		#args_list = []
		for index in range(parts):
			i1, i2 = index * part_length, min(length, (index +1) * part_length)
			yield i1, i2

def linspace_centers(start, stop, N):
	return np.arange(N) / (N+0.) * (stop-start) + float(stop-start)/N/2 + start

def multisum(a, axes):
	correction = 0
	for axis in axes:
		a = np.nansum(a, axis=axis-correction)
		correction += 1
	return a

def disjoined(data):
	# create marginalized distributions and multiple them together
	data_disjoined = None
	dim = len(data.shape)
	for d in range(dim):
		axes = list(range(dim))
		axes.remove(d)
		data1d = multisum(data, axes)
		shape = [1 for k in range(dim)]
		shape[d] = len(data1d)
		data1d = data1d.reshape(tuple(shape))
		if d == 0:
			data_disjoined = data1d
		else:
			data_disjoined = data_disjoined * data1d
	return data_disjoined



def get_data_file(filename):
	try: # this works for egg like stuff, but fails for py2app apps
		from pkg_resources import Requirement, resource_filename
		path = resource_filename(Requirement.parse("vaex"), filename)
		if os.path.exists(path):
			return path
	except:
		pass
	# this is where we expect data to be in normal installations
	for extra in ["", "data", "data/dist"]:
		path = os.path.join(os.path.dirname(__file__), "..", "..", extra, filename)
		#print "try", path
		if os.path.exists(path):
			return path
		path = os.path.join(sys.prefix, extra, filename)
		#print "try", path
		if os.path.exists(path):
			return path
		# if all fails..
		path = os.path.join(get_root_path(), extra, filename)
		#print "try", path
		if os.path.exists(path):
			return path

def get_root_path():
	osname = platform.system().lower()
	#if (osname == "linux") and is_frozen: # we are using pyinstaller
	if is_frozen: # we are using pyinstaller or py2app
		return os.path.dirname(sys.argv[0])
	else:
		return os.path.abspath(".")
	
	
def os_open(document):
	"""Open document by the default handler of the OS, could be a url opened by a browser, a text file by an editor etc"""
	osname = platform.system().lower()
	if osname == "darwin":
		os.system("open \"" +document +"\"")
	if osname == "linux":
		cmd = "xdg-open \"" +document +"\"&"
		os.system(cmd)
	if osname == "windows":
		os.system("start \"" +document +"\"")

def filesize_format(value):
	for unit in ['bytes','KiB','MiB','GiB']:
		if value < 1024.0:
			return "%3.1f%s" % (value, unit)
		value /= 1024.0
	return "%3.1f%s" % (value, 'TiB')



log_timer = True
class Timer(object):
	def __init__(self, name=None, logger=None):
		self.name = name
		self.logger = logger

	def __enter__(self):
		if log_timer:
			if self.logger:
				self.logger.debug("%s starting" % self.name)
			else:
				print(('[%s starting]...' % self.name))
			self.tstart = time.time()

	def __exit__(self, type, value, traceback):
		if log_timer:
			msg = "%s done, %ss elapsed" % (self.name, time.time() - self.tstart)
			if self.logger:
				self.logger.debug(msg)
			else:
				print(msg)
			if type or value or traceback:
				print((type, value, traceback))
		return False