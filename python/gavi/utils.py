# -*- coding: utf-8 -*-

import time

def filesize_format(value):
	for unit in ['bytes','KiB','MiB','GiB']:
		if value < 1024.0:
			return "%3.1f%s" % (value, unit)
		value /= 1024.0
	return "%3.1f%s" % (num, 'TiB')



class Timer(object):
	def __init__(self, name=None):
		self.name = name

	def __enter__(self):
		print '[%s starting]...' % self.name
		self.tstart = time.time()

	def __exit__(self, type, value, traceback):
		if self.name:
			print '[%s]' % self.name,
		print 'Elapsed: %s' % (time.time() - self.tstart)
		print type, value, traceback
		return False