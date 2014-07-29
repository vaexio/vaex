# -*- coding: utf-8 -*-

import time


class Timer(object):
	def __init__(self, name=None):
		self.name = name

	def __enter__(self):
		print '[%s starting]...' % self.name,
		self.tstart = time.time()

	def __exit__(self, type, value, traceback):
		if self.name:
			print '[%s]' % self.name,
		print 'Elapsed: %s' % (time.time() - self.tstart)
