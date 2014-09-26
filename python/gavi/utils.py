# -*- coding: utf-8 -*-

import time
import platform
import os


def os_open(document):
	"""Open document by the default handler of the OS, could be a url opened by a browser, a text file by an editor etc"""
	osname = platform.system().lower()
	if osname == "darwin":
		os.system("open \"" +document +"\"")
	if osname == "linux":
		os.system("xdg-open \"" +document +"\"&")
	if osname == "windows":
		os.system("start \"" +document +"\"")

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