# -*- coding: utf-8 -*-
from __future__ import absolute_import
#from .... import logging

#import logging
import logging
#print(logging.__file__)
from logging import getLogger as _getLogger

#logging.basicConfig(level=logging.ERROR)
logging.basicConfig(level=logging.ERROR)
#logging.basicConfig(level=logging.DEBUG, filename="logging.txt")
#print "das" * 1000
rootlogger = logging.getLogger('vaex')
rootlogger.setLevel(logging.DEBUG)

def getLogger(*args, **kwargs):
	return _getLogger(*args, **kwargs)
	#return rootlogger.getLogger(*args, **kwargs)

LEVELS = {'debug': logging.DEBUG,
          'info': logging.INFO,
          'warning': logging.WARNING,
          'error': logging.ERROR,
          'critical': logging.CRITICAL}
def add_options(parser, defaultloglevel="info"):
	parser.add_option("--log-level", help="level of logging: debug,info,warning,error,critical [default: %default]", default=defaultloglevel)
	
def configure(opts):
	rootlogger.setLevel(LEVELS[opts.log_level])
	fh = logging.handlers.RotatingFileHandler("test.log", maxBytes=1024*100, backupCount=5)
	formatter = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s")
	fh.setLevel(logging.DEBUG)
	fh.setFormatter(formatter)
	rootlogger.addHandler(fh)

def configure_model(opts):
	rootlogger.setLevel(LEVELS[opts.log_level])
	fh = logging.handlers.RotatingFileHandler("test.log", maxBytes=1024*100, backupCount=5)
	formatter = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s")
	fh.setLevel(logging.DEBUG)
	fh.setFormatter(formatter)
	rootlogger.addHandler(fh)

if False:
	formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(name)s:%(message)s")
	
	ch = logging.StreamHandler()
	ch.setLevel(logging.DEBUG)
	ch.setFormatter(formatter)
	rootlogger.addHandler(ch)
	
	#iclogger = logging.Logger('schw.ic')
	#rootlogger.setLevel(DEBUG)
	print((dir(rootlogger)))
#rootlogger.basicConfig()
