"""
Vaex is...
"""# -*- coding: utf-8 -*-
from __future__ import print_function

try:
	from . import version
except:
	import sys
	print("version file not found, please run git/hooks/post-commit or git/hooks/post-checkout and/or install them as hooks (see git/README)", file=sys.stderr)
	raise

__version__ = version.versionstring
#__pre_release_name__ = version.pre_release
__version_tuple__ = version.versiontuple
__program_name__ = "vaex"
#__version_name__ = version.versiontring
#__release_name_ = version.versiontring[:]
#__clean_release__ = "%d.%d.%d" % (__version_tuple__)
__full_name__ = __program_name__ + "-" + __version__
#__clean_name__ =  __program_name__ + "-" + __clean_release__

__build_name__ = __full_name__ + "-" +version.osname


import vaex.dataset
#import vaex.plot
from vaex.dataset import Dataset
from vaex.remote import ServerRest
del ServerRest, Dataset

def app(*args, **kwargs):
	"""Create a vaex app, the QApplication mainloop must be started.

	In ipython notebook/jupyter do the following:
	import vaex.ui.main # this causes the qt api level to be set properly
	import vaex as xs
	Next cell:
	%gui qt
	Next cell
	app = vx.app()

	From now on, you can run the app along with jupyter

	"""


	import vaex.ui.main
	return vaex.ui.main.VaexApp()
def open(path, *args, **kwargs):
	"""Open a dataset from file given by path

	:param str path: local or absolute path to file
	:param args: extra arguments for file readers that need it
	:param kwargs: extra keyword arguments
	:return: return dataset if file is supported, otherwise None
	:rtype: Dataset

	:Example:

	>>> import vaex as vx
	>>> vx.open('myfile.hdf5')
	<vaex.dataset.Hdf5MemoryMapped at 0x1136ee3d0>
	>>> vx.open('gadget_file.hdf5', 3) # this will read only particle type 3
	<vaex.dataset.Hdf5MemoryMappedGadget at 0x1136ef3d0>
	"""
	if path in aliases:
		path = aliases[path]
	if path.startswith("http://") or path.startswith("ws://"): # TODO: think about https and wss
		server, dataset = path.rsplit("/", 1)
		server = vaex.server(server, **kwargs)
		datasets = server.datasets(as_dict=True)
		if dataset not in datasets:
			raise KeyError("no such dataset '%s' at server, possible dataset names: %s" %  (dataset, " ".join(datasets.keys())))
		return datasets[dataset]
	else:
		return vaex.dataset.load_file(path, *args, **kwargs)

def open_many(filenames):
	"""Open a list of filenames, and return a dataset with all datasets cocatenated

	:param list[str] filenames: list of filenames/paths
	:rtype: Dataset
	"""
	datasets = []
	for filename in filenames:
		datasets.append(open(filename.strip()))
	return vaex.dataset.DatasetConcatenated(datasets=datasets)

def open_samp_single():
	"""Connect to a SAMP Hub and wait for a single table load event, disconnect, download the table and return the dataset

	Useful if you want to send a single table from say TOPCAT to vaex in a python console or notebook
	"""
	from vaex.samp import SampSingle
	from astropy.table import Table
	#samp = Samp(daemon=False)
	#samp.tableLoadCallbacks.
	samp = SampSingle()
	url = samp.wait_for_table()
	table = Table.read(url)
	return vaex.dataset.DatasetAstropyTable(table=table)



def from_arrays(name="array", **arrays):
	"""Create an in memory dataset from numpy arrays


	:param: str name: name of dataset
	:param: arrays: keyword arguments with arrays

	:Example:
	>>> x = np.arange(10)
	>>> y = x ** 2
	>>> dataset = vx.from_arrays("test", x=x, y=y)


	"""
	dataset = vaex.dataset.DatasetArrays(name)
	for name, array in arrays.items():
		dataset.add_column(name, array)
	return dataset

import vaex.settings
aliases = vaex.settings.main.auto_store_dict("aliases")

# py2/p3 compatibility
try:
	from urllib.parse import urlparse
except ImportError:
	from urlparse import urlparse

def server(url, **kwargs):
	"""Connect to hostname supporting the vaex web api

	:param str hostname: hostname or ip address of server
	:return vaex.dataset.ServerRest: returns a server object, note that it does not connect to the server yet, so this will always succeed
	:rtype: ServerRest
	"""
	url = urlparse(url)
	if url.scheme == "ws":
		websocket = True
	else:
		websocket = False
	assert url.scheme in ["ws", "http"]
	port = url.port
	base_path = url.path
	hostname = url.hostname
	return vaex.remote.ServerRest(hostname, base_path=base_path, port=port, websocket=websocket, **kwargs)

def example():
	"""Returns an example dataset which comes with vaex for testing/learning purposes

	:rtype: vaex.dataset.Dataset
	"""
	from . import utils
	return open(utils.get_data_file("helmi-dezeeuw-2000-10p.hdf5"))


def zeldovich(dim=2, N=256, n=-2.5, t=None, scale=1, seed=None):
	"""Creates a zeldovich dataset
	"""
	return vaex.dataset.Zeldovich(dim=dim, N=N, n=n, t=t, scale=scale)

def set_log_level_debug():
	"""set log level to debug"""
	import logging
	logging.getLogger("vaex").setLevel(logging.DEBUG)

def set_log_level_info():
	"""set log level to info"""
	import logging
	logging.getLogger("vaex").setLevel(logging.INFO)

def set_log_level_warning():
	"""set log level to warning"""
	import logging
	logging.getLogger("vaex").setLevel(logging.WARNING)

def set_log_level_exception():
	"""set log level to exception"""
	import logging
	logging.getLogger("vaex").setLevel(logging.FATAL)

def set_log_level_off():
	"""Disabled logging"""
	import logging
	logging.disable(logging.CRITICAL)


import logging
logging.basicConfig(level=logging.ERROR)
set_log_level_info()
import os
import_script = os.path.expanduser("~/.vaex/vaex_import.py")
if os.path.exists(import_script):
	try:
		execfile(import_script)
	except:
		import traceback
		traceback.print_tb()