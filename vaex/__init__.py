"""
Vaex is a library for dealing with big tabular data.

The most important class (datastructure) in vaex is the :class:`.Dataset`. A dataset is obtained by either, opening
the example dataset:

>>> import vaex as vx
>>> t = vx.example()

Or opening a file:

>>> t1 = vx.open("somedata.hdf5")
>>> t2 = vx.open("somedata.fits")
>>> t3 = vx.open("somedata.csv")

Or connecting to a remove server:

>>> tbig = vx.open("http://bla.com/bigtable")

The main purpose of vaex is to provide statistics, such as mean, count, sum, standard deviation, per columns, possibly
with a selection, and on a regular grid.

To count the number of rows:

>>> t = vx.examples()
>>> t.count()
330000.0

Or the number of valid values, which for this dataset is the same:

>>> t.count("x")
330000.0

Count them on a regular grid:

>>> t.count("x", binby=["x", "y"], shape=(4,4))
array([[   902.,   5893.,   5780.,   1193.],
       [  4097.,  71445.,  75916.,   4560.],
       [  4743.,  71131.,  65560.,   4108.],
       [  1115.,   6578.,   4382.,    821.]])

Visualise it using matplotlib:

>>> t.plot("x", "y", show=True)
<matplotlib.image.AxesImage at 0x1165a5090>


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
import vaex.file
import vaex.export

import vaex.datasets
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
	import vaex
	if path in aliases:
		path = aliases[path]
	if path.startswith("http://") or path.startswith("ws://"): # TODO: think about https and wss
		server, dataset = path.rsplit("/", 1)
		server = vaex.server(server, **kwargs)
		datasets = server.datasets(as_dict=True)
		if dataset not in datasets:
			raise KeyError("no such dataset '%s' at server, possible dataset names: %s" %  (dataset, " ".join(datasets.keys())))
		return datasets[dataset]
	if path.startswith("cluster"):
		import vaex.distributed
		return vaex.distributed.open(path, *args, **kwargs)
	else:
		return vaex.file.open(path, *args, **kwargs)

def open_many(filenames):
	"""Open a list of filenames, and return a dataset with all datasets cocatenated

	:param list[str] filenames: list of filenames/paths
	:rtype: Dataset
	"""
	datasets = []
	for filename in filenames:
		filename = filename.strip()
		if filename and filename[0] != "#":
			datasets.append(open(filename))
	return vaex.dataset.DatasetConcatenated(datasets=datasets)

def from_samp(username=None, password=None):
	"""Connect to a SAMP Hub and wait for a single table load event, disconnect, download the table and return the dataset

	Useful if you want to send a single table from say TOPCAT to vaex in a python console or notebook
	"""
	print("Waiting for SAMP message...")
	import vaex.samp
	t = vaex.samp.single_table(username=username, password=password)
	return from_astropy_table(t.to_table())


def from_astropy_table(table):
	return vaex.file.other.DatasetAstropyTable(table=table)

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

def from_scalars(name="scalars", **kwargs):
	"""Similar to from_arrays, but convenient for a dataset of length 1

	>>> ds = vx.from_arrays("test", x=1, y=2)
	"""
	import numpy as np
	return from_arrays(name, **{k:np.array([v]) for k, v in kwargs.items()})


def from_pandas(df, name="pandas"):
	"""Create an in memory dataset from a pandas dataframe

	:param: pandas.DataFrame df: Pandas dataframe
	:param: name: unique for the dataset

	>>> import pandas as pd
	>>> df = pd.from_csv("test.csv")
	>>> ds = vx.from_pandas(df, name="test")
	"""
	dataset = vaex.dataset.DatasetArrays(name)
	for name in df.columns:
		dataset.add_column(name, df[name].values)
	return dataset

def from_ascii(path, seperator=None, names=True, skip_lines=0, skip_after=0, **kwargs):
	import vaex.ext.readcol as rc
	ds = vaex.dataset.DatasetArrays(path)
	if names not in [True, False]:
		namelist = names
		names = False
	else:
		namelist = None
	data = rc.readcol(path, fsep=seperator, asdict=namelist is None, names=names, skipline=skip_lines, skipafter=skip_after, **kwargs)
	if namelist:
		for name, array in zip(namelist, data.T):
			ds.add_column(name, array)
	else:
		for name, array in data.items():
			ds.add_column(name, array)
	return ds

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
	path = utils.get_data_file("helmi-dezeeuw-2000-10p.hdf5")
	return open(path) if path else None


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
format = "%(levelname)s:%(threadName)s:%(name)s:%(message)s"
logging.basicConfig(level=logging.INFO, format=format)
#logging.basicConfig(level=logging.DEBUG)
set_log_level_warning()
import os
import_script = os.path.expanduser("~/.vaex/vaex_import.py")
if os.path.exists(import_script):
	try:
		execfile(import_script)
	except:
		import traceback
		traceback.print_tb()