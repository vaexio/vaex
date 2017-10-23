"""
Vaex is a library for dealing with big tabular data.

The most important class (datastructure) in vaex is the :class:`.Dataset`. A dataset is obtained by either, opening
the example dataset:

>>> import vaex
>>> t = vaex.example()

Or using :func:`open` or :func:`from_csv`, to open a file:

>>> t1 = vaex.open("somedata.hdf5")
>>> t2 = vaex.open("somedata.fits")
>>> t3 = vaex.from_csv("somedata.csv")

Or connecting to a remove server:

>>> tbig = vaex.open("http://bla.com/bigtable")

The main purpose of vaex is to provide statistics, such as mean, count, sum, standard deviation, per columns, possibly
with a selection, and on a regular grid.

To count the number of rows:

>>> t = vaex.example()
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


import glob
import vaex.dataset
from . import stat
#import vaex.file
#import vaex.export
from .delayed import delayed

import vaex.datasets
#import vaex.plot
#from vaex.dataset import Dataset
#del ServerRest, Dataset

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
	try:
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
			import vaex.file
			return vaex.file.open(path, *args, **kwargs)
	except:
		logging.getLogger("vaex").error("error opening %r" % path)
		raise

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
	import vaex.file.other
	return vaex.file.other.DatasetAstropyTable(table=table)

def from_items(*items):
	"""Create an in memory dataset from numpy arrays, in contrast to from_arrays this keeps the order of columns intact

	:param: items: list of [(name, numpy array), ...]

	:Example:
	>>> x = np.arange(10)
	>>> y = x ** 2
	>>> dataset = vx.from_items(('x', x), ('y', y))


	"""
	dataset = vaex.dataset.DatasetArrays("array")
	for name, array in items:
		dataset.add_column(name, array)
	return dataset

def from_arrays(**arrays):
	"""Create an in memory dataset from numpy arrays


	:param: arrays: keyword arguments with arrays

	:Example:
	>>> x = np.arange(10)
	>>> y = x ** 2
	>>> dataset = vx.from_arrays(x=x, y=y)


	"""
	dataset = vaex.dataset.DatasetArrays("array")
	for name, array in arrays.items():
		dataset.add_column(name, array)
	return dataset

def from_scalars(**kwargs):
	"""Similar to from_arrays, but convenient for a dataset of length 1

	>>> ds = vx.from_scalars(x=1, y=2)
	"""
	import numpy as np
	return from_arrays( **{k:np.array([v]) for k, v in kwargs.items()})


def from_pandas(df, name="pandas", copy_index=True, index_name="index"):
	"""Create an in memory dataset from a pandas dataframe

	:param: pandas.DataFrame df: Pandas dataframe
	:param: name: unique for the dataset

	>>> import pandas as pd
	>>> df = pd.from_csv("test.csv")
	>>> ds = vx.from_pandas(df, name="test")
	"""
	import six
	dataset = vaex.dataset.DatasetArrays(name)
	def add(name, column):
		values = column.values
		if isinstance(values[0], six.string_types):
			values = values.astype("S")
		try:
			dataset.add_column(name, values)
		except Exception as e:
			print("could not convert column %s, error: %r, will try to convert it to string" % (name, e))
			try:
				values = values.astype("S")
				dataset.add_column(name, values)
			except Exception as e:
				print("Giving up column %s, error: %r" (name, e))
	for name in df.columns:
		add(name, df[name])
	if copy_index:
		add(index_name, df.index)
	return dataset

def from_ascii(path, seperator=None, names=True, skip_lines=0, skip_after=0, **kwargs):
	"""
	Create an in memory dataset from an ascii file (whitespace seperated by default).

	>>> ds = vx.from_ascii("table.asc")
	>>> ds = vx.from_ascii("table.csv", seperator=",", names=["x", "y", "z"])

	:param path: file path
	:param seperator: value seperator, by default whitespace, use "," for comma seperated values.
	:param names: If True, the first line is used for the column names, otherwise provide a list of strings with names
	:param skip_lines: skip lines at the start of the file
	:param skip_after: skip lines at the end of the file
	:param kwargs:
	:return:
	"""

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

def from_csv(filename_or_buffer, **kwargs):
	"""Shortcut to read a csv file using pandas and convert to a dataset directly"""
	import pandas as pd
	return from_pandas(pd.read_csv(filename_or_buffer, **kwargs))

def read_csv(filepath_or_buffer, **kwargs):
	'''Alias to from_csv'''
	return from_csv(filenames, **kwargs)

def _convert_name(filenames, shuffle=False):
	'''Convert a filename (or list of) to a filename with .hdf5 and optionally a -shuffle suffix'''
	if not isinstance(filenames, (list, tuple)):
		filenames = [filenames]
	base = filenames[0]
	if shuffle:
		base += '-shuffle'
	if len(filenames) > 1:
		return base + "_and_{}_more.hdf5".format(len(filenames))
	else:
		return base + ".hdf5"

def read_csv_and_convert(path, shuffle=False, copy_index=True, **kwargs):
	'''Convert a path (or glob pattern) to a single hdf5 file, will open the hdf5 file if exists

	Example:
		>>> vaex.read_csv_and_convert('test-*.csv', shuffle=True)  # this may take a while
		>>> vaex.read_csv_and_convert('test-*.csv', shuffle=True)  # 2nd time it is instant

	:param str path: path of file or glob pattern for multiple files
	:param bool shuffle: shuffle dataset when converting to hdf5
	:param bool copy_index: by default pandas will create an index (row number), set to false if you want to drop that
	:param kwargs: parameters passed to pandas' read_cvs

	'''
	from concurrent.futures import ProcessPoolExecutor
	import pandas as pd
	filenames = glob.glob(path)
	if len(filenames) > 1:
		filename_hdf5 = _convert_name(filenames, shuffle=shuffle)
		filename_hdf5_noshuffle = _convert_name(filenames, shuffle=False)
		if not os.path.exists(filename_hdf5):
			if not os.path.exists(filename_hdf5_noshuffle):
				#with ProcessPoolExecutor() as executor:
				#	executor.submit(read_csv_and_convert, filenames, shuffle=shuffle, **kwargs)
				for filename in filenames:
					read_csv_and_convert(filename, shuffle=shuffle, copy_index=copy_index, **kwargs)
				ds = open_many([_convert_name(k, shuffle=shuffle) for k in filenames])
			else:
				ds = open(filename_hdf5_noshuffle)
			ds.export_hdf5(filename_hdf5, shuffle=shuffle)
		return open(filename_hdf5)
	else:
		filename = filenames[0]
		filename_hdf5 = _convert_name(filename, shuffle=shuffle)
		filename_hdf5_noshuffle = _convert_name(filename, shuffle=False)
		if not os.path.exists(filename_hdf5):
			if not os.path.exists(filename_hdf5_noshuffle):
				df = pd.read_csv(filename, **kwargs)
				ds = from_pandas(df, copy_index=copy_index)
			else:
				ds = open(filename_hdf5_noshuffle)
			ds.export_hdf5(filename_hdf5, shuffle=shuffle)
		return open(filename_hdf5)




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
	from vaex.remote import ServerRest
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

def example(download=True):
	"""Returns an example dataset which comes with vaex for testing/learning purposes

	:rtype: vaex.dataset.Dataset
	"""
	from . import utils
	path = utils.get_data_file("helmi-dezeeuw-2000-10p.hdf5")
	if path is None and download:
		return vaex.datasets.helmi_de_zeeuw_10percent.fetch()
	return open(path) if path else None


def zeldovich(dim=2, N=256, n=-2.5, t=None, scale=1, seed=None):
	"""Creates a zeldovich dataset
	"""
	import vaex.file
	return vaex.file.other.Zeldovich(dim=dim, N=N, n=n, t=t, scale=scale)

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

import logging
logger = logging.getLogger('vaex')

import pkg_resources
for entry in pkg_resources.iter_entry_points(group='vaex.namespace'):
	logger.debug('adding vaex namespace: ' + entry.name)
	try:
	    add_namespace = entry.load()
	    add_namespace()
	except Exception:
		logger.exception('issue loading ' + entry.name)

for entry in pkg_resources.iter_entry_points(group='vaex.plugin'):
	logger.debug('adding vaex plugin: ' + entry.name)
	try:
	    add_namespace = entry.load()
	    add_namespace()
	except Exception:
		logger.exception('issue loading ' + entry.name)
