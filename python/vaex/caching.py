__author__ = 'maartenbreddels'
from cachetools import Cache, LRUCache
import os
import numpy as np
import logging

logger = logging.getLogger("vaex.caching")

class MemoryCache(LRUCache):
	def __init__(self, maxsize, missing=None, getsizeof=None, delitem=lambda key: None):
		LRUCache.__init__(self, maxsize, missing, getsizeof)
		self.__delitem = delitem

	def __delitem__(self, key):
		self.__delitem(key)
		super(MemoryCache, self).__delitem__(key)

class FileWrapped(object):
	def __init__(self, maxsize, missing=None, getsizeof=None, delitem=lambda key: None):
		LRUCache.__init__(self, maxsize, missing, getsizeof)
		self.__delitem = delitem

	def __delitem__(self, key):
		self.__delitem(key)
		super(FileWrapped, self).__delitem__(key)

	def __setitem__(self, key, value, cache_setitem=Cache.__setitem__):
		cache_setitem(self, key, value)


MB = 1024**2
GB = MB * 1024
class SelectionCache(object):
	"""
	:type LRUCache: memory_cache
	:type LRUCache: file_cache
	"""
	def __init__(self, max_memory_size=1*GB, max_disk_size=4*GB):
		#self.file_cache = MemoryCache(max_disk_size, getsizeof=self.getsizeof, delitem=self.on_delete_file)
		#self.memory_cache = MemoryCache(max_memory_size, missing=self.get_from_disk, delitem=self.on_delete_memory, )
		self.memory_cache = LRUCache(max_memory_size)

	def __getitem__(self, key):
		return self.memory_cache[key]

	def __setitem__(self, key, value):
		self.memory_cache[key] = value

	def __contains__(self, key):
		return key in self.memory_cache or key in self.file_cache


import collections
class NumpyFileDict(object): #collections.MutableMapping):
	def __init__(self):
		self.key_to_path = {}

	def __filename(self, key):
		#print "key", key
		return "_".join(map(str, key)) + ".npy"
	def path(self, key):
		if isinstance(key, tuple):
			key = self.__filename(key)
		else:
			key = repr(key)
		#print "key=", key
		return key

	#def __len__(self):
	#	return len(self.key_to_path)

	def __iter__(self):
		for key in self.key_to_path.keys():
			yield key #self[key]
		#return iter(self.key_to_path)

	def __contains__(self, key):
		return key in self.key_to_path

	def __getitem__(self, key):
		#print("get", key)
		if key in self.key_to_path:
			path = self.path(key)
			return np.load(path)
		else:
			raise KeyError, key

	def __setitem__(self, key, value):
		logger.debug("set %r", key)
		path = self.path(key)
		np.save(path, value)
		self.key_to_path[key] = path

	def __delitem__(self, key):
		logger.debug("delete %r", key)
		path = self.path(key)
		os.remove(path)
		del self.key_to_path[key]

if __name__ == "__main__":
	logger.setLevel("DEBUG")
	def f(key):
		a, b = key
		return np.arange(a, b)
	np_dict = NumpyFileDict()
	#np_dict[(1,2)] = np.arange(10)
	cache = Cache(2, missing=f, dict_value=np_dict)
	print "cache[1,3] =", cache[(1, 3)], "..."
	print "cache[1,13] =", cache[(1, 13)], "..."
	print cache[(2, 3)]
	print cache[(3, 4)]
	print cache[(3, 5)]
	print cache[(3, 6)]
	print "keys", cache.keys()
	for ar in np_dict:
		print "-->", ar


