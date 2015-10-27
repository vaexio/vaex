__author__ = 'breddels'
import numpy as np
from . import logging
import threading
from .dataset import Dataset, Subspace, Task
logger = logging.getLogger("vaex.remote")
import vaex.promise

#from twisted.internet import reactor
#from twisted.web.client import Agent
#from twisted.web.http_headers import Headers
from tornado.httpclient import AsyncHTTPClient, HTTPClient
from tornado.concurrent import Future

def wrap_future_with_promise(future):
	if isinstance(future, vaex.promise.Promise): # TODO: not so nice, sometimes we pass a promise
		return future
	promise = vaex.promise.Promise()
	def callback(future):
		print(("callback", future))
		e = future.exception()
		if e:
			print(("reject", e))
			promise.reject(e)
		else:
			promise.fulfill(future.result())
	future.add_done_callback(callback)
	return promise

import tornado.ioloop

import threading
import json
#import urllib.request, urllib.parse, urllib.error

try:
	from urllib.request import urlopen
	from urllib.parse import urlparse, urlencode
except ImportError:
	from urlparse import urlparse
	from urllib import urlopen, urlencode

import threading

class ServerRest(object):
	def __init__(self, hostname, port=5000, base_path="/", background=False, thread_mover=None):
		self.hostname = hostname
		self.port = port
		self.base_path = base_path
		#if async:
		event = threading.Event()
		self.thread_mover = thread_mover

		if True:
			#print "not running async"
			#if not tornado.ioloop.IOLoop.initialized():
			if True:
				#print "not init"
				#tornado.ioloop.IOLoop.clear_current()
				#tornado.ioloop.IOLoop.clear_instance()
				#tornado.ioloop.IOLoop.current().run_sync()
				def ioloop():
					#print "creating io loop"
					self.io_loop = tornado.ioloop.IOLoop.current(instance=False)
					if self.io_loop is None:
						event.set()
						return
						self.io_loop = tornado.ioloop.IOLoop.instance()
					event.set()
					self.io_loop.make_current()
					#print "starting"
					self.io_loop.start()
				thread = threading.Thread(target=ioloop)
				thread.setDaemon(True)
				thread.start()
			#else:
			#	print "already initialized"

			#print "waiting for io loop to be created"
			event.wait()
			#print self.io_loop
		#self.io_loop.make_current()
		#if async:
		self.http_client_async = AsyncHTTPClient()
		self.http_client = HTTPClient()
		#else:
		#self.async = async

	def wait(self):
		io_loop = tornado.ioloop.IOLoop.instance()
		io_loop.start()

	def fetch(self, url, transform, async=False, **kwargs):
		if async:
			future = self.http_client_async.fetch(url, **kwargs)
			return wrap_future_with_promise(future).then(transform).then(self._move_to_thread)
		else:
			return transform(self.http_client.fetch(url, **kwargs))


	def datasets(self, async=False):
		def wrap(result):
			#print "body", repr(result.body), result
			data = json.loads(result.body)
			return [DatasetRest(self, **kwargs) for kwargs in data["datasets"]]
		url = self._build_url("datasets")
		logger.debug("fetching: %r", url)
		return self.fetch(url, wrap, async=async)
		#return self._return(result, wrap)

	def _build_url(self, method):
		return "http://%s:%d%s%s" % (self.hostname, self.port, self.base_path, method)

	def _list_columns(self, name, async=False):
		def wrap(result):
			list = json.loads(result.body)
			return list
		url = self._build_url("datasets/%s/columns" % name)
		logger.debug("fetching: %r", url)
		#result = self.http_client.fetch(url)
		#return self._return(result, wrap)
		self.fetch(url, wrap, async=async)

	def _info(self, name, async=False):
		def wrap(result):
			list = json.loads(result.body)
			return list
		url = self._build_url("datasets/%s/info" % name)
		logger.debug("fetching: %r", url)
		return self.fetch(url, wrap, async=async)

	def open(self, name, async=False):
		def wrap(info):
			column_names = info["column_names"]
			full_length = info["length"]
			return DatasetRest(self, name, column_names, full_length)
		if async:
			return self._info(name, async=True).then(wrap)
		else:
			return wrap(self._info(name, async=False))
		#3result = self._info(name)
		#return self._return(result, wrap)

	def _async(self, promise):
		if self.async:
			return promise
		else:
			return promise.get()

	def minmax(self, expr, dataset_name, expressions, **kwargs):
		return self._simple(expr, dataset_name, expressions, "minmax", **kwargs)
		def wrap(result):
			data = json.loads(result.body)
			print(("data", data))
			return np.array([[data[expression]["min"], data[expression]["max"]] for expression in expressions])
		columns = "/".join(expressions)
		url = self._build_url("datasets/%s/minmax/%s" % (dataset_name, columns))
		logger.debug("fetching: %r", url)
		return self.fetch(url, wrap, async=async)
		#return self._return(result, wrap, async=async)

	def mean(self, expr, dataset_name, expressions, **kwargs):
		return self._simple(expr, dataset_name, expressions, "mean", **kwargs)
	def var(self, expr, dataset_name, expressions, **kwargs):
		return self._simple(expr, dataset_name, expressions, "var", **kwargs)
	def sum(self, expr, dataset_name, expressions):
		return self._simple(expr, dataset_name, expressions, "sum")
	def limits_sigma(self, expr, dataset_name, expressions, **kwargs):
		return self._simple(expr, dataset_name, expressions, "limits_sigma", **kwargs)

	def _simple(self, expr, dataset_name, expressions, name, async=False, **kwargs):
		def wrap(result):
			return np.array(json.loads(result.body)["result"])
		url = self._build_url("datasets/%s/%s" % (dataset_name, name))
		post_data = {key:json.dumps(value) for key, value in list(dict(kwargs).items())}
		post_data["masked"] = json.dumps(expr.is_masked)
		post_data.update(dict(expressions=json.dumps(expressions)))
		body = urlencode(post_data)
		return self.fetch(url+"?"+body, wrap, async=async, method="GET")
		#return self._return(result, wrap)

	def histogram(self, expr, dataset_name, expressions, size, limits, weight=None, async=False):
		def wrap(result):
			data = np.fromstring(result.body)
			shape = (size,) * len(expressions)
			data = data.reshape(shape)
			return data
		url = self._build_url("datasets/%s/histogram" % (dataset_name,))
		logger.debug("fetching: %r", url)
		post_data = dict(expressions=json.dumps(expressions), size=json.dumps(size),
						 weight=json.dumps(weight),
						 limits=json.dumps(limits.tolist()), masked=json.dumps(expr.is_masked))
		body = urlencode(post_data)
		return self.fetch(url+"?"+body, wrap, async=async, method="GET")
		#return self._return(result, wrap)

	def __return(self, response_or_future, transform):
		if self.async:
			future = response_or_future
			return wrap_future_with_promise(future).then(transform).then(self._move_to_thread)
		else:
			response = response_or_future
			return transform(response)

	def _move_to_thread(self, result):
		promise = Promise()
		#def do(value):
		#	return value
		#promise.then(do)
		logger.debug("the other thread should fulfil the result to this promise")
		self.thread_mover(promise, result)
		return promise


	def select(self, dataset_name, expression, async=False, **kwargs):
		name = "select"
		def wrap(result):
			return np.array(json.loads(result.body))
		url = self._build_url("datasets/%s/%s" % (dataset_name, name))
		post_data = {key:json.dumps(value) for key, value in list(dict(kwargs).items())}
		post_data.update(dict(expression=json.dumps(expression)))
		body = urlencode(post_data)
		return self.http_client.fetch(url+"?"+body, wrap, async=async, method="GET")
		#return self._return(result, wrap)


class SubspaceRemote(Subspace):
	def toarray(self, list):
		return np.array(list)

	@property
	def dimension(self):
		return len(self.expressions)

	def _promise(self, promise):
		"""Helper function for returning tasks results, result when immediate is True, otherwise the task itself, which is a promise"""
		if self.async:
			return promise
		else:
			return promise

	def minmax(self):
		return self._promise(self.dataset.server.minmax(self, self.dataset.name, self.expressions, async=self.async))
		#return self._task(task)

	def histogram(self, limits, size=256, weight=None):
		return self._promise(self.dataset.server.histogram(self, self.dataset.name, self.expressions, size=size, limits=limits, weight=weight, async=self.async))

	def mean(self):
		return self.dataset.server.mean(self, self.dataset.name, self.expressions, async=self.async)

	def var(self, means=None):
		return self.dataset.server.var(self, self.dataset.name, self.expressions, means=means, async=self.async)

	def sum(self):
		return self.dataset.server.sum(self, self.dataset.name, self.expressions, async=self.async)

	def limits_sigma(self, sigmas=3, square=False):
		return self.dataset.server.limits_sigma(self, self.dataset.name, self.expressions, sigmas=sigmas, square=square, async=self.async)

	def plot_(self, grid=None, limits=None, center=None, f=lambda x: x,**kwargs):
		import pylab
		if limits is None:
			limits = self.limits_sigma()
		if center is not None:
			limits = np.array(limits) - np.array(center).reshape(2,1)
		if grid is None:
			grid = self.histogram(limits=limits)
		pylab.imshow(f(grid), extent=np.array(limits).flatten(), origin="lower", **kwargs)



class DatasetRemote(Dataset):
	def __init__(self, name, server, column_names):
		self.is_local = False
		super(DatasetRemote, self).__init__(name, column_names)
		self.server = server

class DatasetRest(DatasetRemote):
	def __init__(self, server, name, column_names, full_length):
		DatasetRemote.__init__(self, name, server.hostname, column_names)
		self.server = server
		self.name = name
		self.column_names = column_names
		self._full_length = full_length
		self.filename = "http://%s:%s/%s" % (server.hostname, server.port, name)
		#self.host = host
		#self.http_client = AsyncHTTPClient()
		#future = http_client.fetch(self._build_url("datasets"))
		#fetch_future.add_done_callback(
		self.fraction = 1

		self.executor = DummyExecutor()

	def __call__(self, *expressions, **kwargs):
		return SubspaceRemote(self, expressions, self.executor, async=kwargs.get("async", False))

	def select(self, expression):
		return self.server.select(self.name, expression)

	def set_fraction(self, f):
		# TODO: implement fractions for remote
		self.fraction = f

	def __len__(self):
		return self._full_length

	def full_length(self):
		return self._full_length


# we may get rid of this when we group together tasks
class DummyExecutor(object):
	def execute(self):
		print("dummy execute")


