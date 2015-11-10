__author__ = 'breddels'
import numpy as np
from . import logging
import threading
import __builtin__
from .dataset import Dataset, Subspace, Task
import vaex.promise
import vaex.settings
try:
	import Cookie  # py2
except ImportError:
	import http.cookies as Cookie  # py3
import cookielib
#from twisted.internet import reactor
#from twisted.web.client import Agent
#from twisted.web.http_headers import Headers
from tornado.httpclient import AsyncHTTPClient, HTTPClient
import tornado.httputil
from tornado.concurrent import Future
from tornado import gen

logger = logging.getLogger("vaex.remote")

DEFAULT_REQUEST_TIMEOUT = 60 * 5 # 5 minutes

def wrap_future_with_promise(future):
	if isinstance(future, vaex.promise.Promise): # TODO: not so nice, sometimes we pass a promise
		return future
	promise = vaex.promise.Promise()
	def callback(future):
		#print(("callback", future, future.result()))
		e = future.exception()
		if e:
			#print(("reject", e))
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

def _check_error(object):
	if "error" in object:
		raise RuntimeError, "Server responded with error: %r" % object["error"]


class ServerRest(object):
	def __init__(self, hostname, port=5000, base_path="/", background=False, thread_mover=None):
		self.hostname = hostname
		self.port = port
		self.base_path = base_path
		#if async:
		event = threading.Event()
		self.thread_mover = thread_mover

		def ioloop_threaded():
			#print "creating io loop"
			logger.debug("creating tornado io_loop")
			#self.io_loop = tornado.ioloop.IOLoop.instance() #tornado.ioloop.IOLoop.current(instance=True)
			self.io_loop = tornado.ioloop.IOLoop().instance()
			#if self.io_loop is None:
				#logger.debug("creating tornado io_loop")
				#event.set()
				#return
				#self.io_loop = tornado.ioloop.IOLoop.instance()
			event.set()
			#self.io_loop.make_current()
			#print "starting"
			logger.debug("started tornado io_loop...")

			self.io_loop.start()
			logger.debug("stopped tornado io_loop")

		io_loop = tornado.ioloop.IOLoop.current(instance=False)
		if io_loop is None:
			logger.debug("no current io loop, starting it in thread")
			thread = threading.Thread(target=ioloop_threaded)
			thread.setDaemon(True)
			thread.start()
			event.wait()
		else:
			self.io_loop = io_loop

		self.io_loop.make_current()
		#if async:
		self.http_client_async = AsyncHTTPClient()
		self.http_client = HTTPClient()
		self.user_id = vaex.settings.webclient.get("cookie.user_id")
		#self.cookiejar = cookielib.FileCookieJar()
		#else:
		#self.async = async

	def wait(self):
		io_loop = tornado.ioloop.IOLoop.instance()
		io_loop.start()

	def fetch(self, url, transform, async=False, no_user=False, **kwargs):
		logger.debug("fetch %s, async=%r", url, async)
		headers = tornado.httputil.HTTPHeaders()
		if self.user_id is None and not no_user:
			raise ValueError, "user id not set, call datasets() first"
		elif self.user_id is not None:
			headers.add("Cookie", "user_id=%s" % self.user_id)
			logger.debug("adding user_id %s to request", self.user_id)
		if async:
			# tornado doesn't like that we call fetch while ioloop is running in another thread, we should use ioloop.add_callbacl
			promise = vaex.promise.Promise()
			def do():
				future = self.http_client_async.fetch(url, headers=headers, request_timeout=DEFAULT_REQUEST_TIMEOUT, **kwargs)
				promise.fulfill(wrap_future_with_promise(future).then(transform).then(self._move_to_thread))
			self.io_loop.add_callback(do)
			return promise
		else:
			return transform(self.http_client.fetch(url, headers=headers, request_timeout=DEFAULT_REQUEST_TIMEOUT, **kwargs))


	def datasets(self, as_dict=False, async=False):
		def wrap(result):
			#print "body", repr(result.body), result
			data = json.loads(result.body)
			cookie = Cookie.SimpleCookie()
			for cookieset in result.headers.get_list("Set-Cookie"):
				cookie.load(cookieset)
				logger.debug("cookie load: %r", cookieset)
			logger.debug("cookie: %r", cookie)
			if "user_id" in cookie:
				user_id = cookie["user_id"].value
				logger.debug("user_id: %s", user_id)
				if self.user_id != user_id:
					self.user_id = user_id
					vaex.settings.webclient.store("cookie.user_id", self.user_id)
			datasets = [DatasetRest(self, **kwargs) for kwargs in data["datasets"]]
			return datasets if not as_dict else dict([(ds.name, ds) for ds in datasets])
		url = self._build_url("datasets")
		logger.debug("fetching: %r", url)
		return self.fetch(url, wrap, async=async, no_user=True)
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
			_check_error(info)
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
			data = json.loads(self._check_exception(result.body))
			return np.array([[data[expression]["min"], data[expression]["max"]] for expression in expressions])
		columns = "/".join(expressions)
		url = self._build_url("datasets/%s/minmax/%s" % (dataset_name, columns))
		logger.debug("fetching: %r", url)
		return self.fetch(url, wrap, async=async)
		#return self._return(result, wrap, async=async)

	def mean(self, subspace, dataset_name, expressions, **kwargs):
		return self._simple(subspace, dataset_name, expressions, "mean", **kwargs)
	def var(self, subspace, dataset_name, expressions, **kwargs):
		return self._simple(subspace, dataset_name, expressions, "var", **kwargs)
	def sum(self, subspace, dataset_name, expressions, **kwargs):
		return self._simple(subspace, dataset_name, expressions, "sum", **kwargs)
	def limits_sigma(self, subspace, dataset_name, expressions, **kwargs):
		return self._simple(subspace, dataset_name, expressions, "limits_sigma", **kwargs)

	def _simple(self, subspace, dataset_name, expressions, name, async=False, **kwargs):
		def wrap(result):
			result = self._check_exception(json.loads(result.body))["result"]
			# try to return is as numpy array
			try:
				return np.array(result)
			except ValueError:
				return result
		url = self._build_url("datasets/%s/%s" % (dataset_name, name))
		post_data = {key:json.dumps(value.tolist() if hasattr(value, "tolist") else value) for key, value in list(dict(kwargs).items())}
		post_data["masked"] = json.dumps(subspace.is_masked)
		post_data["active_fraction"] = json.dumps(subspace.dataset.get_active_fraction())
		post_data.update(dict(expressions=json.dumps(expressions)))
		body = urlencode(post_data)
		return self.fetch(url+"?"+body, wrap, async=async, method="GET")
		#return self._return(result, wrap)

	def histogram(self, subspace, dataset_name, expressions, size, limits, weight=None, async=False, **kwargs):
		def wrap(result):
			# TODO: don't do binary transfer, just json, now we cannot handle exception
			data = np.fromstring(result.body)
			shape = (size,) * len(expressions)
			data = data.reshape(shape)
			return data
		url = self._build_url("datasets/%s/histogram" % (dataset_name,))
		logger.debug("fetching: %r", url)
		limits = np.array(limits)
		post_data = dict(expressions=json.dumps(expressions), size=json.dumps(size),
						 weight=json.dumps(weight),
						 limits=json.dumps(limits.tolist()), masked=json.dumps(subspace.is_masked))
		post_data["active_fraction"] = json.dumps(subspace.dataset.get_active_fraction())
		post_data.update({key:json.dumps(value) for key, value in list(dict(kwargs).items())})
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
		promise = vaex.promise.Promise()
		#def do(value):
		#	return value
		#promise.then(do)
		if self.thread_mover:
			logger.debug("the other thread should fulfil the result to this promise")
			self.thread_mover(promise, result)
			return promise
		else:
			return result


	def select(self, dataset, dataset_name, boolean_expression, mode, async=False, **kwargs):
		name = "select"
		def wrap(result):
			return np.array(self._check_exception(json.loads(result.body)))
		url = self._build_url("datasets/%s/%s" % (dataset_name, name))
		post_data = {key:json.dumps(value) for key, value in list(dict(kwargs).items())}
		post_data["active_fraction"] = json.dumps(dataset.get_active_fraction())
		post_data.update(dict(boolean_expression=json.dumps(boolean_expression), mode=json.dumps(mode)))
		body = urlencode(post_data)
		return self.fetch(url+"?"+body, wrap, async=async, method="GET")
		#return self._return(result, wrap)

	def call_ondataset(self, method_name, dataset_remote, async, **kwargs):
		def wrap(result):
			result = self._check_exception(json.loads(result.body))["result"]
			try:
				return np.array(result)
			except ValueError:
				return result
		url = self._build_url("datasets/%s/%s" % (dataset_remote.name, method_name))
		post_data = {key:json.dumps(self._to_json_compatible(value)) for key, value in dict(kwargs).items()}
		post_data["active_fraction"] = json.dumps(dataset_remote.get_active_fraction())
		post_data["variables"] = json.dumps(dataset_remote.variables.items())
		post_data["virtual_columns"] = json.dumps(dataset_remote.virtual_columns.items())
		#post_data["selection_name"] = json.dumps(dataset_remote.get_selection_name())
		body = urlencode(post_data)
		return self.fetch(url+"?"+body, wrap, async=async, method="GET")
		#return self._return(result, wrap)

	def call(self, method_name, arg, async, **kwargs):
		def wrap(result):
			result = self._check_exception(json.loads(result.body))["result"]
			try:
				return np.array(result)
			except ValueError:
				return result
		url = self._build_url("%s/%s" % (method_name, arg))
		post_data = {key:json.dumps(self._to_json_compatible(value)) for key, value in dict(kwargs).items()}
		body = urlencode(post_data)
		return self.fetch(url+"?"+body, wrap, async=async, method="GET")

	def _to_json_compatible(self, obj):
		if hasattr(obj, "tolist"):
			obj = obj.tolist()
		return obj




	def _check_exception(self, reply_json):
		if "exception" in reply_json:
			logger.error("exception happened at server side: %r", reply_json)
			class_name = reply_json["exception"]["class"]
			msg = reply_json["exception"]["msg"]
			raise getattr(__builtin__, class_name)(msg)
		else:
			return reply_json

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

	def sleep(self, seconds, async=False):
		return self.dataset.server.call("sleep", seconds, async=async)

	def minmax(self):
		return self._promise(self.dataset.server.minmax(self, self.dataset.name, self.expressions, async=self.async, selection_name=self.get_selection_name()))
		#return self._task(task)

	def histogram(self, limits, size=256, weight=None):
		return self._promise(self.dataset.server.histogram(self, self.dataset.name, self.expressions, size=size, limits=limits, weight=weight, async=self.async, selection_name=self.get_selection_name()))

	def nearest(self, point, metric=None):
		point = point if not hasattr(point, "tolist") else point.tolist()
		result = self.dataset.server._simple(self, self.dataset.name, self.expressions, "nearest", async=self.async, point=point, metric=metric, selection_name=self.get_selection_name())
		return self._promise(result)

	def mean(self):
		return self.dataset.server.mean(self, self.dataset.name, self.expressions, async=self.async, selection_name=self.get_selection_name())

	def correlation(self, means, vars):
		return self.dataset.server._simple(self, self.dataset.name, self.expressions, "correlation", means=means, vars=vars, async=self.async, selection_name=self.get_selection_name())

	def var(self, means=None):
		return self.dataset.server.var(self, self.dataset.name, self.expressions, means=means, async=self.async, selection_name=self.get_selection_name())

	def sum(self):
		return self.dataset.server.sum(self, self.dataset.name, self.expressions, async=self.async, selection_name=self.get_selection_name())

	def limits_sigma(self, sigmas=3, square=False):
		return self.dataset.server.limits_sigma(self, self.dataset.name, self.expressions, sigmas=sigmas, square=square, async=self.async, selection_name=self.get_selection_name())

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
		super(DatasetRemote, self).__init__(name, column_names)
		self.server = server

class DatasetRest(DatasetRemote):
	def __init__(self, server, name, column_names, full_length):
		DatasetRemote.__init__(self, name, server.hostname, column_names)
		self.server = server
		self.name = name
		self.column_names = column_names
		self._full_length = full_length
		self._length = full_length
		self.filename = "http://%s:%s/%s" % (server.hostname, server.port, name)
		#self.host = host
		#self.http_client = AsyncHTTPClient()
		#future = http_client.fetch(self._build_url("datasets"))
		#fetch_future.add_done_callback(
		self.fraction = 1

		self.executor = DummyExecutor()

	def is_local(self): return False

	def __repr__(self):
		name = self.__class__.__module__ + "." +self.__class__.__name__
		return "<%s(server=%r, name=%r, column_names=%r, __len__=%r)> instance at 0x%x" % (name, self.server, self.name, self.column_names, len(self), id(self))

	def __call__(self, *expressions, **kwargs):
		return SubspaceRemote(self, expressions, self.executor, async=kwargs.get("async", False))

	def select(self, boolean_expression, mode="replace", async=False, selection_name="default"):
		def emit(result):
			# bit dirty to put this in the signal handler, but here we know we succeeded
			self._has_selection = boolean_expression is not None
			self.signal_selection_changed.emit(self)
			return result

		result = self.server.select(self, self.name, boolean_expression, mode, async=async, selection_name=selection_name)
		if async:
			return result.then(emit)
		else:
			emit(None)
			return result

	def lasso_select(self, expression_x, expression_y, xsequence, ysequence, mode="replace", async=False):
		def emit(result):
			# bit dirty to put this in the signal handler, but here we know we succeeded
			self._has_selection = True
			self.signal_selection_changed.emit(self)
			return result
		result = self.server.call_ondataset("lasso_select", self, expression_x=expression_x, expression_y=expression_y,
											xsequence=xsequence, ysequence=ysequence, mode=mode, async=async)
		if async:
			return result.then(emit)
		else:
			emit(None)
			return result

	def evaluate(self, expression, i1=None, i2=None, out=None, async=False):
		result = self.server.call_ondataset("evaluate", self, expression=expression, i1=i1, i2=i2, async=async)
		# TODO: we ignore out
		return result

	#def set_fraction(self, f):
	#	# TODO: implement fractions for remote
	#	self.fraction = f

	#def __len__(self):
	#	return self._full_length

	#def full_length(self):
	#	return self._full_length


# we may get rid of this when we group together tasks
class DummyExecutor(object):
	def __init__(self):
		self.signal_begin = vaex.events.Signal("begin")
		self.signal_progress = vaex.events.Signal("progress")
		self.signal_end = vaex.events.Signal("end")
		self.signal_cancel = vaex.events.Signal("cancel")

	def execute(self):
		print("dummy execute")


if __name__ == "__main__":
	import vaex
	import sys
	vaex.set_log_level_debug()
	server = vaex.server(sys.argv[1], port=int(sys.argv[2]))
	datasets = server.datasets()
	print datasets
	dataset = datasets[0]
	dataset = vaex.example()
	print dataset("x").minmax()
	dataset.select("x < 0")
	print dataset.selected_length(), len(dataset)
	print dataset("x").selected().is_masked
	print dataset("x").selected().minmax()
