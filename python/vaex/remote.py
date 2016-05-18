__author__ = 'breddels'
import numpy as np
import logging
import threading
import uuid

try:
	import __builtin__
except ImportError:
	import builtins as __builtin__

from .dataset import Dataset, Subspace, Task
import vaex.promise
import vaex.settings
import vaex.utils
try:
	import Cookie  # py2
except ImportError:
	import http.cookies as Cookie  # py3


from tornado.httpclient import AsyncHTTPClient, HTTPClient
import tornado.httputil
import tornado.websocket
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


try:
	from urllib.request import urlopen
	from urllib.parse import urlparse, urlencode
except ImportError:
	from urlparse import urlparse
	from urllib import urlopen, urlencode

import threading

def _check_error(object):
	if "error" in object:
		raise RuntimeError("Server responded with error: %r" % object["error"])


class ServerRest(object):
	def __init__(self, hostname, port=5000, base_path="/", background=False, thread_mover=None, websocket=True):
		self.hostname = hostname
		self.port = port
		self.base_path = base_path if base_path.endswith("/") else (base_path + "/")
		#if async:
		event = threading.Event()
		self.thread_mover = thread_mover or (lambda fn, *args, **kwargs: fn(*args, **kwargs))
		logger.debug("thread mover: %r", self.thread_mover)

		# jobs maps from uid to tasks
		self.jobs = {}

		def ioloop_threaded():
			logger.debug("creating tornado io_loop")
			self.io_loop = tornado.ioloop.IOLoop().instance()
			event.set()
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
		self.use_websocket = websocket
		self.websocket = None

		self.submit = self.submit_http

		if self.use_websocket:
			self.submit = self.submit_websocket
			self._websocket_connect()

	def _websocket_connect(self):
		def connected(websocket):
			logger.debug("connected to websocket: %s" % self._build_url(""))
		def failed(reason):
			logger.error("failed to connect to %s" % self._build_url(""))
		self.websocket_connected = vaex.promise.Promise()
		self.websocket_connected.then(connected, failed)
		if 0:
			connected = wrap_future_with_promise(tornado.websocket.websocket_connect(self._build_url("websocket"), on_message_callback=self._on_websocket_message))
			connected.get()
			self.websocket_connected.fulfill(connected)
		def do():
			try:
				connected = wrap_future_with_promise(tornado.websocket.websocket_connect(self._build_url("websocket"), on_message_callback=self._on_websocket_message))
				self.websocket_connected.fulfill(connected)
			except:
				logger.exception("error connecting")
				#raise
		self.io_loop.add_callback(do)
		result = self.websocket_connected.get()
		if self.websocket_connected.isRejected:
			raise self.websocket.reason




	def _on_websocket_message(self, msg):
		response = json.loads(msg)
		import sys
		if sys.getsizeof(msg) > 1024*4:
			logger.debug("socket read message: <large amount of data>",)
		else:
			logger.debug("socket read message: %s", msg)
			logger.debug("json response: %r", response)
		# for the moment, job == task, in the future a job can be multiple tasks
		job_id = response.get("job_id")
		if job_id:
			try:
				phase = response["job_phase"]
				logger.debug("job update %r, phase=%r", job_id, phase)
				if phase == "COMPLETED":
					result = response["result"]#[0]
					#logger.debug("completed job %r, result=%r", job_id, result)
					task = self.jobs[job_id]
					logger.debug("completed job %r (async=%r, thread_mover=%r)", job_id, task.async, self.thread_mover)
					processed_result = task.post_process(result)
					if task.async:
						self.thread_mover(task.fulfill, processed_result)
					else:
						task.fulfill(processed_result)
				elif phase == "EXCEPTION":
					logger.error("exception happened at server side: %r", response)
					class_name = response["exception"]["class"]
					msg = response["exception"]["msg"]
					exception = getattr(__builtin__, class_name)(msg)
					logger.debug("error in job %r, exception=%r", job_id, exception)
					task = self.jobs[job_id]
					if task.async:
						self.thread_mover(task.reject, exception)
					else:
						task.reject(exception)
				elif phase == "ERROR":
					logger.error("error happened at server side: %r", response)
					msg = response["error"]
					exception = RuntimeError("error at server: %r" % msg)
					task = self.jobs[job_id]
					if task.async:
						self.thread_mover(task.reject, exception)
					else:
						task.reject(exception)
				elif phase == "PENDING":
					fraction = response["progress"]
					logger.debug("pending?: %r", phase)
					task = self.jobs[job_id]
					if task.async:
						self.thread_mover(task.signal_progress.emit, fraction)
					else:
						task.signal_progress.emit(fraction)
			except Exception as e:
				logger.exception("error in handling job", e)
				task = self.jobs[job_id]
				if task.async:
					self.thread_mover(task.reject, e)
				else:
					task.reject(e)


	def wait(self):
		io_loop = tornado.ioloop.IOLoop.instance()
		io_loop.start()

	def submit_websocket(self, path, arguments, async=False, post_process=lambda x: x):
		assert self.use_websocket

		task = TaskServer(post_process=post_process, async=async)
		logger.debug("created task: %r, %r (async=%r)" % (path, arguments, async))
		job_id = str(uuid.uuid4())
		self.jobs[job_id] = task
		arguments["job_id"] = job_id
		arguments["path"] = path
		arguments["user_id"] = self.user_id
		def listify(value):
			if isinstance(value, list):
				value = list([listify(item) for item in value])
			if hasattr(value, "tolist"):
				value = value.tolist()
			return value

		#arguments = dict({key: (value.tolist() if hasattr(value, "tolist") else value) for key, value in arguments.items()})
		arguments = dict({key: listify(value) for key, value in arguments.items()})

		def do():
			def write(socket):
				try:
					logger.debug("write to websocket: %r", arguments)
					socket.write_message(json.dumps(arguments))
					return
				except:
					import traceback
					traceback.print_exc()
			#return
			logger.debug("will schedule a write to the websocket")
			self.websocket_connected.then(write).end()#.then(task.fulfill)

		self.io_loop.add_callback(do)
		logger.debug("we can continue (main thread is %r)", threading.currentThread())
		if async:
			return task
		else:
			return task.get()

	def submit_http(self, path, arguments, post_process, async, **kwargs):
		def pre_post_process(response):
			cookie = Cookie.SimpleCookie()
			for cookieset in response.headers.get_list("Set-Cookie"):
				cookie.load(cookieset)
				logger.debug("cookie load: %r", cookieset)
			logger.debug("cookie: %r", cookie)
			if "user_id" in cookie:
				user_id = cookie["user_id"].value
				logger.debug("user_id: %s", user_id)
				if self.user_id != user_id:
					self.user_id = user_id
					vaex.settings.webclient.store("cookie.user_id", self.user_id)
			data = json.loads(response.body)
			self._check_exception(data)
			return post_process(data["result"])

		arguments = {key: (value.tolist() if hasattr(value, "tolist") else value) for key, value in arguments.items()}
		arguments_json = {key:json.dumps(value) for key, value in arguments.items()}
		headers = tornado.httputil.HTTPHeaders()

		url = self._build_url(path +"?" + urlencode(arguments_json))
		logger.debug("fetch %s, async=%r", url, async)

		if self.user_id is not None:
			headers.add("Cookie", "user_id=%s" % self.user_id)
			logger.debug("adding user_id %s to request", self.user_id)
		if async:
			task = TaskServer(pre_post_process, async=async)
			# tornado doesn't like that we call fetch while ioloop is running in another thread, we should use ioloop.add_callbacl
			def do():
				self.thread_mover(task.signal_progress.emit, 0.5)
				future = self.http_client_async.fetch(url, headers=headers, request_timeout=DEFAULT_REQUEST_TIMEOUT, **kwargs)
				def thread_save_succes(value):
					self.thread_mover(task.signal_progress.emit, 1.0)
					self.thread_mover(task.fulfill, value)
				def thread_save_failure(value):
					self.thread_mover(task.reject, value)
				wrap_future_with_promise(future).then(pre_post_process).then(thread_save_succes, thread_save_failure)
			self.io_loop.add_callback(do)
			return task #promise.then(self._move_to_thread)
		else:
			return pre_post_process(self.http_client.fetch(url, headers=headers, request_timeout=DEFAULT_REQUEST_TIMEOUT, **kwargs))

	def datasets(self, as_dict=False, async=False):
		def post(result):
			logger.debug("datasets result: %r", result)
			datasets = [DatasetRest(self, **kwargs) for kwargs in result]
			logger.debug("datasets: %r", datasets)
			return datasets if not as_dict else dict([(ds.name, ds) for ds in datasets])
		return self.submit(path="datasets", arguments={}, post_process=post, async=async)

	def _build_url(self, method):
		protocol = "ws" if self.use_websocket else "http"
		return "%s://%s:%d%s%s" % (protocol, self.hostname, self.port, self.base_path, method)

	def _call_subspace(self, method_name, subspace, **kwargs):
		def post_process(result):
			if method_name == "histogram": # histogram is the exception..
				# TODO: don't do binary transfer, just json, now we cannot handle exception
				import base64
				logger.debug("result: %r" % result)
				result = base64.b64decode(result) #.decode("base64")
				data = np.fromstring(result)
				shape = (kwargs["size"],) * subspace.dimension
				data = data.reshape(shape)
				return data
			else:
				try:
					return np.array(result)
				except ValueError:
					return result
		dataset_name = subspace.dataset.name
		expressions = subspace.expressions
		async = subspace.async
		path = "datasets/%s/%s" % (dataset_name, method_name)
		url = self._build_url(path)
		arguments = dict(kwargs)
		if not subspace.dataset.get_auto_fraction():
			arguments["active_fraction"] = subspace.dataset.get_active_fraction()
		selection = subspace.get_selection()
		if selection is not None:
			arguments["selection"] = selection.to_dict()
		arguments["variables"] = list(subspace.dataset.variables.items())
		arguments["virtual_columns"] = list(subspace.dataset.virtual_columns.items())
		arguments.update(dict(expressions=expressions))
		return self.submit(path, arguments, post_process=post_process, async=async)

	def _call_dataset(self, method_name, dataset_remote, async, **kwargs):
		def post_process(result):
			#result = self._check_exception(json.loads(result.body))["result"]
			try:
				return np.array(result)
			except ValueError:
				return result
		path = "datasets/%s/%s" % (dataset_remote.name, method_name)
		arguments = dict(kwargs)
		if not dataset_remote.get_auto_fraction():
			arguments["active_fraction"] = dataset_remote.get_active_fraction()
		arguments["variables"] = list(dataset_remote.variables.items())
		arguments["virtual_columns"] = list(dataset_remote.virtual_columns.items())
		#arguments["selection_name"] = json.dumps(dataset_remote.get_selection_name())
		body = urlencode(arguments)

		return self.submit(path, arguments, post_process=post_process, async=async)
		#return self.fetch(url+"?"+body, wrap, async=async, method="GET")
		#return self._return(result, wrap)


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

	def _task(self, promise):
		"""Helper function for returning tasks results, result when immediate is True, otherwise the task itself, which is a promise"""
		if self.async:
			return promise
		else:
			return promise

	def sleep(self, seconds, async=False):
		return self.dataset.server.call("sleep", seconds, async=async)

	def minmax(self):
		return self._task(self.dataset.server._call_subspace("minmax", self))
		#return self._task(task)

	def histogram(self, limits, size=256, weight=None):
		return self._task(self.dataset.server._call_subspace("histogram", self, size=size, limits=limits, weight=weight))

	def nearest(self, point, metric=None):
		point = vaex.utils.make_list(point)
		result = self.dataset.server._call_subspace("nearest", self, point=point, metric=metric)
		return self._task(result)

	def mean(self):
		return self.dataset.server._call_subspace("mean", self)

	def correlation(self, means=None, vars=None):
		return self.dataset.server._call_subspace("correlation", self, means=means, vars=vars)

	def var(self, means=None):
		return self.dataset.server._call_subspace("var", self, means=means)

	def sum(self):
		return self.dataset.server._call_subspace("sum", self)

	def limits_sigma(self, sigmas=3, square=False):
		return self.dataset.server._call_subspace("limits_sigma", self, sigmas=sigmas, square=square)

	def mutual_information(self, limits=None, size=256):
		return self.dataset.server._call_subspace("mutual_information", self, limits=limits, size=size)

class DatasetRemote(Dataset):
	def __init__(self, name, server, column_names):
		super(DatasetRemote, self).__init__(name, column_names)
		self.server = server

import astropy.units

class DatasetRest(DatasetRemote):
	def __init__(self, server, name, column_names, dtypes, ucds, descriptions, units, description, full_length, virtual_columns=None):
		DatasetRemote.__init__(self, name, server.hostname, column_names)
		self.server = server
		self.name = name
		self.column_names = column_names
		self.dtypes = {name: np.zeros(1, dtype=getattr(np, dtype)).dtype for name, dtype in dtypes.items()}
		self.units = {name: astropy.units.Unit(unit) for name, unit in units.items()}
		self.virtual_columns.update(virtual_columns or {})
		self.ucds = ucds
		self.descriptions = descriptions
		self.description = description
		self._full_length = full_length
		self._length = full_length
		#self.filename = #"http://%s:%s/%s" % (server.hostname, server.port, name)
		self.path = self.filename = self.server._build_url("%s" % name)


		self.fraction = 1

		self.executor = ServerExecutor()

	def dtype(self, expression):
		if expression in self.get_column_names():
			return self.dtypes[expression]
		else:
			return np.zeros(1, dtype=np.float64).dtype

	def is_local(self): return False

	def __repr__(self):
		name = self.__class__.__module__ + "." +self.__class__.__name__
		return "<%s(server=%r, name=%r, column_names=%r, __len__=%r)> instance at 0x%x" % (name, self.server, self.name, self.column_names, len(self), id(self))

	def __call__(self, *expressions, **kwargs):
		return SubspaceRemote(self, expressions, kwargs.get("executor") or self.executor, async=kwargs.get("async", False))

	def evaluate(self, expression, i1=None, i2=None, out=None, async=False):
		result = self.server._call_dataset("evaluate", self, expression=expression, i1=i1, i2=i2, async=async)
		# TODO: we ignore out
		return result


# we may get rid of this when we group together tasks
class ServerExecutor(object):
	def __init__(self):
		self.signal_begin = vaex.events.Signal("begin")
		self.signal_progress = vaex.events.Signal("progress")
		self.signal_end = vaex.events.Signal("end")
		self.signal_cancel = vaex.events.Signal("cancel")

	def execute(self):
		logger.debug("dummy execute")

from vaex.dataset import Task
class TaskServer(Task):
	def __init__(self, post_process, async):
		vaex.dataset.Task.__init__(self, None, [])
		self.post_process = post_process
		self.async = async


if __name__ == "__main__":
	import vaex
	import sys
	vaex.set_log_level_debug()
	server = vaex.server(sys.argv[1], port=int(sys.argv[2]))
	datasets = server.datasets()
	print(datasets)
	dataset = datasets[0]
	dataset = vaex.example()
	print(dataset("x").minmax())
	dataset.select("x < 0")
	print(dataset.selected_length(), len(dataset))
	print(dataset("x").selected().is_masked)
	print(dataset("x").selected().minmax())
