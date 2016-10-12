from __future__ import absolute_import
__author__ = 'maartenbreddels'

import tornado.ioloop
import tornado.web
import tornado.httpserver
import tornado.websocket
import tornado.auth
import tornado.gen
import threading
import logging
import vaex as vx
import vaex.utils
import json
import inspect
import yaml
import argparse
import os
import time
import numpy as np

logger = logging.getLogger("vaex.webserver")
job_index = 0
class JobFlexible(object):
	def __init__(self, cost, fn=None, args=None, kwargs=None, index=None):
		global job_index
		self.cost = cost
		if index is None:
			self.index = job_index
			job_index += 1
		else:
			self.index = index
		self.fraction = None
		self.time_elapsed_goal = None
		self.time_created = time.time()
		self.fn = fn or (lambda *args, **kwargs: None)
		self.args = args
		self.kwargs = kwargs

	def set_fraction(self, fraction):
		self.fraction = fraction

	def set_time_elapsed_goal(self, time_elapsed_goal):
		self.time_elapsed_goal = time_elapsed_goal

	def execute(self):
		try:
			result = self.fn(*self.args, **self.kwargs)
			return result
		finally:
			self.done()

	def done(self):
		self.time_done = time.time()
		self.time_elapsed = self.time_done - self.time_created
import vaex.events
class JobQueue(object):
	def __init__(self):
		self.queue = []
		self.queue_delay = 1. # 1 second to get rid of the queue
		self.seconds_per_cost = 1.
		self.cost_modifier_per_miss = 0.1
		self.fraction_minimum = 1e-4
		self.signal_job_finished = vaex.events.Signal("job finished")

	def add(self, job):
		self.queue.append(job)

	def get_next(self):
		if len(self.queue) == 0:
			raise IndexError("queue empty")
		total_cost = sum(job.cost for job in self.queue)
		job_count = len(self.queue)
		job = self.queue.pop(0)
		logger.debug("queue length is %d, total cost of queue is %f, job index %d", len(self.queue), total_cost, job.index)
		# if the total cost equals seconds, and we want the queue to be finished in 1 second
		# this will be the best fraction
		elapsed = time.time() - job.time_created
		if elapsed > self.queue_delay * 1.3: # we already spent way too long time, the math doesn't work out, so what now?
			# we scale the cost up, since it apearently doesn't work out fully
			#raise ValueError, "elapsed = %f" % elapsed
			logger.error("queue clogging up")
		# assuming the queue costs will stay constant we want to finished this whole queue in max self.queue_delay
		# seconds. Therefore the current job should only calculate a fraction as computed by:
		fraction = (self.queue_delay - elapsed) / (total_cost * self.seconds_per_cost) * 0.9
		fraction = min(max(self.fraction_minimum, fraction), 1)
		time_elapsed_goal = elapsed + job.cost * self.seconds_per_cost * fraction / 0.9#elapsed + fraction * (self.seconds_per_cost)
		logger.debug("to finish the queue we set the next job's fraction to %f (time elapsed = %f)", fraction, elapsed)
		#print fraction
		job.set_fraction(fraction)
		job.set_time_elapsed_goal(time_elapsed_goal)
		return job

	def finished(self, job):
		ratio = job.time_elapsed/ job.time_elapsed_goal # / job.time_elapsed_goal
		if ratio > 1.2:
			logger.debug("job should have taken %f seconds, took %f seconds, we should go faster", job.time_elapsed_goal, job.time_elapsed)
			self.seconds_per_cost *= (1 + self.cost_modifier_per_miss)
		if ratio < 0.75:
			self.seconds_per_cost *= (1 - self.cost_modifier_per_miss)
			logger.debug("job should have taken %f seconds, took %f seconds, we can slow down", job.time_elapsed_goal, job.time_elapsed)
		logger.debug("seconds_per_cost set to: %f", self.seconds_per_cost)
		self.signal_job_finished.emit(job)

		#seconds_goal = job.set_fraction
		#fraction_goal = job.set_fraction
		#fraction_achieved = job.ti
		pass

#import concurrent.future

class JobExecutor(object):
	def __init__(self, job_queue):
		self.job_queue = job_queue
		#self.thread_pool_executor = concurrent.

	def empty_queue(self):
		while True:
			try:
				job = self.job_queue.get_next()
				job.execute()
				self.job_queue.finished(job)
			except IndexError:
				return

class GoogleOAuth2LoginHandler(tornado.web.RequestHandler,
                               tornado.auth.GoogleOAuth2Mixin):
    @tornado.gen.coroutine
    def get(self):
        if self.get_argument('code', False):
            user = yield self.get_authenticated_user(
                redirect_uri='http://your.site.com/auth/google',
                code=self.get_argument('code'))
            # Save the user with e.g. set_secure_cookie
        else:
            yield self.authorize_redirect(
                redirect_uri='http://your.site.com/auth/google',
                client_id='cid',
                scope=['profile', 'email'],
                response_type='code',
                extra_params={'approval_prompt': 'auto'})

def task_invoke(subspace, method_name, **arguments):
	method = getattr(subspace, method_name)
	args, varargs, kwargs, defaults = inspect.getargspec(method)
	#print inspect.getargspec(method)
	#args_required = args[:len(args)-len(defaults)]
	#kwargs = {}
	#for arg in args:
	#	if arg in arguments:
	#		kwargs[arg] = arguments[arg]
	print("!!!!!! calling %s with arguments %r" % (method_name, arguments))
	values = method(**arguments)
	return values

import collections
import concurrent.futures
import vaex.execution
import vaex.multithreading
import tornado.escape
class ListHandler(tornado.web.RequestHandler):
	def initialize(self, webserver, submit_threaded, cache, cache_selection, datasets=None):
		self.webserver = webserver
		self.submit_threaded = submit_threaded
		self.datasets = datasets or [vx.example()]
		self.datasets_map = collections.OrderedDict([(ds.name,ds) for ds in self.datasets])
		self.cache = cache
		self.cache_selection = cache_selection

	def set_default_headers(self):
		self.set_header("Access-Control-Allow-Origin", "*")

	@tornado.gen.coroutine
	def get(self):
		user_id = None
		if False: # disable user_id for the moment
			user_id = self.get_cookie("user_id")
			logger.debug("user_id in cookie: %r", user_id)
			if user_id is None:
				import uuid
				user_id = str(uuid.uuid4())
				self.set_cookie("user_id", user_id)
			logger.debug("user_id: %r", user_id)
		# tornado retursn a list of values, just use the first value
		arguments = {key:json.loads(tornado.escape.to_basestring(self.request.arguments[key][0])) for key in self.request.arguments.keys()}
		key = (self.request.path, "-".join([str(v) for v in sorted(arguments.items())]))
		response = self.cache.get(key)
		if response is None:
			response = yield self.submit_threaded(process, self.webserver, user_id, self.request.path, **arguments)
			try:
				self.cache[key] = response
				pass
			except ValueError:
				pass # raised when it doesn't fit in cache
		else:
			logger.debug("cache hit for key: %r", key)
		#logger.debug("response is: %r", response)
		if response is None:
			response = self.error("unknown request or error")
		if isinstance(response, (np.ndarray, float)):
			response = np.array(response)
			self.set_header("Content-Type", "application/numpy-array")
			self.write(str(response.shape)+"\n")
			self.write(str(response.dtype)+"\n")
			self.write(response.tobytes())
		else:
			if isinstance(response, str):
				self.set_header("Content-Type", "application/octet-stream")
			self.write(response)


class ProgressWebSocket(tornado.websocket.WebSocketHandler):
	def initialize(self, webserver, submit_threaded, cache, cache_selection, datasets=None):
		self.webserver = webserver
		self.submit_threaded = submit_threaded
		self.datasets = datasets or [vx.example()]
		self.datasets_map = collections.OrderedDict([(ds.name,ds) for ds in self.datasets])
		self.cache = cache
		self.cache_selection = cache_selection

	def check_origin(self, origin):
		return True

	def open(self):
		logger.debug("WebSocket opened")

	@tornado.gen.coroutine
	def on_message(self, message):
		logger.debug("websocket message: %r", message)
		arguments = json.loads(message)
		path = arguments["path"]
		arguments.pop("path")
		user_id = arguments.pop("user_id", None)
		job_id = arguments.pop("job_id")


		if False: # disable user_id for the moment
			user_id = self.get_cookie("user_id")
			if user_id == None:
				user_id = arguments.pop("user_id")
				if user_id == None:
					self.write_json(error="no user id, do an http get request first")
					return
		last_progress = [None]
		def progress(f):
			def do():
				logger.debug("progress: %r", f)
				last_progress[0] = f
				self.write_json(job_id=job_id, job_phase="PENDING", progress=f)
			if last_progress[0] is None or (f - last_progress[0]) > 0.05 or f == 1.0:
				tornado.ioloop.IOLoop.current().add_callback(do)
			return True
		key = (path, "-".join([str(v) for v in sorted(arguments.items())]))
		response = self.cache.get(key)
		progress(0)
		if response is None:
			#response = yield self.submit_threaded(process, self.webserver, user_id, self.request.path, **arguments)
			response = yield self.submit_threaded(process, self.webserver, user_id, path, progress=progress,
												  **arguments)
			try:
				self.cache[key] = response
				pass
			except ValueError:
				pass # raised when it doesn't fit in cache
		progress(1)

		if response is None:
			response = self.error("unknown request or error")

		meta_data = dict(job_id=job_id)
		if isinstance(response, (np.ndarray, float)):
			response = np.array(response)
			#self.set_header("Content-Type", "application/numpy-array")
			meta_data["shape"] = str(response.shape)
			meta_data["dtype"] = str(response.dtype)
			meta_data["job_phase"] = "COMPLETED"
			data_response = response.tobytes()
		else: # it should be a dict
			if "result" in response:
				response["job_phase"] = "COMPLETED"
			if "error" in response:
				response["job_phase"] = "ERROR"
			if "exception" in response:
				response["job_phase"] = "EXCEPTION"
			meta_data.update(response)
			data_response = b""
		text_response = json.dumps(meta_data).encode("utf8")\
						+ b"\n" + data_response

		self.write_message(text_response, binary=True)

	def write_json(self, **kwargs):
		logger.debug("writing json: %r", kwargs)
		self.write_message(json.dumps(kwargs) + "\n", binary=True)

	def on_close(self):
		logger.debug("WebSocket closed")

class QueueHandler(tornado.web.RequestHandler):
	def initialize(self, datasets):
		self.datasets = datasets
		self.datasets_map = dict([(ds.name,ds) for ds in self.datasets])

	def get(self):
        #self.write("Hello, world")
		self.write(dict(datasets=[{"name":ds.name, "length":len(ds)} for ds in self.datasets]))

def exception(exception):
	logger.exception("handled exception at server, all fine")
	return ({"exception": {"class":str(exception.__class__.__name__), "msg": str(exception)} })

def error(msg):
	return ({"error": msg}) #, "result":None})

def process(webserver, user_id, path, fraction=None, progress=None, **arguments):
	if not hasattr(webserver.thread_local, "executor"):
		logger.debug("creating thread pool and executor")
		webserver.thread_local.thread_pool = vaex.multithreading.ThreadPoolIndex(nthreads=webserver.threads_per_job)
		webserver.thread_local.executor = vaex.execution.Executor(thread_pool=webserver.thread_local.thread_pool)
		webserver.thread_pools.append(webserver.thread_local.thread_pool)

	progress = progress or (lambda x: True)
	progress(0)
	# TODO: mem leak and other issues if we don't disconnect this
	webserver.thread_local.executor.signal_progress.connect(progress)
	#return ("Hello, world")
	#print request.path
	try:
		parts = [part for part in path.split("/") if part]
		logger.debug("request: %r" % parts)
		if parts[0] == "sleep":
			seconds = float(parts[1])
			import time
			time.sleep(min(1, seconds))
			return ({"result":seconds})
		elif parts[0] == "datasets":
			if len(parts) == 1:
				#
				response = dict(result=[{"name":ds.name, "full_length":ds.full_length(), "column_names":ds.get_column_names(strings=True),
										 "description": ds.description, "descriptions":ds.descriptions,
										 "ucds":ds.ucds, "units":{name:str(unit) for name, unit in ds.units.items()},
										 "dtypes":{name:str(ds.columns[name].dtype) for name in ds.get_column_names(strings=True)},
										 "virtual_columns":dict(ds.virtual_columns),
										 } for ds in webserver.datasets])
				logger.debug("response: %r", response)
				return response
			else:

				dataset_name = parts[1]
				logger.debug("dataset: %s", dataset_name)
				if dataset_name not in webserver.datasets_map:
					error("dataset does not exist: %r, possible options: %r" % (dataset_name, webserver.datasets_map.keys()))
				else:
					if len(parts) > 2:
						method_name = parts[2]
						logger.debug("method: %r args: %r" % (method_name, arguments))
						if "expressions" in arguments:
							expressions = arguments["expressions"]
						else:
							expressions = None
						# make a shallow copy, such that selection and active_fraction is not shared
						dataset = webserver.datasets_map[dataset_name].shallow_copy(virtual=False)
	
						if dataset.mask is not None:
							logger.debug("selection: %r", dataset.mask.sum())
						if "active_fraction" in arguments:
							active_fraction = arguments["active_fraction"]
							logger.debug("setting active fraction to: %r", active_fraction)
							dataset.set_active_fraction(active_fraction)
						else:
							if fraction is not None:
								dataset.set_active_fraction(fraction)
								logger.debug("auto fraction set to %f", fraction)
						if "active_start_index" in arguments and "active_end_index" in arguments:
							i1, i2 = arguments["active_start_index"], arguments["active_end_index"]
							logger.debug("setting active range to: %r", (i1, i2))
							dataset.set_active_range(i1, i2)

						if "variables" in arguments:
							variables = arguments["variables"]
							logger.debug("setting variables to: %r", variables)
							for key, value in variables:
								dataset.set_variable(key, value)
						if "virtual_columns" in arguments:
							virtual_columns = arguments["virtual_columns"]
							logger.debug("setting virtual_columns to: %r", virtual_columns)
							for key, value in virtual_columns:
								dataset.add_virtual_column(key, value)
							for key, value in virtual_columns:
								try:
									dataset.validate_expression(value)
								except (SyntaxError, KeyError, NameError) as e:
									logger.exception("state was: %r", arguments)
									return exception(e)
						if expressions:
							for expression in expressions:
								try:
									dataset.validate_expression(expression)
								except (SyntaxError, KeyError, NameError) as e:
									return exception(e)
						subspace = dataset(*expressions, executor=webserver.thread_local.executor) if expressions else None
						if subspace:
							# old stype selection
							if "selection" in arguments:
								selection_values = arguments["selection"]
								if selection_values:
									selection = vaex.dataset.selection_from_dict(dataset, selection_values)
									dataset.set_selection(selection, executor=webserver.thread_local.executor)
						else:
							if "selections" in arguments:
								selection_values = arguments["selections"]
								if selection_values:
									for name, value in selection_values.items():
										selection = vaex.dataset.selection_from_dict(dataset, value)
										dataset.set_selection(selection, name=name)
						try:
							if subspace:
								if "selection" in arguments:
									subspace = subspace.selected()
							if subspace is None:
								for name in "job_id expressions active_fraction selections variables virtual_columns active_start_index active_end_index".split():
									arguments.pop(name, None)
							else:
								if subspace is not None:
									for name in "job_id expressions active_fraction selection selections variables virtual_columns active_start_index active_end_index".split():
										arguments.pop(name, None)
							logger.debug("subspace: %r", subspace)
							if subspace is None and method_name in "count cov correlation covariance mean std minmax min max sum var".split():
								grid = task_invoke(dataset, method_name, **arguments)
								return grid
							elif method_name in ["minmax", "image_rgba_url", "var", "mean", "sum", "limits_sigma", "nearest", "correlation", "mutual_information"]:
								#print "expressions", expressions
								values = task_invoke(subspace, method_name, **arguments)
								logger.debug("result: %r", values)
								#print values, expressions
								values = values.tolist() if hasattr(values, "tolist") else values

								#for value in values:
								#	logger.debug("value: %r" % type(value))
								#try:
								#	values = [(k.tolist() if hasattr(k, "tolist") else k) for k in values]
								#except:
								#	pass
								return ({"result": values})
							elif method_name == "histogram":
								grid = task_invoke(subspace, method_name, **arguments)
								return grid
								#self.set_header("Content-Type", "application/octet-stream")
								#return (grid.tostring())
								#import base64
								#return grid.tostring()
								#result = base64.b64encode(grid.tostring()).decode("ascii")
								#logger.debug("type: %s" % type(result))
								#return {"result": result}
							elif method_name in ["select", "lasso_select"]:
								dataset.mask = webserver.cache_selection.get((dataset.path, user_id))
								result = task_invoke(dataset, method_name, **arguments)
								result = result.tolist() if hasattr(result, "tolist") else result
								webserver.cache_selection[(dataset.path, user_id)] = dataset.mask
								return ({"result": result})
							elif method_name in ["evaluate"]:
								result = task_invoke(dataset, method_name, **arguments)
								result = result.tolist() if hasattr(result, "tolist") else result
								return ({"result": result})
							else:
								logger.error("unknown method: %r", method_name)
								return error("unknown method: " + method_name)
						except (SyntaxError, KeyError, NameError) as e:
							return exception(e)
	except Exception as e:
		logger.exception("unknown issue")
		return exception(e)
	return error("unknown request")

from cachetools import Cache, LRUCache
import sys
MB = 1024**2
GB = MB * 1024

class WebServer(threading.Thread):
	def __init__(self, address="localhost", port=9000, webserver_thread_count=2, cache_byte_size=500*MB,
				 cache_selection_byte_size=500*MB, datasets=[], compress=True, development=False, threads_per_job=4):
		threading.Thread.__init__(self)
		self.setDaemon(True)
		self.address = address
		self.port = port
		self.started = threading.Event()
		self.datasets = datasets
		self.datasets_map = dict([(ds.name,ds) for ds in self.datasets])

		self.webserver_thread_count = webserver_thread_count
		self.threads_per_job = threads_per_job

		self.thread_pool = concurrent.futures.ThreadPoolExecutor(self.webserver_thread_count)
		self.thread_local = threading.local()
		self.thread_pools = []

		self.job_queue = JobQueue()

		self.cache = LRUCache(cache_byte_size, getsizeof=sys.getsizeof)
		self.cache_selection = LRUCache(cache_selection_byte_size, getsizeof=sys.getsizeof)

		self.options = dict(webserver=self, datasets=datasets, submit_threaded=self.submit_threaded, cache=self.cache,
							cache_selection=self.cache_selection)


		#tornado.web.GZipContentEncoding.MIN_LENGTH = 1
		tornado.web.GZipContentEncoding.CONTENT_TYPES.add("application/octet-stream")
		self.application = tornado.web.Application([
			(r"/queue", QueueHandler, self.options),
			(r"/auth", GoogleOAuth2LoginHandler, {}),
			(r"/websocket", ProgressWebSocket, self.options),
			(r"/.*", ListHandler, self.options),
		], compress_response=compress, debug=development)
		logger.debug("compression set to %r", compress)
		logger.debug("cache size set to %s", vaex.utils.filesize_format(cache_byte_size))
		logger.debug("thread count set to %r", self.webserver_thread_count)

	def submit_threaded(self, callable, *args, **kwargs):
		job = JobFlexible(4., callable, args=args, kwargs=kwargs)
		self.job_queue.add(job)
		def execute():
			job = self.job_queue.get_next()
			job.kwargs["fraction"] = job.fraction # add fraction keyword argument to the callback
			try:
				result = job.execute()
			finally:
				self.job_queue.finished(job)
			return result
		future = self.thread_pool.submit(execute) #, *args, **kwargs)
		return future

	def serve(self):
		self.mainloop()

	def serve_threaded(self):
		logger.debug("start thread")
		self.start()
		logger.debug("wait for thread to run")
		self.started.wait()
		logger.debug("make tornado io loop the main thread's current")
		# this will make the main thread use this ioloop as current
		self.ioloop.make_current()

	def run(self):
		self.mainloop()

	def mainloop(self):
		logger.info("serving at http://%s:%d" % (self.address, self.port))
		self.ioloop = tornado.ioloop.IOLoop.current()
		# listen doesn't return a server object, which we need to close
		#self.application.listen(self.port, address=self.address)
		from tornado.httpserver import HTTPServer
		self.server = HTTPServer(self.application)
		try:
			self.server.listen(self.port, self.address)
		except:
			self.started.set()
			raise
		self.started.set()
		#self.ioloop.add_callback(self.started.set)
		#if not self.ioloop.is_running():
		try:
			self.ioloop.start()
		except RuntimeError:
			pass # TODO: not sure why this happens in the unittest
		#self.ioloop.stop()
		#try:
		#	self.ioloop.close()
		#except ValueError:
		#	pass
		#self.ioloop.clear_current()

	def stop_serving(self):
		logger.debug("stop server")
		self.server.stop()
		logger.debug("stop io loop")
		self.ioloop.stop()
		#self.ioloop.close()
		#self.ioloop.clear_current()
		for thread_pool in self.thread_pools:
			thread_pool.close()

defaults_yaml = """
address: 0.0.0.0
port: 9000
filenames: []
verbose: 2
cache: 500000000
compress: true
filename: []
development: False
threads_per_job: 4
"""

def main(argv):

	parser = argparse.ArgumentParser(argv[0])
	parser.add_argument("filename", help="filename for dataset", nargs='*')
	parser.add_argument("--address", help="address to bind the server to (default: %(default)s)", default="0.0.0.0")
	parser.add_argument("--port", help="port to listen on (default: %(default)s)", type=int, default=9000)
	parser.add_argument('--verbose', '-v', action='count', default=2)
	parser.add_argument('--cache', help="cache size in bytes for requests, set to zero to disable (default: %(default)s)", type=int, default=500000000)
	parser.add_argument('--compress', help="compress larger replies (default: %(default)s)", default=True, action='store_true')
	parser.add_argument('--no-compress', dest="compress", action='store_false')
	parser.add_argument('--development', default=False, action='store_true', help="enable development features (auto reloading)")
	parser.add_argument('--threads-per-job', default=4, type=int, help="threads per job (default: %(default)s)")
	#config = layeredconfig.LayeredConfig(defaults, env, layeredconfig.Commandline(parser=parser, commandline=argv[1:]))
	config = parser.parse_args(argv[1:])

	verbosity = ["ERROR", "WARNING", "INFO", "DEBUG"]
	logging.getLogger("vaex").setLevel(verbosity[config.verbose])
	#import vaex
	#vaex.set_log_level_debug()
	from vaex.settings import webserver as settings

	#filenames = config.filenames
	filenames = []
	filenames = config.filename
	datasets = []
	for filename in filenames:
		ds = vx.open(filename)
		if ds is None:
			print("error opening file: %r" % filename)
		else:
			datasets.append(ds)
	datasets = datasets or [vx.example()]
	#datasets = [ds for ds in datasets if ds is not None]
	logger.info("datasets:")
	for dataset in datasets:
		logger.info("\thttp://%s:%d/%s or ws://%s:%d/%s", config.address, config.port, dataset.name, config.address, config.port, dataset.name)
	server = WebServer(datasets=datasets, address=config.address, port=config.port, cache_byte_size=config.cache,
					   compress=config.compress, development=config.development,
					   threads_per_job=config.threads_per_job)
	server.serve()

if __name__ == "__main__":
	#logger.setLevel(logging.logging.DEBUG)
	#print sys.argv
	main(sys.argv)
	#3_threaded()
	#import time
	#time.sleep(10)
	#server.stop_serving()