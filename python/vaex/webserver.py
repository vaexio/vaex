from __future__ import absolute_import
__author__ = 'maartenbreddels'

import tornado.ioloop
import tornado.web
import tornado.httpserver
import tornado.auth
import threading
from . import logging
import vaex as vx
import json
import inspect

logger = logging.getLogger("vaex.webserver")


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

def task_invoke(subspace, method_name, request):
	method = getattr(subspace, method_name)
	args, varargs, kwargs, defaults = inspect.getargspec(method)
	#print inspect.getargspec(method)
	#args_required = args[:len(args)-len(defaults)]
	kwargs = {}
	for arg in args:
		if arg in request.arguments:
			#print arg, repr(request.arguments[arg][0])
			kwargs[arg] = json.loads(request.arguments[arg][0])
	values = method(**kwargs)
	return values

import collections

class ListHandler(tornado.web.RequestHandler):
	def initialize(self, datasets=None):
		self.datasets = datasets or [vx.example()]
		self.datasets_map = collections.OrderedDict([(ds.name,ds) for ds in self.datasets])

	def get(self):
        #self.write("Hello, world")
		#print self.request.path
		parts = [part for part in self.request.path.split("/") if part]
		logger.debug("request: %r" % parts)
		#print parts
		#print self.request.headers
		user_id = self.get_cookie("user_id")
		logger.debug("user_id in cookie: %r", user_id)
		if user_id is None:
			import uuid
			user_id = str(uuid.uuid4())
			self.set_cookie("user_id", user_id)
		logger.debug("user_id: %r", user_id)
		#print parts
		if parts[0] == "sleep":
			seconds = float(parts[1])
			import time
			time.sleep(seconds)
			self.write({"result":seconds})
		elif parts[0] == "datasets":
			if len(parts) == 1:
				response = dict(datasets=[{"name":ds.name, "full_length":len(ds), "column_names":ds.get_column_names()} for ds in self.datasets])
				self.write(response)
			else:
				dataset_name = parts[1]
				if dataset_name not in self.datasets_map:
					self.error("dataset does not exist: %r, possible options: %r" % (dataset_name, self.datasets_map.keys()))
				else:
					if len(parts) > 2:
						method_name = parts[2]
						logger.debug("method: %r args: %r" % (method_name, self.request.arguments))
						if "expressions" in self.request.arguments:
							expressions = json.loads(self.request.arguments["expressions"][0])
						else:
							expressions = None
						dataset = self.datasets_map[dataset_name]
						if dataset.mask is not None:
							logger.debug("selection: %r", dataset.mask.sum())
						if "active_fraction" in self.request.arguments:
							active_fraction = json.loads(self.request.arguments["active_fraction"][0])
							logger.debug("setting active fraction to: %r", active_fraction)
							dataset.set_active_fraction(active_fraction)
						if "variables" in self.request.arguments:
							variables = json.loads(self.request.arguments["variables"][0])
							logger.debug("setting variables to: %r", variables)
							for key, value in variables:
								dataset.set_variable(key, value)
						if "virtual_columns" in self.request.arguments:
							virtual_columns = json.loads(self.request.arguments["virtual_columns"][0])
							logger.debug("setting virtual_columns to: %r", virtual_columns)
							for key, value in virtual_columns:
								dataset.add_virtual_column(key, value)
						subspace = dataset(*expressions) if expressions else None
						try:
							if subspace:
								if "selection_name" in self.request.arguments:
									selection_name = json.loads(self.request.arguments["selection_name"][0])
									logger.debug("selection_name = %r", selection_name)
									if selection_name == "default":
										logger.debug("taking selected")
										subspace = subspace.selected()
							logger.debug("subspace: %r", subspace)
							if method_name in ["minmax", "var", "mean", "sum", "limits_sigma", "nearest"]:
								#print "expressions", expressions
								values = task_invoke(subspace, method_name, self.request)
								logger.debug("result: %r", values)
								#print values, expressions
								values = values.tolist() if hasattr(values, "tolist") else values
								self.write({"result": values})
							elif method_name == "histogram":
								grid = task_invoke(subspace, method_name, self.request)
								self.set_header("Content-Type", "application/octet-stream")
								self.write(grid.tostring())
							elif method_name in ["select", "evaluate", "lasso_select"]:
								result = task_invoke(dataset, method_name, self.request)
								result = result.tolist() if hasattr(result, "tolist") else result
								self.write({"result": result})
							else:
								logger.error("unknown method: %r", method_name)
								self.error("unknown method: " + method_name)
						except (SyntaxError, KeyError) as e:
							self.exception(e)

	def exception(self, exception):
		logger.exception("handled exception at server, all fine")
		self.write({"exception": {"class":str(exception.__class__.__name__), "msg": str(exception)} })
	def error(self, msg):
		self.write({"error": msg}) #, "result":None})



class QueueHandler(tornado.web.RequestHandler):
	def initialize(self, datasets):
		self.datasets = datasets
		self.datasets_map = dict([(ds.name,ds) for ds in self.datasets])

	def get(self):
        #self.write("Hello, world")
		self.write(dict(datasets=[{"name":ds.name, "length":len(ds)} for ds in self.datasets]))



class WebServer(threading.Thread):
	def __init__(self, address="localhost", port=9000, datasets=[]):
		threading.Thread.__init__(self)
		self.setDaemon(True)
		self.address = address
		self.port = port
		self.started = threading.Event()
		self.options = dict(datasets=datasets)

		self.application = tornado.web.Application([
			(r"/queue", QueueHandler, self.options),
			(r"/auth", GoogleOAuth2LoginHandler, {}),
			(r"/.*", ListHandler, self.options),
		])

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
		self.server.listen(self.port, self.address)
		self.started.set()
		#self.ioloop.add_callback(self.started.set)
		#if not self.ioloop.is_running():
		try:
			self.ioloop.start()
		except RuntimeError:
			pass # TODO: not sure why this happens in the unittest
		self.ioloop.clear_current()

	def stop_serving(self):
		logger.debug("stop server")
		self.server.stop()
		logger.debug("stop io loop")
		#self.ioloop.stop()
		self.ioloop.clear_current()

if __name__ == "__main__":
	#logger.setLevel(logging.logging.DEBUG)
	import vaex
	vaex.set_log_level_debug()
	import sys
	from vaex.settings import webserver as settings
	filenames = settings.get("datasets.filenames", [])
	filenames += sys.argv[1:]
	print filenames
	datasets = [vx.open(filename) for filename in filenames]
	datasets = datasets or [vx.example()]
	server = WebServer(datasets=datasets, address=settings.get("server.address", "0.0.0.0"), port=settings.get("server.port", 9000))
	server.serve()

	#3_threaded()
	#import time
	#time.sleep(10)
	#server.stop_serving()