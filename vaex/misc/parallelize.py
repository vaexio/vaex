from __future__ import print_function
# -*- coding: utf-8 -*-
import threading
import queue
import sys
import os
import pickle
from io import StringIO
import traceback
import time
import signal
import multiprocessing




pickle_encoding = "hex"
pickle_protocol = 2

def log(*args):
	#print "LOG: " + " ".join([str(k) for k in args])
	pass
	
def serialize(obj):
	data = pickle.dumps(obj, pickle_protocol)
	rawdata = data.encode(pickle_encoding)
	return rawdata
	
def deserialize(rawdata):
	data = rawdata.decode(pickle_encoding)
	obj = pickle.loads(data)
	return obj
	
class InfoThread(threading.Thread):
	def __init__(self, fullsize, executions):
		threading.Thread.__init__(self)
		self.fullsize = fullsize
		self.executions = executions
		self.setDaemon(True)
		#self.
		
	def n_done(self):
		n = 0
		for execution in self.executions:
			n += len(execution.results)
		return n
		
	def run(self):
		while 1:
			count = self.n_done() #self.fullsize - self.taskQueue.qsize()
			print("%d out of %d tasks completed (%5.2f%%)" % (count, self.fullsize, float(count)/self.fullsize*100))
			time.sleep(0.1)
			
class Watchdog(threading.Thread):
	def __init__(self, executions):
		threading.Thread.__init__(self)
		#self.fullsize = fullsize
		self.executions = executions
		#self.setDaemon(True):
		self.setDaemon(True)
		
	def run(self):
		while 1:
			time.sleep(0.5)
			for execution in self.executions:
				print(execution.isAlive(), execution.error)
		

import gavi.progressbar
class InfoThreadProgressBar(InfoThread):
	def __init__(self, fullsize, executions):
		InfoThread.__init__(self, fullsize, executions)
		self.bar = gavi.progressbar.ProgressBar(0, fullsize)
		
	def run(self):
		done = False
		error = False
		while (not done) and (not error):
			count = self.n_done() #self.fullsize - self.taskQueue.qsize()
			self.bar.update(count)
			time.sleep(0.1)
			done = count == self.fullsize
			for execution in self.executions:
				if execution.error:
					error = True 

class LocalExecutor(object):
	def __init__(self, thread, function):
		self.thread = thread
		self.function = function
		
	def stop(self):
		pass
		
	def init(self):
		pass
		
	def __call__(self, *args, **kwargs):
		info = "ok"
		exc_info = None
		result = None
		try:
			result = self.function(*args, **kwargs)
		except:
			info = "exception"
			exc_info = traceback.format_exc()
		return info, exc_info, result

class IOExecutor(object):
	def __init__(self, thread, function, init, id_number):
		self.thread = thread
		self.function = function
		self.initf = init
		self.id_number = id_number
		self.createIO()
		self.error = False

	def __call__(self, *args, **kwargs):
		self.log("p: sending execute command")
		#os.close(w) # we are not gonna write
		self.output.write("execute\n")
		self.output.write(serialize((args, kwargs)))
		self.output.write("\n")
		self.output.flush()
		self.log("p: reading response")
		response = self.input.readline().strip()
		#self.log("p: response:", `response`)
		error, exc_info, result = deserialize(response)
		return error, exc_info, result
		
	def init(self):
		return True
	
	def stop(self, error=False):
		self.error = error
		self.log("p: sending stop command")
		self.output.write("stop\n")
		self.output.flush()
		self.log("waitpid")
		self.join()
		self.log("waitpid done")
		self.input.close()
		self.output.close()
		
	def join(self):
		os.waitpid(self.pid, 0)
	
	def log(self, *args):
		log(self.thread.getName()+":", " ".join([str(k) for k in args]))

class ForkExecutor(IOExecutor):
	#def __init__(self, thread, function):
	#	IOExecutor.__init__(self, thread, function)
		
	def createIO(self):
		self.log("forking")
		#self.r, self.w = os.pipe()
		#self.output = os.fdopen(self.w, 'w')
		#self.input = os.fdopen(self.r, 'r')
		self.parent_input_p, self.child_output_p = os.pipe()
		self.child_input_p, self.parent_output_p = os.pipe()
		#self.r_c, self.w_c = os.pipe()
		#self.output_p = os.fdopen(self.w_p, 'w')
		#self.input_p = os.fdopen(self.r_p, 'r')
		#self.output_c = os.fdopen(self.w_c, 'w')
		#self.input_c = os.fdopen(self.r_c, 'r')
		self.pid = os.fork()
		if not self.pid: # we are the child
			#if self.id_number != 0:
			#	signal.signal(signal.SIGINT, signal.SIG_IGN)
			self.child_output = os.fdopen(self.child_output_p, 'w')
			self.child_input = os.fdopen(self.child_input_p, 'r')
			self.output = self.child_output
			self.input = self.child_input
			os.close(self.parent_output_p)
			os.close(self.parent_input_p)
			try:
				self.childPart()
			except BaseException:
				print("oops")
			self.input.close()
			self.output.close()
			os._exit(0)
		else:
			self.parent_output = os.fdopen(self.parent_output_p, 'w')
			self.parent_input = os.fdopen(self.parent_input_p, 'r')
			self.output = self.parent_output
			self.input = self.parent_input
			os.close(self.child_output_p)
			os.close(self.child_input_p)
		# else just continue
			
			
	def childPart(self):
		done = False
		self.log("c: child")
		if self.initf is not None:
			self.initf(self.id_number)
		
		while not done:
			self.log("c: waiting for command...")
			command = self.input.readline().strip()
			self.log("command:", command)
			if command == "execute":
				# execute...
				response = self.input.readline().strip()
				self.log("c: args:", repr(response))
				args, kwargs = deserialize(response)
				info = "ok"
				exc_info = None
				result = None
				try:
					#self.log("c: calling")
					result = self.function(*args, **kwargs)
					#self.log("c: result" +result)
				except BaseException:
					info = "exception"
					exc_info = traceback.format_exc()
					#raise "bla"
					#done = True
				except KeyboardInterrupt:
					info = "exception"
					exc_info = traceback.format_exc()
					#raise "bla"
					#done = True
				# encode result
				self.log("c: pickling")
				self.output.write(serialize((info, exc_info, result)))
				self.output.write("\n")
				self.output.flush()
				self.log("c: closing")
			elif command == "stop":
				self.log("c: stopping...")
				done = True
				self.log("c: closed, exiting")
			else:
				done = True
				self.log("c: unknown command", repr(command))
				# and exit
				#os._exit(0) # os._exit is used for children
	
	
		
	def join(self):
		os.waitpid(self.pid, 0)
	
class Execution(threading.Thread):
	def __init__(self, taskQueue, fork, function, init, id_number, args, kwargs):
		self.taskQueue = taskQueue
		self.fork = fork
		self.function = function
		self.init = init
		self.id_number = id_number
		self.args = args
		self.kwargs = kwargs
		self.results = []
		self.done = False
		self.error = False
		threading.Thread.__init__(self)
		if self.fork:
			self.executor = ForkExecutor(self, self.function, self.init, self.id_number)
		else:
			self.executor = LocalExecutor(self, self.function)
	
	def log(self, *args):
		log(self.getName()+":", " ".join([str(k) for k in args]))
		
	
	def run(self):
		task = None
		if self.executor.init():
			task = None
			self.log("starting")
			try:
				task = self.taskQueue.get(False)
			except queue.Empty:
				self.log("empty at first try")
			while task is not None and self.error is False:
				tasknr, args = task
				args = list(args)
				common_args = list(self.args)
				args.extend(common_args)
				kwargs = dict(self.kwargs)
				
				info, exc_info, result = self.executor(*args, **kwargs)
				if info == "exception":
					print(exc_info)
					self.error = True
				self.log("r: got result")
				self.results.append((tasknr, result))
				try:
					task = self.taskQueue.get(False)
				except queue.Empty:
					self.log("empty queue")
					task = None
			#if not self.error:
			self.executor.stop(self.error)
			self.log("done")
		else:
			self.log("failure starting slave")
			self.error = True
		self.done = True

def countcores():
	stdin, stdout = os.popen2("cat /proc/cpuinfo | grep processor")
	lines = stdout.readlines()
	return len(lines)

def timed(f):
	def execute(*args, **kwargs):
		t0 = time.time()
		utime0, stime0, child_utime0, child_stime0, walltime0 = os.times()
		result = f(*args, **kwargs)
		dt = time.time() - t0
		utime, stime, child_utime, child_stime, walltime = os.times()
		#name = f.func_name
		print()
		print("user time:            % 9.3f sec." % (utime-utime0))
		print("system time:          % 9.3f sec." % (stime-stime0))
		print("user time(children):  % 9.3f sec." % (child_utime-child_utime0))
		print("system time(children):% 9.3f sec." % (child_stime-child_stime0))
		print()
		dt_total = child_utime-child_utime0 + child_stime-child_stime0+utime-utime0+stime-stime0
		print("total cpu time:       % 9.3f sec. (time it would take on a single cpu/core)" % (dt_total))
		print("elapsed time:         % 9.3f sec. (normal wallclock time it took)" % (walltime-walltime0))
		dt = walltime-walltime0
		if dt == 0:
			eff = 0.
		else:
			eff = dt_total/(dt)
		print("efficiency factor     % 9.3f      (ratio of the two above ~= # cores)" % eff)
		return result
	return execute

def parallelize(cores=None, fork=True, flatten=False, info=False, infoclass=InfoThreadProgressBar, init=None, *args, **kwargs):
	"""Function decorater that executes the function in parallel
	
	Usage::

		@parallelize(cores=10, info=True)
		def f(x):
			return x**2
		
		x = numpy.arange(0, 100, 0.1)
		y = f(x) # this gets executed parallel

	:param cores: number of cpus/cores to use (if None, it counts the cores using /proc/cpuinfo)
	:param fork: fork a process (should always be true since of the GIT, but can be used with c modules that release the GIT)
	:param flatten: if False and each return value is a list, final result will be a list of lists, if True, all lists are combined to one big list
	:param info: show progress bar (see infoclass)
	:param infoclass: class to instantiate that shows the progress (default shows progressbar)
	:param init: function to be called in each forked process before executing, can be used to set the seed, takes a integer as parameter (number that identifies the process)
	:param args: extra arguments passed to function
	:param kwargs: extra keyword arguments passed to function
			
	Example::
			
		@parallelize(cores=10, info=True, n=2)
		def f(x, n):
			return x**n
		
		x = numpy.arange(0, 100, 0.1)
		y = f(x) # this gets executed parallel
			
	
	
	"""
	if cores == None:
		cores = multiprocessing.cpu_count()
	def wrapper(f):
		def execute(*multiargs):
			results = []
			len(list(zip(*multiargs)))
			N = len(multiargs[0])
			if info:
				print("running %i jobs on %i cores" % (N, cores))
			taskQueue = queue.Queue(len(multiargs[0]))
			#for timenr in range(times):
			#	taskQueue.put(timenr)
			for tasknr, _args in enumerate(zip(*multiargs)):
				taskQueue.put((tasknr, list(_args)))
			#for timenr in range(times):
			#	result = f(*args, **kwargs)
			#	results.append(result)
			executions = [Execution(taskQueue, fork, f, init, corenr, args, kwargs) for corenr in range(cores)]
			if info:
				infoobj = infoclass(len(multiargs[0]), executions)
				infoobj.start()
			for i, execution in enumerate(executions):
				execution.setName("T-%d" % i)
				execution.start()
			#if 1:
			#	watchdog = Watchdog(executions)
			#	watchdog.start()
			error = False
			for execution in executions:
				log("joining:",execution.getName())
				try:
					execution.join()
				except BaseException:
					error = True
				results.extend(execution.results)
				if execution.error:
					error = True 
			if info:
				infoobj.join()
			if error:
				print("error", file=sys.stderr)
				results = None
				raise Exception("error in one or more of the executors")
			else:
				results.sort(cmp=lambda a, b: cmp(a[0], b[0]))
				results = [k[1] for k in results]
				#print "bla", results
				if flatten:
					flatresults = []
					for result in results:
						flatresults.extend(result)
					results = flatresults
			return results
		return execute
	return wrapper
			
			
if __name__ == "__main__":
	@timed
	@parallelize(cores=6, fork=True, flatten=True, text="common argument")
	def testprime(from_nr, to_nr, text=""):
		#print text, from_nr, to_nr
		primes = []
		from_nr = max(from_nr, 2)
		for p in range(from_nr, to_nr):
			isprime = True
			time.sleep(1)
			
			for i in range(2, p):
				if p % i == 0:
					isprime = False
					break
			if isprime:
				primes.append(p)
		return primes
			
	
	testnumbers = list(range(0, 10001, 100))
	from_nrs = testnumbers[:-1]
	to_nrs = testnumbers[1:]
	results = testprime(from_nrs, to_nrs)
	#print results
	#print countcores()