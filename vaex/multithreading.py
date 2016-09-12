# -*- coding: utf-8 -*-
import threading
import queue
import math
import multiprocessing
import time
import sys
import vaex.utils
import logging
import traceback

thread_count_default = multiprocessing.cpu_count()# * 2 + 1
logger = logging.getLogger("vaex.multithreading")


cancel_lock = threading.Lock()
print_lock = threading.Lock()
class MiniJob(object):
	def __init__(self, callable, queue_out, args):
		self.thread_index = None #thread_index
		self.callable = callable
		self.queue_out = queue_out
		self.args = args
		self.result = None
		self.exc_info = None
		self.cancelled = False
		self.lock = threading.Lock()

	def cancel(self):
		with self.lock:
			self.cancelled = True

	def __call__(self, thread_index):
		if not self.cancelled:
			try:
				result = self.callable(thread_index, *self.args)
				with self.lock:
					if not self.cancelled:
						self.result = result
						self.queue_out.put(self)
					#else:
					#	self.queue_out(None)
			except Exception as e:
				exc_info = sys.exc_info()
				with self.lock:
					if not self.cancelled:
						self.exc_info = exc_info
						self.queue_out.put(self)
					#else:
					#	self.queue_out(None)

class ThreadPoolIndex(object):
	def __init__(self, nthreads=None):
		self.nthreads = nthreads or thread_count_default
		self.jobs = []
		self.threads = [threading.Thread(target=self.execute, kwargs={"index":i}) for i in range(self.nthreads)]
		self.thread_locks = [threading.Lock() for i in range(self.nthreads)]
		self.lock = threading.Lock()
		self.queue_in = [] #queue.Queue()
		self.queue_out = queue.Queue()
		self.new_jobs_event = threading.Event()
		for thread in self.threads:
			thread.setDaemon(True)
			thread.start()
		self.exception_occurred = False
		self._working = False
		self.lock = threading.Lock()

	def map(self, callable, iterator, on_error=None, progress=None, cancel=None):
		results = []
		progress = progress or (lambda x: True)
		cancel = cancel or (lambda: True)
		try:
			if self._working:
				raise RuntimeError("reentered ThreadPoolIndex.map")
			with self.lock:
				self._working = True
				count = 0
				self.jobs = []
				alljobs = []
				for element in iterator:
					job = MiniJob(callable=callable, args=element, queue_out=self.queue_out)
					self.jobs.append(job)
					alljobs.append(job)
					count +=1
				self.new_jobs_event.set()
				done = False
				yielded = 0
				while not done:
					job = self.queue_out.get()
					if 1:
						if job.exc_info is not None:
							for other_job in alljobs:
								other_job.cancel()
							while not self.queue_out.empty():
								self.queue_out.get()
							if sys.version_info >= (3, 0):
								raise job.exc_info[1].with_traceback(job.exc_info[2])
							else:
								logger.error("real traceback: %s", traceback.format_tb(job.exc_info[2]))
								raise job.exc_info[1]#(element[2])
						results.append(job.result)
						yielded += 1
						if progress(yielded/float(count)) == False:
							for other_job in alljobs:
								other_job.cancel()
							while not self.queue_out.empty():
								self.queue_out.get()
							done = True
						else:
							done = yielded == count
			return results
		finally:
			self._working = False
			#self.jobs = []
			self.new_jobs_event.clear()

	def execute(self, index):
		done = False
		while not done:
			#print "waiting..", index
			t0 = time.time()
			empty = True
			while empty:
				try:
					job = self.jobs.pop()
					empty = False
				except IndexError:
					self.new_jobs_event.wait()
			job(index)

	def _map(self, callable, iterator, on_error=None, progress=None, cancel=None):
		results = []
		progress = progress or (lambda x: True)
		cancel = cancel or (lambda: True)

		try:
			if self._working:
				raise RuntimeError("reentered ThreadPoolIndex.map")
			with self.lock:
				self._working = True
				self.exception_occurred = False
				self.callable = callable
				count = 0
				for element in iterator:
					#print "put in queue", element
					#logger.debug("put in queue: %r", element)
					self.queue_in.append(element)
					#$$self.queue_in.unfinished_tasks += 1
					count +=1
				self.new_jobs_event.set()
				#self.queue_in.put(element)
				#self.queue_in.put(element)
				#try:
				#for element in iterator:
				#	self.queue_in.not_empty.notify()
				#except:
				#	pass

				def stop():
					self.exception_occurred = True
					logger.debug("wait for queue_in")
					#self.queue_in.join()
					self.queue_in = []
					# now we know all threads are waiting for the queue_in, so they will not fill queue_out
					# but, it ma still contain elements, like exceptions, so flush it
					logger.debug("flush queue_out")
					while not self.queue_out.empty():
						self.queue_out.get()
					logger.debug("flush queue_out done")
				done = False
				yielded = 0
				while not done:
					#logger.debug("...")
					element = self.queue_out.get()
					#logger.debug("got queue element")
					#print "get from queue", element
					if isinstance(element, tuple) and len(element) > 1 and isinstance(element[1], Exception):
						if on_error:
							on_error(element[1])
							done = True
						else:
							#print "RAISE"
							stop()
							#print("********************")
							#print(element)
							#raise element[1]
							if sys.version_info >= (3, 0):
								raise element[1].with_traceback(element[2])
							else:
								raise element[1]#(element[2])
							#raise element[0](element[1])
							#raise element[0].__class__(None, element[2])
							#TODO: 2to3 gave this suggestion: raise element[1].with_traceback(element[2])
							done = True
					else:
						#yield element
						results.append(element)
						yielded += 1
						if progress(yielded/float(count)) == False:
							stop()
							cancel()
							done = True
						else:
							done = yielded == count
			return results
		finally:
			self._working = False
			self.new_jobs_event.clear()


	def close(self):
		self.callable = None
		#print "closing threads"
		for index in range(self.nthreads):
			self.queue_in.append(None)
		self.new_jobs_event.set()


	def _execute(self, index):
		done = False
		while not done:
			#print "waiting..", index
			t0 = time.time()
			empty = True
			while empty:
				try:
					args = self.queue_in.pop()
					empty = False
				except IndexError:
					self.new_jobs_event.wait()
			#logger.debug("took %f to get a job", time.time() - t0)
			try:
				if self.exception_occurred:
					pass # just eat the whole queue after an exception
					#print "thread: %s just eat queue item %r" % (index, args)
				else:
					if self.callable is None:
						#print "ending thread.."
						done = True
					else:
						#print "running..", index
						try:
							#lock.acquire()
							result = self.callable(index, *args)
							#self.queue_in.task_done()
							#lock.release()
						except Exception as e:
							exc_info = sys.exc_info()
							self.queue_out.put(exc_info)
						else:
							self.queue_out.put(result)
			finally:
				#self.queue_in.task_done()
				pass

					#print "done..", index
					#self.semaphore_outrelease()
			#print "thread closed"



	def run_blocks(self, callabble, total_length, parts=10, on_error=None):
		for result in self.map(callabble, vaex.utils.subdivide(total_length, parts), on_error=on_error):
			yield result


main_pool = None #ThreadPoolIndex()
def get_main_pool():
	global main_pool
	if main_pool is None:
		main_pool = ThreadPoolIndex()
	return main_pool

class ThreadPool(object):

	def __init__(self, nthreads=thread_count_default):
		self.nthreads = nthreads
		self.threads = [threading.Thread(target=self.execute, kwargs={"index":i}) for i in range(nthreads)]
		#self.semaphores_in = [threading.Semaphore(0) for i in range(nthreads)]
		self.queues_in = [queue.Queue(1) for i in range(nthreads)]
		self.queues_out = [queue.Queue(1) for i in range(nthreads)]
		for thread in self.threads:
			thread.setDaemon(True)
			thread.start()

	def close(self):
		self.callable = None
		#print "closing threads"
		for index in range(self.nthreads):
			self.queues_in[index].put(None)

	def execute(self, index):
		#print "index", index
		done = False
		while not done:
			#print "waiting..", index
			args = self.queues_in[index].get()
			if self.callable is None:
				#print "ending thread.."
				done = True
			else:
				#print "running..", index
				try:
					#lock.acquire()
					result = self.callable(index, *args)
					#lock.release()
				except Exception as e:
					exc_info = sys.exc_info()
					self.queues_out[index].put(exc_info)
				else:
					self.queues_out[index].put(result)
				#print "done..", index
				#self.semaphore_outrelease()
		#print "thread closed"

	def run_parallel(self, callable, args_list=[]):
		#lock.acquire()
		self.callable = callable
		for index in range(self.nthreads):
			if index < len(args_list) :
				args = args_list[index]
			else:
				args = []
			self.queues_in[index].put(args)
		#for thread in self.threads:
		#	self.queue_out.get()
		results = [queue.get() for queue in self.queues_out]
		#lock.release()
		return results

	def run_blocks(self, callable, total_length):
		subblock_size = int(math.ceil(float(total_length)/self.nthreads))
		#subblock_count = math.ceil(total_length/subblock_size)
		args_list = []
		for index in range(self.nthreads):
			i1, i2 = index * subblock_size, (index +1) * subblock_size
			if i2 > total_length: # last one can be a bit longer
				i2 = total_length
			args_list.append((i1, i2))
		results = self.run_parallel(callable, args_list)
		for result in results:
			#print result, isinstance(result, tuple)#, len(result) > 1, isinstance(result[1], Exception)
			if isinstance(result, tuple) and len(result) > 1 and isinstance(result[1], Exception):
				raise result[1](None, result[2])
				# TODO: 2to3 gave this suggestion raise result[1].with_traceback(result[2])
		return results


#pool = ThreadPool()
