# -*- coding: utf-8 -*-
import threading
import queue
import math
import multiprocessing
import sys
import vaex.utils
from . import logging
lock = threading.Lock()

thread_count_default = multiprocessing.cpu_count()
logger = logging.getLogger("vaex.multithreading")


class ThreadPoolIndex(object):
	def __init__(self, nthreads=thread_count_default):
		self.nthreads = nthreads
		self.threads = [threading.Thread(target=self.execute, kwargs={"index":i}) for i in range(nthreads)]
		self.lock = threading.Lock()
		self.queue_in = queue.Queue()
		self.queue_out = queue.Queue()
		for thread in self.threads:
			thread.setDaemon(True)
			thread.start()
		self.exception_occurred = False

	def map(self, callable, iterator, on_error=None):
		results = []
		with lock:
			self.exception_occurred = False
			self.callable = callable
			count = 0
			for element in iterator:
				#print "put in queue", element
				self.queue_in.put(element)
				count +=1

			done = False
			yielded = 0
			while not done:
				element = self.queue_out.get()
				#print "get from queue", element
				if isinstance(element, tuple) and len(element) > 1 and isinstance(element[1], Exception):
					if on_error:
						on_error(element[1])
						done = True
					else:
						#print "RAISE"
						self.exception_occurred = True
						logger.debug("wait for queue_in")
						self.queue_in.join()
						# now we know all threads are waiting for the queue_in, so they will not fill queue_out
						# but, it ma still contain elements, like exceptions, so flush it
						logger.debug("flush queue_out")
						while not self.queue_out.empty():
							self.queue_out.get()
						#print("********************")
						#print(element)
						raise element[0](element[1]) #element[0].__class__(None, element[2])
						#TODO: 2to3 gave this suggestion: raise element[1].with_traceback(element[2])
						done = True
				else:
					#yield element
					results.append(element)
					yielded += 1
				done = yielded == count
		return results


	def close(self):
		self.callable = None
		#print "closing threads"
		for index in range(self.nthreads):
			self.queue_in.put(None)

	def execute(self, index):
		done = False
		while not done:
			#print "waiting..", index
			args = self.queue_in.get()
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
							#lock.release()
						except Exception as e:
							exc_info = sys.exc_info()
							self.queue_out.put(exc_info)
						else:
							self.queue_out.put(result)
			finally:
				self.queue_in.task_done()

					#print "done..", index
					#self.semaphore_outrelease()
			#print "thread closed"



	def run_blocks(self, callabble, total_length, parts=10, on_error=None):
		for result in self.map(callabble, vaex.utils.subdivide(total_length, parts), on_error=on_error):
			yield result


pool = ThreadPoolIndex()

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
