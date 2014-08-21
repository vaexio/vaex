# -*- coding: utf-8 -*-
import threading
import Queue
import math

lock = threading.Lock()

class ThreadPool(object):
	
	def __init__(self, nthreads=4):
		self.nthreads = nthreads
		self.threads = [threading.Thread(target=self.execute, kwargs={"index":i}) for i in range(nthreads)]
		#self.semaphores_in = [threading.Semaphore(0) for i in range(nthreads)]
		self.queues_in = [Queue.Queue(1) for i in range(nthreads)]
		self.queues_out = [Queue.Queue(1) for i in range(nthreads)]
		for thread in self.threads:
			thread.setDaemon(True)
			thread.start()
			
	def close(self):
		self.callable = None
		print "closing threads"
		for index in range(self.nthreads):
			self.queues_in[index].put(None)
			
	def execute(self, index):
		print "index", index
		done = False
		while not done:
			#print "waiting..", index
			args = self.queues_in[index].get()
			if self.callable is None:
				print "ending thread.."
				done = True
			else:
				#print "running..", index
				try:
					#lock.acquire()
					result = self.callable(index, *args)
					#lock.release()
				except Exception, e:
					self.queues_out[index].put(e)
				else:
					self.queues_out[index].put(result)
				#print "done..", index
				#self.semaphore_outrelease()
		print "thread closed"
		
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
		subblock_size = math.ceil(total_length/self.nthreads)
		#subblock_count = math.ceil(total_length/subblock_size)
		args_list = []
		for index in range(self.nthreads):
			i1, i2 = index * subblock_size, (index +1) * subblock_size
			if i2 > total_length: # last one can be a bit longer
				i2 = total_length
			args_list.append((i1, i2))
		results = self.run_parallel(callable, args_list)
		for result in results:
			if isinstance(result, Exception):
				raise result
		
		

pool = ThreadPool()