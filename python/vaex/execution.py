import functools
import vaex.vaexfast
import numpy as np
from vaex.utils import Timer
import vaex.events
from . import multithreading
import time
import math
import vaex.ui.expressions as expr
import numexpr as ne
from functools import reduce

__author__ = 'breddels'

buffer_size_default = 1e6 # TODO: this should not be fixed, larger means faster but also large memory usage

import threading
import queue
import math
import multiprocessing
import sys
import collections
lock = threading.Lock()
import vaex.multithreading

import logging
logger = logging.getLogger("vaex.execution")

thread_count_default = multiprocessing.cpu_count()

class UserAbort(Exception):
	def __init__(self, reason):
		super(UserAbort, self).__init__(reason)
class Column(collections.namedtuple('Column', ['dataset', 'expression'])):
	def needs_copy(self):
		return not \
			(self.expression in self.dataset.column_names  \
			and not isinstance(self.dataset.columns[self.expression], vaex.dataset._ColumnConcatenatedLazy)\
			and self.dataset.columns[self.expression].dtype.type==np.float64 \
			and self.dataset.columns[self.expression].strides[0] == 8 \
			and self.expression not in self.dataset.virtual_columns)
				#and False:

class Job(object):
	def __init__(self, task, order):
		self.task = task
		self.order = order

# mutex for numexpr, it is not thread save
ne_lock = threading.Lock()

class Executor(object):
	def __init__(self, thread_pool=None, buffer_size=None, thread_mover=None):
		self.thread_pool = thread_pool or vaex.multithreading.ThreadPoolIndex()
		self.task_queue = []
		self.buffer_size = buffer_size or buffer_size_default
		self.signal_begin = vaex.events.Signal("begin")
		self.signal_progress = vaex.events.Signal("progress")
		self.signal_end = vaex.events.Signal("end")
		self.signal_cancel = vaex.events.Signal("cancel")
		self.thread_mover = thread_mover or (lambda fn, *args, **kwargs: fn(*args, **kwargs))
		self._is_executing = False

	def schedule(self, task):
		self.task_queue.append(task)
		return task

	def run(self, task):
		logger.debug("added task: %r" % task)
		previous_queue = self.task_queue
		try:
			self.task_queue = [task]
			self.execute()
		finally:
			self.task_queue = previous_queue
		return task._value

	def execute(self):
		if self._is_executing:
			logger.debug("nested execute call")
			# this situation may happen since in this methods, via a callback (to update a progressbar) we enter
			# Qt's eventloop, which may execute code that will call execute again
			# as long as that code is using async tasks (i.e. promises) we can simple return here, since after
			# the execute is almost finished, any new tasks added to the task_queue will get executing
			return
		# u 'column' is uniquely identified by a tuple of (dataset, expression)
		self._is_executing = True
		try:
			t0 = time.time()
			logger.debug("executing queue: %r" % (self.task_queue))
			task_queue_all = list(self.task_queue)
			self.task_queue = []
			#for task in self.task_queue:
			#$	print task, task.expressions_all
			datasets = set(task.dataset for task in task_queue_all)
			cancelled = [False]
			def cancel():
				logger.debug("cancelling")
				self.signal_cancel.emit()
				cancelled[0] = True
			try:
				# process tasks per dataset
				self.signal_begin.emit()
				for dataset in datasets:
					task_queue = [task for task in task_queue_all if task.dataset == dataset]
					expressions = list(set(expression for task in task_queue for expression in task.expressions_all))

					for task in task_queue:
						task._results = []
						task.signal_progress.emit(0)
					block_scopes = [dataset._block_scope(0, self.buffer_size) for i in range(self.thread_pool.nthreads)]
					def process(thread_index, i1, i2):
						if not cancelled[0]:
							block_scope = block_scopes[thread_index]
							block_scope.move(i1, i2)
							#with ne_lock:
							block_dict = {expression:block_scope.evaluate(expression) for expression in expressions}
							for task in task_queue:
								blocks = [block_dict[expression] for expression in task.expressions_all]
								if not cancelled[0]:
									task._results.append(task.map(thread_index, i1, i2, *blocks))
								# don't call directly, since ui's don't like being updated from a different thread
								#self.thread_mover(task.signal_progress, float(i2)/length)
								#time.sleep(0.3)

					length = len(dataset)
					#print self.thread_pool.map()
					for element in self.thread_pool.map(process, vaex.utils.subdivide(length, max_length=self.buffer_size),\
														progress=lambda p: all(self.signal_progress.emit(p)) and\
																all([all(task.signal_progress.emit(p)) for task in task_queue]),
														cancel=cancel):
						pass # just eat all element
					self._is_executing = False
			except:
				# on any error we flush the task queue
				self.signal_cancel.emit()
				logger.exception("error in task, flush task queue")
				raise
			logger.debug("executing took %r seconds" % (time.time() - t0))
			# while processing the self.task_queue, new elements will be added to it, so copy it
			logger.debug("cancelled: %r", cancelled)
			if cancelled[0]:
				logger.debug("execution aborted")
				task_queue = task_queue_all
				for task in task_queue:
					task._result = task.reduce(task._results)
					#task.reject(UserAbort("cancelled"))
					# remove references
					task._result = None
					task._results = None
			else:
				task_queue = task_queue_all
				for task in task_queue:
					task._result = task.reduce(task._results)
					task.fulfill(task._result)
					# remove references
					task._result = None
					task._results = None
				self.signal_end.emit()
				# if new tasks were added as a result of this, execute them immediately
				# TODO: we may want to include infinite recursion protection
				if len(self.task_queue) > 0:
					self.execute()
		finally:
			self._is_executing = False

#default_executor = None
