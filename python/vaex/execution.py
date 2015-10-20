import functools
import vaex.vaexfast
import numpy as np
from vaex.utils import Timer
import vaex.events
import multithreading
import time
import math
import vaex.ui.expressions as expr
import numexpr as ne

__author__ = 'breddels'

buffer_size = 1e6 # TODO: this should not be fixed, larger means faster but also large memory usage

import threading
import Queue
import math
import multiprocessing
import sys
import collections
lock = threading.Lock()

import vaex.logging
logger = vaex.logging.getLogger("vaex.execution")

thread_count_default = multiprocessing.cpu_count()

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
	def __init__(self, dataset, thread_pool):
		self.dataset = dataset
		self.thread_pool = thread_pool
		self.task_queue = []

	def schedule(self, task):
		self.task_queue.append(task)
		return task

	def run(self, task):
		assert task.dataset == self.dataset
		previous_queue = self.task_queue
		try:
			self.task_queue = [task]
			self.execute()
		finally:
			self.task_queue = previous_queue
		return task._value

	def execute(self):
		# u 'column' is uniquely identified by a tuple of (dataset, expression)
		t0 = time.time()
		logger.debug("executing queue: %r" % (self.task_queue))

		expressions = list(set(expression for task in self.task_queue for expression in task.expressions_all))

		for task in self.task_queue:
			task._results = []
		block_scopes = [self.dataset._block_scope(0, buffer_size) for i in range(self.thread_pool.nthreads)]
		def process(thread_index, i1, i2):
			block_scope = block_scopes[thread_index]
			block_scope.move(i1, i2)
			with ne_lock:
				block_dict = {expression:block_scope.evaluate(expression) for expression in expressions}
			for task in self.task_queue:
				blocks = [block_dict[expression] for expression in task.expressions_all]
				task._results.append(task.map(thread_index, i1, i2, *blocks))

		length = len(self.dataset)
		#print self.thread_pool.map()
		for element in self.thread_pool.map(process, vaex.utils.subdivide(length, max_length=buffer_size)):
			pass # just eat all element
		logger.debug("executing took %r seconds" % (time.time() - t0))
		# while processing the self.task_queue, new elements will be added to it, so copy it
		task_queue = list(self.task_queue)
		self.task_queue = []
		for task in task_queue:
			task._result = task.reduce(task._results)
			task.fulfill(task._result)
			# remove references
			task._result = None
			task._results = None
		# if new tasks were added as a result of this, execute them immediately
		# TODO: we may want to include infinite recursion protection
		if len(self.task_queue) > 0:
			self.execute()


	def _execute(self):
		# u 'column' is uniquely identified by a tuple of (dataset, expression)
		t0 = time.time()
		logger.info("executing queue: %r" % (self.task_queue))
		columns = list(set(Column(task.dataset, expression) for task in self.task_queue for expression in task.expressions_all))
		expressions = list(set(expression for task in self.task_queue for expression in task.expressions_all))
		columns_copy = [column for column in columns if column.needs_copy()]
		buffers_needs = len(columns_copy)
		thread_buffers = {} # maps column to a list of buffers
		thread_virtual_buffers = {} # maps column to a list of buffers
		for column in columns_copy:
			thread_buffers[column.expression] = np.zeros((self.thread_pool.nthreads, buffer_size))
		for name in self.dataset.virtual_columns.keys():
			thread_virtual_buffers[name] = np.zeros((self.thread_pool.nthreads, buffer_size))

		for task in self.task_queue:
			task._results = []
			#print "task", task, task._results
		def process(thread_index, i1, i2):
			#print "process", thread_index, i1, i2
			size = i2-i1 # size may be smaller for the last step
			# truncate the buffers accordingly, and pick the right one for this thread
			buffers = {key:buffer[thread_index,0:size] for key,buffer in thread_buffers.items()}
			virtual_buffers = {key:buffer[thread_index,0:size] for key,buffer in thread_virtual_buffers.items()}
			def evaluate(column):
				ne.evaluate(column.expression, local_dict=local_dict)
			local_dict = {} # contains only the real column
			local_dict.update(self.dataset.variables)
			for column in self.dataset.columns.keys():
				local_dict[column] = self.dataset.columns[column][i1:i2]
			# this dict should in the end contains all data blocks for all columns or expressions
			# and virtual columns
			block_dict = local_dict.copy()

			# TODO: do not evaluate ALL virtual columns
			for name, expression in self.dataset.virtual_columns.items():
				with ne_lock:
					ne.evaluate(expression, local_dict=block_dict, out=virtual_buffers[name])
				# only update the dict after evaluating!
				# if not, a virtual column may reference a not yet computed column with zeros or garbage
				block_dict[name] = virtual_buffers[name]
				#print "set", name, "to", virtual_buffers[name]

			# TODO: we copy virtual columns, avoid that
			#block_dict.update(buffers)
			# now we do the dynamic evaluated 'columns'
			for column in columns:
				if column.needs_copy():
					with ne_lock:
						ne.evaluate(column.expression, local_dict=block_dict, out=buffers[column.expression])
						block_dict[column.expression] = buffers[column.expression]
			for task in self.task_queue:
				blocks = [block_dict[expression] for expression in task.expressions_all]
				task._results.append(task.map(thread_index, i1, i2, *blocks))
		length = len(self.dataset)
		#print self.thread_pool.map()
		for element in self.thread_pool.map(process, vaex.utils.subdivide(length, max_length=buffer_size)):
			pass # just eat all element
		logger.debug("executing took %r seconds" % (time.time() - t0))
		# while processing the self.task_queue, new elements will be added to it, so copy it
		task_queue = list(self.task_queue)
		self.task_queue = []
		for task in list(task_queue):
			task._result = task.reduce(task._results)
			task.fulfill(task._result)
		# if new tasks were added as a result of this, execute them immediately
		# TODO: we may want to include infinite recursion protection
		if len(self.task_queue) > 0:
			self.execute()

		#for
		#buffers = {column:[] for column in columns if column.needs_copy()}


default_executor = None


class FakeLogger(object):
	def debug(self, *args):
		pass
	def info(self, *args):
		pass
	def error(self, *args):
		pass
	def exception(self, *args):
		pass

#logger = FakeLogger()

class Job(object):
	def __init__(self, callback, expressions):
		pass

class Job(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class JobsManager(object):
	def __init__(self):
		#self.datasets = datasets
		self.jobs = []
		self.order_numbers = []
		self.after_execute = []
		self.signal_begin = vaex.events.Signal("begin")
		self.signal_end = vaex.events.Signal("end")
		self.signal_cancel = vaex.events.Signal("cancel")
		self.signal_progress = vaex.events.Signal("progress")
		self.progress_total = 0

	def addJob(self, order, callback, dataset, *expressions, **variables):
		args = order, callback, dataset, expressions, variables
		logger.info("job: %r" % (args,))
		if order not in self.order_numbers:
			self.order_numbers.append(order)
		self.progress_total += len(dataset)
		self.jobs.append((order, callback, dataset, [None if e is None or len(e) == 0 else e for e in expressions], variables))


	def add_scalar_jobs(self, dataset, function_per_block, function_merge, function_final, use_mask, expressions, feedback=None):
		#print "-->", self, dataset, function_per_block, function_post_blocks, function_final, use_mask, expressions
		pool = multithreading.pool
		subresults = [None] * len(expressions)

		#print "expressions", expressions
		class Wrapper(object):
			pass
		wrapper = Wrapper()
		wrapper.N_done = 0
		N_total = len(expressions) * (len(dataset)) # if not use_mask else np.sum(dataset.mask))


		def calculate(info, block, index):
			#print block, index
			def subblock(thread_index, sub_i1, sub_i2):
				if use_mask:
					data = block[sub_i1:sub_i2][dataset.mask[info.i1:info.i2][sub_i1:sub_i2]]
				else:
					data = block[sub_i1:sub_i2]
				#print function_per_block, data
				result = function_per_block(data)
				#print "result", result
				#assert result is not None
				if subresults_per_thread[thread_index] is None:
					subresults_per_thread[thread_index] = result
				else:
					subresults_per_thread[thread_index] = reduce(function_merge, [result, subresults_per_thread[thread_index]])

			subresults_per_thread = [None] * pool.nthreads
			pool.run_blocks(subblock, info.size)
			wrapper.N_done += len(block)
			if feedback:
				cancel = feedback(wrapper.N_done*100./N_total)
				if cancel:
					raise Exception, "cancelled"
			block_result = reduce(function_merge, subresults_per_thread)
			if subresults[index] is None:
				subresults[index] = block_result
			else:
				subresults[index] = reduce(function_merge, [block_result, subresults[index]])
		for index in range(len(expressions)):
			self.addJob(0, functools.partial(calculate, index=index), dataset, expressions[index])
		self.execute()
		return np.array([function_final(scalar) for scalar in subresults])


	def calculate_mean(self, dataset, use_mask, expressions, feedback=None):
		assert len(self.jobs) == 0, "leftover jobs exist"
		return self.add_scalar_jobs(dataset, np.sum, lambda a,b: a + b, lambda x: x/len(dataset), use_mask, expressions, feedback=feedback)

	def find_min_max(self, dataset, expressions, use_mask=False, feedback=None):
		assert len(self.jobs) == 0, "leftover jobs exist"
		pool = multithreading.ThreadPool()
		minima = [None] * len(expressions)
		maxima = [None] * len(expressions)
		#ranges = []
		#range_per_thread = [None] * pool.nthreads
		N_total = len(expressions) * len(dataset)
		class Wrapper(object):
			pass
		wrapper = Wrapper()
		wrapper.N_done = 0
		try:
			t0 = time.time()
			def calculate_range(info, block, index):
				minima_per_thread = [None] * pool.nthreads
				maxima_per_thread = [None] * pool.nthreads
				def subblock(thread_index, sub_i1, sub_i2):
					if use_mask:
						data = block[sub_i1:sub_i2][dataset.mask[info.i1:info.i2][sub_i1:sub_i2]]
					else:
						data = block[sub_i1:sub_i2]
					if len(data):
						result = vaex.vaexfast.find_nan_min_max(data)
						mi, ma = result
						#if sub_i1 == 0:
						minima_per_thread[thread_index] = mi if minima_per_thread[thread_index] is None else min(mi, minima_per_thread[thread_index])
						maxima_per_thread[thread_index] = ma if maxima_per_thread[thread_index] is None else max(ma, maxima_per_thread[thread_index])
				if info.error:
					#self.message(info.error_text, index=-1)
					raise Exception, info.error_text
				pool.run_blocks(subblock, info.size)
				#if info.first:
				#	minima[index] = min(minima_per_thread)
				#	maxima[index] = max(maxima_per_thread)
				#else:
				#	minima[index] = min(min(minima_per_thread), minima[index])
				#	maxima[index] = max(max(maxima_per_thread), maxima[index])
				#if info.last:
				#	self.message("min/max[%d] %.2fs" % (axisIndex, time.time() - t0), index=50+axisIndex)
				mins = [k for k in minima_per_thread if k is not None]
				if mins:
					mi = min(mins)
					minima[index] = mi if minima[index] is None else min(mi, minima[index])
				maxs = [k for k in maxima_per_thread if k is not None]
				if maxs:
					ma = max(maxs)
					maxima[index] = ma if maxima[index] is None else max(ma, maxima[index])
				wrapper.N_done += len(block)
				if feedback:
					cancel = feedback(wrapper.N_done*100./N_total)
					if cancel:
						raise Exception, "cancelled"

			for index in range(len(expressions)):
				self.addJob(0, functools.partial(calculate_range, index=index), dataset, expressions[index])
			self.execute()
		finally:
			pool.close()
		return zip(minima, maxima)

	def execute(self):
		self.signal_begin.emit()
		error_text = None
		progress_value = 0
		buffer_size_local = buffer_size #int(buffer_size / len(self.jobs))
		try:
			# keep a local copy to support reentrant calling
			jobs = self.jobs
			order_numbers = self.order_numbers
			self.jobs = []
			self.order_numbers = []

			cancelled = False
			errors = False
			with Timer("init"):
				order_numbers.sort()

				all_expressions_set = set()
				for order, callback, dataset, expressions, variables in jobs:
					for expression in expressions:
						all_expressions_set.add((dataset, expression))
				# for each expresssion keep an array for intermediate storage
				expression_outputs = dict([(key, np.zeros(buffer_size_local, dtype=np.float64)) for key in all_expressions_set])

			class Info(object):
				pass

			# multiple passes, in order
			with Timer("passes"):
				for order in order_numbers:
					if cancelled or errors:
						break
					logger.debug("jobs, order: %r" % order)
					jobs_order = [job for job in jobs if job[0] == order]
					datasets = set([job[2] for job in jobs])
					# TODO: maybe per dataset a seperate variable dict
					variables = {}
					for job in jobs_order:
						variables_job = job[-1]
						for key, value in variables_job.items():
							if key in variables:
								if variables[key] != value:
									raise ValueError, "variable %r cannot have both value %r and %r" % (key, value, variables[key])
							variables[key] = value
					logger.debug("variables: %r" % (variables,))
					# group per dataset
					for dataset in datasets:
						if dataset.current_slice is None:
							index_start = 0
							index_stop = dataset._length
						else:
							index_start, index_stop = dataset.current_slice

						dataset_length = index_stop - index_start
						logger.debug("dataset: %r" % dataset.name)
						jobs_dataset = [job for job in jobs_order if job[2] == dataset]
						# keep a set of expressions, duplicate expressions will only be calculated once
						expressions_dataset = set()
						expressions_translated = dict()
						for order, callback, dataset, expressions, _variables in jobs_dataset:
							for expression in expressions:
								expressions_dataset.add((dataset, expression))
								try:
									#print "vcolumns:", dataset.virtual_columns
									expr_noslice, slice_vars = expr.translate(expression, dataset.virtual_columns)
								except:
									logger.error("translating expression: %r" % (expression,))
									expressions_translated[(dataset, expression)] = (expression, {}) # just pass expression, code below will handle errors
								else:
									expressions_translated[(dataset, expression)] = (expr_noslice, slice_vars)

						logger.debug("expressions: %r" % expressions_dataset)
						# TODO: implement fractions/slices
						n_blocks = int(math.ceil(dataset_length *1.0 / buffer_size_local))
						logger.debug("blocks: %r %r" % (n_blocks, dataset_length *1.0 / buffer_size_local))
						# execute blocks for all expressions, better for Lx cache
						t0 = time.time()
						for block_index in range(n_blocks):
							if cancelled or errors:
								break
							i1 = block_index * buffer_size_local
							i2 = (block_index +1) * buffer_size_local
							last = False
							if i2 >= dataset_length: # round off the sizes
								i2 = dataset_length
								last = True
								logger.debug("last block")
								#for i in range(len(outputs)):
								#	outputs[i] = outputs[i][:i2-i1]
							# local dicts has slices (not copies) of the whole dataset
							logger.debug("block: %r to %r" % (i1, i2))
							local_dict = dict()
							# dataset scope, there will be evaluated in order
							for key, value in dataset.variables.items():
								try:
									local_dict[key] = eval(dataset.variables[key], np.__dict__, local_dict)
								except:
									local_dict[key] = None
							#print "local vars", local_dict
							local_dict.update(variables) # window scope
							for key, value in dataset.columns.items():
								local_dict[key] = value[i1:i2]
							for key, value in dataset.rank1s.items():
								local_dict[key] = value[:,i1:i2]
							for dataset, expression in expressions_dataset:
								if cancelled or errors:
									break
								if expression is None:
									expr_noslice, slice_vars = None, {}
								else:
									expr_noslice, slice_vars = expressions_translated[(dataset, expression)] #expr.translate(expression)
									logger.debug("replacing %r with %r" % (expression, expr_noslice))
									for var, sliceobj in slice_vars.items():
										logger.debug("adding slice %r as var %r (%r:%r)" % (sliceobj, var, sliceobj.var.name, sliceobj.args))
										array = local_dict[sliceobj.var.name]
										#print local_dict.keys()
										slice_evaluated = eval(repr(sliceobj.args), {}, local_dict)
										logger.debug("slicing array of shape %r  with slice %r" % (array.shape, slice_evaluated))
										sliced_array = array[slice_evaluated]
										logger.debug("slice is of type %r and shape %r" % (sliced_array.dtype, sliced_array.shape))
										local_dict[var] = sliced_array[i1:i2]
							info = Info()
							info.index = block_index
							info.size = i2-i1
							info.total_size = dataset_length
							info.length = n_blocks
							info.first = block_index == 0
							info.last = last
							info.i1 = i1
							info.i2 = i2
							info.percentage = i2 * 100. / info.total_size
							info.slice = slice(i1, i2)
							info.error = False
							info.error_text = ""
							info.time_start = t0
							results = {}
							# check for 'snapshots'/sequence array, and get the proper index automatically
							for name, var in local_dict.items():
								if hasattr(var, "shape"):
									#print name, var.shape
									if len(var.shape) == 2:
										local_dict[name] = var[dataset.selected_serie_index]
										#print " to", name, local_dict[name].shape
								else:
									#print name, var
									pass
							# put to
							with Timer("evaluation"):
								for dataset, expression in expressions_dataset:
									logger.debug("expression: %r" % (expression,))
									#expr_noslice, slice_vars = expr.translate(expression)
									expr_noslice, slice_vars = expressions_translated[(dataset, expression)] #
									logger.debug("translated expression: %r" % (expr_noslice,))
									#print "native", dataset.columns[expression].dtype, dataset.columns[expression].dtype.type==np.float64, dataset.columns[expression].dtype.byteorder, native_code, dataset.columns[expression].dtype.byteorder==native_code, dataset.columns[expression].strides[0]
									if expr_noslice is None:
										results[expression] = None
									#elif expression in dataset.column_names and dataset.columns[expression].dtype==np.float64:
									elif expression in dataset.column_names  \
											and dataset.columns[expression].dtype.type==np.float64 \
											and dataset.columns[expression].strides[0] == 8 \
											and expression not in dataset.virtual_columns:
											#and False:
											#and dataset.columns[expression].dtype.byteorder in [native_code, "="] \
										logger.debug("avoided expression, simply a column name with float64")
										#print "fast"
										##yield self.columns[expression][i1:i2], info
										results[expression] = dataset.columns[expression][i1:i2]
									else:
										# same as above, but -i1, since the array stars at zero
										output = expression_outputs[(dataset, expression)][i1-i1:i2-i1]
										try:
											ex = expr_noslice
											if not isinstance(ex, str):
												ex = repr(expr_noslice)
											#print ex, repr(expr_noslice), expr_noslice, local_dict, len(output)
											ne.evaluate(ex, local_dict=local_dict, out=output, casting="unsafe")
										except Exception, e:
											info.error = True
											info.error_text = repr(e) #.message
											error_text = info.error_text
											print "error_text", error_text
											errors = True
											logger.exception("error in expression: %s" % expression)
											break


										results[expression] = output[0:i2-i1]
							# for this order and dataset all values are calculated, now call the callback
							for key, value in results.items():
								if value is None:
									logger.debug("output[%r]: None" % (key,))
								else:
									logger.debug("output[%r]: %r, %r" % (key, value.shape, value.dtype))
							with Timer("callbacks"):
								for _order, callback, dataset, expressions, _variables in jobs_dataset:
									if cancelled or errors:
										break
									#logger.debug("callback: %r" % (callback))
									arguments = [info]
									arguments += [results.get(expression) for expression in expressions]
									cancelled = cancelled or callback(*arguments)
									progress_value += info.i2 - info.i1
									cancelled = cancelled or np.any(self.signal_progress.emit(float(progress_value)/self.progress_total))
									assert progress_value <= self.progress_total
									if cancelled or errors:
										break
							if info.error:
								# if we get an error, no need to go through the whole data
								break
			if not cancelled and not errors:
				self.signal_end.emit()
				for callback in self.after_execute:
					try:
						callback()
					except Exception, e:
						logger.exception("error in post processing callback")
						error_text = str(e)
			else:
				self.signal_cancel.emit()
		finally:
			self.progress_total = 0
			self.jobs = []
			self.order_numbers = []
			pass
		return error_text


main_manager = JobsManager()

def add_job(self, order, callback, dataset, *expressions, **variables):
	main_manager.addJob(order, callback, dataset, *expressions, **variables)
