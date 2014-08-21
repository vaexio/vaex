# -*- coding: utf-8 -*-
import h5py
import mmap
import numpy as np
import os
import numexpr as ne
import math
from gavi.utils import Timer
import time
import itertools

import gavifast
from gavi import multithreading
import functools

import gavi.vaex.expressions as expr

import sys
import platform
frozen = getattr(sys, 'frozen', False)
darwin = "darwin" not in platform.system()
if (not frozen) or darwin: # astropy not working with pyinstaller
	fits = __import__("astropy.io.fits").io.fits
	pass

def error(title, msg):
	print "Error", title, msg


buffer_size = 1e7

import gavi.logging
logger = gavi.logging.getLogger("gavi.vaex")


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
		
class JobsManager(object):
	def __init__(self):
		#self.datasets = datasets
		self.jobs = []
		self.order_numbers = []
		self.after_execute = []
		
	def addJob(self, order, callback, dataset, *expressions, **variables):
		args = order, callback, dataset, expressions, variables
		logger.info("job: %r" % (args,))
		if order not in self.order_numbers:
			self.order_numbers.append(order)
		self.jobs.append((order, callback, dataset, [None if e is None or len(e) == 0 else e for e in expressions], variables))
		
	def find_min_max(self, dataset, expressions, feedback=None):
		assert len(self.jobs) == 0, "leftover jobs exist"
		pool = multithreading.ThreadPool()
		minima = [None] * len(expressions)
		maxima = [None] * len(expressions)
		#ranges = []
		minima_per_thread = [None] * pool.nthreads
		maxima_per_thread = [None] * pool.nthreads
		#range_per_thread = [None] * pool.nthreads
		N_total = len(expressions) * len(dataset)
		class Wrapper(object):
			pass
		wrapper = Wrapper()
		wrapper.N_done = 0
		try:
			t0 = time.time()
			def calculate_range(info, block, index):
				def subblock(thread_index, sub_i1, sub_i2):
					print "^" * 60
					result = gavifast.find_nan_min_max(block[sub_i1:sub_i2])
					print "result", result
					mi, ma = result
					print ">>>", mi, ma, thread_index
					#if sub_i1 == 0:
					minima_per_thread[thread_index] = mi
					maxima_per_thread[thread_index] = ma
					#else:
					#	minima_per_thread[thread_index] = min(mi, minima_per_thread[thread_index])
					#	maxima_per_thread[thread_index] = min(ma, maxima_per_thread[thread_index])
					print ">>>>>>>>", minima_per_thread, maxima_per_thread, thread_index
				#self.message("min/max[%d] at %.1f%% (%.2fs)" % (axisIndex, info.percentage, time.time() - info.time_start), index=50+axisIndex )
				#QtCore.QCoreApplication.instance().processEvents()
				if info.error:
					#print "error", info.error_text
					#self.message(info.error_text, index=-1)
					raise Exception, info.error_text
				pool.run_blocks(subblock, info.size)
				print minima_per_thread
				if info.first:
					minima[index] = min(minima_per_thread)
					maxima[index] = max(maxima_per_thread)
				else:
					minima[index] = min(min(minima_per_thread), minima[index])
					maxima[index] = max(max(maxima_per_thread), maxima[index])
				print "min/max for axis", index, minima[index], maxima[index]
				#if info.last:
				#	self.message("min/max[%d] %.2fs" % (axisIndex, time.time() - t0), index=50+axisIndex)
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
		try:
			with Timer("init"):
				self.order_numbers.sort()
				
				all_expressions_set = set()
				for order, callback, dataset, expressions, variables in self.jobs:
					for expression in expressions:
						all_expressions_set.add((dataset, expression))
				# for each expresssion keep an array for intermediate storage
				expression_outputs = dict([(key, np.zeros(buffer_size, dtype=np.float64)) for key in all_expressions_set])
			
			class Info(object):
				pass
			
			# multiple passes, in order
			with Timer("passes"):
				for order in self.order_numbers:
					logger.debug("jobs, order: %r" % order)
					jobs_order = [job for job in self.jobs if job[0] == order]
					datasets = set([job[2] for job in self.jobs])
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
									expr_noslice, slice_vars = expr.translate(expression)
								except:
									logger.error("translating expression: %r" % (expression,))
									expressions_translated[(dataset, expression)] = (expression, {}) # just pass expression, code below will handle errors
								else:
									expressions_translated[(dataset, expression)] = (expr_noslice, slice_vars)
						
						logger.debug("expressions: %r" % expressions_dataset)
						# TODO: implement fractions/slices
						n_blocks = int(math.ceil(dataset_length *1.0 / buffer_size))
						logger.debug("blocks: %r %r" % (n_blocks, dataset_length *1.0 / buffer_size))
						# execute blocks for all expressions, better for Lx cache
						t0 = time.time()
						for block_index in range(n_blocks):
							i1 = block_index * buffer_size
							i2 = (block_index +1) * buffer_size
							last = False
							if i2 >= dataset_length: # round off the sizes
								i2 = dataset_length
								last = True
								logger.debug("last block")
								#for i in range(len(outputs)):
								#	outputs[i] = outputs[i][:i2-i1]
							# local dicts has slices (not copies) of the whole dataset
							logger.debug("block: %r to %r" % (i1, i2))
							local_dict = dict(variables)
							for key, value in dataset.columns.items():
								local_dict[key] = value[i1:i2]
							for key, value in dataset.rank1s.items():
								local_dict[key] = value[:,i1:i2]
							for dataset, expression in expressions_dataset:
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
							# put to 
							with Timer("evaluation"):
								for dataset, expression in expressions_dataset:
									logger.debug("expression: %r" % (expression,))
									#expr_noslice, slice_vars = expr.translate(expression)
									expr_noslice, slice_vars = expressions_translated[(dataset, expression)] #
									logger.debug("translated expression: %r" % (expr_noslice,))
									if expr_noslice is None:
										results[expression] = None
									#elif expression in dataset.column_names and dataset.columns[expression].dtype==np.float64:
									elif expression in dataset.column_names and dataset.columns[expression].dtype==np.float64 and dataset.columns[expression].strides[0] == 8:
										logger.debug("avoided expression, simply a column name with float64")
										#yield self.columns[expression][i1:i2], info
										results[expression] = dataset.columns[expression][i1:i2]
									else:
										# same as above, but -i1, since the array stars at zero
										output = expression_outputs[(dataset, expression)][i1-i1:i2-i1]
										try:
											ne.evaluate(repr(expr_noslice), local_dict=local_dict, out=output, casting="unsafe")
										except SyntaxError, e:
											info.error = True
											info.error_text = e.message
											logger.exception("error in expression: %s" % (expression,))
											break
										except KeyError, e:
											info.error = True
											info.error_text = "Unknown variable: %r" % (e.message, )
											logger.exception("unknown variable %s in expression: %s" % (e.message, expression))
											break
										except TypeError, e:
											info.error = True
											info.error_text = e.message
											logger.exception("error in expression: %s" % expression)
											break
										except Exception, e:
											info.error = True
											info.error_text = e.message
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
									logger.debug("callback: %r" % (callback))
									arguments = [info]
									arguments += [results.get(expression) for expression in expressions]
									callback(*arguments)
							if info.error:
								# if we get an error, no need to go through the whole data
								break
			for callback in self.after_execute:
				try:
					callback()
				except:
					logger.exception("error in post processing callback")
					
		finally:
			self.jobs = []
			self.order_numbers = []


class Link(object):
	def __init__(self, dataset):
		self.dataset = dataset
		self.listeners = []
		
	def sendExpression(self, expression, receiver):
		for listener in self.listeners:
			if listener != receiver:
				listener.onChangeExpression(expression)
		if expression in self.dataset.global_links:
			# merge the listeners of this link to the other link
			logger.debug("merging with link %r" % expression)
			merging_link = self.dataset.global_links[expression]
			for receiver in merging_link.listeners:
				self.listeners.append(receiver)
		else:
			# add new mapping
			logger.debug("renamed link %r" % expression)
			self.dataset.global_links[expression] = self
		# remove old mapping
		for key, link in self.dataset.global_links.items():
			logger.debug("link[%r] = %r" % (key, link))
			if (link == self) and key != expression: # remove dangling links
				logger.debug("removing link %r" % key)
				del self.dataset.global_links[key]
				
	def sendRanges(self, range_, receiver):
		for listener in self.listeners:
			if listener != receiver:
				listener.onChangeRange(range_)

	def sendRangesShow(self, range_, receiver):
		for listener in self.listeners:
			if listener != receiver:
				listener.onChangeRangeShow(range_)

	#
	@staticmethod
	def sendCompute(links, receivers):
		listener_set = set(list(itertools.chain.from_iterable([link.listeners for link in links])))

		for listener in listener_set:
			if listener not in receivers:
				listener.onCompute()
				
	def sendPlot(self, receiver):
		for listener in self.listeners:
			if listener != receiver:
				listener.onPlot()
				
				
		
		
	
class MemoryMapped(object):
	def link(self, expression, listener):
		if expression not in self.global_links:
			self.global_links[expression] = Link(self)
			logger.debug("creating link object: %r" % self.global_links[expression])
		else:
			
			logger.debug("reusing link object: %r" % self.global_links[expression])

			
		link = self.global_links[expression]
		link.listeners.append(listener)
		return link
	
	def unlink(self, link, receiver):
		link.listeners.remove(receiver)
		
	def full_length(self):
		return self._length
		
	def __len__(self):
		return self._fraction_length
		
		
	# nommap is a hack to get in memory datasets working
	def __init__(self, filename, write=False, nommap=False):
		self.filename = filename
		self.write = write
		self.name = os.path.splitext(os.path.basename(self.filename))[0]
		if not nommap:
			self.file = file(self.filename, "r+" if write else "r")
			self.fileno = self.file.fileno()
			self.mapping = mmap.mmap(self.fileno, 0, prot=mmap.PROT_READ | 0 if not write else mmap.PROT_WRITE )
			self.file_map = {filename: self.file}
			self.fileno_map = {filename: self.fileno}
			self.mapping_map = {filename: self.mapping}
		else:
			self.file_map = {}
			self.fileno_map = {}
			self.mapping_map = {}
		self._length = None
		self._fraction_length = None
		self.nColumns = 0
		self.columns = {}
		self.column_names = []
		self.current_slice = None
		self.fraction = 1.0
		self.rank1s = {}
		self.rank1names = []
		self.selected_row_index = None
		self.selected_serie_index = None
		self.row_selection_listeners = []
		self.serie_index_selection_listeners = []
		self.mask_listeners = []
		self.all_columns = {}
		self.all_column_names = []
		self.mask = None
		self.global_links = {}
		self.offsets = {}
		self.strides = {}
		self.filenames = {}
		self.dtypes = {}
		
	def evaluate(self, callback, *expressions, **variables):
		jobManager = JobsManager()
		logger.debug("evalulate: %r %r" % (expressions,variables))
		jobManager.addJob(0, callback, self, *expressions, **variables)
		jobManager.execute()
		return
		
		
	def old(self):
		class Info(object):
			pass
		outputs = [np.zeros(buffer_size, dtype=np.float64) for _ in expressions]
		n_blocks = int(math.ceil(self._length *1.0 / buffer_size))
		print "blocks", n_blocks, self._length *1.0 / buffer_size
		# execute blocks for all expressions, better for Lx cache
		for block_index in range(n_blocks):
			i1 = block_index * buffer_size
			i2 = (block_index +1) * buffer_size
			if i2 >= self._length: # round off the sizes
				i2 = self._length
				for i in range(len(outputs)):
					outputs[i] = outputs[i][:i2-i1]
			# local dicts has slices (not copies) of the whole dataset
			local_dict = {}
			for key, value in self.columns.items():
				local_dict[key] = value[i1:i2]
			info = Info()
			info.index = block_index
			info.size = i2-i1
			info.length = n_blocks
			info.first = block_index == 0
			info.i1 = i1
			info.i2 = i2
			info.slice = slice(i1, i2)
			results = []
			for output, expression in zip(outputs, expressions):
				if expression in self.column_names and self.columns[expression].dtype == np.float64:
					print "avoided"
					#yield self.columns[expression][i1:i2], info
					results.append(self.columns[expression][i1:i2])
				else:
					ne.evaluate(expression, local_dict=local_dict, out=output, casting="unsafe")
					results.append(output)
			print results, info
			yield tuple(results), info
		
		
	def addFile(self, filename, write=False):
		self.file_map[filename] = file(filename, "r+" if write else "r")
		self.fileno_map[filename] = self.file_map[filename].fileno()
		self.mapping_map[filename] = mmap.mmap(self.fileno_map[filename], 0, prot=mmap.PROT_READ | 0 if not write else mmap.PROT_WRITE )


	def selectMask(self, mask):
		self.mask = mask
		for mask_listener in self.mask_listeners:
			mask_listener(mask)
		
		
	def selectRow(self, index):
		self.selected_row_index = index
		for row_selection_listener in self.row_selection_listeners:
			row_selection_listener(index)
		
	def selectSerieIndex(self, serie_index):
		self.selected_serie_index = serie_index
		for serie_index_selection_listener in self.serie_index_selection_listeners:
			serie_index_selection_listener(serie_index)
			
	def matches_url(self, url):
		filename = url
		print url, self.filename
		if filename.startswith("file:/"):
			filename = filename[5:]
		similar = os.path.splitext(os.path.abspath(self.filename))[0] == os.path.splitext(filename)[0]
		logger.info("matching urls: %r == %r == %r" % (os.path.splitext(self.filename)[0], os.path.splitext(filename)[0], similar) )
		return similar
			
		
		
	def close(self):
		self.file.close()
		self.mapping.close()
		
	def setFraction(self, fraction):
		self.fraction = fraction
		self.current_slice = (0, int(self._length * fraction))
		self._fraction_length = self.current_slice[1]
		self.selectMask(None)
		# TODO: if row in slice, we don't have to remove it
		self.selectRow(None)
		
	def addMemoryColumn(self, name, column):
		length = len(column)
		if self.current_slice is None:
			self.current_slice = (0, length)
			self.fraction = 1.
			self._fraction_length = length
		self._length = length
		#print self.mapping, dtype, length if stride is None else length * stride, offset
		self.columns[name] = column
		self.column_names.append(name)
		self.all_columns[name] = column
		self.all_column_names.append(name)
		#self.column_names.sort()
		self.nColumns += 1
		self.nRows = self._length
		
	def addColumn(self, name, offset=None, length=None, dtype=np.float64, stride=1, filename=None, array=None):
		if filename is None:
			filename = self.filename
		mapping = self.mapping_map[filename]
			
		if array is not None:
			length = len(array)
			
		if self._length is not None and length != self._length:
			error("inconsistent length", "length of column %s is %d, while %d was expected" % (name, length, self._length))
		else:
			if self.current_slice is None:
				self.current_slice = (0, length)
				self.fraction = 1.
				self._fraction_length = length
			self._length = length
			#print self.mapping, dtype, length if stride is None else length * stride, offset
			if array is not None:
				length = len(array)
				mmapped_array = array
				stride = None
				offset = None
				dtype = array.dtype
			else:
				mmapped_array = np.frombuffer(mapping, dtype=dtype, count=length if stride is None else length * stride, offset=offset)
				if stride:
					#import pdb
					#pdb.set_trace()
					mmapped_array = mmapped_array[::stride]
			self.columns[name] = mmapped_array
			self.column_names.append(name)
			self.all_columns[name] = mmapped_array
			self.all_column_names.append(name)
			#self.column_names.sort()
			self.nColumns += 1
			self.nRows = self._length
			self.offsets[name] = offset
			self.strides[name] = stride
			self.filenames[name] = os.path.abspath(filename)
			self.dtypes[name] = dtype
			
	def addRank1(self, name, offset, length, length1, dtype=np.float64, stride=1, stride1=1, filename=None):
		if filename is None:
			filename = self.filename
		mapping = self.mapping_map[filename]
		if self._length is not None and length != self._length:
			error("inconsistent length", "length of column %s is %d, while %d was expected" % (name, length, self._length))
		else:
			if self.current_slice is None:
				self.current_slice = (0, length)
				self.fraction = 1.
			self._length = length
			#print self.mapping, dtype, length if stride is None else length * stride, offset
			rawlength = length * length1
			rawlength *= stride
			rawlength *= stride1
			#print rawlength, offset
			#print rawlength * 8, offset, self.mapping.size()
			#import pdb
			#pdb.set_trace()
			mmapped_array = np.frombuffer(mapping, dtype=dtype, count=rawlength, offset=offset)
			mmapped_array = mmapped_array.reshape((length1*stride1, length*stride))
			mmapped_array = mmapped_array[::stride1,::stride]
			self.rank1s[name] = mmapped_array
			self.rank1names.append(name)
			self.all_columns[name] = mmapped_array
			self.all_column_names.append(name)
			
			#self.column_names.sort()
			#self.nColumns += 1
			#self.nRows = self._length
			
import struct
class HansMemoryMapped(MemoryMapped):
	
	def __init__(self, filename, filename_extra=None):
		super(HansMemoryMapped, self).__init__(filename)
		self.pageSize, \
		self.formatSize, \
		self.numberParticles, \
		self.numberTimes, \
		self.numberParameters, \
		self.numberCompute, \
		self.dataOffset, \
		self.dataHeaderSize = struct.unpack("Q"*8, self.mapping[:8*8])
		print "offset", self.dataOffset, self.formatSize, self.numberParticles, self.numberTimes
		zerooffset = offset = self.dataOffset
		length = self.numberParticles+1
		stride = self.formatSize/8 # stride in units of the size of the element (float64)
		
		# TODO: ask Hans for the self.numberTimes-2
		lastoffset = offset + (self.numberParticles+1)*(self.numberTimes-2)*self.formatSize
		t_index = 3
		names = "x y z vx vy vz".split()
		midoffset = offset + (self.numberParticles+1)*self.formatSize*t_index
		names = "x y z vx vy vz".split()

		for i, name in enumerate(names):
			self.addColumn(name+"_0", offset+8*i, length, dtype=np.float64, stride=stride)
			
		for i, name in enumerate(names):
			self.addColumn(name+"_last", lastoffset+8*i, length, dtype=np.float64, stride=stride)
		
		#for i, name in enumerate(names):
		#	self.addColumn(name+"_mid", midoffset+8*i, length, dtype=np.float64, stride=stride)
		

		names = "x y z vx vy vz".split()
		#import pdb
		#pdb.set_trace()
		if 1:
			stride = self.formatSize/8 
			#stride1 = self.numberTimes #*self.formatSize/8 
			#print (self.numberParticles+1)
			#print stride, stride1
			for i, name in enumerate(names):
				#print name, offset
				# TODO: ask Hans for the self.numberTimes-1
				self.addRank1(name, offset+8*i, length=self.numberParticles+1, length1=self.numberTimes-1, dtype=np.float64, stride=stride, stride1=1)
				
		if filename_extra is not None:
			self.addFile(filename_extra)
			mapping = self.mapping_map[filename_extra]
			names = "J_r J_theta J_phi Theta_r Theta_theta Theta_phi Omega_r Omega_theta Omega_phi r_apo r_peri".split()
			offset = 0
			stride = 11
			#import pdb
			#pdb.set_trace()
			for i, name in enumerate(names):
				#print name, offset
				# TODO: ask Hans for the self.numberTimes-1
				self.addRank1(name, offset+8*i, length=self.numberParticles+1, length1=self.numberTimes-1, dtype=np.float64, stride=stride, stride1=1, filename=filename_extra)
				#print "min/max", np.min(self.rank1s[name]), np.max(self.rank1s[name]), offset+8*i, self.rank1s[name][0][0]
				#for i, name in enumerate(names):
				
				self.addColumn(name+"_0", offset+8*i, length, dtype=np.float64, stride=stride, filename=filename_extra)
				self.addColumn(name+"_last", offset+8*i + (self.numberParticles+1)*(self.numberTimes-2)*11*8, length, dtype=np.float64, stride=stride, filename=filename_extra)
			
			

		#for i, name in enumerate(names):
		#	self.addColumn(name+"_last", offset+8*i + (self.formatSize*(self.numberTimes-1)), length, dtype=np.float64, stride=stride)
		#for i, name in enumerate(names):
		#	self.addRank1(name, offset+8*i, (length, numberTimes), dtype=np.float64, stride=stride)
		
		
		
		
		#uint64 = np.frombuffer(self.mapping, dtype=dtype, count=length if stride is None else length * stride, offset=offset)
		

if __name__ == "__main__":
	path = "/Users/users/buist/research/2014 Simulation Data/12Orbits/Sigma/Orbitorb1.ac0.10000.100.5.orb.omega2"
	path = "/net/pannekoek/data/users/buist/Research/2014 Simulation Data/12Orbits/Integration/Orbitorb9.ac8.10000.100.5.orb.bin"

	hmm = HansMemoryMapped(path)


class FitsBinTable(MemoryMapped):
	def __init__(self, filename, write=False):
		super(FitsBinTable, self).__init__(filename, write=write)
		fitsfile = fits.open(filename)
		for table in fitsfile:
			if isinstance(table, fits.BinTableHDU):
				logger.debug("adding table: %r" % table)
				for column in table.columns:
					column.array[:] # 2nd time it will be a real np array
					self.addColumn(column.name, array=column.array[:])
		#BinTableHDU
		
		
class Hdf5MemoryMapped(MemoryMapped):
	def __init__(self, filename, write=False):
		super(Hdf5MemoryMapped, self).__init__(filename, write=write)
		self.h5file = h5py.File(self.filename, "r+" if write else "r")
		self.load()
		
	def load(self):
		if "data" in self.h5file:
			self.load_columns(self.h5file["/data"])
		if "columns" in self.h5file:
			self.load_columns(self.h5file["/columns"])
			
	def load_columns(self, h5data):
		print h5data
		# make sure x y x etc are first
		first = "x y z vx vy vz".split()
		finished = set()
		for column_name in first + list(h5data):
			if column_name in h5data and column_name not in finished:
				print type(column_name)
				column = h5data[column_name]
				if hasattr(column, "dtype"):
					print column
					offset = column.id.get_offset() 
					shape = column.shape
					if len(shape) == 1:
						self.addColumn(column_name, offset, len(column), dtype=column.dtype)
					else:
						print "rank 1 array", shape
						self.addRank1(column_name, offset, shape[1], length1=shape[0], dtype=column.dtype, stride=1, stride1=1)
						#self.addColumn(column_name+"_0", offset, shape[1], dtype=column.dtype)
						print column.dtype.itemsize
						self.addColumn(column_name+"_last", offset+shape[0]*column.dtype.itemsize, shape[1], dtype=column.dtype)
						#self.addRank1(name, offset+8*i, length=self.numberParticles+1, length1=self.numberTimes-1, dtype=np.float64, stride=stride, stride1=1, filename=filename_extra)
			finished.add(column_name)
			
	def close(self):
		super(Hdf5MemoryMapped, self).close()
		self.h5file.close()
		
	def __expose_array(self, hdf5path, column_name):
		array = self.h5file[hdf5path]
		array[0] = array[0] # without this, get_offset returns None, probably the array isn't really created
		offset = array.id.get_offset() 
		self.remap()
		self.addColumn(column_name, offset, len(array), dtype=array.dtype)
		
	def __add_column(self, column_name, dtype=np.float64, length=None):
		array = self.h5data.create_dataset(column_name, shape=(self._length if length is None else length,), dtype=dtype)
		array[0] = array[0] # see above
		offset = array.id.get_offset() 
		self.h5file.flush()
		self.remap()
		self.addColumn(column_name, offset, len(array), dtype=array.dtype)

class AmuseHdf5MemoryMapped(Hdf5MemoryMapped):
	def __init__(self, filename, write=False):
		super(AmuseHdf5MemoryMapped, self).__init__(filename, write=write)
		
	def load(self):
		particles = self.h5file["/particles"]
		print "amuse", particles
		for group_name in particles:
			#print group
			#import pdb
			#pdb.set_trace()
			group = particles[group_name]
			self.load_columns(group["attributes"])
			
			column_name = "keys"
			column = group[column_name]
			offset = column.id.get_offset() 
			self.addColumn(column_name, offset, len(column), dtype=column.dtype)

class Hdf5MemoryMappedGadget(MemoryMapped):
	def __init__(self, filename, particleName, particleType):
		super(Hdf5MemoryMappedGadget, self).__init__(filename)
		self.particleType = particleType
		self.particleName = particleName
		self.name = self.name + "-" + self.particleName
		h5file = h5py.File(self.filename, 'r')
		#for i in range(1,4):
		key = "/PartType%d" % self.particleType
		if key not in h5file:
			raise KeyError, "%s does not exist" % key
		particles = h5file[key]
		for name in particles.keys():
			print name
			#name = "/PartType%d/Coordinates" % i
			data = particles[name]
			if isinstance(data, h5py.highlevel.Dataset): #array.shape
				array = data
				print array.shape, array.dtype
				shape = array.shape
				if len(shape) == 1:
					offset = array.id.get_offset() 
					self.addColumn(name, offset, data.shape[0], dtype=data.dtype)
				else:
					if name == "Coordinates":
						offset = data.id.get_offset() 
						self.addColumn("x", offset, data.shape[0], dtype=data.dtype, stride=3)
						self.addColumn("y", offset+4, data.shape[0], dtype=data.dtype, stride=3)
						self.addColumn("z", offset+8, data.shape[0], dtype=data.dtype, stride=3)
					elif name == "Velocity":
						offset = data.id.get_offset() 
						self.addColumn("vx", offset, data.shape[0], dtype=data.dtype, stride=3)
						self.addColumn("vy", offset+4, data.shape[0], dtype=data.dtype, stride=3)
						self.addColumn("vz", offset+8, data.shape[0], dtype=data.dtype, stride=3)
					else:
						print "unsupported column: %r of shape %r" % (name, array.shape)
		

class MemoryMappedGadget(MemoryMapped):
	def __init__(self, filename):
		super(MemoryMappedGadget, self).__init__(filename)
		#h5file = h5py.File(self.filename)
		import gavi.file.gadget
		length, posoffset, veloffset, header = gavi.file.gadget.getinfo(filename)
		print length, posoffset, posoffset
		print posoffset, hex(posoffset)
		self.addColumn("x", posoffset, length, dtype=np.float32, stride=3)
		self.addColumn("y", posoffset+4, length, dtype=np.float32, stride=3)
		self.addColumn("z", posoffset+8, length, dtype=np.float32, stride=3)
		
		self.addColumn("vx", veloffset, length, dtype=np.float32, stride=3)
		self.addColumn("vy", veloffset+4, length, dtype=np.float32, stride=3)
		self.addColumn("vz", veloffset+8, length, dtype=np.float32, stride=3)
		