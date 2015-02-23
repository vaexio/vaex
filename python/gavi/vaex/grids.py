__author__ = 'maartenbreddels'
import numpy as np
import gavifast
from gavi.utils import filesize_format
import gavi.logging
total_bytes = 0

logger = gavi.logging.getLogger("gavi.vaex.grids")

def add_mem(bytes, *info):
	global total_bytes
	total_bytes += bytes
	added = filesize_format(bytes)
	total = filesize_format(total_bytes)
	print "MEMORY USAGE: added %s, total %s (%r)" % (added, total, info)

class Grid(object):
	grid_cache = {}

	def __init__(self, grids, max_size, dimensions,weight_expression, dtype=np.float64):
		self.grids = grids
		self.max_size = max_size
		self.dimensions = dimensions
		self.data = None# create on demand using get_grid
		self.data_selection = None #np.zeros((max_size,) * dimensions, dtype=dtype)
		# experimented without using cache, should in theory be faster, but in practive doesn't matter much for 2d
		# for 3d the memory usage is too high
		self.use_cache = True
		self.weight_expression = weight_expression
		self.dtype = dtype

	def get_data(self, size, use_selection):
		data = self.data_selection if use_selection else self.data
		if size == self.max_size:
			return data
		else:
			return gavifast.resize(data, size)

	def check_grid(self):
		compute_selection = self.grids.dataset.mask is not None
		shape = (self.max_size,) * self.dimensions
		if self.data is None:
			self.data = np.zeros(shape, dtype=self.dtype)
			add_mem(self.data.nbytes, self.weight_expression, "normal grid")
			if not self.use_cache:
				self.data_per_thread = np.zeros((self.grids.threadpool.nthreads-1, )  + (self.max_size,) * self.dimensions, dtype=self.dtype)
				add_mem(self.data_per_thread.nbytes, self.weight_expression, "normal grid for threads")
		if compute_selection:
			if self.data_selection is None:
				self.data_selection = np.zeros(shape, dtype=self.dtype)
				add_mem(self.data_selection.nbytes, self.weight_expression, "selection grid")
				if not self.use_cache:
					self.data_selection_per_thread = np.zeros((self.grids.threadpool.nthreads-1, )  + (self.max_size,) * self.dimensions, dtype=self.dtype)
					add_mem(self.data_selection_per_thread.nbytes, self.weight_expression, "selection grid for threads")


	@classmethod
	def get_temp_grid(cls, shape, selection, dtype=np.float64):
		key = shape + (selection, dtype)
		if key not in cls.grid_cache:
			grid = np.zeros(shape, dtype=dtype)
			add_mem(grid.nbytes, "per thread grid", key)
			cls.grid_cache[key] = grid
			return grid
		else:
			return cls.grid_cache[key]



	def bin_block(self, info, *blocks):
		#assert len(blocks) == self.dimensions or len(blocks) == self.dimensions + 1
		#print "block", self.weight_expression, info.i1, info.i2

		compute_selection = self.grids.dataset.mask is not None

		self.check_grid()
		#self.data = self.get_grid(False)
		#self.data_selection = self.get_grid(True) if compute_selection else None

		if info.first:
			self.data.reshape(-1)[:] = 0.
			if compute_selection:
				self.data_selection.reshape(-1)[:] = 0

		data_selection = None
		data_selection_per_thread = None
		# get two unique grids
		# one thread can add it's binning data to the final grid, the rest
		# has to do it in their own 'thread local' grid
		shape = (self.grids.threadpool.nthreads-1, )  + (self.max_size,) * self.dimensions
		if self.use_cache: # if we use the cache we need to clear up the array
			data_per_thread = self.get_temp_grid(shape, None, self.dtype)
			if compute_selection:
				data_selection_per_thread = self.get_temp_grid(shape, "selection", self.dtype)
		else:
			data_per_thread = self.data_per_thread
			if compute_selection:
				data_selection_per_thread = self.data_selection_per_thread

		# private grids only have to be cleared the first time, cached always
		if self.use_cache or info.first:
			data_per_thread.reshape(-1)[:] = 0.
			if compute_selection:
				data_selection_per_thread.reshape(-1)[:] = 0.

		#data_per_thread.reshape(-1)[:] = 0.
		#else:
		#	data_selection_per_thread = None

		ranges_flat = []
		for minimum, maximum in self.grids.ranges:
			ranges_flat.append(minimum)
			if minimum == maximum:
				maximum += 1
			ranges_flat.append(maximum)
		if len(blocks) == self.dimensions + 1:
			blocks, block_weight = blocks[:-1], blocks[-1]
		else:
			block_weight = None
		def bin_subblock(index, sub_i1, sub_i2):
			if index == 0:
				data = self.data
			else:
				data = data_per_thread[index-1]
			#print "\tthread", index, self.weight_expression, sub_i1, sub_i2
			if block_weight is None:
				subblock_weight = None
			else:
				subblock_weight = block_weight[sub_i1:sub_i2]
			subblocks = [block[sub_i1:sub_i2] for block in blocks]
			if self.dimensions == 1:
				gavifast.histogram1d(subblocks[0], subblock_weight, data, *ranges_flat)
			elif self.dimensions == 2:
				gavifast.histogram2d(subblocks[0], subblocks[1], subblock_weight, data, *ranges_flat)
			elif self.dimensions == 3:
				gavifast.histogram3d(subblocks[0], subblocks[1], subblocks[2], subblock_weight, data, *ranges_flat)
			else:
				raise NotImplementedError("TODO")
			if compute_selection:
				if index == 0:
					data = self.data_selection
				else:
					data = data_selection_per_thread[index-1]
				mask = self.grids.dataset.mask[info.i1:info.i2][sub_i1:sub_i2]
				subblocks = [block[mask] for block in subblocks]
				#subblocks = [block[mask[info.i1+sub_i1:info.i1+sub_i2]] for block in blocks]
				#[info.i1+sub_i1:info.i1+sub_i2]
				if subblock_weight is not None:
					subblock_weight = subblock_weight[mask]
				if self.dimensions == 1:
					gavifast.histogram1d(subblocks[0], subblock_weight, data, *ranges_flat)
				elif self.dimensions == 2:
					gavifast.histogram2d(subblocks[0], subblocks[1], subblock_weight, data, *ranges_flat)
				elif self.dimensions == 3:
					gavifast.histogram3d(subblocks[0], subblocks[1], subblocks[2], subblock_weight, data, *ranges_flat)
				else:
					raise NotImplementedError("TODO")
		self.grids.threadpool.run_blocks(bin_subblock, info.size)

		# the cached grids will be used by other grids, so add it to our 'private' grid
		# if we have our own grids, we only have to sum up in the last block, saves time
		if self.use_cache or info.last:
			for i in range(self.grids.threadpool.nthreads-1):
				np.add(self.data, data_per_thread[i], self.data)
			#self.data += np.sum(data_per_thread, axis=0)
			if compute_selection:
				self.data_selection += np.sum(data_selection_per_thread, axis=0)





class Grids(object):
	def __init__(self, dataset, threadpool, *expressions):
		self.dataset = dataset
		self.threadpool = threadpool
		self.grids = {}
		self.expressions = expressions
		self.dimensions = len(self.expressions)
		self.ranges = [None,] * self.dimensions

	def add_jobs(self, jobsManager):
		#self.jobsManager.addJob(1, functools.partial(self.calculate_visuals, compute_counter=compute_counter), self.dataset, *all_expressions, **self.getVariableDict())
		for name, grid in self.grids.items():
			callback = grid.bin_block
			expressions = list(self.expressions)
			if grid.weight_expression is not None:
				expressions.append(grid.weight_expression)
			if name is "counts" or (grid.weight_expression is not None and len(grid.weight_expression ) > 0):
				logger.debug("JOB: expressions: %r" % (expressions,))
				jobsManager.addJob(1, callback, self.dataset, *expressions)

	def set_expressions(self, expressions):
		self.expressions = list(expressions)

	def define_grid(self, name, size, weight_expression):
		grid = self.grids.get(name)
		if grid is None or grid.max_size < size:
			grid = Grid(self, size, self.dimensions, weight_expression)
		#assert grid.dimension == len(expressions), "existing grid is not of proper dimension"
		assert grid.max_size >= size
		# it's ok if a similar grid exists of a bigger size, we can 'downscale' without losing
		# precision, but cannot upscale
		grid.weight_expression = weight_expression
		self.grids[name] = grid

	def __getitem__(self, name):
		return self.grids[name]
