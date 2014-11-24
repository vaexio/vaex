__author__ = 'maartenbreddels'
import numpy as np
import gavifast

class Grid(object):
	def __init__(self, grids, max_size, dimensions,weight_expression, dtype=np.float64):
		self.grids = grids
		self.max_size = max_size
		self.dimensions = dimensions
		self.data = np.zeros((max_size,) * dimensions, dtype=dtype)
		self.data_per_thread = np.zeros((self.grids.threadpool.nthreads, )  + (max_size,) * dimensions, dtype=dtype)
		self.data_selection = np.zeros((max_size,) * dimensions, dtype=dtype)
		self.data_selection_per_thread = np.zeros((self.grids.threadpool.nthreads, )  + (max_size,) * dimensions, dtype=dtype)
		self.weight_expression = weight_expression

	def get_data(self, size, use_selection):
		data = self.data_selection if use_selection else self.data
		if size == self.max_size:
			return data
		else:
			return gavifast.resize(data, size)

	def bin_block(self, info, *blocks):
		#assert len(blocks) == self.dimensions or len(blocks) == self.dimensions + 1
		compute_selection = self.grids.dataset.mask is not None
		if info.first:
			self.data.reshape(-1)[:] = 0.
			if compute_selection:
				self.data_selection.reshape(-1)[:] = 0
		self.data_per_thread.reshape(-1)[:] = 0.
		self.data_selection_per_thread.reshape(-1)[:] = 0.
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
			if block_weight is None:
				subblock_weight = None
			else:
				subblock_weight = block_weight[sub_i1:sub_i2]
			subblocks = [block[sub_i1:sub_i2] for block in blocks]
			if self.dimensions == 2:
				gavifast.histogram2d(subblocks[0], subblocks[1], subblock_weight, self.data_per_thread[index], *ranges_flat)
			elif self.dimensions == 3:
				gavifast.histogram3d(subblocks[0], subblocks[1], subblocks[2], subblock_weight, self.data_per_thread[index], *ranges_flat)
			else:
				raise NotImplementedError("TODO")
			if compute_selection:
				mask = self.grids.dataset.mask
				subblocks = [block[mask[info.i1+sub_i1:info.i1+sub_i2]] for block in blocks]
				if subblock_weight is not None:
					subblock_weight = subblock_weight[mask[info.i1+sub_i1:info.i1+sub_i2]]
				if self.dimensions == 2:
					gavifast.histogram2d(subblocks[0], subblocks[1], subblock_weight, self.data_selection_per_thread[index], *ranges_flat)
				elif self.dimensions == 3:
					gavifast.histogram3d(subblocks[0], subblocks[1], subblocks[2], subblock_weight, self.data_selection_per_thread[index], *ranges_flat)
				else:
					raise NotImplementedError("TODO")
		self.grids.threadpool.run_blocks(bin_subblock, info.size)
		self.data += np.sum(self.data_per_thread, axis=0)
		self.data_selection += np.sum(self.data_selection_per_thread, axis=0)




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
			print "expressions", expressions
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
