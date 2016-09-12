# -*- coding: utf-8 -*-
import vaex.dataset as dataset
import numpy as np
import unittest
import vaex as vx
import vaex.vaexfast

class TestStatisticNd(unittest.TestCase):
	def test_add(self):
		x = np.arange(10, dtype=np.float64)
		grid = np.zeros((10,2), dtype=np.float64)
		w = x * 1
		w[2] = np.nan
		#grid[...,0] = np.inf
		#grid[...,1] = -np.inf
		vaex.vaexfast.statisticNd([x], w, grid, [0.], [10.], 0)
		print(grid)

		grid0 = np.zeros((1,), dtype=np.float64)
		vaex.vaexfast.statisticNd([], w, grid0, [], [], 0)
		print(grid0)

	def test_2(self):
		x = np.arange(10, dtype=np.float64) + 10
		grid = np.zeros((2), dtype=np.float64)
		grid[...,0] = np.inf
		grid[...,1] = -np.inf
		w = x * 1
		w[2] = np.nan
		print(np.nansum(w))
		#grid[...,0] = np.inf
		#grid[...,1] = -np.inf
		vaex.vaexfast.statisticNd([], w, grid, [], [], 2)
		print(grid)


if __name__ == '__main__':
    unittest.main()


