# -*- coding: utf-8 -*-
import vaex.dataset as dataset
import numpy as np
import unittest
import vaex as vx
import vaex.vaexfast

class TestStatisticNd(unittest.TestCase):
	def test_1(self):
		x = np.arange(10, dtype=np.float64)
		grid = np.zeros((10,2), dtype=np.float64)
		w = x * 1
		w[2] = np.nan
		#grid[...,0] = np.inf
		#grid[...,1] = -np.inf
		vaex.vaexfast.statisticNd([x], w, grid, [0.], [10.], 3)
		print grid

	def t_est_2(self):
		x = np.arange(10, dtype=np.float64) + 10
		grid = np.zeros((2), dtype=np.float64)
		grid[...,0] = np.inf
		grid[...,1] = -np.inf
		w = x * 1
		w[2] = np.nan
		print np.nansum(w)
		#grid[...,0] = np.inf
		#grid[...,1] = -np.inf
		vaex.vaexfast.statisticNd([], w, grid, [], [], 2)
		print grid


if __name__ == '__main__':
    unittest.main()


