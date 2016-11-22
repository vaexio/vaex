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


class TestGridInterpolate(unittest.TestCase):
	def test_interpolate(self):
		x = np.array([[0., 1.]])
		y = np.array([2.])
		vaex.vaexfast.grid_interpolate(x, y, 0.5)
		self.assertEqual(y[0], 0.5)

		x = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 1,1,1,1,1]])

		vaex.vaexfast.grid_interpolate(x, y, 0.5)
		self.assertEqual(y[0], 1./4)

		x = np.array([[0, 0.5, 0.75, 1]])

		vaex.vaexfast.grid_interpolate(x, y, 0.5)
		self.assertEqual(y[0], 1./3)

		vaex.vaexfast.grid_interpolate(x, y, 0.75)
		self.assertEqual(y[0], 2./3)

		vaex.vaexfast.grid_interpolate(x, y, 0.5+0.25/2)
		self.assertEqual(y[0], 0.5)


		x = np.array([[0, 0.5, 0.75, 1], [0.5, 0.5, 0.75, 1], [0, 0.5, 1.0, 1]])
		y = np.array([2., 2., 2.])
		vaex.vaexfast.grid_interpolate(x, y, 0.5)
		np.testing.assert_array_almost_equal(y, np.array([1/3., 1./3/2, 1./3.]) )


		x = np.array([[0, 0.5, 0.75, 1], [0.5, 0.5, 0.75, 1], [0, 0.5, 1.0, 1]])
		vaex.vaexfast.grid_interpolate(x, y, 0.75)
		np.testing.assert_array_almost_equal(y, np.array([2 / 3., 2./3, 1./2]))


		print("#######")
		x = np.array([[0, 0.5, 0.75, 1], [0.5, 0.5, 0.75, 1], [0, 0.5, 1.0, 1]])
		vaex.vaexfast.grid_interpolate(x, y, 1.0)
		np.testing.assert_array_almost_equal(y, np.array([1., 1, 5./6]))

if __name__ == '__main__':
    unittest.main()


