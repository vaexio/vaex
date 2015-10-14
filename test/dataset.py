import vaex.dataset as dataset
import numpy as np
import unittest
import vaex as vx

class TestDataset(unittest.TestCase):
	def setUp(self):
		self.dataset = dataset.DatasetArrays()

		x = np.arange(10)
		y = x ** 2
		self.dataset.add_column("x", x)
		self.dataset.add_column("y", y)
		
		#self.jobsManager = dataset.JobsManager()

		x = np.array([0., 1])
		y = np.array([-1., 1])
		self.datasetxy = vx.dataset.DatasetArrays()
		self.datasetxy.add_column("x", x)
		self.datasetxy.add_column("y", y)

		
	def test_length(self):
		assert len(self.dataset) == 10

	def t_est_length_mask(self):
		self.dataset._set_mask(self.dataset.columns['x'] < 5)
		self.assertEqual(self.dataset.length(selection=True), 5)


	def test_virtual_columns_spherical(self):
		alpha = np.array([0.])
		delta = np.array([0.])
		distance = np.array([1.])
		dataset = vx.dataset.DatasetArrays()
		dataset.add_column("alpha", alpha)
		dataset.add_column("delta", delta)
		dataset.add_column("distance", distance)

		dataset.add_virtual_columns_spherical_to_cartesian("alpha", "delta", "distance", "x", "y", "z", radians=False)

		subspace = dataset("x", "y", "z")
		x, y, z = subspace.sum()

		self.assertAlmostEqual(x, 0)
		self.assertAlmostEqual(y, 1)
		self.assertAlmostEqual(z, 0)

	def test_virtual_columns_equatorial(self):
		alpha = np.array([0.])
		delta = np.array([0.])
		distance = np.array([1.])
		dataset = vx.dataset.DatasetArrays()
		dataset.add_column("alpha", alpha)
		dataset.add_column("delta", delta)
		dataset.add_column("distance", distance)

		dataset.add_virtual_columns_equatorial_to_galactic("alpha", "delta", "distance", "x", "y", "z", radians=False)
		dataset.add_virtual_column("r", "sqrt(x**2+y**2+z**2)")

		subspace = dataset("x", "y", "z")
		x, y, z = subspace.sum()

		self.assertAlmostEqual(x**2+y**2+z**2, 1)

		subspace = dataset("r")
		r, = subspace.sum()
		self.assertAlmostEqual(r, 1)

	def test_sum(self):
		x, y = self.datasetxy("x", "y").sum()
		self.assertAlmostEqual(x, 1)
		self.assertAlmostEqual(y, 0)

	def test_mean(self):
		x, y = self.datasetxy("x", "y").mean()
		self.assertAlmostEqual(x, 0.5)
		self.assertAlmostEqual(y, 0)

	def test_var(self):
		x, y = self.datasetxy("x", "y").var()
		self.assertAlmostEqual(x, 0.5)
		self.assertAlmostEqual(y, 1.)

if __name__ == '__main__':
    unittest.main()

