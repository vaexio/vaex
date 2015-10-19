import vaex.dataset as dataset
import numpy as np
import unittest
import vaex as vx
import tempfile
import vaex.webserver

class CallbackCounter(object):
	def __init__(self):
		self.counter = 0

	def __call__(self, *args, **kwargs):
		self.counter += 1

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

		x1 = np.array([1., 3])
		x2 = np.array([2., 3, 4,])
		x3 = np.array([5.])
		self.x_concat = np.concatenate((x1, x2, x3))

		dataset1 = vx.dataset.DatasetArrays()
		dataset2 = vx.dataset.DatasetArrays()
		dataset3 = vx.dataset.DatasetArrays()
		dataset1.add_column("x", x1)
		dataset2.add_column("x", x2)
		dataset3.add_column("x", x3)
		dataset3.add_column("y", x3**2)

		self.dataset_concat = vx.dataset.DatasetConcatenated([dataset1, dataset2, dataset3])


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

		dataset.add_virtual_columns_equatorial_to_galactic_cartesian("alpha", "delta", "distance", "x", "y", "z", radians=False)
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

	def test_concat(self):
		self.assertEqual(self.dataset_concat.get_column_names(), ["x"])
		N = len(self.x_concat)
		# try out every possible slice
		for i1 in range(N-1):
			for i2 in range(i1+1,N):
				#print "***", i1, i2
				a = self.dataset_concat.columns["x"][i1:i2]
				b = self.x_concat[i1:i2]
				#print a, b
				np.testing.assert_array_almost_equal(a, b)

	def test_export(self):

		path = tempfile.mktemp(".hdf5")
		print path

		with self.assertRaises(AssertionError):
			self.dataset.export_hdf5(path, selection=True)

		for fraction in [1, 0.5]:
			self.dataset.set_active_fraction(fraction)
			self.dataset.select("x > 3")
			length = len(self.dataset)
			for column_names in [["x", "y"], ["x"], ["y"], None]:
				for byteorder in "=<>":
					for shuffle in [False, True]:
						for selection in [False, True]:
							print path, column_names, byteorder, shuffle, selection, fraction
							self.dataset.export_hdf5(path, column_names=column_names, byteorder=byteorder, shuffle=shuffle, selection=selection)
							compare = vx.open(path)
							column_names = column_names or ["x", "y"]
							self.assertEqual(compare.get_column_names(), column_names + (["random_index"] if shuffle else []))
							for column_name in column_names:
								if selection:
									self.assertEqual(sorted(compare.columns[column_name]), sorted(self.dataset.columns[column_name][self.dataset.mask]))
								else:
									if shuffle:
										indices = compare.columns["random_index"]
										self.assertEqual(sorted(compare.columns[column_name]), sorted(self.dataset.columns[column_name][indices]))
									else:
										self.assertEqual(sorted(compare.columns[column_name]), sorted(self.dataset.columns[column_name][:length]))

	def test_fraction(self):
		counter_selection = CallbackCounter()
		counter_current_row = CallbackCounter()
		self.dataset.signal_pick.connect(counter_current_row)
		self.dataset.signal_selection_changed.connect(counter_selection)

		self.dataset.set_active_fraction(1.0)
		self.assertEqual(counter_selection.counter, 1)
		self.assertEqual(counter_current_row.counter, 1)
		length = len(self.dataset)

		# test for event and the effect of the length
		self.dataset.set_active_fraction(0.5)
		self.assertEqual(counter_selection.counter, 2)
		self.assertEqual(counter_current_row.counter, 2)
		self.assertEqual(length/2, len(self.dataset))

		self.dataset.select("x > 5")
		self.assertEqual(counter_selection.counter, 3)
		self.assertEqual(counter_current_row.counter, 2)
		self.assert_(self.dataset.has_selection())
		self.dataset.set_active_fraction(0.5)
		self.assertFalse(self.dataset.has_selection())

		self.dataset.set_current_row(1)
		self.assertTrue(self.dataset.has_current_row())
		self.dataset.set_active_fraction(0.5)
		self.assertFalse(self.dataset.has_current_row())

		for dataset in [self.dataset, self.dataset_concat]:
			dataset.set_active_fraction(1.0)
			x = dataset.columns["x"][:] * 1. # make a copy
			dataset.set_active_fraction(0.5)
			length = len(dataset)
			a = x[:length]
			b = dataset.columns["x"][:len(dataset)]
			np.testing.assert_array_almost_equal(a, b)
			self.assertLess(length, dataset.full_length())

		# TODO: test if statistics and histogram work on the active_fraction

	def test_current_row(self):
		counter_current_row = CallbackCounter()
		self.dataset.signal_pick.connect(counter_current_row)
		self.dataset.set_current_row(0)
		self.assertEqual(counter_current_row.counter, 1)

		with self.assertRaises(IndexError):
			self.dataset.set_current_row(-1)
		with self.assertRaises(IndexError):
			self.dataset.set_current_row(len(self.dataset))


	def test_current(self):
		for dataset in [self.dataset, self.dataset_concat]:
			for i in range(len(dataset)):
				dataset.set_current_row(i)
				values = dataset("x", "x**2").current()
				value = dataset.columns["x"][:][i]
				self.assertEqual([value, value**2], values)



test_port = 19999

class TestWebServer(unittest.TestCase):
	def setUp(self):
		self.dataset = dataset.DatasetArrays()

		x = np.arange(10)
		y = x ** 2
		self.dataset.add_column("x", x)
		self.dataset.add_column("y", y)

		self.webserver = vaex.webserver.WebServer(datasets=[self.dataset], port=test_port)
		self.webserver.serve_threaded()
		self.server = vx.server("localhost", port=test_port)
		self.dataset_remote = self.server.datasets()[0]

	def tearDown(self):
		self.webserver.stop_serving()

	def test_list(self):
		datasets = self.server.datasets()
		self.assert_(len(datasets) == 1)
		dataset_remote = datasets[0]
		self.assertEqual(dataset_remote.name, self.dataset.name)
		self.assertEqual(dataset_remote.get_column_names(), self.dataset.get_column_names())
		self.assertEqual(len(dataset_remote), len(self.dataset))

	def test_minmax(self):
		self.assertEqual(self.dataset_remote("x", "y").minmax().tolist(), self.dataset("x", "y").minmax().tolist())

	def test_var(self):
		self.assertEqual(self.dataset_remote("x", "y").var().tolist(), self.dataset("x", "y").var().tolist())

	def test_histogram(self):
		grid1 = self.dataset("x").bounded().gridded(32).grid
		grid2 = self.dataset_remote("x").bounded().gridded(32).grid
		self.assertEqual(grid1.tolist(), grid2.tolist())



if __name__ == '__main__':
    unittest.main()

