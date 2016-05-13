# -*- coding: utf-8 -*-
import os
import vaex.dataset as dataset
import numpy as np
import unittest
import vaex as vx
import tempfile
import vaex.webserver
import astropy.io.fits
import astropy.units

import vaex.execution
a = vaex.execution.buffer_size_default # will crash if we decide to rename it

# this will make the test execute more code and may show up bugs
#vaex.execution.buffer_size_default = 3

vx.set_log_level_exception()
#vx.set_log_level_off()
vx.set_log_level_debug()

class CallbackCounter(object):
	def __init__(self, return_value=None):
		self.counter = 0
		self.return_value = return_value
		self.last_args = None
		self.last_kwargs = None

	def __call__(self, *args, **kwargs):
		self.counter += 1
		self.last_args = args
		self.last_kwargs = kwargs
		return self.return_value

class TestDataset(unittest.TestCase):
	def setUp(self):
		self.dataset = dataset.DatasetArrays("dataset")

		self.x = x = np.arange(10)
		self.y = y = x ** 2
		self.dataset.add_column("x", x)
		self.dataset.add_column("y", y)
		self.dataset.set_variable("t", 1.)
		self.dataset.add_virtual_column("z", "x+t*y")
		self.dataset.units["x"] = astropy.units.Unit("km")
		self.dataset.units["y"] = astropy.units.Unit("km/s")
		self.dataset.units["t"] = astropy.units.Unit("s")
		self.dataset.add_column("f", np.arange(len(self.dataset), dtype=np.float64))
		self.dataset.ucds["x"] = "some;ucd"

		#self.jobsManager = dataset.JobsManager()

		x = np.array([0., 1])
		y = np.array([-1., 1])
		self.datasetxy = vx.dataset.DatasetArrays("datasetxy")
		self.datasetxy.add_column("x", x)
		self.datasetxy.add_column("y", y)

		x1 = np.array([1., 3])
		x2 = np.array([2., 3, 4,])
		x3 = np.array([5.])
		self.x_concat = np.concatenate((x1, x2, x3))

		dataset1 = vx.dataset.DatasetArrays("dataset1")
		dataset2 = vx.dataset.DatasetArrays("dataset2")
		dataset3 = vx.dataset.DatasetArrays("dataset3")
		dataset1.add_column("x", x1)
		dataset2.add_column("x", x2)
		dataset3.add_column("x", x3)
		dataset3.add_column("y", x3**2)
		self.dataset_concat = vx.dataset.DatasetConcatenated([dataset1, dataset2, dataset3], name="dataset_concat")

		self.dataset_concat_dup = vx.dataset.DatasetConcatenated([self.dataset, self.dataset, self.dataset], name="dataset_concat_dup")

	def tearDown(self):
		self.dataset.remove_virtual_meta()
		self.dataset_concat.remove_virtual_meta()
		self.dataset_concat_dup.remove_virtual_meta()


	def histogram_cumulative(self):

		self.dataset("x").histogram()

	def test_units(self):
		assert self.dataset.unit("x") == astropy.units.km
		assert self.dataset.unit("y") == astropy.units.km/astropy.units.second
		assert self.dataset.unit("t") == astropy.units.second
		assert self.dataset.unit("z") == astropy.units.km
		assert self.dataset.unit("x+y") == None

	def test_dtype(self):
		self.assertEqual(self.dataset.dtype("x"), np.int64)
		self.assertEqual(self.dataset.dtype("f"), np.float64)
		self.assertEqual(self.dataset.dtype("x*f"), np.float64)

	def test_byte_size(self):
		self.assertEqual(self.dataset.byte_size(), 8*3*len(self.dataset))
		self.dataset.select("x < 1")
		self.assertEqual(self.dataset.byte_size(selection=True), 8*3)

	def test_ucd_find(self):
		self.dataset.ucds["x"] = "a;b;c"
		self.dataset.ucds["y"] = "b;c;d"
		self.dataset.ucds["z"] = "b;c;d"
		self.assertEqual(self.dataset.ucd_find("a"), "x")
		self.assertEqual(self.dataset.ucd_find("b"), "x")
		self.assertEqual(self.dataset.ucd_find("^b"), "y")
		self.assertEqual(self.dataset.ucd_find("c"), "x")
		self.assertEqual(self.dataset.ucd_find("d"), "y")

		self.assertEqual(self.dataset.ucd_find("b;c"), "x")
		self.assertEqual(self.dataset.ucd_find("^b;c"), "y")

	def test_data_access(self):
		assert (all(self.dataset.data.x == self.dataset.columns["x"]))
		self.assertEqual(self.dataset.data.x.ucd, self.dataset.ucds["x"])

	def test_subspace_basics(self):
		self.assertIsNotNone(repr(self.dataset("x")))
		self.assertIsNotNone(repr(self.dataset("x", "y")))
		self.assertIsNotNone(repr(self.dataset("x", "y", "z")))

		subspace = self.dataset("x", "y")
		for i in range(len(self.dataset)):
			self.assertEqual(subspace.row(0).tolist(), [self.x[0], self.y[0]])

		self.assertEqual(self.dataset.subspace("x", "y").expressions, self.dataset("x", "y").expressions)

	def test_subspaces(self):
		dataset = vaex.from_arrays("arrays", x=[1], y=[2], z=[3])
		subspaces = dataset.subspaces(dimensions=2)
		self.assertEqual(len(subspaces), 3)
		subspaces = dataset.subspaces(dimensions=2, exclude="x")
		self.assertEqual(len(subspaces), 1)
		subspaces = dataset.subspaces(dimensions=2, exclude=["x"])
		self.assertEqual(len(subspaces), 1)
		subspaces = dataset.subspaces(dimensions=2, exclude=[["x", "y"]])
		self.assertEqual(len(subspaces), 2)
		subspaces = dataset.subspaces(dimensions=2, exclude=[["y", "x"]])
		self.assertEqual(len(subspaces), 2)
		subspaces = dataset.subspaces(dimensions=2, exclude=lambda list: "x" in list)
		self.assertEqual(len(subspaces), 1)

		subspaces = self.dataset.subspaces([("x", "y")])
		self.assertEqual(subspaces.names(), ["x y"])
		self.assertEqual(subspaces.expressions_list(), [("x", "y")])

		self.assertIsNotNone(subspaces.selected().subspaces[0].is_masked)

		for async in [False, True]:
			subspaces = self.dataset.subspaces([("x", "y")], async=async)
			result = subspaces.minmax()
			if async:
				subspaces.subspace.executor.execute()
				result = result.get()
			minmax = result
			values = np.array(result).flatten()
			self.assertEqual(values.tolist(), self.dataset("x", "y").minmax().flatten().tolist())

			result = subspaces.limits_sigma()
			if async:
				subspaces.subspace.executor.execute()
				result = result.get()
			values = np.array(result).flatten()
			self.assertEqual(values.tolist(), self.dataset("x", "y").limits_sigma().flatten().tolist())

			result = subspaces.mean()
			if async:
				subspaces.subspace.executor.execute()
				result = result.get()
			means = result
			values = np.array(result).flatten()
			self.assertEqual(values.tolist(), self.dataset("x", "y").mean().flatten().tolist())

			result = subspaces.var()
			if async:
				subspaces.subspace.executor.execute()
				result = result.get()
			vars = result
			values = np.array(result).flatten()
			self.assertEqual(values.tolist(), self.dataset("x", "y").var().flatten().tolist())

			result = subspaces.var(means=means)
			if async:
				subspaces.subspace.executor.execute()
				result = result.get()
			vars = result
			values = np.array(result).flatten()
			self.assertEqual(values.tolist(), self.dataset("x", "y").var(means=means[0]).flatten().tolist())

			#means = [0, 0]
			result = subspaces.var(means=means)
			if async:
				subspaces.subspace.executor.execute()
				result = result.get()
			values = np.array(result).flatten()
			self.assertEqual(values.tolist(), self.dataset("x", "y").var(means=means[0]).flatten().tolist())

			for means_ in [means, None]:
				for vars_ in [vars, None]:
					result = subspaces.correlation(means=means_, vars=vars_)
					if async:
						subspaces.subspace.executor.execute()
						result = result.get()
					values = np.array(result).flatten()
					#print async, means_, vars_
					#print values, self.dataset("x", "y").correlation(), self.dataset("x", "y").correlation(means=means_[0] if means_ else None, vars=vars_[0] if vars_ else None).flatten().tolist()
					self.assertEqual(values.tolist(), self.dataset("x", "y").correlation(means=means_[0] if means_ else None, vars=vars_[0] if vars_ else None).flatten().tolist())


			result = subspaces.mutual_information()
			if async:
				subspaces.subspace.executor.execute()
				result = result.get()
			values = np.array(result).flatten()
			self.assertEqual(values.tolist(), self.dataset("x", "y").mutual_information().flatten().tolist())

			result = subspaces.mutual_information(limits=minmax)
			if async:
				subspaces.subspace.executor.execute()
				result = result.get()
			values = np.array(result).flatten()
			self.assertEqual(values.tolist(), self.dataset("x", "y").mutual_information(limits=minmax[0]).flatten().tolist())

	def test_not_implemented(self):
		subspace = vaex.dataset.Subspace(self.dataset, ["x", "y"], self.dataset.executor, False)
		with self.assertRaises(NotImplementedError):
			subspace.minmax()
		with self.assertRaises(NotImplementedError):
			subspace.mean()
		with self.assertRaises(NotImplementedError):
			subspace.var()
		with self.assertRaises(NotImplementedError):
			subspace.sum()
		with self.assertRaises(NotImplementedError):
			subspace.histogram([])
		with self.assertRaises(NotImplementedError):
			subspace.limits_sigma()

	def test_subspace_gridded(self):
		subspace = self.dataset("x", "y")
		limits = subspace.minmax()
		grid = subspace.histogram(limits)
		subspace_bounded = subspace.bounded_by(limits)
		subspace_gridded = subspace_bounded.gridded()
		assert(np.all(subspace_gridded.grid == grid))

		subspace_bounded = subspace.bounded_by_minmax()
		subspace_gridded = subspace_bounded.gridded()
		assert(np.all(subspace_gridded.grid == grid))

		limits = subspace.limits_sigma()
		grid = subspace.histogram(limits)
		subspace_bounded = subspace.bounded_by(limits)
		subspace_gridded = subspace_bounded.gridded()
		assert(np.all(subspace_gridded.grid == grid))

		subspace_bounded = subspace.bounded_by_sigmas()
		subspace_gridded = subspace_bounded.gridded()
		assert(np.all(subspace_gridded.grid == grid))


		subspace_gridded_vector = subspace_gridded.vector("x", "y")
		gridx = subspace.histogram(subspace_gridded_vector.subspace_bounded.bounds, size=32, weight="x")
		gridy = subspace.histogram(subspace_gridded_vector.subspace_bounded.bounds, size=32, weight="y")

		assert(np.all(subspace_gridded_vector.vx.grid == gridx))
		assert(np.all(subspace_gridded_vector.vy.grid == gridy))



	def test_length(self):
		assert len(self.dataset) == 10

	def t_est_length_mask(self):
		self.dataset._set_mask(self.dataset.columns['x'] < 5)
		self.assertEqual(self.dataset.length(selection=True), 5)

	def test_evaluate(self):
		for t in [2, 3]:
			self.dataset.set_variable("t", t)
			x = self.dataset.evaluate("x")
			y = self.dataset.evaluate("y")
			z = self.dataset.evaluate("z")
			z_test = x + t * y
			np.testing.assert_array_almost_equal(z, z_test)


	def test_subspace_errors(self):

		with self.assertRaises(SyntaxError):
			self.dataset("x/").sum()
		with self.assertRaises((KeyError, NameError)): # TODO: should we have just one error type?
			self.dataset("doesnotexist").sum()

		# that that after a error we can still continue
		self.dataset("x").sum()

		for i in range(100):
			with self.assertRaises(SyntaxError):
				self.dataset("x/").sum()
			with self.assertRaises((KeyError, NameError)): # TODO: should we have just one error type?
				self.dataset("doesnotexist").sum()
			self.dataset("x").sum()

	def test_invalid_expression(self):
		with self.assertRaises(SyntaxError):
			self.dataset.validate_expression("x/")
		with self.assertRaises(NameError):
			self.dataset.validate_expression("hoeba(x)")
		with self.assertRaises(NameError):
			self.dataset.validate_expression("x()")
		self.dataset.validate_expression("sin(x)+tan(y)")
		with self.assertRaises((KeyError, NameError)): # TODO: should we have just one error type?
			self.dataset.validate_expression("doesnotexist")
		self.dataset.validate_expression("x / y * z + x - x - -x")
		self.dataset.validate_expression("x < 0")
		self.dataset.validate_expression("x <= 0")
		self.dataset.validate_expression("x > 0")
		self.dataset.validate_expression("x >= 0")
		self.dataset.validate_expression("x == 0")
		self.dataset.validate_expression("x != 0")

	def test_evaluate_nested(self):
		self.dataset.add_virtual_column("z2", "-z")
		self.dataset.add_virtual_column("z3", "z+z2")
		zeros = self.dataset.evaluate("z3")
		np.testing.assert_array_almost_equal(zeros, np.zeros(len(self.dataset)))


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

		self.assertAlmostEqual(x, 1)
		self.assertAlmostEqual(y, 0)
		self.assertAlmostEqual(z, 0)


		dataset.add_virtual_columns_cartesian_to_spherical("x", "y", "z", "theta", "phi", "r", radians=False)
		theta, phi, r = dataset("theta", "phi", "r").row(0)
		self.assertAlmostEqual(theta, 0)
		self.assertAlmostEqual(phi, 0)
		self.assertAlmostEqual(r, 1)


		dataset.add_virtual_columns_celestial("alpha", "delta", "l", "b")
		# TODO: properly test, with and without radians
		dataset.evaluate("l")
		dataset.evaluate("b")

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

		self.datasetxy.select("x < 1")
		x, y = self.datasetxy("x", "y").selected().sum()
		self.assertAlmostEqual(x, 0)
		self.assertAlmostEqual(y, -1)

	def test_progress(self):
		x, y = self.datasetxy("x", "y").sum()
		task = self.datasetxy("x", "y", async=True).sum()
		counter = CallbackCounter(True)
		task.signal_progress.connect(counter)
		self.datasetxy.executor.execute()
		x2, y2 = task.get()
		self.assertEqual(x, x2)
		self.assertEqual(y, y2)
		self.assertGreater(counter.counter, 0)
		self.assertEqual(counter.last_args[0], 1.0)


	def test_mean(self):
		x, y = self.datasetxy("x", "y").mean()
		self.assertAlmostEqual(x, 0.5)
		self.assertAlmostEqual(y, 0)

		self.datasetxy.select("x < 1")
		x, y = self.datasetxy("x", "y").selected().mean()
		self.assertAlmostEqual(x, 0)
		self.assertAlmostEqual(y, -1)

	def test_minmax(self):
		((xmin, xmax), ) = self.dataset("x").minmax()
		self.assertAlmostEqual(xmin, 0)
		self.assertAlmostEqual(xmax, 9)

		self.dataset.select("x < 5")
		((xmin2, xmax2), ) = self.dataset("x").selected().minmax()
		self.assertAlmostEqual(xmin2, 0)
		self.assertAlmostEqual(xmax2, 4)

	def test_var(self):
		x, y = self.datasetxy("x", "y").var()
		self.assertAlmostEqual(x, 0.5)
		self.assertAlmostEqual(y, 1.)

		x, y = self.dataset("x", "y").var()
		self.assertAlmostEqual(x, np.mean(self.x**2))
		self.assertAlmostEqual(y, np.mean(self.y**2))

		self.dataset.select("x < 5")
		x, y = self.dataset("x", "y").selected().var()
		self.assertAlmostEqual(x, np.mean(self.x[:5]**2))
		self.assertAlmostEqual(y, np.mean(self.y[:5]**2))

	def test_correlation(self):

		subspace = self.datasetxy("y", "y")
		means = subspace.mean()
		vars = subspace.var(means)
		correlation = subspace.correlation(means, vars)
		self.assertAlmostEqual(correlation, 1.0)

		subspace = self.datasetxy("y", "-y")
		means = subspace.mean()
		vars = subspace.var(means)
		correlation = subspace.correlation(means, vars)
		self.assertAlmostEqual(correlation, -1.0)


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

		def concat(*types):
			arrays = [np.arange(3, dtype=dtype) for dtype in types]
			N = len(arrays)
			datasets = [vx.dataset.DatasetArrays("dataset-%i" % i)  for i in range(N)]
			for dataset, array in zip(datasets, arrays):
				dataset.add_column("x", array)
			dataset_concat = vx.dataset.DatasetConcatenated(datasets, name="dataset_concat")
			return dataset_concat

		self.assertEqual(concat(np.float32, np.float64).columns["x"].dtype, np.float64)
		self.assertEqual(concat(np.float32, np.int64).columns["x"].dtype, np.float64)
		self.assertEqual(concat(np.float32, np.byte).columns["x"].dtype, np.float32)
		self.assertEqual(concat(np.float64, np.byte, np.int64).columns["x"].dtype, np.float64)

		ar1 = np.zeros((10, 2))
		ar2 = np.zeros((20))
		arrays = [ar1, ar2]
		N = len(arrays)
		datasets = [vx.dataset.DatasetArrays("dataset1") for i in range(N)]
		for dataset, array in zip(datasets, arrays):
			dataset.add_column("x", array)
		with self.assertRaises(ValueError):
			dataset_concat = vx.dataset.DatasetConcatenated(datasets, name="dataset_concat")


		ar1 = np.zeros((10))
		ar2 = np.zeros((20))
		arrays = [ar1, ar2]
		N = len(arrays)
		datasets = [vx.dataset.DatasetArrays("dataset1") for i in range(N)]
		for dataset, array in zip(datasets, arrays):
			dataset.add_column("x", array)
		dataset_concat = vx.dataset.DatasetConcatenated(datasets, name="dataset_concat")


		dataset_concat1 = vx.dataset.DatasetConcatenated(datasets, name="dataset_concat")
		dataset_concat2 = vx.dataset.DatasetConcatenated(datasets, name="dataset_concat")
		self.assertEqual(len(dataset_concat1.concat(dataset_concat2).datasets), 4)
		self.assertEqual(len(dataset_concat1.concat(datasets[0]).datasets), 3)
		self.assertEqual(len(datasets[0].concat(dataset_concat1).datasets), 3)
		self.assertEqual(len(datasets[0].concat(datasets[0]).datasets), 2)

	def test_export_concat(self):
		x1 = np.arange(1000, dtype=np.float32)
		x2 = np.arange(100, dtype=np.float32)
		self.x_concat = np.concatenate((x1, x2))

		dataset1 = vx.dataset.DatasetArrays("dataset1")
		dataset2 = vx.dataset.DatasetArrays("dataset2")
		dataset1.add_column("x", x1)
		dataset2.add_column("x", x2)

		self.dataset_concat = vx.dataset.DatasetConcatenated([dataset1, dataset2], name="dataset_concat")

		path_hdf5 = tempfile.mktemp(".hdf5")
		self.dataset_concat.export_hdf5(path_hdf5)


	def test_export(self):

		path = path_hdf5 = tempfile.mktemp(".hdf5")
		path_fits = tempfile.mktemp(".fits")
		path_fits_astropy = tempfile.mktemp(".fits")
		#print path

		with self.assertRaises(AssertionError):
			self.dataset.export_hdf5(path, selection=True)

		for dataset in [self.dataset_concat_dup, self.dataset]:
			#print dataset.virtual_columns
			for fraction in [1, 0.5]:
				dataset.set_active_fraction(fraction)
				dataset.select("x > 3")
				length = len(dataset)
				for column_names in [["x", "y", "z"], ["x"], ["y"], ["z"], None]:
					for byteorder in ">=<":
						for shuffle in [False, True]:
							for selection in [False, True]:
								for export in [dataset.export_fits, dataset.export_hdf5] if byteorder == ">" else [dataset.export_hdf5]:
									#print dataset, path, column_names, byteorder, shuffle, selection, fraction, dataset.full_length()
									#print dataset.full_length()
									#print len(dataset)
									if export == dataset.export_hdf5:
										path = path_hdf5
										export(path, column_names=column_names, byteorder=byteorder, shuffle=shuffle, selection=selection, progress=False)
									else:
										path = path_fits
										export(path, column_names=column_names, shuffle=shuffle, selection=selection, progress=False)
										with astropy.io.fits.open(path) as fitsfile:
											# make sure astropy can read the data
											bla = fitsfile[1].data
											try:
												fitsfile.writeto(path_fits_astropy)
											finally:
												os.remove(path_fits_astropy)
									compare = vx.open(path)
									column_names = column_names or ["x", "y", "f", "z"]
									# TODO: does the order matter?
									self.assertEqual(sorted(compare.get_column_names()), sorted(column_names + (["random_index"] if shuffle else [])))
									for column_name in column_names:
										values = dataset.evaluate(column_name)
										if selection:
											self.assertEqual(sorted(compare.columns[column_name]), sorted(values[dataset.mask]))
										else:
											if shuffle:
												indices = compare.columns["random_index"]
												self.assertEqual(sorted(compare.columns[column_name]), sorted(values[indices]))
											else:
												self.assertEqual(sorted(compare.columns[column_name]), sorted(values[:length]))
									compare.close_files()

				# self.dataset_concat_dup references self.dataset, so set it's active_fraction to 1 again
				dataset.set_active_fraction(1)
		import vaex.export
		dataset = self.dataset
		dataset.export_fits(path_fits)
		name = "vaex export"
		#print(path_fits)
		vaex.export.main([name, "--no-progress", "-q", "file", path_fits, path_hdf5])
		backup = vaex.vaex.utils.check_memory_usage
		try:
			vaex.vaex.utils.check_memory_usage = lambda *args: False
			assert vaex.export.main([name, "--no-progress", "-q", "soneira", "--dimension=2", "-m=40", path_hdf5]) == 1
		finally:
			vaex.utils.check_memory_usage = backup
		assert vaex.export.main([name, "--no-progress", "-q", "soneira", "--dimension=2", "-m=20", path_hdf5]) == 0

	def test_fraction(self):
		counter_selection = CallbackCounter()
		counter_current_row = CallbackCounter()
		self.dataset.signal_pick.connect(counter_current_row)
		self.dataset.signal_selection_changed.connect(counter_selection)

		self.dataset.set_active_fraction(1.0) # this shouldn't trigger
		self.assertEqual(counter_selection.counter, 0)
		self.assertEqual(counter_current_row.counter, 0)
		length = len(self.dataset)
		self.dataset.set_active_fraction(0.1) # this should trigger
		self.assertEqual(counter_selection.counter, 1)
		self.assertEqual(counter_current_row.counter, 1)

		# test for event and the effect of the length
		self.dataset.set_active_fraction(0.5)
		self.assertEqual(counter_selection.counter, 2)
		self.assertEqual(counter_current_row.counter, 2)
		self.assertEqual(length/2, len(self.dataset))

		self.dataset.select("x > 5")
		self.assertEqual(counter_selection.counter, 3)
		self.assertEqual(counter_current_row.counter, 2)
		self.assertTrue(self.dataset.has_selection())
		self.dataset.set_active_fraction(0.5) # nothing should happen, still the same
		self.assertTrue(self.dataset.has_selection())
		self.dataset.set_active_fraction(0.4999)
		self.assertFalse(self.dataset.has_selection())

		self.dataset.set_current_row(1)
		self.assertTrue(self.dataset.has_current_row())
		self.dataset.set_active_fraction(0.5)
		self.assertFalse(self.dataset.has_current_row())

		if self.dataset.is_local(): # this part doesn't work for remote datasets
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
		self.dataset.set_active_fraction(1)
		total, = self.dataset("x").sum()
		self.dataset.set_active_fraction(0.5)
		total_half, = self.dataset("x").sum()
		self.assertLess(total_half, total)

		limits = [(-100, 100)]
		self.dataset.set_active_fraction(1)
		total = self.dataset("x").histogram(limits).sum()
		self.dataset.set_active_fraction(0.5)
		total_half = self.dataset("x").histogram(limits).sum()
		self.assertLess(total_half, total)


	def test_histogram(self):
		counts = self.dataset("x").histogram([[0,10]], size=10)
		#import pdb
		#pdb.set_trace()
		self.assertTrue(all(counts == 1), "counts is %r" % counts)

		sums = self.dataset("x").histogram([[0,10]], size=10, weight="y")
		assert(all(sums == self.y))

		self.dataset.select("x < 5")
		mask = self.x < 5

		counts = self.dataset("x").selected().histogram([[0,10]], size=10)
		mod_counts = counts * 1.
		mod_counts[~mask] = 0
		assert(all(counts == mod_counts))

		mod_sums = self.y * 1.
		mod_sums[~mask] = 0
		sums = self.dataset("x").selected().histogram([[0,10]], size=10, weight="y")
		assert(all(sums == mod_sums))


		x = np.array([0, 1, 0, 1])
		y = np.array([0, 0, 1, 1])
		dataset = vx.from_arrays(x=x, y=y)
		counts = dataset("x", "y").histogram([[0.,2.], [0.,2.]], size=2)
		assert(np.all(counts == 1))

		x = np.array([0, 1, 0, 1, 0, 1, 0, 1])
		y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
		z = np.array([0, 0, 0, 0, 1, 1, 1, 1])
		dataset = vx.from_arrays(x=x, y=y, z=z)
		counts = dataset("x", "y", "z").histogram([[0.,2.], [0.,2.], [0.,2.]], size=2)
		assert(np.all(counts == 1))

		x = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
		y = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
		z = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
		w = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,])
		dataset = vx.from_arrays(x=x, y=y, z=z, w=w)
		counts = dataset("x", "y", "z", "w").histogram([[0.,2.], [0.,2.], [0.,2.], [0.,2.]], size=2)
		assert(np.all(counts == 1))



	def test_current_row(self):
		counter_current_row = CallbackCounter()
		self.dataset.signal_pick.connect(counter_current_row)
		self.dataset.set_current_row(0)
		self.assertEqual(counter_current_row.counter, 1)

		with self.assertRaises(IndexError):
			self.dataset.set_current_row(-1)
		with self.assertRaises(IndexError):
			self.dataset.set_current_row(len(self.dataset))


	def t_not_needed_est_current(self):
		for dataset in [self.dataset, self.dataset_concat]:
			for i in range(len(dataset)):
				dataset.set_current_row(i)
				values = dataset("x", "x**2").current()
				value = dataset.columns["x"][:][i]
				self.assertEqual([value, value**2], values)

	def test_selection(self):
		total = self.dataset("x").sum()
		self.dataset.select("x > 5")
		total_subset = self.dataset("x").selected().sum()
		self.assertLess(total_subset, total)
		for mode in vaex.dataset._select_functions.keys():
			self.dataset.select("x > 5")
			self.dataset.select("x > 5", mode)
			self.dataset.select(None)
			self.dataset.select("x > 5", mode)


		self.dataset.select("x > 5")
		total_subset = self.dataset("x").selected().sum()
		self.dataset.select_inverse()
		total_subset_inverse = self.dataset("x").selected().sum()
		self.dataset.select("x <= 5")
		total_subset_inverse_compare = self.dataset("x").selected().sum()
		self.assertEqual(total_subset_inverse, total_subset_inverse_compare)
		self.assertEqual(total_subset_inverse + total_subset, total)


		pass # TODO

	def test_favorite_selections(self):
		self.dataset.select("x > 5")
		total_subset = self.dataset("x").selected().sum()
		self.dataset.add_favorite_selection("test")
		self.dataset.select_nothing()
		with self.assertRaises(ValueError):
			self.dataset.add_favorite_selection("test")
		self.dataset.load_favorite_selections()
		self.dataset.apply_favorite_selection("test")
		total_subset_test = self.dataset("x").selected().sum()
		self.assertEqual(total_subset, total_subset_test)




	def test_selection_history(self):
		self.assertTrue(not self.dataset.has_selection())
		self.assertTrue(not self.dataset.selection_can_undo())
		self.assertTrue(not self.dataset.selection_can_redo())

		self.dataset.select_nothing()
		self.assertTrue(not self.dataset.has_selection())
		self.assertTrue(not self.dataset.selection_can_undo())
		self.assertTrue(not self.dataset.selection_can_redo())


		total = self.dataset("x").sum()
		self.assertTrue(not self.dataset.has_selection())
		self.assertTrue(not self.dataset.selection_can_undo())
		self.assertTrue(not self.dataset.selection_can_redo())
		self.dataset.select("x > 5")
		self.assertTrue(self.dataset.has_selection())
		total_subset = self.dataset("x").selected().sum()
		self.assertLess(total_subset, total)
		self.assertTrue(self.dataset.selection_can_undo())
		self.assertTrue(not self.dataset.selection_can_redo())

		self.dataset.select("x < 7", mode="and")
		total_subset2 = self.dataset("x").selected().sum()
		self.assertLess(total_subset2, total_subset)
		self.assertTrue(self.dataset.selection_can_undo())
		self.assertTrue(not self.dataset.selection_can_redo())

		self.dataset.selection_undo()
		total_subset_same = self.dataset("x").selected().sum()
		self.assertEqual(total_subset, total_subset_same)
		self.assertTrue(self.dataset.selection_can_undo())
		self.assertTrue(self.dataset.selection_can_redo())

		self.dataset.selection_redo()
		total_subset2_same = self.dataset("x").selected().sum()
		self.assertEqual(total_subset2, total_subset2_same)
		self.assertTrue(self.dataset.selection_can_undo())
		self.assertTrue(not self.dataset.selection_can_redo())

		self.dataset.selection_undo()
		self.dataset.selection_undo()
		self.assertTrue(not self.dataset.has_selection())
		self.assertTrue(not self.dataset.selection_can_undo())
		self.assertTrue(self.dataset.selection_can_redo())

		self.dataset.selection_redo()
		self.assertTrue(self.dataset.has_selection())
		self.assertTrue(self.dataset.selection_can_undo())
		self.assertTrue(self.dataset.selection_can_redo())
		self.dataset.select("x < 7", mode="and")
		self.assertTrue(self.dataset.selection_can_undo())
		self.assertTrue(not self.dataset.selection_can_redo())

		self.dataset.select_nothing()
		self.assertTrue(not self.dataset.has_selection())
		self.assertTrue(self.dataset.selection_can_undo())
		self.assertTrue(not self.dataset.selection_can_redo())
		self.dataset.selection_undo()
		self.assertTrue(self.dataset.selection_can_undo())
		self.assertTrue(self.dataset.selection_can_redo())

	def test_selection_serialize(self):
		selection_lasso = vaex.dataset.SelectionLasso(self.dataset, "x", "y", [0, 10, 0], [-1, -1, 1], None, "replace")
		selection_expression = vaex.dataset.SelectionExpression(self.dataset, "x > 5", None, "and")
		self.dataset.set_selection(selection_expression)
		total_subset = self.dataset("x").selected().sum()

		self.dataset.select("x > 5")
		total_subset_same = self.dataset("x").selected().sum()
		self.assertEqual(total_subset, total_subset_same)

		values = selection_expression.to_dict()
		self.dataset.set_selection(vaex.dataset.selection_from_dict(self.dataset, values))
		total_subset_same2 = self.dataset("x").selected().sum()
		self.assertEqual(total_subset, total_subset_same2)


	def test_nearest(self):
		index, distance, (value,) = self.dataset("x").nearest([3])
		self.assertEqual(index, 3)
		self.assertEqual(distance, 0)
		self.assertEqual(value, 3)

		index, distance, (value,) = self.dataset("x").nearest([3.7])
		self.assertEqual(index, 4)
		self.assertAlmostEqual(distance, 0.3)
		self.assertEqual(value, 4)

		self.dataset.select("x > 5")
		index, distance, (value,) = self.dataset("x").selected().nearest([3.7])
		self.assertEqual(index, 6)
		self.assertEqual(distance, 2.3)
		self.assertEqual(value, 6)


	def test_lasso(self):
		# this doesn't really test much, just that the code gets executed
		self.x = x = np.arange(10)
		self.y = y = x ** 2

		x = [-0.1, 5.1, 5.1, -0.1]
		y = [-0.1, -0.1, 4.1, 4.1]
		self.dataset.select_lasso("x", "y", x, y)
		sumx, sumy = self.dataset("x", "y").selected().sum()
		self.assertAlmostEqual(sumx, 0+1+2)
		self.assertAlmostEqual(sumy, 0+1+4)



# allow multiple python versions on one machine to run the test
import sys
test_port = 29110 + sys.version_info[0] * 10 + sys.version_info[1]

class TestDatasetRemote(TestDataset):
	use_websocket = True
	def setUp(self):
		global test_port
		# run all tests from TestDataset, but now served at the server
		super(TestDatasetRemote, self).setUp()
		self.dataset_local = self.dataset
		self.datasetxy_local = self.datasetxy
		self.dataset_concat_local = self.dataset_concat
		self.dataset_concat_dup_local = self.dataset_concat_dup

		datasets = [self.dataset_local, self.datasetxy_local, self.dataset_concat_local, self.dataset_concat_dup_local]
		self.webserver = vaex.webserver.WebServer(datasets=datasets, port=test_port)
		#print "serving"
		self.webserver.serve_threaded()
		#print "getting server object"
		scheme = "ws" if self.use_websocket else "http"
		self.server = vx.server("%s://localhost:%d" % (scheme, test_port))
		test_port += 1
		#print "get datasets"
		datasets = self.server.datasets(as_dict=True)
		#print "got it", datasets

		self.dataset = datasets["dataset"]
		self.datasetxy = datasets["datasetxy"]
		self.dataset_concat = datasets["dataset_concat"]
		self.dataset_concat_dup = datasets["dataset_concat_dup"]
		#print "all done"



	def tearDown(self):
		TestDataset.tearDown(self)
		#print "stop serving"
		self.webserver.stop_serving()

	def test_export(self):
		pass # we can't export atm

	def test_concat(self):
		pass # doesn't make sense to test this for remote

	def test_data_access(self):
		pass

	def test_byte_size(self):
		pass # we don't know the selection's length for dataset remote..


"""

class T_estDatasetRemoteHttp(T_estDatasetRemote):
	use_websocket = False


class T_stWebServer(unittest.TestCase):
	def setUp(self):
		self.dataset = dataset.DatasetArrays()

		self.x = x = np.arange(10)
		self.y = y = x ** 2
		self.dataset.add_column("x", x)
		self.dataset.add_column("y", y)

		self.webserver = vaex.webserver.WebServer(datasets=[self.dataset], port=test_port)
		self.webserver.serve_threaded()
		self.server = vx.server("http://localhost:%d" % test_port)
		self.dataset_remote = self.server.datasets()[0]

	def tearDown(self):
		self.webserver.stop_serving()

	def test_list(self):
		datasets = self.server.datasets()
		self.assertTrue(len(datasets) == 1)
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

"""

if __name__ == '__main__':
    unittest.main()

