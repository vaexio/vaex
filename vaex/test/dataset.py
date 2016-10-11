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

basedir = os.path.dirname(__file__)
# this will make the test execute more code and may show up bugs
#vaex.execution.buffer_size_default = 3

vx.set_log_level_exception()
#vx.set_log_level_off()
#vx.set_log_level_debug()

def from_scalars(**kwargs):
	return vx.from_arrays("test", **{k:np.array([v]) for k, v in kwargs.items()})

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

		name = np.array(list(map(lambda x: str(x) + "bla", self.x)), dtype='S') #, dtype=np.string_)
		self.names = self.dataset.get_column_names()
		self.dataset.add_column("name", np.array(name))


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
		self.dataset_local = self.dataset
		self.datasetxy_local = self.datasetxy
		self.dataset_concat_local = self.dataset_concat
		self.dataset_concat_dup_local = self.dataset_concat_dup

		np.random.seed(0) # fix seed so that test never fails randomly

	def test_amuse(self):
		ds = vx.open(os.path.join(basedir, "files", "default_amuse_plummer.hdf5"))
		self.assertGreater(len(ds), 0)
		self.assertGreater(len(ds.get_column_names()), 0)
		self.assertIsNotNone(ds.unit("x"))
		self.assertIsNotNone(ds.unit("vx"))
		self.assertIsNotNone(ds.unit("mass"))

	def tearDown(self):
		self.dataset.remove_virtual_meta()
		self.dataset_concat.remove_virtual_meta()
		self.dataset_concat_dup.remove_virtual_meta()

	def test_mixed_endian(self):

		x = np.arange(10., dtype=">f8")
		y = np.arange(10, dtype="<f8")
		ds = vx.from_arrays("mixed", x=x, y=y)
		ds.count()
		ds.count(binby=["x", "y"])

	def test_uncertainty_propagation(self):

		N = 100000
		# distance
		parallaxes = np.random.normal(1, 0.1, N)
		ds_many = vx.from_arrays("test", parallax=parallaxes)
		ds_many.add_virtual_columns_distance_from_parallax("parallax", "distance")
		distance_std_est = ds_many.std("distance").item()

		ds_1 = vx.from_arrays("test", parallax=np.array([1.]), parallax_uncertainty=np.array([0.1]))
		ds_1.add_virtual_columns_distance_from_parallax("parallax", "distance", "parallax_uncertainty")
		distance_std = ds_1.evaluate("distance_uncertainty")[0]
		self.assertAlmostEqual(distance_std, distance_std_est,2)

	def test_add_virtual_columns_cartesian_velocities_to_polar(self):
		if 1:
			def datasets(x, y, velx, vely):
				ds_1 = from_scalars(x=x, y=y, vx=velx, vy=vely, x_e=0.01, y_e=0.02, vx_e=0.03, vy_e=0.04)
				#sigmas = ["alpha_e**2", "delta_e**2", "pm_a_e**2", "pm_d_e**2"]
				#cov = [[sigmas[i] if i == j else "" for i in range(4)] for j in range(4)]
				ds_1.add_virtual_columns_cartesian_velocities_to_polar(cov_matrix_x_y_vx_vy="auto")
				N = 100000
				# distance
				x =        np.random.normal(x, 0.01, N)
				y =        np.random.normal(y, 0.02, N)
				velx =        np.random.normal(velx, 0.03, N)
				vely =        np.random.normal(vely, 0.04, N)
				ds_many = vx.from_arrays(x=x, y=y, vx=vely, vy=vely)
				ds_many.add_virtual_columns_cartesian_velocities_to_polar()
				return ds_1, ds_many
			ds_1, ds_many = datasets(0, 2, 3, 4)

			vr_polar_e = ds_1.evaluate("vr_polar_uncertainty")[0]
			vphi_polar_e = ds_1.evaluate("vphi_polar_uncertainty")[0]
			self.assertAlmostEqual(vr_polar_e, ds_many.std("vr_polar").item(), delta=0.02)
			self.assertAlmostEqual(vphi_polar_e, ds_many.std("vphi_polar").item(), delta=0.02)

			# rotation is anti clockwise
			ds_1 = from_scalars(x=0, y=2, vx=0, vy=2)
			ds_1.add_virtual_columns_cartesian_velocities_to_polar()
			vr_polar = ds_1.evaluate("vr_polar")[0]
			vphi_polar = ds_1.evaluate("vphi_polar")[0]
			self.assertAlmostEqual(vr_polar, 2)
			self.assertAlmostEqual(vphi_polar, 0)

			ds_1 = from_scalars(x=0, y=2, vx=-2, vy=0)
			ds_1.add_virtual_columns_cartesian_velocities_to_polar()
			vr_polar = ds_1.evaluate("vr_polar")[0]
			vphi_polar = ds_1.evaluate("vphi_polar")[0]
			self.assertAlmostEqual(vr_polar, 0)
			self.assertAlmostEqual(vphi_polar, 2)

	def test_add_virtual_columns_cartesian_to_polar(self):
		for radians in [True, False]:
			def datasets(x, y, radians=radians):
				ds_1 = from_scalars(x=x, y=y, x_e=0.01, y_e=0.02)
				#sigmas = ["alpha_e**2", "delta_e**2", "pm_a_e**2", "pm_d_e**2"]
				#cov = [[sigmas[i] if i == j else "" for i in range(4)] for j in range(4)]
				ds_1.add_virtual_columns_cartesian_to_polar(cov_matrix_x_y="auto", radians=radians)
				N = 100000
				# distance
				x =        np.random.normal(x, 0.01, N)
				y =        np.random.normal(y, 0.02, N)
				ds_many = vx.from_arrays(x=x, y=y)
				ds_many.add_virtual_columns_cartesian_to_polar(radians=radians)
				return ds_1, ds_many
			ds_1, ds_many = datasets(0, 2)

			r_polar_e = ds_1.evaluate("r_polar_uncertainty")[0]
			phi_polar_e = ds_1.evaluate("phi_polar_uncertainty")[0]
			self.assertAlmostEqual(r_polar_e, ds_many.std("r_polar").item(), delta=0.02)
			self.assertAlmostEqual(phi_polar_e, ds_many.std("phi_polar").item(), delta=0.02)

			# rotation is anti clockwise
			r_polar = ds_1.evaluate("r_polar")[0]
			phi_polar = ds_1.evaluate("phi_polar")[0]
			self.assertAlmostEqual(r_polar, 2)
			self.assertAlmostEqual(phi_polar, np.pi/2 if radians else 90)

	def test_add_virtual_columns_proper_motion_eq2gal(self):
		for radians in [True, False]:
			def datasets(alpha, delta, pm_a, pm_d, radians=radians):
				ds_1 = from_scalars(alpha=alpha, delta=delta, pm_a=pm_a, pm_d=pm_d, alpha_e=0.01, delta_e=0.02, pm_a_e=0.003, pm_d_e=0.004)
				sigmas = ["alpha_e**2", "delta_e**2", "pm_a_e**2", "pm_d_e**2"]
				cov = [[sigmas[i] if i == j else "" for i in range(4)] for j in range(4)]
				ds_1.add_virtual_columns_proper_motion_eq2gal("alpha", "delta", "pm_a", "pm_d", "pm_l", "pm_b", cov_matrix_alpha_delta_pma_pmd=cov, radians=radians)
				N = 100000
				# distance
				alpha =        np.random.normal(0, 0.01, N)  + alpha
				delta =        np.random.normal(0, 0.02, N)  + delta
				pm_a =         np.random.normal(0, 0.003, N)  + pm_a
				pm_d =         np.random.normal(0, 0.004, N)  + pm_d
				ds_many = vx.from_arrays(alpha=alpha, delta=delta, pm_a=pm_a, pm_d=pm_d)
				ds_many.add_virtual_columns_proper_motion_eq2gal("alpha", "delta", "pm_a", "pm_d", "pm_l", "pm_b", radians=radians)
				return ds_1, ds_many
			ds_1, ds_many = datasets(0, 0, 1, 2)

			if 0: # only for testing the test
				c1_e = ds_1.evaluate("c1_uncertainty")[0]
				c2_e = ds_1.evaluate("c2_uncertainty")[0]
				self.assertAlmostEqual(c1_e, ds_many.std("__proper_motion_eq2gal_C1").item(), delta=0.02)
				self.assertAlmostEqual(c2_e, ds_many.std("__proper_motion_eq2gal_C2").item(), delta=0.02)

			pm_l_e = ds_1.evaluate("pm_l_uncertainty")[0]
			pm_b_e = ds_1.evaluate("pm_b_uncertainty")[0]
			self.assertAlmostEqual(pm_l_e, ds_many.std("pm_l").item(), delta=0.02)
			self.assertAlmostEqual(pm_b_e, ds_many.std("pm_b").item(), delta=0.02)

	def test_add_virtual_columns_proper_motion2vperpendicular(self):
		def datasets(distance, pm_l, pm_b):
			ds_1 = from_scalars(pm_l=pm_l, pm_b=pm_b, distance=distance, distance_e=0.1, pm_long_e=0.3, pm_lat_e=0.4)
			sigmas = ["distance_e**2", "pm_long_e**2", "pm_lat_e**2"]
			cov = [[sigmas[i] if i == j else "" for i in range(3)] for j in range(3)]
			ds_1.add_virtual_columns_proper_motion2vperpendicular(cov_matrix_distance_pm_long_pm_lat=cov)
			N = 100000
			# distance
			distance = np.random.normal(0, 0.1, N)  + distance
			pm_l =     np.random.normal(0, 0.3, N)  + pm_l
			pm_b =     np.random.normal(0, 0.4, N)  + pm_b
			ds_many = vx.from_arrays(pm_l=pm_l, pm_b=pm_b, distance=distance)
			ds_many.add_virtual_columns_proper_motion2vperpendicular()
			return ds_1, ds_many
		ds_1, ds_many = datasets(2, 3, 4)

		vl_e = ds_1.evaluate("vl_uncertainty")[0]
		vb_e = ds_1.evaluate("vb_uncertainty")[0]
		self.assertAlmostEqual(vl_e, ds_many.std("vl").item(), delta=0.02)
		self.assertAlmostEqual(vb_e, ds_many.std("vb").item(), delta=0.02)
		k = 4.74057
		self.assertAlmostEqual(ds_1.evaluate("vl")[0], 2*k*3)
		self.assertAlmostEqual(ds_1.evaluate("vb")[0], 2*k*4)

	def test_virtual_columns_lbrvr_proper_motion2vcartesian(self):
		for radians in [True, False]:
			def datasets(l, b, distance, vr, pm_l, pm_b, radians=radians):
				ds_1 = from_scalars(l=l, b=b, pm_l=pm_l, pm_b=pm_b, vr=vr, distance=distance, distance_e=0.1, vr_e=0.2, pm_long_e=0.3, pm_lat_e=0.4)
				sigmas = ["vr_e**2", "distance_e**2", "pm_long_e**2", "pm_lat_e**2"]
				cov = [[sigmas[i] if i == j else "" for i in range(4)] for j in range(4)]
				ds_1.add_virtual_columns_lbrvr_proper_motion2vcartesian(cov_matrix_vr_distance_pm_long_pm_lat=cov, radians=radians)
				N = 100000
				# distance
				l =        np.random.normal(0, 0.1, N) * 0 + l
				b =        np.random.normal(0, 0.1, N) * 0 + b
				distance = np.random.normal(0, 0.1, N)  + distance
				vr =       np.random.normal(0, 0.2, N)  + vr
				pm_l =     np.random.normal(0, 0.3, N)  + pm_l
				pm_b =     np.random.normal(0, 0.4, N)  + pm_b
				ds_many = vx.from_arrays(l=l, b=b, pm_l=pm_l, pm_b=pm_b, vr=vr, distance=distance)
				ds_many.add_virtual_columns_lbrvr_proper_motion2vcartesian(radians=radians)
				return ds_1, ds_many
			ds_1, ds_many = datasets(0, 0, 1, 1, 2, 3)

			vx_e = ds_1.evaluate("vx_uncertainty")[0]
			vy_e = ds_1.evaluate("vy_uncertainty")[0]
			vz_e = ds_1.evaluate("vz_uncertainty")[0]
			self.assertAlmostEqual(vx_e, ds_many.std("vx").item(), delta=0.02)

			self.assertAlmostEqual(vy_e, ds_many.std("vy").item(), delta=0.02)
			self.assertAlmostEqual(vz_e, ds_many.std("vz").item(), delta=0.02)
			self.assertAlmostEqual(vx_e, 0.2,2)
			self.assertAlmostEqual(ds_1.evaluate("vx")[0], 1)
			k = 4.74057
			self.assertAlmostEqual(ds_1.evaluate("vy")[0], k*2)
			self.assertAlmostEqual(ds_1.evaluate("vz")[0], k*3)

		ds = vx.from_scalars(l=90, b=0, pm_l=-1, pm_b=0, distance=1, vr=0)
		ds.add_virtual_columns_lbrvr_proper_motion2vcartesian()
		self.assertAlmostEqual(ds.evaluate("vx")[0], k*1)
		self.assertAlmostEqual(ds.evaluate("vy")[0], 0)
		ds = vx.from_scalars(l=90, b=0, pm_l=-1, pm_b=0, distance=2, vr=0)
		ds.add_virtual_columns_lbrvr_proper_motion2vcartesian()
		self.assertAlmostEqual(ds.evaluate("vx")[0], k*2)
		self.assertAlmostEqual(ds.evaluate("vy")[0], 0)

		ds = vx.from_scalars(l=0, b=90, pm_l=0, pm_b=-1, distance=1, vr=0)
		ds.add_virtual_columns_lbrvr_proper_motion2vcartesian()
		self.assertAlmostEqual(ds.evaluate("vx")[0], k*1)
		self.assertAlmostEqual(ds.evaluate("vy")[0], 0)
		ds = vx.from_scalars(l=0, b=90, pm_l=0, pm_b=-1, distance=2, vr=0)
		ds.add_virtual_columns_lbrvr_proper_motion2vcartesian()
		self.assertAlmostEqual(ds.evaluate("vx")[0], k*2)
		self.assertAlmostEqual(ds.evaluate("vy")[0], 0)

		ds = vx.from_scalars(l=90, b=0, pm_l=0, pm_b=0, distance=1, vr=1)
		ds.add_virtual_columns_lbrvr_proper_motion2vcartesian()
		self.assertAlmostEqual(ds.evaluate("vx")[0], 0)
		self.assertAlmostEqual(ds.evaluate("vy")[0], 1)

		ds = vx.from_scalars(l=90, b=0, pm_l=0, pm_b=0, distance=2, vr=1)
		ds.add_virtual_columns_lbrvr_proper_motion2vcartesian()
		self.assertAlmostEqual(ds.evaluate("vx")[0], 0)
		self.assertAlmostEqual(ds.evaluate("vy")[0], 1)


	def test_strings(self):
		# TODO: concatenated datasets with strings of different length
		self.assertEqual(["x", "y", "f"], self.dataset.get_column_names())

		names = ["x", "y", "f", "name"]
		self.assertEqual(names, self.dataset.get_column_names(strings=True))

		if self.dataset.is_local():
			# check if strings are exported
			path_hdf5 = tempfile.mktemp(".hdf5")
			self.dataset.export_hdf5(path_hdf5, virtual=False)

			exported_dataset = vx.open(path_hdf5)
			self.assertEqual(names, exported_dataset.get_column_names(strings=True))

			path_fits = tempfile.mktemp(".fits")
			self.dataset.export_fits(path_fits, virtual=False)

			exported_dataset = vx.open(path_fits)
			self.assertEqual(names, exported_dataset.get_column_names(strings=True))

			path_fits_astropy = tempfile.mktemp(".fits")
			with astropy.io.fits.open(path_fits) as fitsfile:
				# make sure astropy can read the data
				bla = fitsfile[1].data
				try:
					fitsfile.writeto(path_fits_astropy)
				finally:
					os.remove(path_fits_astropy)


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

	def test_subspace_basics(self):
		self.assertIsNotNone(repr(self.dataset("x")))
		self.assertIsNotNone(repr(self.dataset("x", "y")))
		self.assertIsNotNone(repr(self.dataset("x", "y", "z")))

		subspace = self.dataset("x", "y")
		for i in range(len(self.dataset)):
			self.assertEqual(subspace.row(0).tolist(), [self.x[0], self.y[0]])

		self.assertEqual(self.dataset.subspace("x", "y").expressions, self.dataset("x", "y").expressions)

	def test_subspaces(self):
		dataset = vaex.from_arrays("arrays", x=np.array([1]), y=np.array([2]), z=np.array([3]))
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
		dataset = from_scalars(alpha=0, delta=0, distance=1)
		dataset.add_virtual_columns_spherical_to_cartesian("alpha", "delta", "distance", "x", "y", "z", radians=False)

		subspace = dataset("x", "y", "z")
		x, y, z = subspace.sum()

		self.assertAlmostEqual(x, 1)
		self.assertAlmostEqual(y, 0)
		self.assertAlmostEqual(z, 0)

		for radians in [True, False]:
			def datasets(alpha, delta, distance, radians=radians):
				ds_1 = from_scalars(alpha=alpha, delta=delta, distance=distance, alpha_e=0.1, delta_e=0.2, distance_e=0.3)
				sigmas = ["alpha_e**2", "delta_e**2", "distance_e**2"]
				cov = [[sigmas[i] if i == j else "" for i in range(3)] for j in range(3)]
				ds_1.add_virtual_columns_spherical_to_cartesian("alpha", "delta", "distance", cov_matrix_alpha_delta_distance=cov, radians=radians)
				N = 1000000
				# distance
				alpha =        np.random.normal(0, 0.1, N) + alpha
				delta =        np.random.normal(0, 0.2, N) + delta
				distance =     np.random.normal(0, 0.3, N) + distance
				ds_many = vx.from_arrays(alpha=alpha, delta=delta, distance=distance)
				ds_many.add_virtual_columns_spherical_to_cartesian("alpha", "delta", "distance", radians=radians)
				return ds_1, ds_many

			ds_1, ds_many = datasets(0, 0, 1.)
			x_e = ds_1.evaluate("x_uncertainty")[0]
			y_e = ds_1.evaluate("y_uncertainty")[0]
			z_e = ds_1.evaluate("z_uncertainty")[0]
			self.assertAlmostEqual(x_e, ds_many.std("x").item(), delta=0.02)

			self.assertAlmostEqual(y_e, ds_many.std("y").item(), delta=0.02)
			self.assertAlmostEqual(z_e, ds_many.std("z").item(), delta=0.02)
			self.assertAlmostEqual(x_e, 0.3)

		# TODO: from cartesian tot spherical errors


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

	def test_sum_old(self):
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


	def test_count(self):
		self.dataset.select("x < 5")
		np.testing.assert_array_almost_equal(self.dataset.count("x", selection=None), 10)
		np.testing.assert_array_almost_equal(self.dataset.count("x", selection=True), 5)

		# convert to float
		self.dataset_local.columns["x"] = self.dataset_local.columns["x"] * 1.
		self.dataset_local.columns["x"][0] = np.nan
		self.dataset.select("x < 5")
		np.testing.assert_array_almost_equal(self.dataset.count("x", selection=None), 9)
		np.testing.assert_array_almost_equal(self.dataset.count("x", selection=True), 4)
		np.testing.assert_array_almost_equal(self.dataset.count("y", selection=None), 10)
		np.testing.assert_array_almost_equal(self.dataset.count("y", selection=True), 4)
		np.testing.assert_array_almost_equal(self.dataset.count(selection=None), 10)
		# we modified the data.. so actually this should be 4..
		np.testing.assert_array_almost_equal(self.dataset.count(selection=True), 4)
		np.testing.assert_array_almost_equal(self.dataset.count("*", selection=None), 10)
		np.testing.assert_array_almost_equal(self.dataset.count("*", selection=True), 4)

		task = self.dataset.count("x", selection=True, async=True)
		self.dataset.executor.execute()
		np.testing.assert_array_almost_equal(task.get(), 4)


		np.testing.assert_array_almost_equal(self.dataset.count("x", selection=None, binby=["x"], limits=[0, 10], shape=1), [9])
		np.testing.assert_array_almost_equal(self.dataset.count("x", selection=True, binby=["x"], limits=[0, 10], shape=1), [4])
		np.testing.assert_array_almost_equal(self.dataset.count("*", selection=None, binby=["x"], limits=[0, 10], shape=1), [9])
		np.testing.assert_array_almost_equal(self.dataset.count("*", selection=True, binby=["x"], limits=[0, 10], shape=1), [4])
		np.testing.assert_array_almost_equal(self.dataset.count("*", selection=None, binby=["y"], limits=[0, 9**2+1], shape=1), [10])
		np.testing.assert_array_almost_equal(self.dataset.count("*", selection=True, binby=["y"], limits=[0, 9**2+1], shape=1), [4])
		np.testing.assert_array_almost_equal(self.dataset.count("x", selection=None, binby=["y"], limits=[0, 9**2+1], shape=1), [9])
		np.testing.assert_array_almost_equal(self.dataset.count("x", selection=True, binby=["y"], limits=[0, 9**2+1], shape=1), [4])

		np.testing.assert_array_almost_equal(self.dataset.count("x", selection=None, binby=["x"], limits=[0, 10], shape=2), [4, 5])
		np.testing.assert_array_almost_equal(self.dataset.count("x", selection=True, binby=["x"], limits=[0, 10], shape=2), [4, 0])

		ds = self.dataset
		a = ds.count("x", binby="y", limits=[0, 100], shape=2)
		ds.select("(y >= 0) & (y < 50)")
		b = ds.count("x", selection=True)
		ds.select("(y >= 50) & (y < 100)")
		c = ds.count("x", selection=True)
		np.testing.assert_array_almost_equal(a, [b, c])

	def test_sum(self):
		self.dataset.select("x < 5")
		np.testing.assert_array_almost_equal(self.dataset.sum("x", selection=None), np.nansum(self.x))
		np.testing.assert_array_almost_equal(self.dataset.sum("x", selection=True), np.nansum(self.x[:5]))

		# convert to float
		x = self.dataset_local.columns["x"] = self.dataset_local.columns["x"] * 1.
		y = self.y
		self.dataset_local.columns["x"][0] = np.nan
		np.testing.assert_array_almost_equal(self.dataset.sum("x", selection=None), np.nansum(x))
		np.testing.assert_array_almost_equal(self.dataset.sum("x", selection=True), np.nansum(x[:5]))

		task = self.dataset.sum("x", selection=True, async=True)
		self.dataset.executor.execute()
		np.testing.assert_array_almost_equal(task.get(), np.nansum(x[:5]))


		np.testing.assert_array_almost_equal(self.dataset.sum("x", selection=None, binby=["x"], limits=[0, 10], shape=1), [np.nansum(x)])
		np.testing.assert_array_almost_equal(self.dataset.sum("x", selection=True, binby=["x"], limits=[0, 10], shape=1), [np.nansum(x[:5])])
		np.testing.assert_array_almost_equal(self.dataset.sum("x", selection=None, binby=["y"], limits=[0, 9**2+1], shape=1), [np.nansum(x)])
		np.testing.assert_array_almost_equal(self.dataset.sum("x", selection=True, binby=["y"], limits=[0, 9**2+1], shape=1), [np.nansum(x[:5])])

		np.testing.assert_array_almost_equal(self.dataset.sum("x", selection=None, binby=["x"], limits=[0, 10], shape=2), [np.nansum(x[:5]), np.nansum(x[5:])])
		np.testing.assert_array_almost_equal(self.dataset.sum("x", selection=True, binby=["x"], limits=[0, 10], shape=2), [np.nansum(x[:5]), 0])

		i = 7
		np.testing.assert_array_almost_equal(self.dataset.sum("x", selection=None, binby=["y"], limits=[0, 9**2+1], shape=2), [np.nansum(x[:i]), np.nansum(x[i:])])
		np.testing.assert_array_almost_equal(self.dataset.sum("x", selection=True, binby=["y"], limits=[0, 9**2+1], shape=2), [np.nansum(x[:5]), 0])

		i = 5
		np.testing.assert_array_almost_equal(self.dataset.sum("y", selection=None, binby=["x"], limits=[0, 10], shape=2), [np.nansum(y[:i]), np.nansum(y[i:])])
		np.testing.assert_array_almost_equal(self.dataset.sum("y", selection=True, binby=["x"], limits=[0, 10], shape=2), [np.nansum(y[:5]), 0])

	def test_cov(self):
		# convert to float
		x = self.dataset_local.columns["x"] = self.dataset_local.columns["x"] * 1.
		y = self.y
		def cov(*args):
			return np.cov(args, bias=1)
		self.dataset.select("x < 5")


		np.testing.assert_array_almost_equal(self.dataset.cov("x", "y", selection=None), cov(x, y))
		np.testing.assert_array_almost_equal(self.dataset.cov("x", "y", selection=True), cov(x[:5], y[:5]))

		#self.dataset.columns["x"][0] = np.nan
		np.testing.assert_array_almost_equal(self.dataset.cov("x", "y", selection=None), cov(x, y))
		np.testing.assert_array_almost_equal(self.dataset.cov("x", "y", selection=True), cov(x[:5], y[:5]))

		task = self.dataset.cov("x", "y", selection=True, async=True)
		self.dataset.executor.execute()
		np.testing.assert_array_almost_equal(task.get(), cov(x[:5], y[:5]))


		np.testing.assert_array_almost_equal(self.dataset.cov("x", "y", selection=None, binby=["x"], limits=[0, 10], shape=1), [cov(x, y)])
		np.testing.assert_array_almost_equal(self.dataset.cov("x", "y", selection=True, binby=["x"], limits=[0, 10], shape=1), [cov(x[:5], y[:5])])
		np.testing.assert_array_almost_equal(self.dataset.cov("x", "y", selection=None, binby=["y"], limits=[0, 9**2+1], shape=1), [cov(x, y)])
		np.testing.assert_array_almost_equal(self.dataset.cov("x", "y", selection=True, binby=["y"], limits=[0, 9**2+1], shape=1), [cov(x[:5], y[:5])])

		nan22 = [[np.nan, np.nan], [np.nan, np.nan]]
		np.testing.assert_array_almost_equal(self.dataset.cov("x", "y", selection=None, binby=["x"], limits=[0, 10], shape=2), [cov(x[:5], y[:5]), cov(x[5:], y[5:])])
		np.testing.assert_array_almost_equal(self.dataset.cov("x", "y", selection=True, binby=["x"], limits=[0, 10], shape=2), [cov(x[:5], y[:5]), nan22])

		i = 7
		np.testing.assert_array_almost_equal(self.dataset.cov("x", "y", selection=None, binby=["y"], limits=[0, 9**2+1], shape=2), [cov(x[:i], y[:i]), cov(x[i:], y[i:])])
		np.testing.assert_array_almost_equal(self.dataset.cov("x", "y", selection=True, binby=["y"], limits=[0, 9**2+1], shape=2), [cov(x[:5], y[:5]), nan22])

		i = 5
		np.testing.assert_array_almost_equal(self.dataset.cov("x", "y", selection=None, binby=["x"], limits=[0, 10], shape=2), [cov(x[:i], y[:i]), cov(x[i:], y[i:])])
		np.testing.assert_array_almost_equal(self.dataset.cov("x", "y", selection=True, binby=["x"], limits=[0, 10], shape=2), [cov(x[:i], y[:i]), nan22])

		# include 3rd varialble
		self.dataset.add_virtual_column("z", "x*y")
		z = self.dataset.evaluate("z")
		np.testing.assert_array_almost_equal(self.dataset.cov(["x", "y", "z"], selection=None), cov(x, y, z))

		nan33 = [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]
		np.testing.assert_array_almost_equal(self.dataset.cov(["x", "y", "z"], selection=None, binby=["x"], limits=[0, 10], shape=2), [cov(x[:5], y[:5], z[:5]), cov(x[5:], y[5:], z[5:])])
		np.testing.assert_array_almost_equal(self.dataset.cov(["x", "y", "z"], selection=True, binby=["x"], limits=[0, 10], shape=2), [cov(x[:5], y[:5], z[:5]), nan33])

		i = 7
		np.testing.assert_array_almost_equal(self.dataset.cov(["x", "y", "z"], selection=None, binby=["y"], limits=[0, 9**2+1], shape=2), [cov(x[:i], y[:i], z[:i]), cov(x[i:], y[i:], z[i:])])
		np.testing.assert_array_almost_equal(self.dataset.cov(["x", "y", "z"], selection=True, binby=["y"], limits=[0, 9**2+1], shape=2), [cov(x[:5], y[:5], z[:5]), nan33])

		i = 5
		np.testing.assert_array_almost_equal(self.dataset.cov(["x", "y", "z"], selection=None, binby=["x"], limits=[0, 10], shape=2), [cov(x[:i], y[:i], z[:i]), cov(x[i:], y[i:], z[i:])])
		np.testing.assert_array_almost_equal(self.dataset.cov(["x", "y", "z"], selection=True, binby=["x"], limits=[0, 10], shape=2), [cov(x[:i], y[:i], z[:i]), nan33])

	def test_correlation(self):
		# convert to float
		x = self.dataset_local.columns["x"] = self.dataset_local.columns["x"] * 1.
		y = self.y
		def correlation(x, y):
			c = np.cov([x, y], bias=1)
			return c[0,1] / (c[0,0] * c[1,1])**0.5

		np.testing.assert_array_almost_equal(self.dataset.correlation([["x", "y"], ["x", "x**2"]], selection=None), [correlation(x, y), correlation(x, x**2)])
		return

		self.dataset.select("x < 5")
		np.testing.assert_array_almost_equal(self.dataset.correlation("x", "y", selection=None), correlation(x, y))
		np.testing.assert_array_almost_equal(self.dataset.correlation("x", "y", selection=True), correlation(x[:5], y[:5]))

		#self.dataset.columns["x"][0] = np.nan
		np.testing.assert_array_almost_equal(self.dataset.correlation("x", "y", selection=None), correlation(x, y))
		np.testing.assert_array_almost_equal(self.dataset.correlation("x", "y", selection=True), correlation(x[:5], y[:5]))

		task = self.dataset.correlation("x", "y", selection=True, async=True)
		self.dataset.executor.execute()
		np.testing.assert_array_almost_equal(task.get(), correlation(x[:5], y[:5]))


		np.testing.assert_array_almost_equal(self.dataset.correlation("x", "y", selection=None, binby=["x"], limits=[0, 10], shape=1), [correlation(x, y)])
		np.testing.assert_array_almost_equal(self.dataset.correlation("x", "y", selection=True, binby=["x"], limits=[0, 10], shape=1), [correlation(x[:5], y[:5])])
		np.testing.assert_array_almost_equal(self.dataset.correlation("x", "y", selection=None, binby=["y"], limits=[0, 9**2+1], shape=1), [correlation(x, y)])
		np.testing.assert_array_almost_equal(self.dataset.correlation("x", "y", selection=True, binby=["y"], limits=[0, 9**2+1], shape=1), [correlation(x[:5], y[:5])])

		np.testing.assert_array_almost_equal(self.dataset.correlation("x", "y", selection=None, binby=["x"], limits=[0, 10], shape=2), [correlation(x[:5], y[:5]), correlation(x[5:], y[5:])])
		np.testing.assert_array_almost_equal(self.dataset.correlation("x", "y", selection=True, binby=["x"], limits=[0, 10], shape=2), [correlation(x[:5], y[:5]), np.nan])

		i = 7
		np.testing.assert_array_almost_equal(self.dataset.correlation("x", "y", selection=None, binby=["y"], limits=[0, 9**2+1], shape=2), [correlation(x[:i], y[:i]), correlation(x[i:], y[i:])])
		np.testing.assert_array_almost_equal(self.dataset.correlation("x", "y", selection=True, binby=["y"], limits=[0, 9**2+1], shape=2), [correlation(x[:5], y[:5]), np.nan])

		i = 5
		np.testing.assert_array_almost_equal(self.dataset.correlation("x", "y", selection=None, binby=["x"], limits=[0, 10], shape=2), [correlation(x[:i], y[:i]), correlation(x[i:], y[i:])])
		np.testing.assert_array_almost_equal(self.dataset.correlation("x", "y", selection=True, binby=["x"], limits=[0, 10], shape=2), [correlation(x[:i], y[:i]), np.nan])

		np.testing.assert_array_almost_equal(self.dataset.correlation("x", "y", selection=True, binby=["x"], limits=[[0, 10]], shape=2), [correlation(x[:i], y[:i]), np.nan])

		self.assertGreater(self.dataset.correlation("x", "y", selection=None, binby=["x"], shape=1), 0)

		self.assertGreater(self.dataset.correlation("x", "y", selection=None, binby=["x"], limits="90%", shape=1), 0)
		self.assertGreater(self.dataset.correlation("x", "y", selection=None, binby=["x"], limits=["90%"], shape=1), 0)
		self.assertGreater(self.dataset.correlation("x", "y", selection=None, binby=["x"], limits="minmax", shape=1), 0)

	def test_covar(self):
		# convert to float
		x = self.dataset_local.columns["x"] = self.dataset_local.columns["x"] * 1.
		y = self.y
		def covar(x, y):
			mask = np.isfinite(x * y)
			#w = np.isfinite(x * y) * 1.0
			x = x[mask]
			y = y[mask]
			return np.cov([x, y], bias=1)[1,0]
		self.dataset.select("x < 5")
		np.testing.assert_array_almost_equal(self.dataset.covar("x", "y", selection=None), covar(x, y))
		np.testing.assert_array_almost_equal(self.dataset.covar("x", "y", selection=True), covar(x[:5], y[:5]))

		#self.dataset.columns["x"][0] = np.nan
		np.testing.assert_array_almost_equal(self.dataset.covar("x", "y", selection=None), covar(x, y))
		np.testing.assert_array_almost_equal(self.dataset.covar("x", "y", selection=True), covar(x[:5], y[:5]))

		task = self.dataset.covar("x", "y", selection=True, async=True)
		self.dataset.executor.execute()
		np.testing.assert_array_almost_equal(task.get(), covar(x[:5], y[:5]))


		np.testing.assert_array_almost_equal(self.dataset.covar("x", "y", selection=None, binby=["x"], limits=[0, 10], shape=1), [covar(x, y)])
		np.testing.assert_array_almost_equal(self.dataset.covar("x", "y", selection=True, binby=["x"], limits=[0, 10], shape=1), [covar(x[:5], y[:5])])
		np.testing.assert_array_almost_equal(self.dataset.covar("x", "y", selection=None, binby=["y"], limits=[0, 9**2+1], shape=1), [covar(x, y)])
		np.testing.assert_array_almost_equal(self.dataset.covar("x", "y", selection=True, binby=["y"], limits=[0, 9**2+1], shape=1), [covar(x[:5], y[:5])])

		np.testing.assert_array_almost_equal(self.dataset.covar("x", "y", selection=None, binby=["x"], limits=[0, 10], shape=2), [covar(x[:5], y[:5]), covar(x[5:], y[5:])])
		np.testing.assert_array_almost_equal(self.dataset.covar("x", "y", selection=True, binby=["x"], limits=[0, 10], shape=2), [covar(x[:5], y[:5]), np.nan])

		i = 7
		np.testing.assert_array_almost_equal(self.dataset.covar("x", "y", selection=None, binby=["y"], limits=[0, 9**2+1], shape=2), [covar(x[:i], y[:i]), covar(x[i:], y[i:])])
		np.testing.assert_array_almost_equal(self.dataset.covar("x", "y", selection=True, binby=["y"], limits=[0, 9**2+1], shape=2), [covar(x[:5], y[:5]), np.nan])

		i = 5
		np.testing.assert_array_almost_equal(self.dataset.covar("x", "y", selection=None, binby=["x"], limits=[0, 10], shape=2), [covar(x[:i], y[:i]), covar(x[i:], y[i:])])
		np.testing.assert_array_almost_equal(self.dataset.covar("x", "y", selection=True, binby=["x"], limits=[0, 10], shape=2), [covar(x[:i], y[:i]), np.nan])


	def t_est_percentile(self):
		self.dataset.select("x < 5")
		np.testing.assert_array_almost_equal(self.dataset.percentile("x", selection=None, percentile_limits="minmax"), np.median(self.x))
		return
		np.testing.assert_array_almost_equal(self.dataset.percentile("x", selection=True), np.median(self.x[:5]))

		# convert to float
		x = self.dataset.columns["x"] = self.dataset.columns["x"] * 1.
		y = self.y
		self.dataset.columns["x"][0] = np.nan
		np.testing.assert_array_almost_equal(self.dataset.sum("x", selection=None), np.nansum(x))
		np.testing.assert_array_almost_equal(self.dataset.sum("x", selection=True), np.nansum(x[:5]))

		task = self.dataset.sum("x", selection=True, async=True)
		self.dataset.executor.execute()
		np.testing.assert_array_almost_equal(task.get(), np.nansum(x[:5]))


		np.testing.assert_array_almost_equal(self.dataset.sum("x", selection=None, binby=["x"], limits=[0, 10], shape=1), [np.nansum(x)])
		np.testing.assert_array_almost_equal(self.dataset.sum("x", selection=True, binby=["x"], limits=[0, 10], shape=1), [np.nansum(x[:5])])
		np.testing.assert_array_almost_equal(self.dataset.sum("x", selection=None, binby=["y"], limits=[0, 9**2+1], shape=1), [np.nansum(x)])
		np.testing.assert_array_almost_equal(self.dataset.sum("x", selection=True, binby=["y"], limits=[0, 9**2+1], shape=1), [np.nansum(x[:5])])

		np.testing.assert_array_almost_equal(self.dataset.sum("x", selection=None, binby=["x"], limits=[0, 10], shape=2), [np.nansum(x[:5]), np.nansum(x[5:])])
		np.testing.assert_array_almost_equal(self.dataset.sum("x", selection=True, binby=["x"], limits=[0, 10], shape=2), [np.nansum(x[:5]), 0])

		i = 7
		np.testing.assert_array_almost_equal(self.dataset.sum("x", selection=None, binby=["y"], limits=[0, 9**2+1], shape=2), [np.nansum(x[:i]), np.nansum(x[i:])])
		np.testing.assert_array_almost_equal(self.dataset.sum("x", selection=True, binby=["y"], limits=[0, 9**2+1], shape=2), [np.nansum(x[:5]), 0])

		i = 5
		np.testing.assert_array_almost_equal(self.dataset.sum("y", selection=None, binby=["x"], limits=[0, 10], shape=2), [np.nansum(y[:i]), np.nansum(y[i:])])
		np.testing.assert_array_almost_equal(self.dataset.sum("y", selection=True, binby=["x"], limits=[0, 10], shape=2), [np.nansum(y[:5]), 0])

	def test_mean(self):
		x, y = self.datasetxy("x", "y").mean()
		self.assertAlmostEqual(x, 0.5)
		self.assertAlmostEqual(y, 0)

		self.datasetxy.select("x < 1")
		x, y = self.datasetxy("x", "y").selected().mean()
		self.assertAlmostEqual(x, 0)
		self.assertAlmostEqual(y, -1)


		np.testing.assert_array_almost_equal(self.datasetxy.mean(["x", "y"], selection=None), [0.5, 0])
		np.testing.assert_array_almost_equal(self.datasetxy.mean(["x", "y"], selection=True), [0, -1])

	def test_minmax(self):
		((xmin, xmax), ) = self.dataset("x").minmax()
		self.assertAlmostEqual(xmin, 0)
		self.assertAlmostEqual(xmax, 9)

		np.testing.assert_array_almost_equal(self.dataset.minmax("x"), [0, 9.])
		np.testing.assert_array_almost_equal(self.dataset.minmax("y"), [0, 9.**2])
		np.testing.assert_array_almost_equal(self.dataset.minmax(["x", "y"]), [[0, 9.], [0, 9.**2]])

		self.dataset.select("x < 5")
		((xmin2, xmax2), ) = self.dataset("x").selected().minmax()
		self.assertAlmostEqual(xmin2, 0)
		self.assertAlmostEqual(xmax2, 4)

		np.testing.assert_array_almost_equal(self.dataset.minmax("x", selection=True), [0, 4])
		np.testing.assert_array_almost_equal(self.dataset.minmax("y", selection=True), [0, 4**2])
		np.testing.assert_array_almost_equal(self.dataset.minmax(["x", "y"], selection=True), [[0, 4], [0, 4**2]])

		task = self.dataset.minmax("x", selection=True, async=True)
		self.dataset.executor.execute()
		np.testing.assert_array_almost_equal(task.get(), [0, 4])


		np.testing.assert_array_almost_equal(self.dataset.minmax("x", selection=None, binby=["x"], limits="minmax", shape=1), [[0, 8]])
		np.testing.assert_array_almost_equal(self.dataset.minmax("x", selection=True, binby=["x"], limits="minmax", shape=1), [[0, 3]])

		np.testing.assert_array_almost_equal(self.dataset.minmax("x", selection=None, binby=["x"], limits="minmax", shape=2), [[0, 4], [5, 8]])
		np.testing.assert_array_almost_equal(self.dataset.minmax("x", selection=True, binby=["x"], limits="minmax", shape=2), [[0, 1], [2, 3]])

	def test_var_and_std(self):
		# subspaces var uses non-central
		x, y = self.datasetxy("x", "y").var()
		self.assertAlmostEqual(x, 0.5)
		self.assertAlmostEqual(y, 1.)

		# newstyle var uses central
		self.assertAlmostEqual(self.datasetxy.var("x"), 0.5**2)
		self.assertAlmostEqual(self.datasetxy.var("y"), 1.)
		self.assertAlmostEqual(self.datasetxy.std("x"), 0.5)
		self.assertAlmostEqual(self.datasetxy.std("y"), 1.)


		x, y = self.dataset("x", "y").var()
		self.assertAlmostEqual(x, np.mean(self.x**2))
		self.assertAlmostEqual(y, np.mean(self.y**2))

		x, y = self.dataset.var(["x", "y"])
		self.assertAlmostEqual(x, np.var(self.x))
		self.assertAlmostEqual(y, np.var(self.y))
		x, y = self.dataset.std(["x", "y"])
		self.assertAlmostEqual(x, np.std(self.x))
		self.assertAlmostEqual(y, np.std(self.y))

		self.dataset.select("x < 5")
		x, y = self.dataset("x", "y").selected().var()
		self.assertAlmostEqual(x, np.mean(self.x[:5]**2))
		self.assertAlmostEqual(y, np.mean(self.y[:5]**2))

		x, y = self.dataset.var(["x", "y"], selection=True)
		self.assertAlmostEqual(x, np.var(self.x[:5]))
		self.assertAlmostEqual(y, np.var(self.y[:5]))

		x, y = self.dataset.std(["x", "y"], selection=True)
		self.assertAlmostEqual(x, np.std(self.x[:5]))
		self.assertAlmostEqual(y, np.std(self.y[:5]))


	def test_correlation_old(self):

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

	def test_limits(self):
		np.testing.assert_array_almost_equal(self.dataset.limits("x", "minmax"), self.dataset.minmax("x"))
		np.testing.assert_array_almost_equal(self.dataset.limits("x"), self.dataset.limits_percentage("x"))
		np.testing.assert_array_almost_equal(self.dataset.limits(["x", "y"], "minmax"), self.dataset.minmax(["x", "y"]))
		np.testing.assert_array_almost_equal(self.dataset.limits(["x", "y"], ["minmax", "minmax"]), self.dataset.minmax(["x", "y"]))

		np.testing.assert_array_almost_equal(self.dataset.limits("x", [0, 10]), [0, 10])

		np.testing.assert_array_almost_equal(self.dataset.limits("x", "90%"), self.dataset.limits_percentage("x", 90.))
		np.testing.assert_array_almost_equal(self.dataset.limits([["x", "y"], ["x", "z"]], "minmax"),\
										 [self.dataset.minmax(["x", "y"]), self.dataset.minmax(["x", "z"])])
		np.testing.assert_array_almost_equal(
			self.dataset.limits( [["x", "y"], ["x", "z"]], [[[0, 10], [0, 20]], "minmax"]),\
											 [[[0, 10], [0, 20]], self.dataset.minmax(["x", "z"])])

		#np.testing.assert_array_almost_equal(self.dataset.limits(["x"], [0, 10]), [[0, 10]])
		if 0:
			#print(">>>>>", self.dataset.limits("x", "minmax"), self.dataset.minmax("x"))
			print(">>>>>", self.dataset.limits(["x", "y"], ["minmax", "minmax"]), self.dataset.minmax(["x", "y"]))







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
								for virtual in [False, True]:
									for export in [dataset.export_fits, dataset.export_hdf5] if byteorder == ">" else [dataset.export_hdf5]:
										#print (">>>", dataset, path, column_names, byteorder, shuffle, selection, fraction, dataset.full_length(), virtual)
										#print dataset.full_length()
										#print len(dataset)
										if export == dataset.export_hdf5:
											path = path_hdf5
											export(path, column_names=column_names, byteorder=byteorder, shuffle=shuffle, selection=selection, progress=False)
										else:
											path = path_fits
											export(path, column_names=column_names, shuffle=shuffle, selection=selection, progress=False, virtual=virtual)
											with astropy.io.fits.open(path) as fitsfile:
												# make sure astropy can read the data
												bla = fitsfile[1].data
												try:
													fitsfile.writeto(path_fits_astropy)
												finally:
													os.remove(path_fits_astropy)
										compare = vx.open(path)
										if column_names is None:
											column_names = ["x", "y", "f", "z", "name"] if virtual else ["x", "y", "f", "name"]
										#if not virtual:
										#	if "z" in column_names:
										#		column_names.remove("z")
										# TODO: does the order matter?
										self.assertEqual(sorted(compare.get_column_names(strings=True)), sorted(column_names + (["random_index"] if shuffle else [])))
										for column_name in column_names:
											values = dataset.evaluate(column_name)
											if selection:
												mask = dataset.evaluate_selection_mask(selection, 0, len(dataset))
												self.assertEqual(sorted(compare.columns[column_name]), sorted(values[mask]))
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
		self.dataset.select("x <= 5", name="inverse")


		counts = self.dataset.count("x", selection=["default", "inverse", "x > 5", "default | inverse"])
		np.testing.assert_array_almost_equal(counts, [4, 6, 4, 10])


		self.dataset.select("x <= 1", name="inverse", mode="subtract")
		counts = self.dataset.count("x", selection=["default", "inverse"])
		np.testing.assert_array_almost_equal(counts, [4, 4])

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

	def test_selection_in_handler(self):
		self.dataset.select("x > 5")
		# in the handler, we should know there is not selection
		def check(*ignore):
			self.assertFalse(self.dataset.has_selection())
		self.dataset.signal_selection_changed.connect(check)
		self.dataset.select_nothing()

	def test_favorite_selections(self):
		self.dataset.select("x > 5")
		total_subset = self.dataset("x").selected().sum()
		self.dataset.selection_favorite_add("test")
		self.dataset.select_nothing()
		with self.assertRaises(ValueError):
			self.dataset.selection_favorite_add("test")
		self.dataset.selections_favorite_load()
		self.dataset.selection_favorite_apply("test")
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

#class A:#class estDatasetRemote(TestDataset):
#class TestDatasetRemote(TestDataset):
class A:
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
		self.webserver = vaex.webserver.WebServer(datasets=datasets, port=test_port, cache_byte_size=0)
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
		self.server.close()
		self.webserver.stop_serving()

	def test_amuse(self):
		pass # no need

	def test_export(self):
		pass # we can't export atm

	def test_concat(self):
		pass # doesn't make sense to test this for remote

	def test_data_access(self):
		pass

	def test_byte_size(self):
		pass # we don't know the selection's length for dataset remote..

	def test_selection(self):
		pass

	#def test_count(self):
	#	pass
	#def test_sum(self):
	#	pass
	#def test_cov(self):
	#	pass
	#def test_correlation(self):
	#	pass
	#def test_covar(self):
	#	pass
	#def test_mean(self):
	#	pass
	#def test_minmax(self):
	#	pass
	#def test_var_and_std(self):
	#	pass
	#def test_limits(self):
	#	pass

import vaex.distributed
class A:#class T_estDatasetDistributed(unittest.TestCase):
	use_websocket = False
	def setUp(self):
		global test_port
		self.dataset_local = self.dataset = dataset.DatasetArrays("dataset")

		self.x = x = np.arange(10)
		self.y = y = x ** 2
		self.dataset.add_column("x", x)
		self.dataset.add_column("y", y)
		datasets = [self.dataset]
		self.webserver1 = vaex.webserver.WebServer(datasets=datasets, port=test_port)
		self.webserver1.serve_threaded()
		test_port += 1
		self.webserver2 = vaex.webserver.WebServer(datasets=datasets, port=test_port)
		self.webserver2.serve_threaded()
		test_port += 1

		scheme = "ws" if self.use_websocket else "http"
		self.server1 = vx.server("%s://localhost:%d" % (scheme, test_port-2))
		self.server2 = vx.server("%s://localhost:%d" % (scheme, test_port-1))
		test_port += 1
		datasets1 = self.server1.datasets(as_dict=True)
		datasets2 = self.server2.datasets(as_dict=True)
		self.datasets = [datasets1["dataset"], datasets2["dataset"]]
		self.dataset = vaex.distributed.DatasetDistributed(self.datasets)



	def tearDown(self):
		#TestDataset.tearDown(self)
		#print "stop serving"
		self.webserver1.stop_serving()
		self.webserver2.stop_serving()

	def test_histogram(self):
		#print self.dataset, self.dataset.__call__
		#print self.dataset.subspace("x")
		#self.dataset_local.set_active_range(5, 10)
		counts = self.dataset("x").histogram([[0,10]], size=10)
		#import pdb
		#pdb.set_trace()
		self.assertTrue(all(counts == 1), "counts is %r" % counts)
		return

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
		return

#class TestDatasetRemotePlain(TestDatasetRemote):
#	use_websocket = False
"""



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

