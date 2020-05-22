# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import vaex.dataset as dataset
import numpy as np
import unittest
import vaex as vx
import tempfile
import vaex.server.tornado_server
import astropy.io.fits
import astropy.units
import pandas as pd
import vaex.execution
import contextlib
a = vaex.execution.buffer_size_default # will crash if we decide to rename it

basedir = os.path.dirname(__file__)
# this will make the test execute more code and may show up bugs
#vaex.execution.buffer_size_default = 3


@contextlib.contextmanager
def small_buffer(ds, size=3):
	if ds.is_local():
		previous = ds.executor.buffer_size
		ds.executor.buffer_size = size
		ds._invalidate_caches()
		try:
			yield
		finally:
			ds.executor.buffer_size = previous
	else:
		yield # for remote dfs we don't support this ... or should we?


# these need to be global for pickling
def function_upper(x):
	return np.array(x.upper())
import vaex.serialize
@vaex.serialize.register
class Multiply:
	def __init__(self, scale=0): self.scale = scale
	@classmethod
	def state_from(cls, state, trusted=True):
		return cls(scale=state)
	def state_get(self): return self.scale
	def __call__(self, x): return x * self.scale


vx.set_log_level_exception()
#vx.set_log_level_off()
#vx.set_log_level_debug()

def from_scalars(**kwargs):
	return vx.from_arrays(**{k:np.array([v]) for k, v in kwargs.items()})

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


		# x is non-c
		# same as np.arange(10, dtype=">f8")., but with strides == 16, instead of 8
		use_filtering = True
		if use_filtering:
			self.zero_index = 2
			self.x = x = np.arange(-2, 40, dtype=">f8").reshape((-1,21)).T.copy()[:,0]
			self.y = y = x ** 2
			self.ints = np.arange(-2,19, dtype="i8")
			self.ints[1] = 2**62+1
			self.ints[2] = -2**62+1
			self.ints[3] = -2**62-1
			self.ints[1+10] = 2**62+1
			self.ints[2+10] = -2**62+1
			self.ints[3+10] = -2**62-1
			self.dataset.add_column("x", x)
			self.dataset.add_column("y", y)
			mo = m = np.arange(-2, 40, dtype=">f8").reshape((-1,21)).T.copy()[:,0]
			ma_value = 77777
			m[-1+10+2] = ma_value
			m[-1+20] = ma_value
		else:
			self.x = x = np.arange(20, dtype=">f8").reshape((-1,10)).T.copy()[:,0]
			self.y = y = x ** 2
			self.ints = np.arange(10, dtype="i8")
			self.ints[0] = 2**62+1
			self.ints[1] = -2**62+1
			self.ints[2] = -2**62-1
			self.dataset.add_column("x", x)
			self.dataset.add_column("y", y)
			m = x.copy()
			ma_value = 77777
			m[-1] = ma_value
		self.m = m = np.ma.array(m, mask=m==ma_value)
		self.mi = mi = np.ma.array(m.data.astype(np.int64), mask=m.data==ma_value, fill_value=88888)
		self.dataset.add_column("m", m)
		self.dataset.add_column("mi", mi)
		self.dataset.add_column("ints", self.ints)
		self.dataset.set_variable("t", 1.)
		self.dataset.add_virtual_column("z", "x+t*y")
		self.dataset.units["x"] = astropy.units.Unit("km")
		self.dataset.units["y"] = astropy.units.Unit("km/s")
		self.dataset.units["t"] = astropy.units.Unit("s")
		self.dataset.add_column("f", np.arange(len(self.dataset), dtype=np.float64))
		self.dataset.ucds["x"] = "some;ucd"


		name = np.array(list(map(lambda x: str(x) + "bla" + ('_' * int(x)), self.x)), dtype='U') #, dtype=np.string_)
		self.names = self.dataset.get_column_names()
		self.dataset.add_column("name", np.array(name))
		self.dataset.add_column("name_arrow", vaex.string_column(name))
		if use_filtering:
			self.dataset.select('(x >= 0) & (x < 10)', name=vaex.dataset.FILTER_SELECTION_NAME)
			self.x = x = self.x[2:12]
			self.y = y = self.y[2:12]
			self.m = m = self.m[2:12]
			self.ints = ints = self.ints[2:12]

		# TODO; better virtual and variables support
		# TODO: this is a copy since concatenated dfs do not yet support
		# global selections

		# a 'deep' copy
		self.dataset_no_filter = vaex.from_items(*self.dataset.to_items(virtual=False, strings=True))
		self.dataset_no_filter.add_virtual_column("z", "x+t*y")
		self.dataset_no_filter.set_variable("t", 1.)


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

		self.dataset_concat_dup = vx.dataset.DatasetConcatenated([self.dataset_no_filter, self.dataset_no_filter, self.dataset_no_filter], name="dataset_concat_dup")
		self.dataset_local = self.dataset
		self.datasetxy_local = self.datasetxy
		self.dataset_concat_local = self.dataset_concat
		self.dataset_concat_dup_local = self.dataset_concat_dup

		np.random.seed(0) # fix seed so that test never fails randomly

		self.df = self.dataset.to_pandas_df()

	def test_function(self):
		def multiply(factor=2):
			def f(x):
				return x*factor
			return f
		ds = self.dataset
		f = ds.add_function('mul2', multiply(2))
		ds['x2'] = f(ds.x)
		self.assertEqual((self.x * 2).tolist(), ds.evaluate('x2').tolist())
		ds.state_get()


	def test_apply(self):
		ds_copy = self.dataset.copy()
		ds = self.dataset
		with small_buffer(ds, 2):
			upper = ds.apply(function_upper, arguments=[ds['name']])
			ds['NAME'] = upper

			name = ds.evaluate('NAME')
			self.assertEqual(name[0], u'0.0BLA')
			ds_copy.state_set(ds.state_get())
			name = ds_copy.evaluate('NAME')
			self.assertEqual(name[0], u'0.0BLA')

		ds['a1'] = ds.apply(lambda x: x+1, arguments=['x'])
		ds['a2'] = ds.apply(lambda x: x+2, arguments=['x'])
		assert (ds['a1']+1).evaluate().tolist() == ds['a2'].evaluate().tolist()


	def test_filter(self):
		ds = self.dataset
		if ds.is_local():  # remote doesn't have a cache
			ds._invalidate_selection_cache()
		with small_buffer(ds):
			ds1 = ds.copy()
			ds1.select(ds1.x > 4, name=vaex.dataset.FILTER_SELECTION_NAME, mode='and')
			ds1._invalidate_caches()

			ds2 = ds[ds.x > 4]
			ds1.x.evaluate()
			# self.assertEqual(ds1.x.evaluate().tolist(), ds2.x.evaluate().tolist())

			ds2.select(ds.x < 6)
			x = ds2.x.evaluate(selection=True)
			self.assertEqual(x.tolist(), [5])
		# print("=" * 70)

	def test_default_selection(self):
		ds = self.dataset
		ds._invalidate_selection_cache()
		with small_buffer(ds):
			indices = ds._filtered_range_to_unfiltered_indices(0, 2)
			self.assertEqual(indices.tolist(), [self.zero_index+0, self.zero_index+1])

			ds = ds[ds.x > 2]
			indices = ds._filtered_range_to_unfiltered_indices(0, 2)
			assert indices.tolist() == [self.zero_index+3, self.zero_index+4]

			x = ds.x.evaluate(0, 2)
			indices = ds._filtered_range_to_unfiltered_indices(0, 2)
			assert len(x) == 2
			assert x[0] == 3

			x = ds.x.evaluate(4, 7)
			indices = ds._filtered_range_to_unfiltered_indices(4, 7)
			assert len(x) == 3
			assert x[0] == 3+4

	def test_unique(self):
		ds = vaex.from_arrays(x=np.array([2,2,1,0,1,1,2]))
		with small_buffer(ds):
		    classes = ds.unique('x')
		    assert np.sort(classes).tolist() == [0, 1, 2]

	def test_amuse(self):
		ds = vx.open(os.path.join(basedir, "files", "default_amuse_plummer.hdf5"))
		self.assertGreater(len(ds), 0)
		self.assertGreater(len(ds.get_column_names()), 0)
		self.assertIsNotNone(ds.unit("x"))
		self.assertIsNotNone(ds.unit("vx"))
		self.assertIsNotNone(ds.unit("mass"))
		ds.close_files()

	def test_masked_array_output(self):
		fn = tempfile.mktemp(".hdf5")
		print(fn)
		self.dataset.export_hdf5(fn, sort="x")
		output = vaex.open(fn)
		self.assertEqual(self.dataset.sum("m"), output.sum("m"))

		table = self.dataset.to_astropy_table()
		fn = tempfile.mktemp(".vot")
		print(fn)
		from astropy.io.votable import from_table, writeto
		votable = from_table(table)
		writeto(votable, fn)
		output = vaex.open(fn)
		self.assertEqual(self.dataset.sum("m"), output.sum("m"))

	def test_formats(self):
		return  # TODO: not workign ATM because of fits + strings
		ds_fits = vx.open(os.path.join(basedir, "files", "gaia-small-fits-basic.fits"))
		ds_fits_plus = vx.open(os.path.join(basedir, "files", "gaia-small-fits-plus.fits"))
		ds_colfits = vx.open(os.path.join(basedir, "files", "gaia-small-colfits-basic.fits"))
		ds_colfits_plus = vx.open(os.path.join(basedir, "files", "gaia-small-colfits-plus.fits"))
		ds_vot = vx.open(os.path.join(basedir, "files", "gaia-small-votable.vot"))
		# FIXME: the votable gives issues
		dslist = [ds_fits, ds_fits_plus, ds_colfits, ds_colfits_plus]#, ds_vot]
		for ds1 in dslist:
			path_hdf5 = tempfile.mktemp(".hdf5")
			ds1.export_hdf5(path_hdf5)
			ds2 = vx.open(path_hdf5)
			diff, missing, types, meta = ds1.compare(ds2)
			self.assertEqual(diff, [], "difference between %s and %s" % (ds1.path, ds2.path))
			self.assertEqual(missing, [], "missing columns %s and %s" % (ds1.path, ds2.path))
			self.assertEqual(meta, [], "meta mismatch between columns %s and %s" % (ds1.path, ds2.path))

			path_fits = tempfile.mktemp(".fits")
			ds1.export_fits(path_fits)
			ds2 = vx.open(path_fits)
			diff, missing, types, meta = ds1.compare(ds2)
			self.assertEqual(diff, [], "difference between %s and %s" % (ds1.path, ds2.path))
			self.assertEqual(missing, [], "missing columns %s and %s" % (ds1.path, ds2.path))
			self.assertEqual(meta, [], "meta mismatch between columns %s and %s" % (ds1.path, ds2.path))

		if 0:
			N = len(dslist)
			for i in range(N):
				for j in range(i+1, N):
					ds1 = dslist[i]
					ds2 = dslist[j]
					diff, missing, types, meta = ds1.compare(ds2)
					self.assertEqual(diff, [], "difference between %s and %s" % (ds1.path, ds2.path))
					self.assertEqual(missing, [], "missing columns %s and %s" % (ds1.path, ds2.path))
			self.assertEqual(meta, [], "meta mismatch between columns %s and %s" % (ds1.path, ds2.path))

	def test_to(self):
		def test_equal(ds1, ds2, units=True, ucds=True, description=True, descriptions=True, skip=[]):
			if description:
				self.assertEqual(ds1.description, ds2.description)
			for name in ds1.get_column_names(virtual=False):
				if name in skip:
					continue
				self.assertIn(name, ds2.get_column_names(virtual=False))
				np.testing.assert_array_equal(ds1.evaluate(name), ds2.evaluate(name), err_msg='mismatch in ' +name)
				if units:
					self.assertEqual(ds1.units.get(name), ds2.units.get(name))
				if ucds:
					self.assertEqual(ds1.ucds.get(name), ds2.ucds.get(name))
				if descriptions:
					self.assertEqual(ds1.descriptions.get(name), ds2.descriptions.get(name))

		# as numpy dict
		ds2 = vx.from_arrays(**self.dataset.to_dict())
		test_equal(self.dataset, ds2, ucds=False, units=False, description=False, descriptions=False)

		# as pandas
		ds2 = vx.from_pandas(self.dataset.to_pandas_df())
		# skip masked arrays, pandas doesn't understand that, converts it to nan, so we can't compare
		test_equal(self.dataset, ds2, ucds=False, units=False, description=False, descriptions=False, skip=['m', 'mi'])

		df = self.dataset.to_pandas_df(index_name="name")
		ds2 = vx.from_pandas(df, index_name="name", copy_index=True)
		test_equal(self.dataset, ds2, ucds=False, units=False, description=False, descriptions=False, skip=['m', 'mi'])

		ds2 = vx.from_pandas(self.dataset.to_pandas_df(index_name="name"))
		assert "name" not in ds2.get_column_names()

		# as astropy table
		ds2 = vx.from_astropy_table(self.dataset.to_astropy_table())
		test_equal(self.dataset, ds2)

		# as arrow table
		ds2 = vx.from_arrow_table(self.dataset.to_arrow_table())
		test_equal(self.dataset, ds2, ucds=False, units=False, description=False, descriptions=False, )

		# return a copy
		ds2 = self.dataset.to_copy(virtual=True)
		assert "z" not in ds2.columns
		assert "z" in ds2.virtual_columns
		test_equal(self.dataset, ds2)

	def test_add_column(self):
		columns = self.dataset.get_column_names()
		self.dataset.add_column("x", self.dataset.data.x)
		self.assertSequenceEqual(columns, self.dataset.get_column_names())
		self.dataset.add_column("extra", self.dataset.data.x)
		extra = self.dataset.evaluate("extra")
		np.testing.assert_array_almost_equal(extra, self.dataset.data.x[self.zero_index:self.zero_index+10])
		with self.assertRaises(ValueError):
			self.dataset.add_column("unequal", self.dataset.data.x[:10])
		with self.assertRaises(ValueError):
			self.dataset.add_column("unequal", self.dataset.data.x[:11])


	def test_rename_column(self):
		self.dataset.rename("x", "xx")
		self.assertNotIn("x", self.dataset.columns)
		self.assertNotIn("x", self.dataset.column_names)
		self.assertNotIn("x", self.dataset.units)
		self.assertNotIn("x", self.dataset.ucds)
		self.assertIn("xx", self.dataset.columns)
		self.assertIn("xx", self.dataset.column_names)
		self.assertIn("xx", self.dataset.units)
		self.assertIn("xx", self.dataset.ucds)

	def test_csv(self):
		separator = ","
		fn = tempfile.mktemp(".csv")
		#print(fn)
		with open(fn, "w") as f:
			print(separator.join(["x", "y", "m", "mi",  "name", "name_arrow", "ints", "f"]), file=f)
			values = [self.dataset.evaluate(k) for k in 'x y m mi name name_arrow ints f'.split()]
			for x, y, m, mi, name, name_arrow, i, f_ in zip(*values):#zip(self.x, self.y, self.dataset.data.m, self.dataset.data.mi, self.dataset.data.name, self.dataset.data.ints, self.dataset.data.f):
				print(separator.join(map(str, [x, y, m, mi, name, name_arrow, i, f_])), file=f)
		ds = vx.from_csv(fn, index_col=False, copy_index=True)
		changes = self.dataset.compare(ds, report_difference=True)
		diff = changes[0]
		#print(diff)
		self.assertEqual(changes[0], [], "changes in dataset")
		self.assertEqual(changes[1], ['index'], "mssing columns")

	def test_ascii(self):
		for seperator in " 	\t,":
			for use_header in [True, False]:
				#print(">>>", repr(seperator), use_header)
				fn = tempfile.mktemp("asc")
				with open(fn, "w") as f:
					if use_header:
						print(seperator.join(["x", "y"]), file=f)
					for x, y, name in zip(self.x, self.y, self.dataset.data.name):
						print(seperator.join(map(repr, [x, y])), file=f)
				#with open(fn) as f:
				#	print(f.read())
				sep = seperator
				if seperator == " ":
					sep = None
				if use_header:
					ds = vx.from_ascii(fn, seperator=sep)
				else:
					ds = vx.from_ascii(fn, seperator=seperator, names="x y".split())

				np.testing.assert_array_almost_equal(ds.data.x, self.x)
				np.testing.assert_array_almost_equal(ds.data.y, self.y)
				#np.testing.assert_array_equal(ds.data.names, self.dataset.data.name)
				#if seperator == ",":
				#	df = pd.read_csv(fn)
				#	ds = vx.from_pandas(df)
				#	np.testing.assert_array_almost_equal(ds.data.x, self.x)
				#	np.testing.assert_array_almost_equal(ds.data.y, self.y)
					#np.testing.assert_array_equal(ds.data.names, self.dataset.data.name)

	def tearDown(self):
		self.dataset.remove_virtual_meta()
		self.dataset_concat.remove_virtual_meta()
		self.dataset_concat_dup.remove_virtual_meta()

	def test_mixed_endian(self):

		x = np.arange(10., dtype=">f8")
		y = np.arange(10, dtype="<f8")
		ds = vx.from_arrays(x=x, y=y)
		ds.count()
		ds.count(binby=["x", "y"])

	def test_join(self):
		np.random.seed(42)
		x = np.arange(10, dtype=np.float64)
		indices = np.arange(10)
		i = x.astype(np.int64)
		np.random.shuffle(indices)
		xs = x[indices]
		y = x**2
		z = ys = y[indices]
		names = np.array(list(map(lambda x: str(x) + "bla", self.x)), dtype='S')[indices]
		ds = vaex.from_arrays(x=x, y=y)
		ds2 = vaex.from_arrays(x=xs, z=ys, i=i, names=names)
		ds.join(ds2[['x', 'z', 'i', 'names']], 'x', rsuffix='r', inplace=True)
		self.assertEqual(ds.sum('x*y'), np.sum(x*y))
		self.assertEqual(ds.sum('x*z'), np.sum(x*y))
		self.assertEqual(ds.sum('x*y'), np.sum(x[indices]*z))
		self.assertEqual(ds.sum('x*y'), np.sum(x[indices]*z))

		# test with incomplete data
		ds = vaex.from_arrays(x=x, y=y)
		ds2 = vaex.from_arrays(x=xs[:4], z=ys[:4], i=i[:4], names=names[:4])
		ds.join(ds2, 'x', rsuffix='r', inplace=True)
		self.assertEqual(ds.sum('x*y'), np.sum(x*y))
		self.assertEqual(ds.sum('x*z'), np.sum(x[indices][:4]*y[indices][:4]))

		# test with incomplete data, but other way around
		ds = vaex.from_arrays(x=x[:4], y=y[:4])
		ds2 = vaex.from_arrays(x=xs, z=ys, i=i, names=names)
		ds.join(ds2, 'x', inplace=True, rsuffix='r')
		self.assertEqual(ds.sum('x*y'), np.sum(x[:4]*y[:4]))
		self.assertEqual(ds.sum('x*z'), np.sum(x[:4]*y[:4]))


	def test_healpix_count(self):
		# only test when healpy is present
		try:
			import healpy as hp
		except ImportError:
			return
		max_order = 6
		nside = hp.order2nside(max_order)
		npix = hp.nside2npix(nside)
		healpix = np.arange(npix)
		ds = vx.from_arrays(healpix=healpix)
		for order in range(max_order):
			counts = ds.healpix_count(healpix_expression="healpix", healpix_max_level=max_order, healpix_level=order)
			scaling = 4**(max_order-order)
			ones = np.ones(npix//scaling) * scaling
			np.testing.assert_array_almost_equal(counts, ones)
			self.assertEqual(counts.sum(), npix)

	def test_uncertainty_propagation(self):

		N = 100000
		# distance
		parallaxes = np.random.normal(1, 0.1, N)
		ds_many = vx.from_arrays(parallax=parallaxes)
		ds_many.add_virtual_columns_distance_from_parallax("parallax", "distance")
		distance_std_est = ds_many.std("distance").item()

		ds_1 = vx.from_arrays(parallax=np.array([1.]), parallax_uncertainty=np.array([0.1]))
		ds_1.add_virtual_columns_distance_from_parallax("parallax", "distance", "parallax_uncertainty")
		distance_std = ds_1.evaluate("distance_uncertainty")[0]
		self.assertAlmostEqual(distance_std, distance_std_est,2)

	def test_virtual_column_storage(self):
		self.dataset.write_meta()
		ds = vaex.zeldovich()
		ds.write_meta()

	def test_add_virtual_columns_cartesian_velocities_to_polar(self):
		if 1:
			def dfs(x, y, velx, vely):
				ds_1 = from_scalars(x=x, y=y, vx=velx, vy=vely, x_e=0.01, y_e=0.02, vx_e=0.03, vy_e=0.04)
				ds_1.add_virtual_columns_cartesian_velocities_to_polar(propagate_uncertainties=True)
				N = 100000
				# distance
				x =        np.random.normal(x, 0.01, N)
				y =        np.random.normal(y, 0.02, N)
				velx =        np.random.normal(velx, 0.03, N)
				vely =        np.random.normal(vely, 0.04, N)
				ds_many = vx.from_arrays(x=x, y=y, vx=vely, vy=vely)
				ds_many.add_virtual_columns_cartesian_velocities_to_polar()
				return ds_1, ds_many
			ds_1, ds_many = dfs(0, 2, 3, 4)

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


	def test_add_virtual_columns_cartesian_velocities_to_spherical(self):
		if 0: # TODO: errors in spherical velocities
			pass

		def test(vr_expect, vlong_expect, vlat_expect, **kwargs):
			ds_1 = from_scalars(**kwargs)
			ds_1.add_virtual_columns_cartesian_velocities_to_spherical()
			vr, vlong, vlat = ds_1.evaluate("vr")[0], ds_1.evaluate("vlong")[0], ds_1.evaluate("vlat")[0]
			self.assertAlmostEqual(vr, vr_expect)
			self.assertAlmostEqual(vlong, vlong_expect)
			self.assertAlmostEqual(vlat, vlat_expect)

		test(0, -1,  0, x=1, y=0, z=0, vx=0, vy=-1, vz=0)
		test(0, -1,  0, x=10, y=0, z=0, vx=0, vy=-1, vz=0)
		test(0,  0,  1, x=1, y=0, z=0, vx=0, vy= 0, vz=1)
		test(1,  0,  0, x=1, y=0, z=0, vx=1, vy= 0, vz=0)
		a = 1./np.sqrt(2.)
		test(0,  0,  1, x=a, y=0, z=a, vx=-a, vy= 0, vz=a)

	def test_add_virtual_columns_cartesian_velocities_to_pmvr(self):
		if 0: # TODO: errors in spherical velocities
			pass

		def test(vr_expect, pm_long_expect, pm_lat_expect, **kwargs):
			ds_1 = from_scalars(**kwargs)
			ds_1.add_variable("k", 1) # easier for comparison
			ds_1.add_virtual_columns_cartesian_velocities_to_pmvr()
			vr, pm_long, pm_lat = ds_1.evaluate("vr")[0], ds_1.evaluate("pm_long")[0], ds_1.evaluate("pm_lat")[0]
			self.assertAlmostEqual(vr, vr_expect)
			self.assertAlmostEqual(pm_long, pm_long_expect)
			self.assertAlmostEqual(pm_lat, pm_lat_expect)

		test(0, -1,  0, x=1, y=0, z=0, vx=0, vy=-1, vz=0)
		test(0, -0.1,  0, x=10, y=0, z=0, vx=0, vy=-1, vz=0)
		test(0,  0,  1, x=1, y=0, z=0, vx=0, vy= 0, vz=1)
		test(1,  0,  0, x=1, y=0, z=0, vx=1, vy= 0, vz=0)
		a = 1./np.sqrt(2.)
		test(0,  0,  1, x=a, y=0, z=a, vx=-a, vy= 0, vz=a)
		test(0,  0,  1*10, x=a/10, y=0, z=a/10, vx=-a, vy= 0, vz=a)

	def test_add_virtual_columns_cartesian_to_polar(self):
		for radians in [True, False]:
			def dfs(x, y, radians=radians):
				ds_1 = from_scalars(x=x, y=y, x_e=0.01, y_e=0.02)
				ds_1.add_virtual_columns_cartesian_to_polar(propagate_uncertainties=True, radians=radians)
				N = 100000
				# distance
				x =        np.random.normal(x, 0.01, N)
				y =        np.random.normal(y, 0.02, N)
				ds_many = vx.from_arrays(x=x, y=y)
				ds_many.add_virtual_columns_cartesian_to_polar(radians=radians)
				return ds_1, ds_many
			ds_1, ds_many = dfs(0, 2)

			r_polar_e = ds_1.evaluate("r_polar_uncertainty")[0]
			phi_polar_e = ds_1.evaluate("phi_polar_uncertainty")[0]
			self.assertAlmostEqual(r_polar_e, ds_many.std("r_polar").item(), delta=0.02)
			self.assertAlmostEqual(phi_polar_e, ds_many.std("phi_polar").item(), delta=0.02)

			# rotation is anti clockwise
			r_polar = ds_1.evaluate("r_polar")[0]
			phi_polar = ds_1.evaluate("phi_polar")[0]
			self.assertAlmostEqual(r_polar, 2)
			self.assertAlmostEqual(phi_polar, np.pi/2 if radians else 90)


	def test_add_virtual_columns_proper_motion2vperpendicular(self):
		def dfs(distance, pm_l, pm_b):
			ds_1 = from_scalars(pm_l=pm_l, pm_b=pm_b, distance=distance, distance_e=0.1, pm_l_e=0.3, pm_b_e=0.4)
			ds_1.add_virtual_columns_proper_motion2vperpendicular(propagate_uncertainties=True)
			N = 100000
			# distance
			distance = np.random.normal(0, 0.1, N)  + distance
			pm_l =     np.random.normal(0, 0.3, N)  + pm_l
			pm_b =     np.random.normal(0, 0.4, N)  + pm_b
			ds_many = vx.from_arrays(pm_l=pm_l, pm_b=pm_b, distance=distance)
			ds_many.add_virtual_columns_proper_motion2vperpendicular()
			return ds_1, ds_many
		ds_1, ds_many = dfs(2, 3, 4)

		vl_e = ds_1.evaluate("vl_uncertainty")[0]
		vb_e = ds_1.evaluate("vb_uncertainty")[0]
		self.assertAlmostEqual(vl_e, ds_many.std("vl").item(), delta=0.02)
		self.assertAlmostEqual(vb_e, ds_many.std("vb").item(), delta=0.02)
		k = 4.74057
		self.assertAlmostEqual(ds_1.evaluate("vl")[0], 2*k*3)
		self.assertAlmostEqual(ds_1.evaluate("vb")[0], 2*k*4)

	def test_virtual_columns_lbrvr_proper_motion2vcartesian(self):
		for radians in [True, False]:
			def dfs(l, b, distance, vr, pm_l, pm_b, radians=radians):
				ds_1 = from_scalars(l=l, b=b, pm_l=pm_l, pm_b=pm_b, vr=vr, distance=distance, distance_e=0.1, vr_e=0.2, pm_l_e=0.3, pm_b_e=0.4)
				ds_1.add_virtual_columns_lbrvr_proper_motion2vcartesian(propagate_uncertainties=True, radians=radians)
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
			ds_1, ds_many = dfs(0, 0, 1, 1, 2, 3)

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

	def test_state(self):
		mul = Multiply(3)
		ds = self.dataset
		copy = ds.copy(virtual=False)
		statefile = tempfile.mktemp('.json')
		ds.select('x > 5', name='test')
		ds.add_virtual_column('xx', 'x**2')
		fmul = ds.add_function('fmul', mul)
		ds['mul'] = fmul(ds.x)
		count = ds.count('x', selection='test')
		sum = ds.sum('xx', selection='test')
		summul = ds.sum('mul', selection='test')
		ds.state_write(statefile)
		copy.state_load(statefile)
		self.assertEqual(count, copy.count('x', selection='test'))
		self.assertEqual(sum, copy.sum('xx', selection='test'))
		self.assertEqual(summul, copy.sum('3*x', selection='test'))
		self.assertEqual(summul, copy.sum('mul', selection='test'))

	def test_strings(self):
		# TODO: concatenated dfs with strings of different length
		self.assertEqual(["x", "y", "m", "mi", "ints", "f"], self.dataset.get_column_names(virtual=False, strings=False))

		names = ["x", "y", "m", "mi", "ints", "f", "name", "name_arrow"]
		self.assertEqual(names, self.dataset.get_column_names(strings=True, virtual=False))

		if self.dataset.is_local():
			# check if strings are exported
			path_hdf5 = tempfile.mktemp(".hdf5")
			self.dataset.export_hdf5(path_hdf5, virtual=False)
			exported_dataset = vx.open(path_hdf5)
			self.assertEqual(names, exported_dataset.get_column_names(strings=True))

			path_arrow = tempfile.mktemp(".arrow")
			self.dataset.export_arrow(path_arrow, virtual=False)
			exported_dataset = vx.open(path_arrow)
			self.assertEqual(names, exported_dataset.get_column_names(strings=True))

			# for fits we do not support arrow like strings
			# TODO: get back support for SXX (string/binary) format
			self.dataset.drop("name_arrow", inplace=True)
			self.dataset.drop("name", inplace=True)
			names.remove("name_arrow")
			names.remove("name")
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
		self.assertEqual(self.dataset.data_type("x"), np.dtype(">f8"))
		self.assertEqual(self.dataset.data_type("f"), np.float64)
		self.assertEqual(self.dataset.data_type("x*f"), np.float64)

	def test_byte_size(self):
		arrow_size = self.dataset.columns['name_arrow'].nbytes
		self.assertEqual(self.dataset.byte_size(), (8*6 + 2 + self.dataset.columns['name'].dtype.itemsize)*len(self.dataset) + arrow_size)
		self.dataset.select("x < 1")
		self.assertEqual(self.dataset.byte_size(selection=True), 8*6 + 2 + self.dataset.columns['name'].dtype.itemsize + arrow_size)

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


	def test_not_implemented(self):
		subspace = vaex.legacy.Subspace(self.dataset, ["x", "y"], self.dataset.executor, False)
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

	def test_length(self):
		self.assertEqual(len(self.dataset), 10)

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
		x = self.dataset.evaluate("x", selection="x < 4")
		self.assertEqual(x.tolist(), x[:4].tolist())


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



	def test_count(self):
		self.dataset.select("x < 5")
		ds = self.dataset[self.dataset.x < 5]
		df = self.df[self.df.x < 5]
		np.testing.assert_array_almost_equal(self.dataset.count("x", selection=None), 10)
		np.testing.assert_array_almost_equal(self.dataset.count("x", selection=True), 5)
		np.testing.assert_array_almost_equal(self.dataset.x.count(selection=True), 5)
		np.testing.assert_array_almost_equal(self.dataset['x'].count(), 10)
		np.testing.assert_array_almost_equal(self.df['x'].count(), 10)
		np.testing.assert_array_almost_equal(ds['x'].count(), 5)
		np.testing.assert_array_almost_equal(df['x'].count(), 5)

		self.dataset.select("x >= 5")
		ds = self.dataset[self.dataset.x >= 5]
		df = self.df[self.df.x >= 5]
		np.testing.assert_array_almost_equal(self.dataset.count("m", selection=None), 9)
		np.testing.assert_array_almost_equal(self.dataset.count("m", selection=True), 4)
		np.testing.assert_array_almost_equal(self.dataset['m'].count(), 9)
		np.testing.assert_array_almost_equal(self.df['m'].count(), 9)
		np.testing.assert_array_almost_equal(ds['m'].count(), 4)
		np.testing.assert_array_almost_equal(df['m'].count(), 4)

		# convert to float
		self.dataset_local.columns["x"] = self.dataset_local.columns["x"] * 1.
		self.dataset_local.columns["x"][self.zero_index] = np.nan
		if self.dataset.is_local():
			self.dataset._invalidate_caches()
		else:
			if hasattr(self, 'webserver1'):
				self.webserver1.dfs[0]._invalidate_caches()
				self.webserver2.dfs[0]._invalidate_caches()
			self.dataset_local._invalidate_caches()

		self.df = self.dataset_local.to_pandas_df()
		self.dataset.select("x < 5")
		ds = self.dataset[self.dataset.x < 5]
		df = self.df[self.df.x < 5]
		# import pdb
		# pdb.set_trace()
		np.testing.assert_array_almost_equal(self.dataset.count("x", selection=None), 9)
		np.testing.assert_array_almost_equal(self.dataset['x'].count(), 9)
		np.testing.assert_array_almost_equal(self.df['x'].count(), 9)
		np.testing.assert_array_almost_equal(self.dataset.count("x", selection=True), 4)
		np.testing.assert_array_almost_equal(ds['x'].count(), 4)
		np.testing.assert_array_almost_equal(df['x'].count(), 4)
		np.testing.assert_array_almost_equal(self.dataset.count("y", selection=None), 9)  # this is because of the filter x<10
		np.testing.assert_array_almost_equal(self.dataset_no_filter.count("y", selection=None), 10)
		np.testing.assert_array_almost_equal(self.dataset['y'].count(), 9)
		np.testing.assert_array_almost_equal(self.dataset_no_filter['y'].count(), 10)
		np.testing.assert_array_almost_equal(self.df['y'].count(), 9)
		np.testing.assert_array_almost_equal(self.dataset.count("y", selection=True), 4)
		np.testing.assert_array_almost_equal(ds['y'].count(), 4)
		np.testing.assert_array_almost_equal(df['y'].count(), 4)
		np.testing.assert_array_almost_equal(self.dataset.count(selection=None), 9)
		np.testing.assert_array_almost_equal(self.dataset_no_filter.count(selection=None), 10)
		np.testing.assert_array_almost_equal(self.dataset.count(), 9)
		np.testing.assert_array_almost_equal(self.dataset_no_filter.count(), 10)
		#np.testing.assert_array_almost_equal(self.df.count(), 9) # TODO: this is different in pandas
		# we modified the data.. so actually this should be 4..
		np.testing.assert_array_almost_equal(self.dataset.count(selection=True), 4)
		np.testing.assert_array_almost_equal(ds.count(), 4)
		np.testing.assert_array_almost_equal(self.dataset.count("*", selection=None), 9)
		np.testing.assert_array_almost_equal(self.dataset_no_filter.count("*", selection=None), 10)
		np.testing.assert_array_almost_equal(self.dataset.count(), 9)
		np.testing.assert_array_almost_equal(self.dataset_no_filter.count(), 10)
		np.testing.assert_array_almost_equal(self.dataset.count("*", selection=True), 4)
		np.testing.assert_array_almost_equal(ds.count(), 4)

		task = self.dataset.count("x", selection=True, delay=True)
		self.dataset.execute()
		np.testing.assert_array_almost_equal(task.get(), 4)


		np.testing.assert_array_almost_equal(self.dataset.count("x", selection=None, binby=["x"], limits=[0, 10], shape=1), [9])
		np.testing.assert_array_almost_equal(self.dataset.count("x", selection=True, binby=["x"], limits=[0, 10], shape=1), [4])
		np.testing.assert_array_almost_equal(self.dataset.count("*", selection=None, binby=["x"], limits=[0, 10], shape=1), [9])
		np.testing.assert_array_almost_equal(self.dataset.count("*", selection=True, binby=["x"], limits=[0, 10], shape=1), [4])
		np.testing.assert_array_almost_equal(self.dataset          .count("*", selection=None, binby=["y"], limits=[0, 9**2+1], shape=1), [9])
		np.testing.assert_array_almost_equal(self.dataset_no_filter.count("*", selection=None, binby=["y"], limits=[0, 9**2+1], shape=1), [10])
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

		ds = self.dataset[(self.dataset.y >= 0) & (self.dataset.y < 50)]
		b = ds.count('x')
		ds = self.dataset[(self.dataset.y >= 50) & (self.dataset.y < 100)]
		c = ds.count('x')
		np.testing.assert_array_almost_equal(a, [b, c])


		df = self.df[(self.df.y >= 0) & (self.df.y < 50)]
		b = df['x'].count()
		df = self.df[(self.df.y >= 50) & (self.df.y < 100)]
		c = df['x'].count()
		np.testing.assert_array_almost_equal(a, [b, c])


	def test_cov(self):
		# convert to float
		x = self.dataset_local.columns["x"][:10] = self.dataset_local.columns["x"][:10] * 1.
		y = self.y
		def cov(*args):
			return np.cov(args, bias=1)
		self.dataset.select("x < 5")


		np.testing.assert_array_almost_equal(self.dataset.cov("x", "y", selection=None), cov(x, y))
		np.testing.assert_array_almost_equal(self.dataset.cov("x", "y", selection=True), cov(x[:5], y[:5]))
		np.testing.assert_array_almost_equal(self.dataset.cov("x", "y", selection=[False, True]), [cov(x, y), cov(x[:5], y[:5])])

		#self.dataset.columns["x"][0] = np.nan
		np.testing.assert_array_almost_equal(self.dataset.cov("x", "y", selection=None), cov(x, y))
		np.testing.assert_array_almost_equal(self.dataset.cov("x", "y", selection=True), cov(x[:5], y[:5]))

		task = self.dataset.cov("x", "y", selection=True, delay=True)
		self.dataset.execute()
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

		# including nan
		n = np.arange(-self.zero_index, 20.-1)
		n[self.zero_index+1] = np.nan
		self.dataset_local.add_column('n', n)
		assert not np.any(np.isnan(self.dataset.cov("x", "n")))


	def test_covar(self):
		# convert to float
		x = self.dataset_local.columns["x"][:10] = self.dataset_local.columns["x"][:10] * 1.
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

		task = self.dataset.covar("x", "y", selection=True, delay=True)
		self.dataset.execute()
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


	def test_percentile(self):

		ds = vx.example()
		#ds.median_approx('z', binby=['x'], limits=[-10, 10], shape=16)
		#ds.median_approx('z', binby=['x', 'y'], limits=[-10, 10], shape=4)
		#m = ds.median_approx('z+x/10', binby=['x'], limits=[-10, 10], shape=32, percentile_shape=128*10 , percentile_limits=[-10,10])
		m = ds.median_approx('z+x/10', binby=['x'], limits=[6.875000, 7.500000], shape=1, percentile_shape=128*10 , percentile_limits=[-10,10])
		mc = ds.median_approx("z+x/10", selection='(x > 6.875000) & (x <= 7.500000)', percentile_shape=128*10 , percentile_limits=[-10,10])


		#print(m, m[32-5], mc)
		print(m, mc)

		return
		dsodd = vx.from_arrays(x=np.arange(3)) # 0,1,2
		dseven = vx.from_arrays(x=np.arange(4)) # 0,1,2,3
		self.dataset.select("x < 5")
		o = 0#10/30/2.

		#x = dsodd.data.x
		ds = dsodd
		#ds = dseven
		x = ds.data.x
		print("median", np.median(x))
		for offset in [-0.99, -0.5, 0.0]:#[0:1]:
			print()
			print("offset", offset)
			limits = [0+offset, x.max()+1+offset]
			print(">>>", ds.percentile_approx("x", selection=None, percentile_limits=limits, percentile_shape=len(x)),)
			#np.testing.assert_array_almost_equal(
			#	ds.percentile_approx("x", selection=None, percentile_limits=limits, percentile_shape=4),
			#	np.median(x), decimal=2)
		#return


		np.testing.assert_array_almost_equal(
			self.dataset.percentile_approx("x", selection=None, percentile_limits=[0-o, 10-o], percentile_shape=100),
			np.median(self.x), decimal=1)
		np.testing.assert_array_almost_equal(
			self.dataset.percentile_approx("x", selection=None, percentile_limits=[0-o, 10-o], percentile_shape=1000),
			np.median(self.x), decimal=2)
		np.testing.assert_array_almost_equal(
			self.dataset.percentile_approx(["x", "y"], selection=None, percentile_shape=10000),
			[np.median(self.x), np.median(self.y)],
			decimal=3)
		return
		np.testing.assert_array_almost_equal(self.dataset.percentile_approx("x", selection=True), np.median(self.x[:5]))

		# convert to float
		x = self.dataset.columns["x"] = self.dataset.columns["x"] * 1.
		y = self.y
		self.dataset.columns["x"][0] = np.nan
		np.testing.assert_array_almost_equal(self.dataset.sum("x", selection=None), np.nansum(x))
		np.testing.assert_array_almost_equal(self.dataset.sum("x", selection=True), np.nansum(x[:5]))

		task = self.dataset.sum("x", selection=True, delay=True)
		self.dataset.execute()
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

	def test_concat(self):
		dc = self.dataset_concat_dup
		self.assertEqual(len(self.dataset_concat_dup), len(self.dataset)*3)


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
			dfs = [vx.dataset.DatasetArrays("dataset-%i" % i)  for i in range(N)]
			for dataset, array in zip(dfs, arrays):
				dataset.add_column("x", array)
			dataset_concat = vx.dataset.DatasetConcatenated(dfs, name="dataset_concat")
			return dataset_concat

		self.assertEqual(concat(np.float32, np.float64).columns["x"].dtype, np.float64)
		self.assertEqual(concat(np.float32, np.int64).columns["x"].dtype, np.float64)
		self.assertEqual(concat(np.float32, np.byte).columns["x"].dtype, np.float32)
		self.assertEqual(concat(np.float64, np.byte, np.int64).columns["x"].dtype, np.float64)

		ar1 = np.zeros((10, 2))
		ar2 = np.zeros((20))
		arrays = [ar1, ar2]
		N = len(arrays)
		dfs = [vx.dataset.DatasetArrays("dataset1") for i in range(N)]
		for dataset, array in zip(dfs, arrays):
			dataset.add_column("x", array)
		with self.assertRaises(ValueError):
			dataset_concat = vx.dataset.DatasetConcatenated(dfs, name="dataset_concat")


		ar1 = np.zeros((10))
		ar2 = np.zeros((20))
		arrays = [ar1, ar2]
		N = len(arrays)
		dfs = [vx.dataset.DatasetArrays("dataset1") for i in range(N)]
		for dataset, array in zip(dfs, arrays):
			dataset.add_column("x", array)
		dataset_concat = vx.dataset.DatasetConcatenated(dfs, name="dataset_concat")


		dataset_concat1 = vx.dataset.DatasetConcatenated(dfs, name="dataset_concat")
		dataset_concat2 = vx.dataset.DatasetConcatenated(dfs, name="dataset_concat")
		self.assertEqual(len(dataset_concat1.concat(dataset_concat2).dfs), 4)
		self.assertEqual(len(dataset_concat1.concat(dfs[0]).dfs), 3)
		self.assertEqual(len(dfs[0].concat(dataset_concat1).dfs), 3)
		self.assertEqual(len(dfs[0].concat(dfs[0]).dfs), 2)

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

	def test_export_sorted(self):
		self.dataset.add_column("s", 100-self.dataset.data.x)
		path_hdf5 = tempfile.mktemp(".hdf5")
		self.dataset.export_hdf5(path_hdf5, sort="s")
		ds2 = vaex.open(path_hdf5)
		np.testing.assert_array_equal(self.dataset.data.x[self.zero_index:self.zero_index+10], ds2.data.x[::-1])

	def test_export_sorted_arrow(self):
		self.dataset.add_column("s", 100-self.dataset.data.x)
		path_arrow = tempfile.mktemp(".arrow")
		self.dataset.export_arrow(path_arrow, sort="s")
		ds2 = vaex.open(path_arrow)
		np.testing.assert_array_equal(self.dataset.data.x[self.zero_index:self.zero_index+10], ds2.data.x[::-1])

	def test_export(self):
		path = path_hdf5 = tempfile.mktemp(".hdf5")
		path_fits = tempfile.mktemp(".fits")
		path_fits_astropy = tempfile.mktemp(".fits")
		#print path

		#with self.assertRaises(AssertionError):
		#	self.dataset.export_hdf5(path, selection=True)

		for dataset in [self.dataset_concat_dup, self.dataset]:
			#print dataset.virtual_columns
			for fraction in [1, 0.5]:
				dataset.set_active_fraction(fraction)
				dataset.select("x > 3")
				dataset.select("x > 2", name="named")
				length = len(dataset)
				for column_names in [["x", "y", "z"], ["x"], ["y"], ["z"], None]:
					for byteorder in "<=>":
						for shuffle in [False, True]:
							for selection in [False, True, "named"]:
								for virtual in [False, True]:
									for export in [dataset.export_fits, dataset.export_hdf5, dataset.export_arrow, dataset.export_parquet]: #if byteorder == ">" else [dataset.export_hdf5]:
										#print (">>>", dataset, path, column_names, byteorder, shuffle, selection, fraction, dataset.length_unfiltered(), virtual)
										#byteorder = "<"
										if export == dataset.export_fits and byteorder != ">":
											#print("skip", export == dataset.export_fits, byteorder != ">", byteorder)
											continue # fits only does big endian
										if export == dataset.export_fits and byteorder == ">":
											continue # arrow only little endian
										if vx.utils.osname == "windows" and export == dataset.export_hdf5 and byteorder == ">":
											#print("skip", vx.utils.osname)
											continue # TODO: IS this a bug for h5py on win32?, leads to an open file
										# same issue on windows for arrow, closing the mmapped file does not help
										# for the moment we create a new temp file
										path_arrow = tempfile.mktemp(".arrow")
										path_parquet = tempfile.mktemp(".parquet")
										#print dataset.length_unfiltered()
										#print len(dataset)
										if export == dataset.export_hdf5:
											path = path_hdf5
											export(path, column_names=column_names, byteorder=byteorder, shuffle=shuffle, selection=selection, progress=False, virtual=virtual)
										elif export == dataset.export_arrow:
											path = path_arrow
											export(path, column_names=column_names, byteorder=byteorder, shuffle=shuffle, selection=selection, progress=False, virtual=virtual)
										elif export == dataset.export_parquet:
											path = path_parquet
											export(path, column_names=column_names, byteorder=byteorder, shuffle=shuffle, selection=selection, progress=False, virtual=virtual)
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
											column_names = ["x", "y", "m", "mi", "ints", "f", "z", "name", "name_arrow"] if virtual else ["x", "y", "m", "mi", "ints", "f", "name", "name_arrow"]
										#if not virtual:
										#	if "z" in column_names:
										#		column_names.remove("z")
										# TODO: does the order matter?
										self.assertEqual((compare.get_column_names(strings=True)), (column_names + (["random_index"] if shuffle else [])))
										def make_masked(ar):
											if export == dataset.export_fits: # for fits the missing values will be filled in with nan
												if ar.dtype.kind == "f":
													nanmask = np.isnan(ar)
													if np.any(nanmask):
														ar = np.ma.array(ar, mask=nanmask)
											return ar

										for column_name in column_names:
											#values = dataset.columns[column_name][dataset._index_start:dataset._index_end] if column_name in dataset.get_column_names(virtual=False) else dataset.evaluate(column_name)
											values = dataset.evaluate(column_name, filtered=False)
											if selection:
												values = dataset.evaluate(column_name)
												mask = dataset.evaluate_selection_mask(selection)#, 0, len(dataset))
												if len(values[::]) != len(mask):
													import pdb
													pdb.set_trace()
												# for concatenated columns, we get a plain numpy array copy using [::]
												a = np.ma.compressed(make_masked(compare.evaluate(column_name)))
												b = np.ma.compressed(make_masked(values[::][mask]))
												if len(a) != len(b):
													import pdb
													pdb.set_trace()
												self.assertEqual(sorted(a), sorted(b))
											else:
												values = dataset.evaluate(column_name, filtered=True)
												if shuffle:
													indices = compare.columns["random_index"]
													a = np.ma.compressed(make_masked(compare.evaluate(column_name)))
													b = np.ma.compressed(make_masked(values[::][indices]))
													self.assertEqual(sorted(a), sorted(b))
												else:
													dtype = compare.columns[column_name].dtype # we don't want any casting
													compare_values = compare.columns[column_name]
													if isinstance(compare_values, vaex.column.Column):
														compare_values = compare_values.to_numpy()
													np.testing.assert_array_equal(compare_values, values[:length].astype(dtype))
										compare.close_files()
										#os.remove(path)

				# self.dataset_concat_dup references self.dataset, so set it's active_fraction to 1 again
				dataset.set_active_fraction(1)
		path_arrow = tempfile.mktemp(".arrow")
		path_hdf5 = tempfile.mktemp(".hdf5")
		dataset = self.dataset
		dataset.export(path_arrow)
		name = "vaex export"
		#print(path_fits)
		vaex.export.main([name, "--no-progress", "-q", "file", path_arrow, path_hdf5])
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
		# the active_fraction only applies to the underlying length, which is 20
		self.dataset.set_active_fraction(0.25)
		self.assertEqual(counter_selection.counter, 2)
		self.assertEqual(counter_current_row.counter, 2)
		self.assertEqual(length/2 - self.zero_index, len(self.dataset))

		self.dataset.select("x > 5")
		self.assertEqual(counter_selection.counter, 3)
		self.assertEqual(counter_current_row.counter, 2)
		self.assertTrue(self.dataset.has_selection())
		self.dataset.set_active_fraction(0.25) # nothing should happen, still the same
		self.assertTrue(self.dataset.has_selection())
		self.dataset.set_active_fraction(0.4999)
		self.assertFalse(self.dataset.has_selection())

		self.dataset.set_current_row(1)
		self.assertTrue(self.dataset.has_current_row())
		self.dataset.set_active_fraction(0.25)
		self.assertFalse(self.dataset.has_current_row())

		if self.dataset.is_local(): # this part doesn't work for remote dfs
			for dataset in [self.dataset, self.dataset_concat]:
				dataset.set_active_fraction(1.0)
				x = dataset.columns["x"][:] * 1. # make a copy
				dataset.set_active_fraction(0.25)
				length = len(dataset)
				a = x[:length]
				b = dataset.columns["x"][:len(dataset)]
				np.testing.assert_array_almost_equal(a, b)
				self.assertLess(length, dataset.length_original())

		# TODO: test if statistics and histogram work on the active_fraction
		self.dataset.set_active_fraction(1)
		total = self.dataset["x"].sum()
		self.dataset.set_active_fraction(0.25)
		total_half = self.dataset["x"].sum()
		self.assertLess(total_half, total)

		limits = [(-100, 100)]
		self.dataset.set_active_fraction(1)
		total = self.dataset["x"].count(limits=limits).sum()
		self.dataset.set_active_fraction(0.25)
		total_half = self.dataset["x"].count(limits=limits).sum()
		self.assertLess(total_half, total)


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


	def test_dropna(self):
		ds = self.dataset
		ds.select_non_missing(column_names=['m'])
		self.assertEqual(ds.count(selection=True), 9)
		ds.select_non_missing(drop_masked=False, column_names=['m'])
		self.assertEqual(ds.count(selection=True), 10)

		self.dataset_local.data.x[self.zero_index] = np.nan
		ds.select_non_missing(column_names=['x'])
		self.assertEqual(ds.count(selection=True), 9)
		ds.select_non_missing(drop_nan=False, column_names=['x'])
		if ds.is_local():
			self.assertEqual(ds.count(selection=True), 10)
		else:
			# TODO: on the server, the filter selection gets re-executed (x < 10)
			# causing it to skip the nan anyway, find a good way to test this?
			self.assertEqual(ds.count(selection=True), 9)

		ds.select_non_missing()
		self.assertEqual(ds.count(selection=True), 8)
		ds.select_non_missing(drop_masked=False)
		self.assertEqual(ds.count(selection=True), 9)
		ds.select_non_missing(drop_nan=False)
		if ds.is_local():
			self.assertEqual(ds.count(selection=True), 9)
		else:
			# TODO: same as above
			self.assertEqual(ds.count(selection=True), 8)


	def test_selection_in_handler(self):
		self.dataset.select("x > 5")
		# in the handler, we should know there is not selection
		def check(*ignore):
			self.assertFalse(self.dataset.has_selection())
		self.dataset.signal_selection_changed.connect(check)
		self.dataset.select_nothing()



	def test_nearest(self):
		index, distance, (value,) = self.dataset("x").nearest([3])
		self.assertEqual(index, 3 + self.zero_index)
		self.assertEqual(distance, 0)
		self.assertEqual(value, 3)

		index, distance, (value,) = self.dataset("x").nearest([3.7])
		self.assertEqual(index, 4 + self.zero_index)
		self.assertAlmostEqual(distance, 0.3)
		self.assertEqual(value, 4)

		self.dataset.select("x > 5")
		index, distance, (value,) = self.dataset("x").selected().nearest([3.7])
		self.assertEqual(index, 6 + self.zero_index)
		self.assertEqual(distance, 2.3)
		self.assertEqual(value, 6)


	def test_select_circle(self):
		# Circular selection
		self.dataset.select_circle('x', 'y', 0.5, 0.5, 1, name='circ')
		# Assert
		np.testing.assert_equal(2, self.dataset.count(selection='circ'))

	def test_select_ellipse(self):
		# Ellipse election
		self.dataset.select_ellipse('x', 'y', 3, 10, 2, 15, -10, name='elli')
		# Assert
		np.testing.assert_equal(3, self.dataset.count(selection='elli'))


# allow multiple python versions on one machine to run the test
import sys
test_port = 29110 + sys.version_info[0] * 10 + sys.version_info[1]

#class A:#class estDatasetRemote(TestDataset):
class TestDatasetRemote(TestDataset):
#class A:
	use_websocket = True

	@classmethod
	def setUpClass(cls):
		global test_port
		cls.webserver = vaex.server.WebServer(dfs=[], port=test_port, cache_byte_size=0).tornado_server
		#print "serving"
		cls.webserver.serve_threaded()
		#print "getting server object"
		scheme = "ws" if cls.use_websocket else "http"
		cls.server = vx.server("%s://localhost:%d" % (scheme, test_port))
		test_port += 1


	@classmethod
	def tearDownClass(cls):
		cls.server.close()
		cls.webserver.stop_serving()


	def setUp(self):
		# run all tests from TestDataset, but now served at the server
		super(TestDatasetRemote, self).setUp()
		# for the webserver we don't support filters on top of filters
		# so the server always uses the full dataset
		# self.dataset_no_filter.name = 'dataset'
		# self.dataset = self.dataset_no_filter

		self.dataset_local = self.dataset
		self.datasetxy_local = self.datasetxy
		self.dataset_concat_local = self.dataset_concat
		self.dataset_concat_dup_local = self.dataset_concat_dup

		dfs = [self.dataset_local, self.datasetxy_local, self.dataset_concat_local, self.dataset_concat_dup_local]
		#print "get dfs"
		self.webserver.set_dfs(dfs)
		dfs = self.server.dfs(as_dict=True)
		#print "got it", dfs

		self.dataset = dfs["dataset"]
		self.datasetxy = dfs["datasetxy"]
		self.dataset_concat = dfs["dataset_concat"]
		self.dataset_concat_dup = dfs["dataset_concat_dup"]
		#print "all done"



	def tearDown(self):
		TestDataset.tearDown(self)
		#print "stop serving"

	def test_to(self):
		pass # not supported

	def test_amuse(self):
		pass # no need

	def test_ascii(self):
		pass  # no need

	def test_csv(self):
		pass # no need
	def test_export(self):
		pass # we can't export atm

	def test_concat(self):
		pass # doesn't make sense to test this for remote

	def test_data_access(self):
		pass

	def test_byte_size(self):
		pass # we don't know the selection's length for dataset remote..

	def test_add_column(self):
		pass # can't add column to remove objects

	def test_rename_column(self):
		pass # TODO: we cannot do that now

	def test_masked_array_output(self):
		pass # cannot test exporting

	def test_export_sorted(self):
		pass # cannot test exporting

	def test_formats(self):
		pass # cannot test exporting

	def test_default_selection(self):
		pass # uses local information

	#def test_selection(self):
	#	pass

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

# import vaex.distributed
#class A:#class T_estDatasetDistributed(unittest.TestCase):
#class TestDatasetDistributed(unittest.TestCase):
class TestDatasetDistributed(TestDatasetRemote):

	use_websocket = False

	def setUp(self):
		TestDataset.setUp(self)
		global test_port
		# self.dataset_local = self.dataset = dataset.DatasetArrays("dataset")
		self.dataset_local = self.dataset

		dfs = [self.dataset]
		dfs_copy = [k.copy() for k in dfs] # otherwise we share the selection cache
		for ds in dfs_copy:
			ds.name = self.dataset_local.name
		dfs = [self.dataset]
		self.webserver1 = vaex.server.WebServer(dfs=dfs, port=test_port).tornado_server
		self.webserver1.serve_threaded()
		test_port += 1
		self.webserver2 = vaex.server.WebServer(dfs=dfs_copy, port=test_port).tornado_server
		self.webserver2.serve_threaded()
		test_port += 1

		scheme = "ws" if self.use_websocket else "http"
		self.server1 = vx.server("%s://localhost:%d" % (scheme, test_port-2))
		self.server2 = vx.server("%s://localhost:%d" % (scheme, test_port-1))
		test_port += 1
		dfs1 = self.server1.dfs(as_dict=True)
		dfs2 = self.server2.dfs(as_dict=True)
		self.dfs = [dfs1["dataset"], dfs2["dataset"]]
		self.dataset = vaex.distributed.DatasetDistributed(self.dfs)



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

		self.webserver = vaex.server.WebServer(dfs=[self.dataset], port=test_port).tornado_server
		self.webserver.serve_threaded()
		self.server = vx.server("http://localhost:%d" % test_port)
		self.dataset_remote = self.server.dfs()[0]

	def tearDown(self):
		self.webserver.stop_serving()

	def test_list(self):
		dfs = self.server.dfs()
		self.assertTrue(len(dfs) == 1)
		dataset_remote = dfs[0]
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
