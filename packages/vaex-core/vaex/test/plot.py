__author__ = 'maartenbreddels'
import unittest
import os
import tempfile
import logging
import shutil

import numpy as np
import PIL.Image
import PIL.ImageChops
import pylab as plt

import vaex as vx
import vaex.utils

try:
	raw_input = input
except:
	pass # py2/3 fix

base_path = os.path.dirname(__file__)
def get_comparison_image(name):
	osname = vaex.utils.osname
	return os.path.join(base_path, "images", name+"_" + osname + ".png")

overwrite_images = False
class check_output(object):
	def __init__(self, name):
		self.name = name
		self.fn = get_comparison_image(name)


	def __enter__(self):
		plt.figure()

	def __exit__(self, exc_type, exc_val, exc_tb):
		if exc_type is not None:
			plt.close()
			return
		fn = tempfile.mktemp(".png")
		plt.savefig(fn)
		if not os.path.exists(self.fn):
			print("comparison image did not exist, copying to %s" % self.fn)
			shutil.copy(fn, self.fn)

		image1 = PIL.Image.open(self.fn)
		image2 = PIL.Image.open(fn)
		diff = PIL.ImageChops.difference(image1, image2)
		extrema = diff.getextrema()
		for i, (vmin, vmax) in enumerate(extrema):
			msg = "difference found between {im1} and {im2} in band {band}\n $ cp {im2} {im1}".format(im1=self.fn, im2=fn,
																									  band=i)
			if vmin != vmax and overwrite_images:
				image1.show()
				image2.show()
				done = False
				while not done:
					answer = raw_input("is the new image ok? [y/N]").lower().strip()
					if answer == "n":
						self.assertEqual(vmin, 0, msg)
						return
					if answer == "y":
						shutil.copy(fn, self.fn)
						return
			assert vmin == 0, msg
			assert vmax == 0, msg
		plt.close()


class TestPlot(unittest.TestCase):
	def setUp(self):
		self.dataset = vx.example()

	def tearDown(self):
		if vx.utils.osname != "osx":
			self.dataset.close_files()

	def test_single(self):
		with check_output("single_xy"):
			self.dataset.plot("x", "y", title="face on")

	def test_single_nan(self):
		with check_output("single_xy_no_nan"):
			self.dataset.plot("x", "y", f="log")
		cm = plt.cm.inferno
		cm.set_bad("orange")
		with check_output("single_xy_nan"):
			self.dataset.plot("x", "y", f="log", colormap=cm)

	def test_multiplot(self):
		with check_output("multiplot_xy"):
			self.dataset.plot([["x", "y"], ["x", "z"]], title="Face on and edge on", figsize=(10, 4));

	def test_multistat(self):
		with check_output("multistat"):
			self.dataset.plot("x", "y", what=["count(*)", "mean(vx)", "correlation(vy, vz)"], title="Different statistics", figsize=(10,5));

	def test_multiplot_multiwhat(self):
		with check_output("multiplot_multiwhat"):
			self.dataset.plot([["x", "y"], ["x", "z"], ["y", "z"]],
				  what=["count(*)", "mean(vx)", "correlation(vx, vy)", "correlation(vx, vz)"],
				  title="Different statistics and plots", figsize=(14, 12));

	def test_multistat_multiwhat_swapped(self):
		with check_output("multistat_multiwhat_swapped"):
			self.dataset.plot([["x", "y"], ["x", "z"], ["y", "z"]],
				what=["count(*)", "mean(vx)", "correlation(vx, vy)", "correlation(vx, vz)"],
				visual=dict(row="what", column="subspace"),
				title="Different statistics and plots", figsize=(14,12));

	def test_slice(self):
		with check_output("slice"):
			self.dataset.plot("Lz", "E", z="FeH:-3,-1,10", visual=dict(row="z"), figsize=(12,8), f="log", wrap_columns=3);

	def test_plot1d(self):
		with check_output("plot1d"):
			self.dataset.plot1d("Lz");

	def test_scatter(self):
		self.dataset.set_active_fraction(0.01)
		with check_output("scatter"):
			self.dataset.scatter("Lz", "E", length_check=False);
		with check_output("scatter_xerr"):
			self.dataset.scatter("Lz", "Lz", xerr="abs(Lz*0.1)", length_check=False);
		with check_output("scatter_xerr_yerr"):
			self.dataset.scatter("Lz", "Lz", xerr="abs(Lz*0.1)", yerr="abs(Lz*0.4)", length_check=False);
		with check_output("scatter_xerr_yerr_asym"):
			self.dataset.scatter("Lz", "Lz", xerr=["abs(Lz*0.1)", "abs(Lz*0.2)"], yerr=["abs(Lz*0.2)", "abs(Lz)"], length_check=False);


	def test_healpix(self):
		self.dataset.add_virtual_columns_cartesian_to_spherical()
		self.dataset.add_column_healpix(longitude="l", latitude="b")
		with check_output("plot_healpix"):
			self.dataset.healpix_plot("healpix");

if __name__ == '__main__':
    unittest.main()
