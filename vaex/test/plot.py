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
		pass

	def __exit__(self, exc_type, exc_val, exc_tb):
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
			msg = "difference found between {im1} and {im2} in band {band}\n $ cp {im1} {im2}".format(im1=self.fn, im2=fn,
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


class TestPlot(unittest.TestCase):
	def setUp(self):
		self.dataset = vx.example()

	def tearDown(self):
		if vx.utils.osname != "osx":
			self.dataset.close_files()

	def test_single(self):
		with check_output("single_xy"):
			self.dataset.plot("x", "y", title="face on")

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
			self.dataset.plot("Lz", "E", z="FeH:-3,-1,10", show=True, visual=dict(row="z"), figsize=(12,8), f="log", wrap_columns=3);

if __name__ == '__main__':
    unittest.main()
