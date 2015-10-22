__author__ = 'maartenbreddels'
import unittest
import os

import vaex as vx
import logging
vx.set_log_level_warning()
#vx.set_log_level_debug()

import vaex.ui
import vaex.ui.main
import vaex.utils

import PIL.Image
import PIL.ImageChops

from vaex.ui.qt import QtGui, QtCore, QtTest

example_path = vaex.utils.get_data_file("helmi-dezeeuw-2000-10p.hdf5")
vaex.ui.hidden = True

qt_app = QtGui.QApplication([])

base_path = os.path.dirname(__file__)
def get_comparison_image(name):
	return os.path.join(base_path, "images", name+".png")

#logging.getLogger("vaex.ui.queue").setLevel(logging.DEBUG)
#logging.getLogger("vaex.ui.layer").setLevel(logging.DEBUG)


class TestUiCreation(unittest.TestCase):
	def test_default(self):
		app = vx.ui.main.VaexApp(open_default=False)
		self.assert_(app.dataset_selector.is_empty())
		self.assertEqual(None, app.current_dataset)

		app = vx.ui.main.VaexApp(open_default=True)
		self.assert_(not app.dataset_selector.is_empty())
		self.assertIsNotNone(app.current_dataset)

	def test_add_dataset(self):
		app = vx.ui.main.VaexApp()
		ds = vx.example()
		app.dataset_selector.add(ds)
		self.assert_(not app.dataset_selector.is_empty())
		self.assertEqual(int(app.dataset_panel.label_length.text().replace(",", "")), len(ds))
		self.assertEqual(ds, app.current_dataset)

	def test_open_dataset(self):
		app = vx.ui.main.VaexApp()
		ds = app.dataset_selector.open(example_path)
		self.assert_(not app.dataset_selector.is_empty())
		self.assertEqual(int(app.dataset_panel.label_length.text().replace(",", "")), len(ds))
		self.assertEqual(ds, app.current_dataset)

class TestPlotPanel(unittest.TestCase):
	def setUp(self):
		self.app = vx.ui.main.VaexApp([], open_default=True)

	def test_open_and_close(self):
		button = self.app.dataset_panel.button_2d
		self.assert_(len(self.app.windows) == 0)
		QtTest.QTest.mouseClick(button, QtCore.Qt.LeftButton)
		self.assertEqual(len(self.app.windows), 1)
		self.assertEqual(self.app.windows[0], self.app.current_window)
		self.assertEqual(self.app.windows[0].dataset, self.app.current_dataset)

		self.app.current_window.close()
		self.assert_(len(self.app.windows) == 0)
		self.assertEqual(None, self.app.current_window)

from vaex.ui.plot_windows import PlotDialog

class TestPlotPanel2d(unittest.TestCase):
	"""
	:type window: PlotDialog
	"""
	def setUp(self):
		self.app = vx.ui.main.VaexApp([], open_default=True)
		button = self.app.dataset_panel.button_2d
		self.assert_(len(self.app.windows) == 0)
		QtTest.QTest.mouseClick(button, QtCore.Qt.LeftButton)
		self.window = self.app.current_window
		self.layer = self.window.current_layer
		self.no_exceptions = True
		import sys
		def testExceptionHook(type, value, tback):
			self.no_exceptions = False
			sys.__excepthook__(type, value, tback)

		sys.excepthook = testExceptionHook

	def tearDown(self):
		self.window.close()

	def compare(self, fn1, fn2):
		image1 = PIL.Image.open(fn1)
		image2 = PIL.Image.open(fn2)
		diff = PIL.ImageChops.difference(image1, image2)
		extrema = diff.getextrema()

		for i, (vmin, vmax) in enumerate(extrema):
			msg = "difference found between %s and %s in band %d" % (fn1, fn2, i)
			self.assertEqual(vmin, 0, msg)
			self.assertEqual(vmax, 0, msg)


	def test_xy(self):
		QtTest.QTest.qWait(self.window.queue_update.default_delay)
		self.window._wait()
		filename = self.window.plot_to_png()
		self.compare(filename, get_comparison_image("example_xy"))

	def test_xr(self):
		self.layer.y = "sqrt(x**2+y**2)"
		self.window._wait()
		filename = self.window.plot_to_png()
		self.compare(filename, get_comparison_image("example_xr"))

	def test_xy_weight_r(self):
		self.layer.weight = "sqrt(x**2+y**2)"
		self.layer.amplitude = "clip(average, 0, 40)"
		self.window._wait()
		filename = self.window.plot_to_png()
		self.compare(filename, get_comparison_image("example_xy_weight_r"))

	def test_xy_vxvy(self):
		self.layer.vx = "vx"
		self.layer.vy = "vy"
		self.window._wait()
		filename = self.window.plot_to_png()
		self.compare(filename, get_comparison_image("example_xy_vxvy"))

		counter = self.window.queue_update.counter
		# the following actions should not cause an update
		self.layer.vx = "vx"
		self.layer.vy = "vy"
		self.assertEqual(counter, self.window.queue_update.counter)
		self.layer.vx = ""
		self.assertEqual(counter, self.window.queue_update.counter)
		self.layer.vx = None
		self.assertEqual(counter, self.window.queue_update.counter)
		self.layer.vx = ""
		self.assertEqual(counter, self.window.queue_update.counter)
		# this should update it
		self.layer.vx = "vx"
		self.assertEqual(counter+1, self.window.queue_update.counter)
		self.window._wait()

	def test_select_by_expression(self):
		vaex.ui.qt.set_choose("x < 0", True)
		QtTest.QTest.mouseClick(self.layer.button_selection_expression, QtCore.Qt.LeftButton)
		self.window._wait()
		self.assertTrue(self.no_exceptions)

	def test_select_by_lasso(self):
		vaex.ui.qt.set_choose("x < 0", True)
		x = [-10, 10, 10, -10]
		y = [-10, -10, 10, 10]
		self.layer.dataset.lasso_select("x", "y", x, y)
		#QtTest.QTest.mouseClick(self.layer.button_selection_expression, QtCore.Qt.LeftButton)
		self.assertLess(self.layer.dataset.selected_length(), len(self.layer.dataset))
		self.window._wait()
		self.assertTrue(self.no_exceptions)




if __name__ == '__main__':
    unittest.main()
