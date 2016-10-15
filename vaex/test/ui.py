__author__ = 'maartenbreddels'
import unittest
import os
import tempfile
import logging
import numpy as np
import PIL.Image
import PIL.ImageChops

import vaex as vx
import vaex.ui
import vaex.ui.main
import vaex.ui.layers
import vaex.utils
import vaex.dataset
import vaex.execution
import vaex.webserver
from vaex.ui.qt import QtGui, QtCore, QtTest

import vaex.ui.qt as dialogs
import random

# this will trigger more code, such as the canceling in between computation
vaex.execution.buffer_size = 10000
#vx.set_log_level_off()

example_path = vaex.utils.get_data_file("helmi-dezeeuw-2000-10p.hdf5")
vaex.ui.hidden = True

qt_app = QtGui.QApplication([])

base_path = os.path.dirname(__file__)
def get_comparison_image(name):
	osname = vaex.utils.osname
	return os.path.join(base_path, "images", name+"_" + osname + ".png")

#logging.getLogger("vaex.ui.queue").setLevel(logging.DEBUG)
#logging.getLogger("vaex.ui").setLevel(logging.DEBUG)
vx.set_log_level_warning()
#vx.set_log_level_debug()
import logging
logger = logging.getLogger("vaex.test.ui")


overwrite_images = False

class CallCounter(object):
	def __init__(self, return_value=None):
		self.counter = 0
		self.return_value = return_value

	def __call__(self, *args, **kwargs):
		self.counter += 1
		return self.return_value

class TestApp(unittest.TestCase):
	def setUp(self):
		self.dataset = vaex.dataset.DatasetArrays("dataset")

		self.x = x = np.arange(10)
		self.y = y = x ** 2
		self.dataset.add_column("x", x)
		self.dataset.add_column("y", y)
		self.dataset.set_variable("t", 1.)
		self.dataset.add_virtual_column("z", "x+t*y")

		self.app = vx.ui.main.VaexApp()

	def test_generate_data(self):
		actions = self.app.menu_data.actions()
		for action in actions:
			if "Soneira Peebles" in action.text():
				action.trigger()
				self.assertIsInstance(self.app.current_dataset, vx.file.other.SoneiraPeebles)
				break

		for action in actions:
			if "Zeldovich" in action.text():
				action.trigger()
				self.assertIsInstance(self.app.current_dataset, vx.file.other.Zeldovich)
				break

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

	def test_export(self):
		path_hdf5 = tempfile.mktemp(".hdf5")
		path_hdf5_ui = tempfile.mktemp(".hdf5")
		path_fits = tempfile.mktemp(".fits")
		path_fits_ui = tempfile.mktemp(".fits")

		for dataset in [self.dataset]:
			self.app.dataset_selector.add(dataset)
			for fraction in [1, 0.5]:
				dataset.set_active_fraction(fraction)
				dataset.select("x > 3")
				length = len(dataset)
				# TODO: gui doesn't export virtual columns, add "z" to this list
				for column_names in [["x", "y"], ["x"], ["y"]]:
					for byteorder in "=<>":
						for shuffle in [False, True]:
							for selection in [False, True]:
								for export in [dataset.export_fits, dataset.export_hdf5] if byteorder == ">" else [dataset.export_hdf5]:
									type = "hdf5" if export == dataset.export_hdf5 else "fits"
									if shuffle and selection:
										continue # TODO: export should fail on this combination
									#print column_names, byteorder, shuffle, selection, type
									if export == dataset.export_hdf5:
										path = path_hdf5
										path_ui = path_hdf5_ui
										export(path, column_names=column_names, byteorder=byteorder, shuffle=shuffle, selection=selection)
									else:
										path = path_fits
										path_ui = path_fits_ui
										export(path, column_names=column_names, shuffle=shuffle, selection=selection)
									compare_direct = vx.open(path)

									dialogs.set_choose(1 if selection else 0).then("=<>".index(byteorder))
									# select columns
									dialogs.set_select_many(True, [name in column_names for name in dataset.get_column_names()])
									counter_confirm = CallCounter(return_value=shuffle)
									counter_info = CallCounter()
									dialogs.dialog_confirm = counter_confirm
									dialogs.dialog_info = counter_info
									dialogs.get_path_save = lambda *args: path_ui
									dialogs.ProgressExecution = dialogs.FakeProgressExecution
									import sys
									sys.stdout.flush()

									self.app.export(type=type)
									compare_ui = vx.open(path_ui)

									column_names = column_names or ["x", "y", "z"]
									self.assertEqual(compare_direct.get_column_names(), compare_ui.get_column_names())
									for column_name in column_names:
										values_ui = compare_ui.evaluate(column_name)
										values = compare_direct.evaluate(column_name)
										self.assertEqual(sorted(values), sorted(values_ui))


class __TestPlotPanel(unittest.TestCase):
	def setUp(self):
		self.app = vx.ui.main.VaexApp([], open_default=True)

	def test_open_and_close(self):
		button = self.app.dataset_panel.button_2d
		self.app.show()
		self.app.hide()
		self.assert_(len(self.app.windows) == 0)
		QtTest.QTest.mouseClick(button, QtCore.Qt.LeftButton)
		self.assertEqual(len(self.app.windows), 1)
		self.assertEqual(self.app.windows[0], self.app.current_window)
		self.assertEqual(self.app.windows[0].dataset, self.app.current_dataset)

		self.app.current_window.close()
		self.assert_(len(self.app.windows) == 0)
		self.assertEqual(None, self.app.current_window)

from vaex.ui.plot_windows import PlotDialog

# hide the class from the main namespace
class NoTest:
	class TestPlotPanel(unittest.TestCase):
		def create_app(self):
			self.app = vx.ui.main.VaexApp([], open_default=True)
		def setUp(self):
			vaex.dataset.main_executor = None
			vaex.dataset.main_executor = None
			vaex.multithreading.main_pool = None
			self.create_app()
			self.app.show()
			self.app.hide()
			self.open_window()
			self.window = self.app.current_window
			#self.window.xlabel = ""
			#self.window.ylabel = ""
			self.window.set_plot_size(512, 512)
			self.window.show()
			self.window.hide()
			self.layer = self.window.current_layer
			self.layer.state.colorbar = False
			self.no_exceptions = True
			import sys
			def testExceptionHook(type, value, tback):
				self.no_exceptions = False
				self.exception_info = (type, value, tback)
				sys.__excepthook__(type, value, tback)

			sys.excepthook = testExceptionHook

			self.no_error_in_field = True
			def error_in_field(*args):
				print(args)
				self.no_error_in_field = False
				previous_error_in_field(*args)
			previous_error_in_field = vaex.ui.layers.LayerTable.error_in_field
			vaex.ui.layers.LayerTable.error_in_field = error_in_field
			def log_error(*args):
				print("dialog error", args)
			dialogs.dialog_error = log_error


		def test_favorite_selections(self):
			with dialogs.assertError():
				self.window.action_selection_add_favorites.trigger()
			self.window.current_layer.dataset.select("x > 5")
			name = "test-selection-" + str(random.random())
			with dialogs.settext(name):
				self.window.action_selection_add_favorites.trigger()
				self.assertIn(name, self.window.current_layer.dataset.favorite_selections)
				found = 0
				found_index = 0
				for i, action in enumerate(self.window._favorite_selections_actions):
					if action.text() == name:
						found += 1
						index = i
				self.assertEqual(found, 1, "expect the entry to occur once, occurred %d" % found)
			with dialogs.setchoose(index):
				self.window.action_selection_remove_favorites.trigger()
				self.assertNotIn(name, self.window.current_layer.dataset.favorite_selections)
				found = 0
				for action in self.window.menu_selection.actions():
					if action.text() == name:
						found += 1
				self.assertEqual(found, 0, "expect the entry to occur be removed, but it occurred %d" % found)
			self.window.dataset.favorite_selections.clear()
			with dialogs.assertError():
				self.window.action_selection_remove_favorites.trigger()
			self.window.remove_layer()
			with dialogs.assertError():
				self.window.action_selection_remove_favorites.trigger()
			with dialogs.assertError():
				self.window.action_selection_add_favorites.trigger()

		#def test_bla(self):
		#	self.window.current_layer.x = "dsadsa"
		#	self.window._wait()
		def tearDown(self):
			#vx.promise.rereaise_unhandled()
			for dataset in self.app.dataset_selector.datasets:
				dataset.close_files()
				dataset.remove_virtual_meta()
			self.window.close()
			if not self.no_exceptions:
				type, value, tback = self.exception_info
				import traceback
				print("printing traceback:")
				traceback.print_exception(type, value, tback)

			self.assertTrue(self.no_exceptions)
			self.assertTrue(self.no_error_in_field)

		def compare(self, fn1, fn2):
			if not os.path.exists(fn2) and overwrite_images:
				import shutil
				shutil.copy(fn1, fn2)
			assert os.path.exists(fn2), "image missing: cp {im1} {im2}".format(im1=fn1, im2=fn2)

			try:
				image1 = PIL.Image.open(fn1)
				image2 = PIL.Image.open(fn2)
				diff = PIL.ImageChops.difference(image1, image2)
				extrema = diff.getextrema()
				for i, (vmin, vmax) in enumerate(extrema):
					msg = "difference found between {im1} and {im2} in band {band}\n $ cp {im1} {im2}".format(im1=fn1, im2=fn2, band=i)
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
								import shutil
								shutil.copy(fn1, fn2)
								return
					self.assertEqual(vmin, 0, msg)
					self.assertEqual(vmax, 0, msg)
			finally:
				pass
				#image1.close()
				#image2.close()

		def test_navigation_history(self):
			self.window._wait()
			self._assert_default_image()

			self._do_zoom()
			self.window._wait()
			self._assert_zoom_image()

			self.window.action_undo.trigger()
			self.window._wait()
			self._assert_default_image()

			self.window.action_redo.trigger()
			self.window._wait()
			self._assert_zoom_image()

		def test_empty_region(self):
			self._move_to_empty_region()
			self._assert_empty_image()
			self.window._wait()


class TestPlotPanel1d(NoTest.TestPlotPanel):
	def _move_to_empty_region(self):
		x = 1e4
		self.window.current_layer.xlim = [-10+x, 10+x]

	def _do_zoom(self):
		self.window.current_layer.xlim = [-10, 10]

	def _assert_default_image(self):
		filename = self.window.plot_to_png()
		self.compare(filename, get_comparison_image("default1d"))

	def _assert_empty_image(self):
		filename = self.window.plot_to_png()
		self.compare(filename, get_comparison_image("empty1d"))

	def _assert_zoom_image(self):
		filename = self.window.plot_to_png()
		self.compare(filename, get_comparison_image("zoom1d"))

	def open_window(self):
		button = self.app.dataset_panel.button_histogram
		self.assert_(len(self.app.windows) == 0)
		QtTest.QTest.mouseClick(button, QtCore.Qt.LeftButton)

	def test_x(self):
		#QtTest.QTest.qWait(self.window.queue_update.default_delay)
		self.window._wait()
		filename = self.window.plot_to_png()
		self.compare(filename, get_comparison_image("example_x"))

	def test_r(self):
		self.layer.x = "sqrt(x**2+y**2)"
		self.window._wait()
		filename = self.window.plot_to_png()
		self.compare(filename, get_comparison_image("example_r"))

class TestPlotPanel2d(NoTest.TestPlotPanel):
	"""
	:type window: PlotDialog
	"""
	def open_window(self):
		button = self.app.dataset_panel.button_2d
		self.assert_(len(self.app.windows) == 0)
		QtTest.QTest.mouseClick(button, QtCore.Qt.LeftButton)

	def _move_to_empty_region(self):
		y = x = 1e4
		self.window.current_layer.xlim = [-10+x, 10+x]
		self.window.current_layer.ylim = [-10+y, 10+y]

	def _do_zoom(self):
		self.window.current_layer.xlim = [-10, 10]
		self.window.current_layer.ylim = [-10, 10]

	def _assert_default_image(self):
		filename = self.window.plot_to_png()
		self.compare(filename, get_comparison_image("default2d"))

	def _assert_empty_image(self):
		filename = self.window.plot_to_png()
		self.compare(filename, get_comparison_image("empty2d"))

	def _assert_zoom_image(self):
		filename = self.window.plot_to_png()
		self.compare(filename, get_comparison_image("zoom2d"))


	def test_xy(self):
		#QtTest.QTest.qWait(self.window.queue_update.default_delay)
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

	def test_xy_vxvy_as_option(self):
		self.window.remove_layer()
		self.window.add_layer(["x", "y"], vx="vx", vy="vy", colorbar="False")
		self.window._wait()
		filename = self.window.plot_to_png()
		self.compare(filename, get_comparison_image("example_xy_vxvy"))

	def test_select_by_expression(self):
		self.window.xlabel = "x"
		self.window.ylabel = "y"
		##self.window._wait() # TODO: is this a bug? if we don't wait and directly do the selection, the ThreadPoolIndex
		## is entered twice, not sure this can happen from the gui
		expression = "x < 0"
		vaex.ui.qt.set_choose(expression, True)
		logger.debug("click mouse")
		QtTest.QTest.mouseClick(self.layer.button_selection_expression, QtCore.Qt.LeftButton)
		logger.debug("clicked mouse")
		return
		self.window._wait()
		self.assertTrue(self.no_exceptions)

		filename = self.window.plot_to_png()
		self.compare(filename, get_comparison_image("example_xy_selection_on_x"))

	def test_selection_options(self):
		self.window.xlabel = "x"
		self.window.ylabel = "y"
		##self.window._wait() # TODO: is this a bug? if we don't wait and directly do the selection, the ThreadPoolIndex
		## is entered twice, not sure this can happen from the gui
		expression = "x < 0"
		vaex.ui.qt.set_choose(expression, True)
		QtTest.QTest.mouseClick(self.layer.button_selection_expression, QtCore.Qt.LeftButton)

		# test if the expressions ends up in the clipboard
		self.window.action_selection_copy.trigger()
		clipboard = QtGui.QApplication.clipboard()
		clipboard_text = clipboard.text()
		self.assertIn(expression, clipboard_text)

		self.window._wait()
		self.assertTrue(self.no_exceptions)
		filename = self.window.plot_to_png()
		self.compare(filename, get_comparison_image("example_xy_selection_on_x"))


		# select nothing
		self.window.action_select_none.trigger()
		self.window._wait()
		filename = self.window.plot_to_png()
		self.compare(filename, get_comparison_image("example_xy"))
		with dialogs.assertError(1):
			self.window.action_selection_copy.trigger()

		# paste the expression back
		print("paste back the expression")
		clipboard.setText(clipboard_text)
		self.window.action_selection_paste.trigger()
		print("waiting....")
		self.window._wait()
		filename = self.window.plot_to_png()
		self.compare(filename, get_comparison_image("example_xy_selection_on_x"))



		#clipboard.setText("junk")
		#with dialogs.assertError(1):
		#	self.window.action_selection_paste.trigger()
			#self.window.action_selection_copy.trigger()




	def test_select_by_lasso(self):
		self.window._wait() # TODO: is this a bug? same as above
		vaex.ui.qt.set_choose("x < 0", True)
		x = [-10, 10, 10, -10]
		y = [-10, -10, 10, 10]
		self.layer.dataset.select_lasso("x", "y", x, y)
		#QtTest.QTest.mouseClick(self.layer.button_selection_expression, QtCore.Qt.LeftButton)
		self.assertLess(self.layer.dataset.selected_length(), len(self.layer.dataset))
		self.window._wait()
		self.assertTrue(self.no_exceptions)

	def test_layers(self):
		self.window.add_layer(["x", "z"])
		self.window._wait()

	def test_resolution(self):
		if 0: # keyClick doesn't work on osx it seems
			self.window.show()
			QtTest.QTest.qWaitForWindowShown(self.window)
			QtTest.QTest.keyClick(self.window, QtCore.Qt.Key_1, QtCore.Qt.ControlModifier|QtCore.Qt.AltModifier) # "Ctrl+Alt+1"should be 32x32
		else:
			self.window.action_resolution_list[0].trigger()
		self.window._wait()
		filename = self.window.plot_to_png()
		self.compare(filename, get_comparison_image("example_xy_32x32"))

	def test_resolution_vector(self):
		self.layer.vx = "vx"
		self.layer.vy = "vy"
		self.window.action_resolution_vector_list[2].trigger()
		self.window._wait()
		filename = self.window.plot_to_png()
		self.compare(filename, get_comparison_image("example_xy_vxvy_32x32"))


	def test_invalid_expression(self):
		self.window._wait()

		with dialogs.assertError(2):
			self.layer.x = "vx*"
			self.layer.y = "vy&"
		with dialogs.assertError(3):
			self.layer.x = "hoeba(vx)"
			self.layer.x = "x(vx)"
			self.layer.y = "doesnotexist"
		with dialogs.assertError(2):
			self.layer.vx = "hoeba(vx)"
			self.layer.vy = "x(vx)"
		with dialogs.assertError(1):
			self.layer.weight = "hoeba(vx)"
		self.layer.x = "x"
		self.layer.y = "y"
		self.layer.weight = "z"
		#self.window._wait()
		# since this will be triggered, overrule it
		self.no_error_in_field = True

import sys
test_port = 29310 + sys.version_info[0] * 10 + sys.version_info[1]

class TestPlotPanel2dRemote(TestPlotPanel2d):
	use_websocket = False
	def create_app(self):
		logger.debug("create app")
		global test_port
		self.app = vx.ui.main.VaexApp([], open_default=False)
		self.dataset_default = vaex.example()
		datasets = [self.dataset_default]
		self.webserver = vaex.webserver.WebServer(datasets=datasets, port=test_port)
		#print "serving"
		logger.debug("serve server")
		self.webserver.serve_threaded()
		#print "getting server object"
		scheme = "ws" if self.use_websocket else "http"
		logger.debug("get from server")
		self.server = vx.server("%s://localhost:%d" % (scheme, test_port), thread_mover=self.app.call_in_main_thread)
		datasets = self.server.datasets(as_dict=True)
		logger.debug("got it")

		self.dataset = datasets[self.dataset_default.name]
		self.app.dataset_selector.add(self.dataset)
		test_port += 1
		logger.debug("create app done")

	def tearDown(self):
		logger.debug("closing all")
		#print "stop serving"
		TestPlotPanel2d.tearDown(self)
		self.webserver.stop_serving()
		self.server.close()


	def test_select_by_lasso(self):
		pass # TODO: cannot test since DatasetRemote.selected_length it not implemented

	#def test_invalid_expression(self): pass
	#def test_resolution_vector(self): pass
	#def test_resolution(self): pass
	#def test_layers(self): pass
	#def test_select_by_lasso(self): pass
	#def test_select_by_expression(self): pass
	#def test_xy_vxvy_as_option(self): pass
	#def test_xy_vxvy(self): pass
	#def test_xy_weight_r(self): pass
	#def test_xr(self): pass
	#def test_xy(self): pass

if __name__ == '__main__':
    unittest.main()
