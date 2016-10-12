from __future__ import absolute_import
__author__ = 'maartenbreddels'
import numpy as np
import threading
import vaex.image
import logging
import vaex as vx
from .common import Job
import os
from IPython.display import HTML, display_html, display_javascript
import bqplot.marks
import bqplot as bq
import bqplot.interacts

logger = logging.getLogger("vaex.ext.bqplot")
base_path = os.path.dirname(__file__)

select_lock = threading.Lock()

@bqplot.marks.register_mark('vaex.ext.bqplot.Image')
class Image(bqplot.marks.Mark):
    src = bqplot.marks.Unicode().tag(sync=True)
    x = bqplot.marks.Float().tag(sync=True)
    y = bqplot.marks.Float().tag(sync=True)
    width = bqplot.marks.Float().tag(sync=True)
    height = bqplot.marks.Float().tag(sync=True)
    preserve_aspect_ratio = bqplot.marks.Unicode('').tag(sync=True)
    _model_module = bqplot.marks.Unicode('vaex.ext.bqplot').tag(sync=True)
    _view_module = bqplot.marks.Unicode('vaex.ext.bqplot').tag(sync=True)

    _view_name = bqplot.marks.Unicode('Image').tag(sync=True)
    _model_name = bqplot.marks.Unicode('ImageModel').tag(sync=True)
    scales_metadata = bqplot.marks.Dict({
        'x': {'orientation': 'horizontal', 'dimension': 'x'},
        'y': {'orientation': 'vertical', 'dimension': 'y'},
    }).tag(sync=True)
    def __init__(self, **kwargs):
        self._drag_end_handlers = bqplot.marks.CallbackDispatcher()
        super(Image, self).__init__(**kwargs)
import warnings

def patch():
	#return
	display_javascript(open(os.path.join(base_path, "bqplot_ext.js")).read(), raw=True)
	#if (bqplot.__version__ == (0, 6, 1)) or (bqplot.__version__ == "0.6.1"):
	#else:
	#	warnings.warn("This version (%s) of bqplot is not supppored" % bqplot.__version__)


class Bqplot(object):
	def __init__(self, subspace, size, limits):
		"""

		:type subspace: Subspace
		"""
		self.size = size
		self.subspace = subspace
		self.limits = limits
		if self.subspace.dataset.has_selection():
			sub = self.subspace.selected()
		else:
			sub = self.subspace

		grid = sub.histogram(size=self.size, limits=self.limits)
		self.create(grid)
		self.callback = self.subspace.dataset.signal_selection_changed.connect(lambda dataset: self._update())
		self.previous_task = None

		self.brush = bqplot.interacts.BrushIntervalSelector(scale=self.scale_x)
		self.brush.observe(self._update_interval, ["selected", "brushing"])
		self.fig.interaction = self.brush

		#updater = vaex.ext.bqplot.DebouncedThreadedUpdater(self.subspace.dataset, im, make_image, progress_widget=progress)

	def _update_interval(self, change):
		if not self.brush.brushing and len(self.brush.selected):
			xmin, xmax = min(self.brush.selected), max(self.brush.selected)
			expression = "({expr} >= {min}) & ({expr} <= {max})".format(expr=self.subspace.expressions[0], min=xmin-0.5, max=xmax+0.5)
			if not select_lock.acquire(False):
				logger.error("lock possible")
			else:
				select_lock.release()
			with select_lock:
				logger.warning("START LOCK _update_interval")
				self.subspace.dataset.select(expression)
				logger.warning("END LOCK _update_interval")
			print((self.brush.selected, self.brush.brushing, expression))
			self._update()



	def create(self, data):
		pass

	def _update(self):
		get_ioloop().add_callback(self.__update)
	def __update(self):
		if not select_lock.acquire(False):
			logger.error("lock possible in _update")
		else:
			select_lock.release()
		with select_lock:
			logger.warning("START LOCK _update")
			logger.info(("updating histogram", self.scale_x.min, self.scale_x.max))
			#self.limits = [[self.scale_x.min, self.scale_x.max]]
			print("limits = %r size=%s" % (self.limits, self.size))
			if self.subspace.dataset.has_selection():
				logger.debug(("has selection"))
				sub = self.subspace.selected()
			else:
				logger.debug(("no selection"))
				sub = self.subspace
			#import traceback
			#	traceback.print_stack()

			sub = sub.asynchronous()
			if self.previous_task:
				logger.debug(("cancel previous task"))
				self.previous_task.cancel()
			logger.debug(("adding task"))
			self.previous_task = grid = sub.histogram(size=self.size, limits=self.limits)
			def do_update(grid):
				logger.debug(("updating histogram grid"))
				self.previous_task = None
				self.update(grid)
			grid.then(do_update).end()
			logger.debug(("added task"))
			self.subspace.executor.execute_threaded()
			logger.warning("END LOCK _update")

	def update(self, data):
		pass

#bq_thread = vaex.ext.common.SingleJobThread()

class BqplotHistogram(Bqplot):
	def __init__(self, subspace, color, size, limits):
		self.color = color
		super(BqplotHistogram, self).__init__(subspace, size, limits)

	def create(self, data):
		size = data.shape[0]
		assert len(data.shape) == 1
		xmin, xmax = self.limits[0]
		dx = (xmax - xmin) / size
		x = np.linspace(xmin, xmax-dx, size)+ dx/2
		#print xmin, xmax, x

		self.scale_x = bq.LinearScale(min=xmin+dx/2, max=xmax-dx/2)
		self.scale_y = bq.LinearScale()

		self.axis_x = bq.Axis(label='X', scale=self.scale_x)
		self.axis_y = bq.Axis(label='Y', scale=self.scale_y, orientation='vertical')
		self.bars = bq.Bars(x=x,
						 y=data, scales={'x': self.scale_x, 'y': self.scale_y}, colors=[self.color])

		self.fig = bq.Figure(axes=[self.axis_x, self.axis_y], marks=[self.bars], padding_x=0)

	def update(self, data):
		self.bars.y = data


def BqplotHistogram2d(Bqplot):
	def __init__(self, subspace, color, size, limits):
		self.color = color
		super(BqplotHistogram, self).__init__(subspace, size, limits)

	def create(self, data):
		pass

import time
def debounced(delay_seconds=0.5):
	def wrapped(f):
		locals = {"counter": 0}
		def execute(*args, **kwargs):
			locals["counter"] += 1
			#print "counter", locals["counter"]
			def debounced_execute(counter=locals["counter"]):
				#$print "counter is", locals["counter"]
				if counter == locals["counter"]:
					logger.info("debounced call")
					f(*args, **kwargs)
				else:
					logger.info("debounced call skipped")
			ioloop = get_ioloop()
			def thread_safe():
				ioloop.add_timeout(time.time() + delay_seconds, debounced_execute)
			ioloop.add_callback(thread_safe)
		return execute
	return wrapped
def get_ioloop():
    import IPython, zmq
    ipython = IPython.get_ipython()
    if ipython and hasattr(ipython, 'kernel'):
        return zmq.eventloop.ioloop.IOLoop.instance()


class DebouncedThreadedUpdater(object):
	def __init__(self, subspace, size, bqplot_image, factory_rgba8, delay=0.5, progress_widget=None):
		self.subspace = subspace

		self.dataset = subspace.dataset
		self.size = size
		self.bqplot_image = bqplot_image
		self.factory_rgba8 = factory_rgba8
		self.delay = delay
		self.progress_widget = progress_widget
		self.previous_task = None
		#self.job_grid = None
		#self.job_select = None
		#self.executor = vx.execution.Executor()

	@debounced(0.5)
	def update_select(self, f):
		def wrapped():
			if not select_lock.acquire(False):
				logger.error("lock possible in _update_select")
			else:
				select_lock.release()
			with select_lock:
				logger.warning("START LOCK _update_select")
				f()
				logger.warning("END LOCK _update_select")
		get_ioloop().add_callback(wrapped)
		return
		def work(job):
			logger.debug("executing selection job for updater %r", self)
			print ("executing selection job for updater %r", self)
			steps = 1
			step = [0]
			def update_progress(fraction):
				if self.progress_widget:
					self.progress_widget.value = fraction/steps + step[0]/float(steps)
				#return True
				if job.cancelled:
					self.executor.signal_progress.disconnect(update_progress)
					return False
				else:
					return True
			try:
				callback = self.executor.signal_progress.connect(update_progress)
				logger.debug("selecting")
				f()
			finally:
				self.executor.signal_progress.disconnect(callback)
		if self.job:
			logger.debug("cancelling selection job")
			self.job.cancel()
		logger.debug("scheduling selection job")
		self.job = Job(work, delay=self.delay)
		self.job.schedule()

	@debounced(0.5)
	def update(self, limits):
		self.limits = limits
		self.update_direct()

	def update_direct_safe(self):
		get_ioloop().add_callback(self.update_direct)

	def update_direct(self):
		if not select_lock.acquire(False):
			logger.error("lock possible in update_direct")
			import traceback
			traceback.print_stack()
		else:
			select_lock.release()
		with select_lock:
			logger.warning("START LOCK update_direct")
			logger.info("updating histogram2d")
			if self.subspace.dataset.has_selection():
				logger.debug(("has selection"))
				sub = self.subspace.selected()
			else:
				logger.debug(("no selection"))
				sub = self.subspace
			sub = sub.asynchronous()
			if self.previous_task:
				self.previous_task.cancel()
			self.previous_task = grid = sub.histogram(size=self.size, limits=self.limits)
			def do_update(grid):
				logger.info("got grid")
				self.previous_task = None
				self.update_im(grid)
			grid.then(do_update).then(None, self.on_error).end()
			self.subspace.executor.execute_threaded()
			logger.warning("END LOCK update_direct")

	def on_error(self, error):
		logger.exception("error occured: %r", error, exc_info=error)
		import traceback
		traceback.print_exc()
		#raise exception
		raise error

	def update_im(self, grid):
		logger.debug("transform to rgba")
		rgba8 = self.subspace.image_rgba(grid=grid, f="log")
		logger.debug("transform to url")
		src = self.subspace.image_rgba_url(rgba8=rgba8)
		logger.debug("updating bqplot image")
		self.bqplot_image.src = src
		xlim, ylim = self.limits
		self.bqplot_image.x = xlim[0]
		self.bqplot_image.y = ylim[1]
		self.bqplot_image.width = xlim[1] - xlim[0]
		self.bqplot_image.height = -(ylim[1] - ylim[0])

	def _old(self):
		xlim, ylim = limits
		def work(job):
			logger.debug("executing job for updater %r", self)
			executor = vx.execution.Executor()
			steps = 1
			step = [0]
			def update_progress(fraction):
				if self.progress_widget:
					self.progress_widget.value = fraction/steps + step[0]/float(steps)
				#return True
				if job.cancelled:
					executor.signal_progress.disconnect(update_progress)
					return False
				else:
					return True
			callback = None
			try:
				if self.progress_widget:
					self.progress_widget.description = "Calculating"
				callback = executor.signal_progress.connect(update_progress)
				logger.debug("creating image")
				rgba8 = self.factory_rgba8(executor, limits)
			finally:
				if callback:
					try:
						executor.signal_progress.disconnect(callback)
					except:
						pass # TODO: why does this sometimes fail??
				if self.progress_widget:
					self.progress_widget.description = "Done"
			src = vaex.image.rgba_to_url(rgba8)
			def update_bqplot():
				logger.debug("updating bqplot image")
				self.bqplot_image.src = src
				self.bqplot_image.x = xlim[0]
				self.bqplot_image.y = ylim[1]
				self.bqplot_image.width = xlim[1] - xlim[0]
				self.bqplot_image.height = -(ylim[1] - ylim[0])
			update_bqplot() # do we need to do this from the proper thread?
		if self.job:
			logger.debug("cancelling job")
			self.job.cancel()
		logger.debug("scheduling job")


job = None
def debounced_threaded_update(dataset, bqplot_image, factory_rgba8, limits, delay=0.5, progress_widget=None):
	global job
	xlim, ylim = limits
	def work(job):
		logger.debug("executing job")
		# TODO, use subspace's exector.. it may not be the same..
		steps = 1
		step = [0]
		def update_progress(fraction):
			if progress_widget:
				progress_widget.value = fraction/steps + step[0]/float(steps)
			if job.cancelled:
				dataset.executor.signal_progress.disconnect(callback)
				return False
			else:
				return True
		try:
			callback = dataset.executor.signal_progress.connect(update_progress)
			logger.debug("creating image")
			rgba8 = factory_rgba8(limits)
		finally:
			dataset.executor.signal_progress.disconnect(callback)
		src = vaex.image.rgba_to_url(rgba8)
		def update_bqplot():
			logger.debug("updating bqplot image")
			bqplot_image.src = src
			bqplot_image.x = xlim[0]
			bqplot_image.y = ylim[1]
			bqplot_image.width = xlim[1] - xlim[0]
			bqplot_image.height = -(ylim[1] - ylim[0])
		update_bqplot() # do we need to do this from the proper thread?
	if job:
		logger.debug("cancelling job")
		job.cancel()
	logger.debug("scheduling job")
	job = Job(work, delay=delay)
	job.schedule()