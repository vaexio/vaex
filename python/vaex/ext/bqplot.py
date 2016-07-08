from __future__ import absolute_import
__author__ = 'maartenbreddels'
import vaex.image
import logging
import vaex as vx
from .common import Job
import os
from IPython.display import HTML, display_html, display_javascript
import bqplot.marks


logger = logging.getLogger("vaex.ext.bqplot")
base_path = os.path.dirname(__file__)


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
	if (bqplot.__version__ == (0, 6, 1)) or (bqplot.__version__ == "0.6.1"):
		display_javascript(file(os.path.join(base_path, "bqplot_ext.js")).read(), raw=True)
	else:
		warnings.warn("This version (%s) of bqplot is not supppored" % bqplot.__version__)
	print(bqplot.__version__)



class DebouncedThreadedUpdater(object):
	def __init__(self, dataset, bqplot_image, factory_rgba8, delay=0.5, progress_widget=None):
		self.dataset = dataset
		self.bqplot_image = bqplot_image
		self.factory_rgba8 = factory_rgba8
		self.delay = delay
		self.progress_widget = progress_widget
		self.job = None
		self.executor = vx.execution.Executor()

	def update_select(self, f):
		def work(job):
			logger.debug("executing selection job for updater %r", self)
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

	def update(self, limits):
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
		self.job = Job(work, delay=self.delay)
		self.job.schedule()
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