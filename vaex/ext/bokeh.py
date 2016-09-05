from __future__ import absolute_import
__author__ = 'maartenbreddels'
import time
import threading
import numpy as np

class Dummy(object):
	pass
progress = Dummy()
progress.value = 0
from bokeh.plotting import curdoc, figure
from functools import partial
def thread_safe_call(doc, f, *args, **kwargs):
	doc.add_next_tick_callback(partial(f, *args, **kwargs))

import logging
logger = logging.getLogger("vaex.ext.bokeh")

job = None
def debounced_threaded_update(bokeh_image, factory_rgba8, limits, delay=0.5):
	global job
	doc = curdoc()
	xlim, ylim = limits
	def factory(job):
		logger.debug("executing job")
		rgba8 = factory_rgba8()
		rgba32 = rgba8.view(dtype=np.uint32).reshape(rgba8.shape[:2])
		def update_bokeh():
			logger.debug("updating bokeh image")
			dw = [xlim[1] - xlim[0]]
			dh = [ylim[1] - ylim[0]]
			x = xlim[0]
			y = ylim[0]
			if 1:
				bokeh_image.data_source.data.update(dict(image=[rgba32], dw=dw, dh=dh, x=[x], y=[y]))
			else:
				bokeh_image.data_source.data["image"] = [rgba32]
				bokeh_image.data_source.data["x"] = x
				bokeh_image.data_source.data["y"] = y
				#bokeh_image.x = xlim[0]
				#bokeh_image.y = ylim[0]
				bokeh_image.data_source.data["dw"] = dw
				bokeh_image.data_source.data["dh"] = dh
			#image = p.image_rgba(image=[img], x=xlim[0], y=ylim[0], dw=[xlim[1]-xlim[0]], dh=[ylim[1]-ylim[0]])
		#thread_safe_call(doc, update_bokeh)
		doc.add_next_tick_callback(update_bokeh)
	if job:
		logger.debug("cancelling job")
		job.cancel()
	logger.debug("scheduling job")
	job = Job(factory, delay=delay)
	job.schedule()



def update(_, selection=False):
	global current_job
	doc = curdoc()
	def work(job, previous_job):
		#print "starting.."
		if previous_job:
			previous_job.cancel()
			previous_job.thread.join(5) # wait for 5 sec max
			#print "wait..."
		if job.cancelled:
			return
		def update_progress(fraction):
			def update_in_main_thread():
				progress.value = fraction/steps + step[0]/float(steps)
			thread_safe_call(doc, update_in_main_thread)
			#print progress.description, fraction, progress.value, job.cancelled#"\r",
			if fraction == 1:
				step[0] += 1
			return not job.cancelled
		try:
			import time
			callback = nyt.executor.signal_progress.connect(update_progress)
			if selection:
				steps = 1
				step = [0]
				progress.description = "Selecting"
				do_selection(job)
			progress.description = "Image"
			steps = 1
			step = [0]
			nyt.executor
			t0 = time.time()
			if not job.cancelled:
				rgba = create_image(job)
			if not job.cancelled:
				progress.description = "Done: %.1fs" % (time.time() - t0)
		except:
			if not job.cancelled:
				progress.description = "error"
			raise
		finally:
			nyt.executor.signal_progress.disconnect(callback)
		def update_widget_in_main_thread():
			url = vaex.image.rgba_to_url(rgba)
			im.src = url
			im.x = x_sc.min
			im.y = y_sc.max
			im.width = (x_sc.max - x_sc.min)
			im.height = -(y_sc.max - y_sc.min)
		if not job.cancelled:
			get_ioloop().add_callback(update_widget_in_main_thread)
	if current_job:
		current_job.cancel()
	current_job = Job(work, previous_job=current_job, delay=0.5)
	current_job.schedule()