from __future__ import absolute_import

__author__ = 'maartenbreddels'
import numpy as np
import vaex.image
import vaex.dataset
import logging
import vaex as vx
import vaex.delayed
from .common import Job
import os
from IPython.display import HTML, display_html, display_javascript, display
import bqplot.marks
import bqplot as bq
import bqplot.interacts
import traitlets
import ipywidgets as widgets
from vaex.notebook import debounced
import vaex.grids
import time


logger = logging.getLogger("vaex.ext.bqplot")
base_path = os.path.dirname(__file__)

@bqplot.marks.register_mark('vaex.ext.bqplot.Image')
class Image(bqplot.marks.Mark):
    src = bqplot.marks.Unicode().tag(sync=True)
    x = bqplot.marks.Float().tag(sync=True)
    y = bqplot.marks.Float().tag(sync=True)
    view_count = traitlets.CInt(0).tag(sync=True)
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

patched = False
def patch(force=False):
    # return
    global patched
    if not patched or force:
        display_javascript(open(os.path.join(base_path, "bqplot_ext.js")).read(), raw=True)
    patched = True

# if (bqplot.__version__ == (0, 6, 1)) or (bqplot.__version__ == "0.6.1"):
# else:
#	warnings.warn("This version (%s) of bqplot is not supppored" % bqplot.__version__)


if 0:
    class _BqplotHistogram(Plot2d):
        def __init__(self, subspace, color, size, limits):
            self.color = color
            super(BqplotHistogram, self).__init__(subspace, size, limits)

        def create(self, data):
            size = data.shape[0]
            assert len(data.shape) == 1
            xmin, xmax = self.limits[0]
            dx = (xmax - xmin) / size
            x = np.linspace(xmin, xmax - dx, size) + dx / 2
            # print xmin, xmax, x

            self.scale_x = bq.LinearScale(min=xmin + dx / 2, max=xmax - dx / 2)
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

