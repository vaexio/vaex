# -*- coding: utf-8 -*-
"""
import traitlets
import ipywidgets as widgets

class VolumeRenderingWidget(widgets.DOMWidget):
    _view_name = Unicode('VolumeRenderingView', sync=True)
    level1 = CInt(0, sync=True)

widgets.IntSlider

from IPython.display import HTML
"""

import os
import logging
import uuid
from base64 import b64encode
import json
import time
import numpy as np
from .utils import debounced, interactive_selection, interactive_cleanup
from IPython.display import HTML, display_html, display_javascript
import IPython
import zmq
import vaex
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

base_path = os.path.dirname(__file__)
logger = logging.getLogger("vaex.jupyter")


def cube_png(grid, file):
    if grid.shape != ((128,) * 3):
        logger.error("only 128**3 cubes are supported")
        return None
    colormap_name = "afmhot"
    import matplotlib.cm
    colormap = matplotlib.cm.get_cmap(colormap_name)
    mapping = matplotlib.cm.ScalarMappable(cmap=colormap)
    # pixmap = QtGui.QPixmap(32*2, 32)
    data = np.zeros((128 * 8, 128 * 16, 4), dtype=np.uint8)

    # mi, ma = 1*10**self.mod1, self.data3d.max()*10**self.mod2
    vmin, vmax = grid.min(), grid.max()
    grid_normalized = (grid - vmin) / (vmax - vmin)
    # intensity_normalized = (np.log(self.data3d + 1.) - np.log(mi)) / (np.log(ma) - np.log(mi));
    import PIL.Image
    for y2d in range(8):
        for x2d in range(16):
            zindex = x2d + y2d * 16
            I = grid_normalized[zindex]
            rgba = mapping.to_rgba(I, bytes=True)  # .reshape(Nx, 4)
            # print rgba.shape
            subdata = data[y2d * 128:(y2d + 1) * 128, x2d * 128:(x2d + 1) * 128]
            for i in range(3):
                subdata[:, :, i] = rgba[:, :, i]
            subdata[:, :, 3] = (grid_normalized[zindex] * 255).astype(np.uint8)  # * 0 + 255
            if 0:
                filename = "cube%03d.png" % zindex
                img = PIL.Image.frombuffer("RGB", (128, 128), subdata[:, :, 0:3] * 1)
                print(("saving to", filename))
                img.save(filename)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = PIL.Image.frombuffer("RGBA", (128 * 16, 128 * 8), data, 'raw')  # , "RGBA", 0, -1)
        img.save(file, "png")


class volr(object):
    def __init__(self, data=None, subspace_gridded=None, **settings):
        self.data = data
        self.subspace_gridded = subspace_gridded
        self.settings = settings

    def set(self, **kwargs):
        settings = json.dumps(kwargs)
        js_code = """var o = $('#%s').data('vr');
        var new_settings = JSON.parse('%s');
        console.log(new_settings);
        console.log($.extend(o.settings, new_settings));
        o.update_transfer_function_array()
        o.drawScene();
        """ % (self.id, settings)
        display_javascript(js_code, raw=True)

    def _ipython_display_(self):
        f = StringIO()
        # filename = os.path.join(base_path, "cube.png")
        if self.data is not None:
            cube_png(self.data, file=f)
        else:
            self.subspace_gridded.cube_png(file=f)
        # cube64 = "'data:image/png;base64," + b64encode(file(filename).read()) + "'"
        cube64 = "'data:image/png;base64," + b64encode(f.getvalue()).decode("ascii") + "'"
        # display_javascript("""
        # window.cube_src = 'data:image/png;base64,%s';
        # """ % (cube64, colormap64), raw=True)

        self.id = id = uuid.uuid1()
        display_html("<canvas id='{id}' width=512 height=512  style='display: inline;'/>".format(**locals()), raw=True)
        display_javascript(""" $('#%s').vr(
                $.extend({cube:%s, colormap:window.colormap_src}, %s)
                )
                """ % (id, cube64, json.dumps(self.settings)), raw=True)


class init(object):
    def __init__(self, ):
        pass

    def _ipython_display_(self):
        display_javascript(open(os.path.join(base_path, "glMatrix-0.9.5.min.js")).read(), raw=True)
        display_javascript(open(os.path.join(base_path, "volumerenderer.js")).read(), raw=True)
        # cube64 = b64encode(file(os.path.join(base_path, "cube.png")).read())
        colormap64 = b64encode(open(os.path.join(base_path, "colormap.png"), "rb").read()).decode("ascii")
        src = """
        window.colormap_src = 'data:image/png;base64,%s';
        """ % (colormap64,)
        # print(src)
        display_javascript(src, raw=True)

        js_code = "window.shader_cache = [];\n"
        for name in ["cube", "texture", "volr"]:
            for type in ["fragment", "vertex"]:
                text = open(os.path.join(base_path, "shaders", name + "-" + type + ".shader")).read()
                text = text.replace("\n", "\\n").replace("'", "\\'")
                js_code += "window.shader_cache['{name}_{type}'] = '{text}';\n".format(**locals())
        display_javascript(js_code, raw=True)
        # print js_code

    def ____ipython_display_(self):
        # base64 = file(os.path.join(base_path, "data.png")).read().encode("base64").replace("\n", "")
        # base64_colormap  = file(os.path.join(base_path, "colormap.png")).read().encode("base64").replace("\n", "")
        # print base64[:10]
        # code = "base64 = '" + base64 + "'; base64_colormap = '" + base64_colormap + "';"
        display_javascript(open(os.path.join(base_path, "all.js")).read(), raw=True)
        display_javascript(file(os.path.join(base_path, "vaex_volumerendering.js")).read(), raw=True)
        display_html(file(os.path.join(base_path, "snippet.js")).read(), raw=True)
        html1 = file(os.path.join(base_path, "snippet.html")).read()
        display_html(html1, raw=True)
        # print "ok"
        display_html("""<div>BLAAT</div> """, raw=True)

        if 0:
            js1 = file(os.path.join(base_path, "snippet.js")).read()
            js2 = file(os.path.join(base_path, "vaex_volumerendering.js")).read()
            js_lib = file(os.path.join(base_path, "all.js")).read()
            html1 = file(os.path.join(base_path, "snippet.html")).read()
            HTML("<script>" + js_lib + "\n" + code + "</script>" + "<script>" + js2 + "</script>" + html1 + js1)


# def get_ioloop():
#     ipython = IPython.get_ipython()
#     if ipython and hasattr(ipython, 'kernel'):
#         return zmq.eventloop.ioloop.IOLoop.instance()

# def debounced(delay_seconds=0.5):
#   """Debounce decorator for Jupyter notebook"""
#   def wrapped(f):
#       locals = {"counter": 0}
#       def execute(*args, **kwargs):
#           locals["counter"] += 1
#           #print "counter", locals["counter"]
#           def debounced_execute(counter=locals["counter"]):
#               #$print "counter is", locals["counter"]
#               if counter == locals["counter"]:
#                   logger.info("debounced call")
#                   f(*args, **kwargs)
#               else:
#                   logger.info("debounced call skipped")
#           ioloop = get_ioloop()
#           def thread_safe():
#               ioloop.add_timeout(time.time() + delay_seconds, debounced_execute)
#           ioloop.add_callback(thread_safe)
#       return execute
#   return wrapped

@vaex.register_dataframe_accessor('widget')
class DataFrameAccessorWidget(object):
    def __init__(self, df):
        self.df = df
        import vaex.jupyter.grid
        self.grid = vaex.jupyter.grid.Grid(df, [])

    def clear(self):
        self.grid = vaex.jupyter.grid.Grid(self.df, [])

    def histogram(self, x, shared=False, **kwargs):
        import vaex.jupyter.state
        import vaex.jupyter.viz
        state = vaex.jupyter.state.VizHistogramState(self.df, x_expression=str(x), **kwargs)
        if shared:
            grid = self.grid
        else:
            grid = vaex.jupyter.grid.Grid(self.df, [])
        grid.state_add(state)
        viz = vaex.jupyter.viz.VizHistogramBqplot(state=state)
        return viz

    def pie(self, x, shared=False, **kwargs):
        import vaex.jupyter.state
        import vaex.jupyter.viz
        state = vaex.jupyter.state.VizHistogramState(self.df, x_expression=str(x), **kwargs)
        if shared:
            grid = self.grid
        else:
            grid = vaex.jupyter.grid.Grid(self.df, [])
        grid.state_add(state)
        viz = vaex.jupyter.viz.VizPieChartBqplot(state=state)
        return viz

    def heatmap(self, x, y, shared=False, **kwargs):
        import vaex.jupyter.state
        import vaex.jupyter.viz
        state = vaex.jupyter.state.VizHeatmapState(self.df, x_expression=str(x), y_expression=str(y), **kwargs)
        if shared:
            grid = self.grid
        else:
            grid = vaex.jupyter.grid.Grid(self.df, [])
        grid.state_add(state)
        viz = vaex.jupyter.viz.VizHeatmapBqplot(state=state)
        return viz



def add_namespace():
    pass
