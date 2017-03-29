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
import bqplot.pyplot as plt
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



type_map = {}
def register_type(name):
    def reg(cls):
        assert cls not in type_map
        type_map[name] = cls
        return cls
    return reg

def get_class(name):
    if name not in type_map:
        raise ValueError("% not found, options are %r" % (name, type_map.keys()))
    return type_map[name]

class PlotBase(widgets.Widget):
    x = traitlets.Unicode(allow_none=False)
    y = traitlets.Unicode(allow_none=True)
    z = traitlets.Unicode(allow_none=True)
    w = traitlets.Unicode(allow_none=True)
    vx = traitlets.Unicode(allow_none=True)
    vy = traitlets.Unicode(allow_none=True)
    vz = traitlets.Unicode(allow_none=True)
    smooth_pre = traitlets.CFloat(None, allow_none=True)
    smooth_post = traitlets.CFloat(None, allow_none=True)

    def __init__(self, dataset, x, y=None, z=None, w=None, grid=None, limits=None, shape=128, what="count(*)", f=None,
                 vshape=16,
                 selection=None, grid_limits=None, normalize=None, colormap="afmhot",
                 figure_key=None, fig=None, what_kwargs={}, grid_before=None, vcount_limits=None, **kwargs):
        super(PlotBase, self).__init__(x=x, y=y, z=z, w=w, **kwargs)
        patch()
        self._dirty = False
        self.vgrids = [None, None, None]
        self.vcount_limits = vcount_limits
        self.vcount = None
        self.dataset = dataset
        self.limits = self.get_limits(limits)
        self.shape = shape
        self.what = what
        self.f = f
        self.selection = selection
        self.grid_limits = grid_limits
        self.normalize = normalize
        self.colormap = colormap
        self.what_kwargs = what_kwargs
        self.grid_before = grid_before
        self.figure_key = figure_key
        self.fig = fig
        self.vshape = vshape

        self._new_progressbar()

        self.output = widgets.Output()
        #with self.output:
        if 1:
            self._cleanups = []

            self.progress = widgets.FloatProgress(value=0.0, min=0.0, max=1.0, step=0.01)
            self.progress.layout.width = "95%"
            self.progress.layout.max_width = '500px'
            self.progress.description = "progress"

            self.create_plot()
            self.control_widget = widgets.VBox()
            self.create_tools()
            self.widget = widgets.VBox([self.control_widget, self.figure, self.progress, self.output])
            if grid is None:
                self.update_grid()
            else:
                self.grid = grid

        #self.update_image() # sometimes bqplot doesn't update the image correcly

    def get_limits(self, limits):
        return self.dataset.limits(self.get_binby(), limits)

    def _on_view_count_change(self, *args):
        with self.output:
            logger.debug("views: %d", self.image.view_count)
            if self._dirty and self.image.view_count > 0:
                try:
                    logger.debug("was dirty, and needs an update")
                    self.update()
                finally:
                    self._dirty = False

    def active_selections(self):
        selections = vaex.dataset._ensure_list(self.selection)
        def translate(selection):
            if selection is False:
                selection = None
            if selection is True:
                selection = "default"
            return selection
        selections = map(translate, selections)
        selections = list([s for s in selections if self.dataset.has_selection(s) or s in [False, None]])
        if not selections:
            selections = [False]
        return selections

    def show(self):
        display(self.widget)

    def create_plot(self):
        self.scale_x = bqplot.LinearScale(min=self.limits[0][0], max=self.limits[0][1])
        self.scale_y = bqplot.LinearScale(min=self.limits[1][0], max=self.limits[1][1])
        self.scale_rotation = bqplot.LinearScale(min=0, max=1)
        self.scale_size = bqplot.LinearScale(min=0, max=1)
        self.scale_opacity = bqplot.LinearScale(min=0, max=1)
        self.scales = {'x': self.scale_x, 'y':self.scale_y, 'rotation':self.scale_rotation,
                       'size':self.scale_size, 'opacity': self.scale_opacity}

        margin = {'bottom': 30, 'left': 60, 'right': 0, 'top': 0}
        self.figure = plt.figure(self.figure_key, fig=self.fig, scales=self.scales, fig_margin=margin)
        plt.figure(fig=self.figure)
        self.figure.padding_y = 0
        x = np.arange(0, 10)
        y = x ** 2
        self._fix_scatter = s = plt.scatter(x, y, visible=False, rotation=x, scales=self.scales)
        self._fix_scatter.visible = False
        #self.scale_rotation = self.scales['rotation']
        src = ""#vaex.image.rgba_to_url(self._create_rgb_grid())
        #self.scale_x.min, self.scale_x.max = self.limits[0]
        #self.scale_y.min, self.scale_y.max = self.limits[1]
        self.image = Image(scales=self.scales, src=src, x=self.scale_x.min, y=self.scale_y.max,
                           width=self.scale_x.max-self.scale_x.min, height=-(self.scale_y.max-self.scale_y.min))
        self.figure.marks = self.figure.marks + [self.image]
        #self.figure.animation_duration = 500
        self.figure.layout.width = '100%'
        self.figure.layout.max_width = '500px'
        self.scatter = s = plt.scatter(x, y, visible=False, rotation=x, scales=self.scales, size=x, marker="arrow")
        self.panzoom = bqplot.PanZoom(scales={'x': [self.scale_x], 'y': [self.scale_y]})
        self.figure.interaction = self.panzoom
        self.figure.axes[0].label = self.x
        self.figure.axes[1].label = self.y

        self.scale_x.observe(self._update_limits, "min")
        self.scale_x.observe(self._update_limits, "max")
        self.scale_y.observe(self._update_limits, "min")
        self.scale_y.observe(self._update_limits, "max")

        self.image.observe(self._on_view_count_change, 'view_count')

    def _update_limits(self, *args):
        with self.output:
            self._progressbar.cancel()
            self.limits[0:2] = [[scale.min, scale.max] for scale in [self.scale_x, self.scale_y]]
            self.update_grid()

    def add_control_widget(self, widget):
        self.control_widget.children += (widget,)

    def create_tools(self):
        self.tools = []
        tool_actions = []
        tool_actions_map = {u"m": self.panzoom}
        tool_actions.append(u"pan/zoom")

        #self.control_widget.set_title(0, "Main")
        self._main_widget = widgets.VBox()
        self._main_widget_1 = widgets.HBox()
        self._main_widget_2 = widgets.HBox()
        if 1:#tool_select:
            self.brush = bqplot.interacts.BrushSelector(x_scale=self.scale_x, y_scale=self.scale_y, color="green")
            tool_actions_map["b"] = self.brush
            tool_actions.append("brush")

            self.brush.observe(self.update_brush, "selected")
            # fig.interaction = brush
            # callback = self.dataset.signal_selection_changed.connect(lambda dataset: update_image())
            callback = self.dataset.signal_selection_changed.connect(lambda *x: self.update_grid())

            def cleanup(callback=callback):
                self.dataset.signal_selection_changed.disconnect(callback=callback)
            self._cleanups.append(cleanup)

            self.button_select_nothing = widgets.Button(description="", icon="trash-o")
            self.button_reset = widgets.Button(description="", icon="refresh")
            import copy
            self.start_limits = copy.deepcopy(self.limits)
            def reset(*args):
                self.limits = copy.deepcopy(self.start_limits)
                self.scale_y.min, self.scale_y.max = self.limits[1]
                self.scale_x.min, self.scale_x.max = self.limits[0]
                self.update_grid()
            self.button_reset.on_click(reset)

            def select_nothing(button):
                self.dataset.select_nothing()

            self.button_select_nothing.on_click(select_nothing)
            self.tools.append(self.button_select_nothing)
            self.modes_names = "replace and or xor subtract".split()
            self.modes_labels = "replace and or xor subtract".split()
            self.button_selection_mode = widgets.Dropdown(description='select', options=self.modes_labels)
            self.tools.append(self.button_selection_mode)

            def change_interact(*args):
                # print "change", args
                self.figure.interaction = tool_actions_map[self.button_action.value]

            tool_actions = ["m", "b"]
            # tool_actions = [("m", "m"), ("b", "b")]
            self.button_action = widgets.ToggleButtons(description='', options=[(action, action) for action in tool_actions],
                                                  icons=["arrows", "pencil-square-o"])
            self.button_action.observe(change_interact, "value")
            self.tools.insert(0, self.button_action)
            self.button_action.value = "m"#""pan/zoom"  # tool_actions[-1]
            if len(self.tools) == 1:
                tools = []
            self._main_widget_1.children += (self.button_reset,)
            self._main_widget_1.children += (self.button_action,)
            self._main_widget_1.children += (self.button_select_nothing,)
            #self._main_widget_2.children += (self.button_selection_mode,)
        self._main_widget.children = [self._main_widget_1, self._main_widget_2]
        self.control_widget.children += (self._main_widget,)
        self._update_grid_counter = 0 # keep track of t
        self._update_grid_counter_scheduled = 0  # keep track of t

    def _progress(self, v):
        self.progress.value = v

    def _new_progressbar(self):
        def update(v):
            with self.output:
                import IPython
                ipython = IPython.get_ipython()
                ipython.kernel.do_one_iteration()
                self.progress.value = v
                return not self._progressbar.cancelled
        self._progressbar = vaex.utils.progressbars(False, next=update, name="bqplot")

    @debounced(0.5, method=True)
    def update_brush(self, *args):
        with self.output:
            self.figure.interaction = None
            if self.brush.selected:
                (x1, y1), (x2, y2) = self.brush.selected
                mode = self.modes_names[self.modes_labels.index(self.button_selection_mode.value)]
                self.select_rectangle(x1, y1, x2, y2, mode=mode)
            else:
                self.dataset.select_nothing()
        self.figure.interaction = self.brush

    def select_rectangle(self, x1, y1, x2, y2, mode="replace"):
        self.dataset.select_rectangle(self.x, self.y, limits=[[x1, x2], [y1, y2]], mode=mode)

    def get_shape(self):
        return vaex.dataset._expand_shape(self.shape, len(self.get_binby()))

    def get_vshape(self):
        return vaex.dataset._expand_shape(self.vshape, len(self.get_binby()))

    @debounced(0.5, method=True)
    def update_grid(self):
        with self.output:
            self._update_grid()

    def _update_grid(self):
        with self.output:
            print("update grid")
            self._progressbar.cancel()
            self._new_progressbar()
            async = True
            promises = []
            pb = self._progressbar.add("grid")
            def c(*args):
                with self.output:
                    print("cancelled")
            pb.oncancel = c
            result = self.dataset._stat(binby=self.get_binby(), what=self.what, limits=self.limits,
                      shape=self.get_shape(), progress=pb,
                                          selection=self.active_selections(), async=True)
            # result = self.dataset.count(binby=self.get_binby(), limits=self.limits,
            #            shape=self.get_shape(), progress=self._progressbar.add("grid"),
            #                                selection=self.active_selections(), async=True)
            if async:
                promises.append(result)
            else:
                self.grid = result

            vs = [self.vx, self.vy, self.vz]
            for i, v in enumerate(vs):
                #print("mean of ", v, self.limits)
                result = None
                if v:
                    result = self.dataset.mean(v, binby=self.get_binby(), limits=self.limits,
                           shape=self.get_vshape(), progress=self._progressbar.add("v"+str(i)),
                           selection=self.active_selections(), async=async)
                if async:
                    promises.append(result)
                else:
                    self.vgrids[i] = result
            result = None
            if any(vs):
                expr = "*".join([v for v in vs if v])
                result = self.dataset.count(expr, binby=self.get_binby(), limits=self.limits,
                                                   shape=self.get_vshape(), progress=self._progressbar.add("vcount"),
                                                   selection=self.active_selections(), async=async)
            if async:
                promises.append(result)
            else:
                self.vgrids[i] = result
            @vaex.delayed.delayed
            def assign(grid, vx, vy, vz, vcount):
                with self.output:
                    self.grid = grid
                    self.vgrids = [vx, vy, vz]
                    self.vcount = vcount
                    self._update_image()
            if async:
                for promise in promises:
                    if promise:
                        promise.end()
                assign(*promises).end()
                self._execute()
            else:
                self._update_image()

    @debounced(0.05, method=True)
    def _execute(self):
        with self.output:
            print("execute")
            self.dataset.executor.execute()

    @debounced(0.5, method=True)
    def update_image(self):
        self._update_image()

    def _update_image(self):
        with self.output:
            grid = self.get_grid()
            if self.smooth_pre:
                grid  = vaex.grids.gf(grid, self.smooth_pre)
            f = vaex.dataset._parse_f(self.f)
            fgrid = f(grid)
            if self.smooth_post:
                fgrid = vaex.grids.gf(fgrid, self.smooth_post)
            ngrid, fmin, fmax = self.normalise(fgrid)
            color_grid = self.colorize(ngrid)
            #print("shape", color_grid.shape)
            if len(color_grid.shape) > 3:
                if len(color_grid.shape) == 4:
                        if color_grid.shape[0] > 1:
                            color_grid = vaex.image.fade(color_grid[::-1])
                        else:
                            color_grid = color_grid[0]
                else:
                    raise ValueError("image shape is %r, don't know what to do with that, expected (L, M, N, 3)" % (color_grid.shape,))
            I = np.transpose(color_grid, (1,0,2)).copy()
            src = vaex.image.rgba_to_url(I)
            self.image.src = src
            #self.scale_x.min, self.scale_x.max = self.limits[0]
            #self.scale_y.min, self.scale_y.max = self.limits[1]
            self.image.x = self.scale_x.min
            self.image.y = self.scale_y.max
            self.image.width = self.scale_x.max - self.scale_x.min
            self.image.height = -(self.scale_y.max - self.scale_y.min)

            vx, vy, vz, vcount = self.get_vgrids()
            if vx is not None and vy is not None and vcount is not None:
                #print(vx.shape)
                vx = vx[-1]
                vy = vy[-1]
                vcount = vcount[-1].flatten()
                vx = vx.flatten()
                vy = vy.flatten()

                xmin, xmax = self.limits[0]
                ymin, ymax = self.limits[1]
                centers_x = np.linspace(xmin, xmax, self.vshape, endpoint=False)
                centers_x += (centers_x[1] - centers_x[0]) / 2
                centers_y = np.linspace(ymin, ymax, self.vshape, endpoint=False)
                centers_y += (centers_y[1] - centers_y[0]) / 2
                #y, x = np.meshgrid(centers_y, centers_x)
                x, y = np.meshgrid(centers_x, centers_y)
                x = x.T
                y = y.T
                x = x.flatten()
                y = y.flatten()
                mask = vcount > 5
                #print(xmin, xmax, x)
                self.scatter.x = x * 1.
                self.scatter.y = y * 1.
                angle = -np.arctan2(vy, vx) + np.pi/2
                self.scale_rotation.min = 0
                self.scale_rotation.max = np.pi
                angle[~mask] = 0
                self.scatter.rotation = angle
                #self.scale.size = mask * 3
                # self.scale.size = mask.asdtype(np.float64) * 3
                self.vmask = mask
                self.scatter.size = self.vmask * 2-1
                # .asdtype(np.float64)

                self.scatter.visible = True
                self.scatter.visible = len(x[mask]) > 0
                #self.scatter.marker = "arrow"
                #print("UpDated")


    def get_grid(self):
        return self.grid

    def get_vgrids(self):
        return self.vgrids[0], self.vgrids[1], self.vgrids[2], self.vcount

    def colorize(self, grid):
        return vaex.dataset._parse_reduction("colormap", self.colormap, [])(grid)

    def normalise(self, grid):
        if self.grid_limits is not None:
            vmin, vmax = self.grid_limits
            grid = grid.copy()
            grid -= vmin
            grid /= (vmax - vmin)
        else:
            n = vaex.dataset._parse_n(self.normalize)
            grid, vmin, vmax = n(grid)
        grid = np.clip(grid, 0, 1)
        return grid, vmin, vmax

@register_type("default")
class Plot2dDefault(PlotBase):
    y = traitlets.Unicode(allow_none=False)
    def __init__(self, **kwargs):
        super(Plot2dDefault, self).__init__(**kwargs)

    def colorize(self, grid):
        if self.z:
            grid = grid.copy()
            grid[~np.isfinite(grid)] = 0
            total = np.sum(grid, axis=-1)
            mask = total == 0
            import matplotlib.cm
            colormap = matplotlib.cm.get_cmap(self.colormap)
            N = grid.shape[-1]
            z = np.linspace(0, 1, N, endpoint=True)
            colors = colormap(z)
            cgrid = np.dot(grid, colors)
            cgrid = (cgrid.T / total.T).T
            #alpha = cgrid[...,3]
            #rgb = cgrid[...,0:2]
            #alpha[mask] = 0
            #rgb[mask] = 0
            n = vaex.dataset._parse_n(self.normalize)
            ntotal, __, __ = n(total)
            #print("totao", ntotal.shape, total.shape)
            #cgrid[...,3] = ntotal
            cgrid[mask,3] = 0
            #print("cgrid", cgrid.shape, total.shape, np.nanmax(grid), np.nanmin(grid))
            #print(cgrid)
            return cgrid
            #colors = vaex.dataset._parse_reduction("colormap", self.colormap, [])(grid)
        else:
            return vaex.dataset._parse_reduction("colormap", self.colormap, [])(grid)
    def get_shape(self):
        if self.z:
            if ":" in self.z:
                shapez = int(self.z.split(":")[1])
                shape = vaex.dataset._expand_shape(self.shape, 2) + (shapez,)
            else:
                shape = vaex.dataset._expand_shape(self.shape, 3)
            return shape
        else:
            return super(Plot2dDefault, self).get_shape()

    def get_binby(self):
        if self.z:
            z = self.z
            if ":" in z:
                z = z.split(":")[0]
            return [self.x, self.y, z]
        else:
            return [self.x, self.y]


@register_type("slice")
class Plot2dSliced(PlotBase):
    z = traitlets.Unicode(allow_none=False)
    z_slice = traitlets.CInt(default_value=0)#.tag(sync=True) # TODO: do linking at python side
    z_shape = traitlets.CInt(default_value=10)
    z_min = traitlets.CFloat(default_value=None, allow_none=True)#.tag(sync=True)
    z_max = traitlets.CFloat(default_value=None, allow_none=True)#.tag(sync=True)
    def __init__(self, **kwargs):
        self.z_min_extreme, self.z_max_extreme = kwargs["dataset"].minmax(kwargs["z"])
        super(Plot2dSliced, self).__init__(**kwargs)

    def get_limits(self, limits):
        limits = self.dataset.limits(self.get_binby(), limits)
        if self.z_min is None:
            self.z_min = limits[2][0]
        if self.z_max is None:
            self.z_max = limits[2][1]
        limits[2][0] = self.z_min
        limits[2][1] = self.z_max
        return limits

    def select_rectangle(self, x1, y1, x2, y2, mode="replace"):
        dz = self.z_max - self.z_min
        z1 = self.z_min + dz * self.z_slice / self.z_shape
        z2 = self.z_min + dz * (self.z_slice+1) / self.z_shape
        spaces = [self.x, self.y, self.z]
        limits = [[x1, x2], [y1, y2], [z1, z2]]
        print(z1, z2)
        self.dataset.select_box(spaces, limits=limits, mode=mode)

    def get_grid(self):
        return self.grid[...,self.z_slice]

    def get_vgrids(self):
        def zsliced(grid):
            return grid[...,self.z_slice] if grid is not None else None
        return [zsliced(grid) for grid in super(Plot2dSliced, self).get_vgrids()]

    def create_tools(self):
        super(Plot2dSliced, self).create_tools()
        self.z_slice_slider = widgets.IntSlider(value=self.z_slice, min=0, max=self.z_shape-1)
        self.tools.append(self.z_slice_slider)
        self.z_slice_slider.observe(self._z_slice_changed, "value")
        self.observe(self._z_slice_changed, "z_slice")

        dz = self.z_max_extreme - self.z_min_extreme

        self.z_range_slider = widgets.FloatRangeSlider(min=self.z_min_extreme, value=[self.z_min, self.z_max],
                                                       max=self.z_max_extreme, step=dz/1000)
        self.z_range_slider.observe(self._z_range_changed_, names=["value"])
        #self.observe(self.z_range_slider, "z_min")

        self.z_control = widgets.VBox([self.z_slice_slider, self.z_range_slider])
        self.add_control_widget(self.z_control)

    def _z_range_changed_(self, changes, **kwargs):
        #print("changes1", changes, repr(changes), kwargs)
        self.limits[2][0], self.limits[2][1] =\
            self.z_min, self.z_max = self.z_range_slider.value = changes["new"]
        self.update_grid()

    def _z_slice_changed(self, changes):
        self.z_slice = self.z_slice_slider.value = changes["new"]
        self._update_image()

    def get_shape(self):
        return vaex.dataset._expand_shape(self.shape, 2) + (self.z_shape, )

    def get_vshape(self):
        return vaex.dataset._expand_shape(self.vshape, 2) + (self.z_shape, )

    def get_binby(self):
        return [self.x, self.y, self.z]

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

