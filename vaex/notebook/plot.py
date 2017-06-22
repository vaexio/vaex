import traitlets
import ipywidgets as widgets
import six
import vaex.utils
import vaex.delayed
import numpy as np
import importlib
from IPython.display import display
import copy
from .utils import debounced

type_map = {}
def register_type(name):
    def reg(cls):
        assert cls not in type_map
        type_map[name] = cls
        return cls
    return reg

def get_type(name):
    if name not in type_map:
        raise ValueError("% not found, options are %r" % (name, type_map.keys()))
    return type_map[name]

backends = {}
backends['ipyleaflet'] = ('vaex.notebook.ipyleaflet', 'IpyleafletBackend')
backends['bqplot'] = ('vaex.notebook.bqplot', 'BqplotBackend')
backends['ipyvolume'] = ('vaex.notebook.ipyvolume', 'IpyvolumeBackend')
backends['matplotlib'] = ('vaex.notebook.ipympl', 'MatplotlibBackend')
backends['ipympl'] = backends['mpl'] = backends['matplotlib']

def create_backend(name):
    if callable(name):
        return name()
    if name not in backends:
        raise NameError("Unknown backend: %s, known ones are: %r" % (name, backends.keys()))
    module_name, class_name = backends[name]
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise NameError("Could not find classname %s in module %s for backend %s" % (class_name, module_name, name))
    return cls()

class BackendBase(widgets.Widget):
    dim = 2
    limits = traitlets.List(traitlets.Tuple(traitlets.CFloat(), traitlets.CFloat()))
    @staticmethod
    def wants_colors():
        return True
    def update_vectors(self, vcount, vgrids, vcount_limits):
        pass

class PlotBase(widgets.Widget):

    x = traitlets.Unicode(allow_none=False).tag(sync=True)
    y = traitlets.Unicode(allow_none=True).tag(sync=True)
    z = traitlets.Unicode(allow_none=True).tag(sync=True)
    w = traitlets.Unicode(allow_none=True).tag(sync=True)
    vx = traitlets.Unicode(allow_none=True).tag(sync=True)
    vy = traitlets.Unicode(allow_none=True).tag(sync=True)
    vz = traitlets.Unicode(allow_none=True).tag(sync=True)
    smooth_pre = traitlets.CFloat(None, allow_none=True).tag(sync=True)
    smooth_post = traitlets.CFloat(None, allow_none=True).tag(sync=True)
    what = traitlets.Unicode(allow_none=False).tag(sync=True)
    vcount_limits = traitlets.List([None, None], allow_none=True).tag(sync=True)

    def __init__(self, backend, dataset, x, y=None, z=None, w=None, grid=None, limits=None, shape=128, what="count(*)", f=None,
                 vshape=16,
                 selection=None, grid_limits=None, normalize=None, colormap="afmhot",
                 figure_key=None, fig=None, what_kwargs={}, grid_before=None, vcount_limits=None,**kwargs):
        super(PlotBase, self).__init__(x=x, y=y, z=z, w=w, what=what, vcount_limits=vcount_limits, **kwargs)
        self.backend = backend
        self.vgrids = [None, None, None]
        self.vcount = None
        self.dataset = dataset
        self.limits = self.get_limits(limits)
        self.shape = shape
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

            self.backend.create_widget(self.output, self, self.dataset, self.limits)
            self.control_widget = widgets.VBox()

            #self.create_tools()
            self.widget = widgets.VBox([self.control_widget, self.backend.widget, self.progress, self.output])
            if grid is None:
                self.update_grid()
            else:
                self.grid = grid

        callback = self.dataset.signal_selection_changed.connect(lambda *x: self.update_grid())
        def _on_limits_change(*args):
            self._progressbar.cancel()
            self.update_grid()
        self.backend.observe(_on_limits_change, "limits")
        for attrname in "x y z vx vy vz".split():
            def _on_change(*args, attrname=attrname):
                limits_index = {'x': 0, 'y': 1, 'z':2}.get(attrname)
                if limits_index is not None:
                    self.backend.limits[limits_index] = None
                self.update_grid()
            self.observe(_on_change, attrname)
        self.observe(lambda *args: self.update_grid(), "what")
        self.observe(lambda *args: self.update_image(), "vcount_limits")

        #self.update_image() # sometimes bqplot doesn't update the image correcly

    def get_limits(self, limits):
        return self.dataset.limits(self.get_binby(), limits)

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

    def add_control_widget(self, widget):
        self.control_widget.children += (widget,)

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

    def select_rectangle(self, x1, y1, x2, y2, mode="replace"):
        self.dataset.select_rectangle(self.x, self.y, limits=[[x1, x2], [y1, y2]], mode=mode)

    def select_lasso(self, x, y, mode="replace"):
        self.dataset.select_lasso(self.x, self.y, x, y, mode=mode)

    def get_shape(self):
        return vaex.dataset._expand_shape(self.shape, len(self.get_binby()))

    def get_vshape(self):
        return vaex.dataset._expand_shape(self.vshape, len(self.get_binby()))

    @debounced(.5, method=True)
    def update_grid(self):
        with self.output:
            limits = self.backend.limits[:self.backend.dim]
            xyz = [self.x, self.y, self.z]
            for i, limit in enumerate(limits):
                if limits[i] is None:
                    limits[i] = self.dataset.limits(xyz[i], async=True)
            @vaex.delayed.delayed
            def limits_done(limits):
                with self.output:
                    self.limits[:self.backend.dim] = np.array(limits).tolist()
                    limits_backend = copy.deepcopy(self.backend.limits)
                    limits_backend[:self.backend.dim] = self.limits[:self.backend.dim]
                    self.backend.limits = limits_backend

                    self._update_grid()
            limits_done(vaex.delayed.delayed_list(limits))
            self._execute()

    def _update_grid(self):
        with self.output:
            self._progressbar.cancel()
            self._new_progressbar()
            current_pb = self._progressbar
            async = True
            promises = []
            pb = self._progressbar.add("grid")
            result = self.dataset._stat(binby=self.get_binby(), what=self.what, limits=self.limits,
                      shape=self.get_shape(), progress=pb,
                                          selection=self.active_selections(), async=True)
            if async:
                promises.append(result)
            else:
                self.grid = result

            vs = [self.vx, self.vy, self.vz]
            for i, v in enumerate(vs):
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
                    if not current_pb.cancelled: # TODO: remote dataset jobs cannot be cancelled
                        self.progress.value = 0
                        self.grid = grid
                        self.vgrids = [vx, vy, vz]
                        self.vcount = vcount
                        #print("assign")
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
            self.dataset.executor.execute()

    @debounced(0.5, method=True)
    def update_image(self):
        self._update_image()

    def _update_image(self):
        with self.output:
            grid = self.get_grid().copy() # we may modify inplace
            if self.smooth_pre:
                for i in range(grid.shape[0]): # seperately for every selection
                    grid[i] = vaex.grids.gf(grid[i], self.smooth_pre)
            f = vaex.dataset._parse_f(self.f)
            with np.errstate(divide='ignore', invalid='ignore'):
                fgrid = f(grid)
            if self.smooth_post:
                for i in range(grid.shape[0]):
                    fgrid[i] = vaex.grids.gf(fgrid[i], self.smooth_post)
            ngrid, fmin, fmax = self.normalise(fgrid)
            if self.backend.wants_colors():
                color_grid = self.colorize(ngrid)
                if len(color_grid.shape) > 3:
                    if len(color_grid.shape) == 4:
                            if color_grid.shape[0] > 1:
                                color_grid = vaex.image.fade(color_grid[::-1])
                            else:
                                color_grid = color_grid[0]
                    else:
                        raise ValueError("image shape is %r, don't know what to do with that, expected (L, M, N, 3)" % (color_grid.shape,))
                I = np.transpose(color_grid, (1,0,2)).copy()
                # if self.what == "count(*)":
                #     I[...,3] = self.normalise(np.sqrt(grid))[0]
                self.backend.update_image(I)
            else:
                self.backend.update_image(ngrid[-1])
            self.backend.update_vectors(self.vcount, self.vgrids, self.vcount_limits)
            return
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
    y = traitlets.Unicode(allow_none=False).tag(sync=True)
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
    z = traitlets.Unicode(allow_none=False).tag(sync=True)
    z_slice = traitlets.CInt(default_value=0).tag(sync=True)#.tag(sync=True) # TODO: do linking at python side
    z_shape = traitlets.CInt(default_value=10).tag(sync=True)
    z_relative = traitlets.CBool(False).tag(sync=True)
    z_min = traitlets.CFloat(default_value=None, allow_none=True).tag(sync=True)#.tag(sync=True)
    z_max = traitlets.CFloat(default_value=None, allow_none=True).tag(sync=True)#.tag(sync=True)
    def __init__(self, **kwargs):
        self.z_min_extreme, self.z_max_extreme = kwargs["dataset"].minmax(kwargs["z"])
        super(Plot2dSliced, self).__init__(**kwargs)
        self.create_tools()

    def get_limits(self, limits):
        limits = self.dataset.limits(self.get_binby(), limits)
        limits = list([list(k) for k in limits])
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
        self.dataset.select_box(spaces, limits=limits, mode=mode)

    def select_lasso(self, x, y, mode="replace"):
        raise NotImplementedError("todo")

    def get_grid(self):
        zslice = self.grid[...,self.z_slice]
        if self.z_relative:
            with np.errstate(divide='ignore', invalid='ignore'):
                zslice = zslice / self.grid.sum(axis=-1)
        return zslice
        #return self.grid[...,self.z_slice]

    def get_vgrids(self):
        def zsliced(grid):
            return grid[...,self.z_slice] if grid is not None else None
        return [zsliced(grid) for grid in super(Plot2dSliced, self).get_vgrids()]

    def create_tools(self):
        #super(Plot2dSliced, self).create_tools()
        self.z_slice_slider = widgets.IntSlider(value=self.z_slice, min=0, max=self.z_shape-1)
        #self.add_control_widget(self.z_slice_slider)
        self.z_slice_slider.observe(self._z_slice_changed, "value")
        self.observe(self._z_slice_changed, "z_slice")

        dz = self.z_max_extreme - self.z_min_extreme

        self.z_range_slider = widgets.FloatRangeSlider(min=min(self.z_min, self.z_min_extreme), value=[self.z_min, self.z_max],
                                                       max=max(self.z_max, self.z_max_extreme), step=dz/1000)
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
