from .bqplot import PlotBase
import ipyvolume.pylab as p3
import ipyvolume.examples
import traitlets
import vaex.dataset
import ipywidgets as widgets
from IPython.display import HTML, display_html, display_javascript, display
import numpy as np

class PlotDefault(PlotBase):
    z = traitlets.Unicode(allow_none=False)

    def _update_image(self):
        with self.output:
            grid = self.get_grid()
            if self.smooth_pre:
                for i in range(grid.shape[0]):
                    grid[i] = vaex.grids.gf(grid[i], self.smooth_pre)
            f = vaex.dataset._parse_f(self.f)
            fgrid = f(grid)
            if self.smooth_post:
                for i in range(grid.shape[0]):
                    fgrid[i] = vaex.grids.gf(fgrid[i], self.smooth_post)
            ngrid, fmin, fmax = self.normalise(fgrid)
            print(ngrid.shape)
            if len(ngrid.shape) == 4:
                #if ngrid.shape[0] == 1:
                ngrid = ngrid[-1]
            p3.volshow(ngrid.T, controls=self._first_time)

            vx, vy, vz = self.vgrids[:3]
            vcount = self.vcount
            if vx is not None and vy is not None and vz is not None and vcount is not None:
                vcount = vcount[-1] # no multivolume render, just take the last selection
                vx = vx[-1]
                vy = vy[-1]
                vz = vz[-1]
                print(vx.shape)
                ok = np.isfinite(vx) & np.isfinite(vy) & np.isfinite(vz)
                vcount_min = None
                vcount_max = None
                if self.vcount_limits is not None:
                    try:
                        vcount_min, vcount_max = self.vcount_limits
                    except:
                        vcount_min = self.vcount_limits
                if vcount_min is not None:
                    ok &= (vcount > vcount_min)
                if vcount_max is not None:
                    ok &= (vcount < vcount_max)
                x, y, z = ipyvolume.examples.xyz(self.get_vshape()[0], limits=self.limits, sparse=False, centers=True)
                v1d = [k[ok] for k in [x, y, z, vx, vy, vz]]
                vsize = 5
                vcolor = "grey"
                if self._first_time:
                    self.quiver = p3.quiver(*v1d, size=vsize, color=vcolor)
                else:
                    self.quiver.x = x[ok]
                    self.quiver.y = y[ok]
                    self.quiver.z = z[ok]
                    self.quiver.vx = vx[ok]
                    self.quiver.vy = vy[ok]
                    self.quiver.vz = vz[ok]

            p3.xlim(*self.limits[0])
            p3.ylim(*self.limits[1])
            p3.zlim(*self.limits[2])
            self._first_time = False

    def create_plot(self):
        self._first_time = True
        self.figure = p3.figure()

    def show(self):
        container = p3.gcc()
        vbox = widgets.VBox([container, self.progress, widgets.VBox(self.tools), self.output])
        display(vbox)

    def create_tools(self):
        self.tools = []

        callback = self.dataset.signal_selection_changed.connect(lambda *x: self.update_grid())

        def cleanup(callback=callback):
            self.dataset.signal_selection_changed.disconnect(callback=callback)

        self._cleanups.append(cleanup)

    def get_binby(self):
        return [self.x, self.y, self.z]