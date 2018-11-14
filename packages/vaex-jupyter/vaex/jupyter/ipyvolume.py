import copy

from .plot import BackendBase
import ipyvolume.pylab as p3
import ipyvolume.examples
import traitlets
import vaex.dataset
import ipywidgets as widgets
from IPython.display import HTML, display_html, display_javascript, display
import numpy as np
from .utils import debounced


def xyz(shape=128, limits=[-3, 3], spherical=False, sparse=True, centers=False):
    dim = 3
    try:
        shape[0]
    except:
        shape = [shape] * dim
    try:
        limits[0][0]
    except:
        limits = [limits] * dim
    if centers:
        v = [slice(vmin + (vmax - vmin) / float(N) / 2, vmax - (vmax - vmin) / float(N) / 4, (vmax - vmin) / float(N)) for (vmin, vmax), N in zip(limits, shape)]
    else:
        v = [slice(vmin, vmax + (vmax - vmin) / float(N) / 2, (vmax - vmin) / float(N - 1)) for (vmin, vmax), N in zip(limits, shape)]
    if sparse:
        x, y, z = np.ogrid.__getitem__(v)
    else:
        x, y, z = np.mgrid.__getitem__(v)
    if spherical:
        r = np.linalg.norm([x, y, z])
        theta = np.arctan2(y, x)
        phi = np.arccos(z / r)
        return x, y, z, r, theta, phi
    else:
        return x, y, z


class IpyvolumeBackend(BackendBase):
    dim = 3

    @staticmethod
    def wants_colors():
        return False

    # def __init__(self):
    #    self._first_time = Tr
    def create_widget(self, output, plot, dataset, limits):
        self.output = output
        self.plot = plot
        self.dataset = dataset
        self.limits = np.array(limits).tolist()
        self._first_time = True
        self._first_time_vector = True
        self.figure = p3.figure()
        self.widget = p3.gcc()

        self.figure.observe(self._update_limits, 'xlim ylim zlim'.split())

    def _update_limits(self, *args):
        with self.output:
            # self._progressbar.cancel()
            limits = copy.deepcopy(self.limits)
            limits[0] = self.figure.xlim
            limits[1] = self.figure.ylim
            limits[2] = self.figure.zlim
            self.limits = limits


    @debounced(0.1, method=True)
    def update_image(self, intensity_image):
        with self.output:
            with self.figure:
                limits = copy.deepcopy(self.limits)
                if self._first_time:
                    self.volume = p3.volshow(intensity_image.T, controls=self._first_time, extent=limits)
                if 1: #hasattr(self.figure, 'extent_original'): # v0.5 check
                    self.volume.data_original = None
                    self.volume.data = intensity_image.T
                    self.volume.extent = copy.deepcopy(self.limits)
                    self.volume.data_min = np.nanmin(intensity_image)
                    self.volume.data_max = np.nanmax(intensity_image)
                    self.volume.show_min = self.volume.data_min
                    self.volume.show_max = self.volume.data_max

                self._first_time = False
                self.figure.xlim = limits[0]
                self.figure.ylim = limits[1]
                self.figure.zlim = limits[2]
                # p3.xlim(*self.limits[0])
                # p3.ylim(*self.limits[1])
                # p3.zlim(*self.limits[2])

    def update_vectors(self, vcount, vgrids, vcount_limits):
        vx, vy, vz = vgrids[:3]
        if vx is not None and vy is not None and vz is not None and vcount is not None:
            vcount = vcount[-1]  # no multivolume render, just take the last selection
            vx = vx[-1]
            vy = vy[-1]
            vz = vz[-1]
            ok = np.isfinite(vx) & np.isfinite(vy) & np.isfinite(vz)
            vcount_min = None
            vcount_max = None
            if vcount_limits is not None:
                try:
                    vcount_min, vcount_max = vcount_limits
                except:
                    vcount_min = self.vcount_limits
            if vcount_min is not None:
                ok &= (vcount > vcount_min)
            if vcount_max is not None:
                ok &= (vcount < vcount_max)
            # TODO: we assume all dimensions are equal length
            x, y, z = xyz(vx.shape[0], limits=self.limits, sparse=False, centers=True)
            v1d = [k[ok] for k in [x, y, z, vx, vy, vz]]
            vsize = 5
            vcolor = "grey"
            if self._first_time_vector:
                self._first_time_vector = False
                self.quiver = p3.quiver(*v1d, size=vsize, color=vcolor)
            else:
                with self.quiver.hold_trait_notifications():
                    self.quiver.x = x[ok]
                    self.quiver.y = y[ok]
                    self.quiver.z = z[ok]
                    self.quiver.vx = vx[ok]
                    self.quiver.vy = vy[ok]
                    self.quiver.vz = vz[ok]

    # def show(self):
    #     container = p3.gcc()
    #     vbox = widgets.VBox([container, self.progress, widgets.VBox(self.tools), self.output])
    #     display(vbox)
    #
    # def create_tools(self):
    #     self.tools = []
    #
    #     callback = self.dataset.signal_selection_changed.connect(lambda *x: self.update_grid())
    #
    #     def cleanup(callback=callback):
    #         self.dataset.signal_selection_changed.disconnect(callback=callback)
    #
    #     self._cleanups.append(cleanup)
    #
    # def get_binby(self):
    #     return [self.x, self.y, self.z]
