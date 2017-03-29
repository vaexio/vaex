from .bqplot import PlotBase
import ipyvolume.pylab as p3
import traitlets
import vaex.dataset
import ipywidgets as widgets
from IPython.display import HTML, display_html, display_javascript, display

class PlotDefault(PlotBase):
    z = traitlets.Unicode(allow_none=False)

    def _update_image(self):
        with self.output:
            grid = self.get_grid()
            f = vaex.dataset._parse_f(self.f)
            fgrid = f(grid)
            ngrid, fmin, fmax = self.normalise(fgrid)
            if len(ngrid.shape) == 4:
                if ngrid.shape[0] == 1:
                    ngrid = ngrid[0]
            print("volshow", ngrid.shape)
            p3.volshow(ngrid)

    def create_plot(self):
        print("figure")
        self.figure = p3.figure()

    def show(self):
        print("show")
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