import numpy as np
import vaex.image
from .plot import BackendBase
import copy
from .utils import debounced
import pylab as plt
import matplotlib.widgets
import ipywidgets as widgets


class MatplotlibBackend(BackendBase):
    def __init__(self):
        self.image = None

    def create_widget(self, output, plot, dataset, limits):
        self.plot = plot
        self.dataset = dataset
        self.output = output
        self.limits = np.array(limits)[:2].tolist()
        self.figure = plt.figure()
        self.canvas = self.figure.canvas
        if not isinstance(self.canvas, widgets.Widget):
            raise ValueError("please use the ipympl backend: not %r" % self.canvas)
        self.canvas.layout.width = '100%'
        self.canvas.layout.max_width = '500px'

        plt.xlim(*list(self.limits[0]))
        plt.ylim(*list(self.limits[1]))
        self.ax = plt.gca()

        def zoom_change(ax):
            with self.output:
                self._update_limits()

        self.ax.callbacks.connect("xlim_changed", zoom_change)
        self.ax.callbacks.connect("ylim_changed", zoom_change)

        self.rectangle_selector = matplotlib.widgets.RectangleSelector(
            self.ax, self._on_select_rectangle, spancoords='data')
        self.lasso = matplotlib.widgets.LassoSelector(self.ax, self._on_lasso)
        self.lasso.set_active(False)
        self.rectangle_selector.set_active(True)
        self.control_widget = widgets.VBox()
        self.widget = widgets.VBox(children=[self.control_widget, self.canvas])

        actions = ["lasso", "rectangle"]
        self.button_action = widgets.ToggleButtons(description='',
                                                   options=[(action, action) for action in actions],
                                                   icons=["rectangle", "pencil-square-o"])

        def change_interact(*args):
            with self.output:
                print(self.button_action.value)
                self.lasso.set_active(self.button_action.value == "lasso")
                self.rectangle_selector.set_active(self.button_action.value == "rectangle")

        self.button_action.observe(change_interact, "value")
        self.control_widget.children = (self.button_action, )
        # TODO: we could maybe support the classical notebook backend
        # self.widget = widgets.Output()
        # with self.widget:
        # plt.show()
        # self.widget = self.canvas

    def _update_limits(self):
        with self.output:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            limits = copy.deepcopy(self.limits)  # could be higher D
            limits[0] = xlim
            limits[1] = ylim
            if self.limits != limits:
                self.limits = limits

    @debounced(0.1, method=True)
    def update_image(self, rgb_image):
        with self.output:
            # print("update image")
            extent = list(self.limits[0]) + list(self.limits[1])
            if self.image:
                self.image.set_data(rgb_image)
                self.image.set_extent(extent)
            else:
                self.image = plt.imshow(rgb_image, extent=extent, origin="lower", aspect="auto")
            plt.figure(self.figure.number)
            plt.draw()

    def _on_lasso(self, vertices):
        with self.output:
            x, y = np.array(vertices).T

            x = np.ascontiguousarray(x, dtype=np.float64)
            y = np.ascontiguousarray(y, dtype=np.float64)
            self.plot.select_lasso(x, y)

    def _on_select_rectangle(self, pos1, pos2):
        with self.output:
            xmin = min(pos1.xdata, pos2.xdata)
            xmax = max(pos1.xdata, pos2.xdata)
            ymin = min(pos1.ydata, pos2.ydata)
            ymax = max(pos1.ydata, pos2.ydata)
            self.plot.select_rectangle(xmin, ymin, xmax, ymax)
