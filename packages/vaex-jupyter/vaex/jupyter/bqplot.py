import copy
import logging
import bqplot.marks
import bqplot as bq
import bqplot.interacts
import ipywidgets as widgets
import vaex
from . import bqplot_image
import bqplot.pyplot as plt
import numpy as np
import vaex.events
from .plot import BackendBase
from .utils import debounced

logger = logging.getLogger("vaex.nb.bqplot")


class BqplotBackend(BackendBase):
    def __init__(self, figure=None, figure_key=None):
        bqplot_image.patch()

        self._dirty = False
        self.figure_key = figure_key
        self.figure = figure
        self.signal_limits = vaex.events.Signal()

        self._cleanups = []

    def update_image(self, rgb_image):
        src = vaex.image.rgba_to_url(rgb_image)
        self.image.src = src
        # self.scale_x.min, self.scale_x.max = self.limits[0]
        # self.scale_y.min, self.scale_y.max = self.limits[1]
        self.image.x = self.scale_x.min
        self.image.y = self.scale_y.max
        self.image.width = self.scale_x.max - self.scale_x.min
        self.image.height = -(self.scale_y.max - self.scale_y.min)

    def create_widget(self, output, plot, dataset, limits):
        self.plot = plot
        self.output = output
        self.dataset = dataset
        self.limits = np.array(limits).tolist()
        self.scale_x = bqplot.LinearScale(min=limits[0][0], max=limits[0][1])
        self.scale_y = bqplot.LinearScale(min=limits[1][0], max=limits[1][1])
        self.scale_rotation = bqplot.LinearScale(min=0, max=1)
        self.scale_size = bqplot.LinearScale(min=0, max=1)
        self.scale_opacity = bqplot.LinearScale(min=0, max=1)
        self.scales = {'x': self.scale_x, 'y': self.scale_y, 'rotation': self.scale_rotation,
                       'size': self.scale_size, 'opacity': self.scale_opacity}

        margin = {'bottom': 30, 'left': 60, 'right': 0, 'top': 0}
        self.figure = plt.figure(self.figure_key, fig=self.figure, scales=self.scales, fig_margin=margin)
        self.figure.layout.min_width = '900px'
        plt.figure(fig=self.figure)
        self.figure.padding_y = 0
        x = np.arange(0, 10)
        y = x ** 2
        self._fix_scatter = s = plt.scatter(x, y, visible=False, rotation=x, scales=self.scales)
        self._fix_scatter.visible = False
        # self.scale_rotation = self.scales['rotation']
        src = ""  # vaex.image.rgba_to_url(self._create_rgb_grid())
        # self.scale_x.min, self.scale_x.max = self.limits[0]
        # self.scale_y.min, self.scale_y.max = self.limits[1]
        self.image = bqplot_image.Image(scales=self.scales, src=src, x=self.scale_x.min, y=self.scale_y.max,
                                           width=self.scale_x.max - self.scale_x.min, height=-(self.scale_y.max - self.scale_y.min))
        self.figure.marks = self.figure.marks + [self.image]
        # self.figure.animation_duration = 500
        self.figure.layout.width = '100%'
        self.figure.layout.max_width = '500px'
        self.scatter = s = plt.scatter(x, y, visible=False, rotation=x, scales=self.scales, size=x, marker="arrow")
        self.panzoom = bqplot.PanZoom(scales={'x': [self.scale_x], 'y': [self.scale_y]})
        self.figure.interaction = self.panzoom
        # self.figure.axes[0].label = self.x
        # self.figure.axes[1].label = self.y

        self.scale_x.observe(self._update_limits, "min")
        self.scale_x.observe(self._update_limits, "max")
        self.scale_y.observe(self._update_limits, "min")
        self.scale_y.observe(self._update_limits, "max")
        self.observe(self._update_scales, "limits")

        self.image.observe(self._on_view_count_change, 'view_count')
        self.control_widget = widgets.VBox()
        self.widget = widgets.VBox(children=[self.control_widget, self.figure])
        self.create_tools()

    def _update_limits(self, *args):
        with self.output:
            limits = copy.deepcopy(self.limits)
            limits[0:2] = [[scale.min, scale.max] for scale in [self.scale_x, self.scale_y]]
            self.limits = limits

    def _update_scales(self, *args):
        with self.scale_x.hold_trait_notifications():
            self.scale_x.min = self.limits[0][0]
            self.scale_x.max = self.limits[0][1]
        with self.scale_y.hold_trait_notifications():
            self.scale_y.min = self.limits[1][0]
            self.scale_y.max = self.limits[1][1]
        # self.update_grid()

    def create_tools(self):
        self.tools = []
        tool_actions = []
        tool_actions_map = {u"pan/zoom": self.panzoom}
        tool_actions.append(u"pan/zoom")

        # self.control_widget.set_title(0, "Main")
        self._main_widget = widgets.VBox()
        self._main_widget_1 = widgets.HBox()
        self._main_widget_2 = widgets.HBox()
        if 1:  # tool_select:
            self.brush = bqplot.interacts.BrushSelector(x_scale=self.scale_x, y_scale=self.scale_y, color="green")
            tool_actions_map["select"] = self.brush
            tool_actions.append("select")

            self.brush.observe(self.update_brush, ["selected", "selected_x"])
            # fig.interaction = brush
            # callback = self.dataset.signal_selection_changed.connect(lambda dataset: update_image())
            # callback = self.dataset.signal_selection_changed.connect(lambda *x: self.update_grid())

            # def cleanup(callback=callback):
            #    self.dataset.signal_selection_changed.disconnect(callback=callback)
            # self._cleanups.append(cleanup)

            self.button_select_nothing = widgets.Button(description="", icon="trash-o")
            self.button_reset = widgets.Button(description="", icon="refresh")
            import copy
            self.start_limits = copy.deepcopy(self.limits)

            def reset(*args):
                self.limits = copy.deepcopy(self.start_limits)
                with self.scale_y.hold_trait_notifications():
                    self.scale_y.min, self.scale_y.max = self.limits[1]
                with self.scale_x.hold_trait_notifications():
                    self.scale_x.min, self.scale_x.max = self.limits[0]
                self.plot.update_grid()
            self.button_reset.on_click(reset)

            self.button_select_nothing.on_click(lambda *ignore: self.plot.select_nothing())
            self.tools.append(self.button_select_nothing)
            self.modes_names = "replace and or xor subtract".split()
            self.modes_labels = "replace and or xor subtract".split()
            self.button_selection_mode = widgets.Dropdown(description='select', options=self.modes_labels)
            self.tools.append(self.button_selection_mode)

            def change_interact(*args):
                # print "change", args
                self.figure.interaction = tool_actions_map[self.button_action.value]

            tool_actions = ["pan/zoom", "select"]
            # tool_actions = [("m", "m"), ("b", "b")]
            self.button_action = widgets.ToggleButtons(description='', options=[(action, action) for action in tool_actions],
                                                       icons=["arrows", "pencil-square-o"])
            self.button_action.observe(change_interact, "value")
            self.tools.insert(0, self.button_action)
            self.button_action.value = "pan/zoom"  # tool_actions[-1]
            if len(self.tools) == 1:
                tools = []
            # self._main_widget_1.children += (self.button_reset,)
            self._main_widget_1.children += (self.button_action,)
            self._main_widget_1.children += (self.button_select_nothing,)
            # self._main_widget_2.children += (self.button_selection_mode,)
        self._main_widget.children = [self._main_widget_1, self._main_widget_2]
        self.control_widget.children += (self._main_widget,)
        self._update_grid_counter = 0  # keep track of t
        self._update_grid_counter_scheduled = 0  # keep track of t

    def _on_view_count_change(self, *args):
        with self.output:
            logger.debug("views: %d", self.image.view_count)
            if self._dirty and self.image.view_count > 0:
                try:
                    logger.debug("was dirty, and needs an update")
                    self.update()
                finally:
                    self._dirty = False

    @debounced(0.5, method=True)
    def update_brush(self, *args):
        with self.output:
            if not self.brush.brushing:  # if we ended brushing, reset it
                self.figure.interaction = None
            if self.brush.selected is not None:
                (x1, y1), (x2, y2) = self.brush.selected
                mode = self.modes_names[self.modes_labels.index(self.button_selection_mode.value)]
                self.plot.select_rectangle(x1, y1, x2, y2, mode=mode)
            else:
                self.dataset.select_nothing()
            if not self.brush.brushing:  # but then put it back again so the rectangle is gone,
                self.figure.interaction = self.brush
