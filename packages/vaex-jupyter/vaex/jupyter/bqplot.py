import bqplot
import ipywidgets as widgets
import numpy as np
import traitlets

import vaex.image


blackish = '#777'
DEBOUNCE_SELECT = 0.5


class _BqplotMixin(traitlets.HasTraits):
    x_min = traitlets.CFloat()
    x_max = traitlets.CFloat()
    y_min = traitlets.CFloat(None, allow_none=True)
    y_max = traitlets.CFloat(None, allow_none=True)
    x_label = traitlets.Unicode()
    y_label = traitlets.Unicode()
    tool = traitlets.Unicode(None, allow_none=True)

    def __init__(self, zoom_y=True, **kwargs):
        super().__init__(**kwargs)
        self.x_scale = bqplot.LinearScale(allow_padding=False)
        self.y_scale = bqplot.LinearScale(allow_padding=False)
        widgets.link((self, 'x_min'), (self.x_scale, 'min'))
        widgets.link((self, 'x_max'), (self.x_scale, 'max'))
        widgets.link((self, 'y_min'), (self.y_scale, 'min'))
        widgets.link((self, 'y_max'), (self.y_scale, 'max'))

        self.x_axis = bqplot.Axis(scale=self.x_scale)
        self.y_axis = bqplot.Axis(scale=self.y_scale, orientation='vertical')
        widgets.link((self, 'x_label'), (self.x_axis, 'label'))
        widgets.link((self, 'y_label'), (self.y_axis, 'label'))
        self.x_axis.color = blackish
        self.y_axis.color = blackish
        self.x_axis.label_color = blackish
        self.y_axis.label_color = blackish
        # self.y_axis.tick_style = {'fill': blackish, 'stroke':'none'}
        self.y_axis.grid_color = blackish
        self.x_axis.grid_color = blackish
        self.x_axis.label_offset = "2em"
        self.y_axis.label_offset = "3em"
        self.x_axis.grid_lines = 'none'
        self.y_axis.grid_lines = 'none'

        self.axes = [self.x_axis, self.y_axis]
        self.scales = {'x': self.x_scale, 'y': self.y_scale}

        self.figure = bqplot.Figure(axes=self.axes)
        self.figure.background_style = {'fill': 'none'}
        self.figure.padding_y = 0
        self.figure.fig_margin = {'bottom': 40, 'left': 60, 'right': 10, 'top': 10}

        self.interacts = {}
        self.interacts['pan-zoom'] = bqplot.PanZoom(scales={'x': [self.x_scale], 'y': [self.y_scale] if zoom_y else []})
        self.interacts['select-rect'] = bqplot.interacts.BrushSelector(x_scale=self.x_scale, y_scale=self.y_scale, color="green")
        self.interacts['select-x'] = bqplot.interacts.BrushIntervalSelector(scale=self.x_scale, color="green")
        self._brush = self.interacts['select-rect']
        self._brush_interval = self.interacts['select-x']

        # TODO: put the debounce in the presenter?
        @vaex.jupyter.debounced(DEBOUNCE_SELECT)
        def update_brush(*args):
            with self.output:
                if not self._brush.brushing:  # if we ended _brushing, reset it
                    self.figure.interaction = None
                if self._brush.selected is not None:
                    x1, x2 = self._brush.selected_x
                    y1, y2 = self._brush.selected_y
                    # (x1, y1), (x2, y2) = self._brush.selected
                    # mode = self.modes_names[self.modes_labels.index(self.button_selection_mode.value)]
                    self.presenter.select_rectangle(x1, x2, y1, y2)
                else:
                    self.presenter.select_nothing()
                if not self._brush.brushing:  # but then put it back again so the rectangle is gone,
                    self.figure.interaction = self._brush

        self._brush.observe(update_brush, ["selected", "selected_x"])

        @vaex.jupyter.debounced(DEBOUNCE_SELECT)
        def update_brush(*args):
            with self.output:
                if not self._brush_interval.brushing:  # if we ended _brushing, reset it
                    self.figure.interaction = None
                if self._brush_interval.selected is not None and len(self._brush_interval.selected):
                    x1, x2 = self._brush_interval.selected
                    self.presenter.select_x_range(x1, x2)
                else:
                    self.presenter.select_nothing()
                if not self._brush_interval.brushing:  # but then put it back again so the rectangle is gone,
                    self.figure.interaction = self._brush_interval

        self._brush_interval.observe(update_brush, ["selected"])

        def tool_change(change=None):
            self.figure.interaction = self.interacts.get(self.tool, None)
        self.observe(tool_change, 'tool')
        self.widget = self.figure


class Histogram(_BqplotMixin):
    opacity = 0.7

    def __init__(self, output, presenter, **kwargs):
        self.output = output
        self.presenter = presenter
        super().__init__(zoom_y=False, **kwargs)
        self.bar = self.mark = bqplot.Bars(x=[1, 2], scales=self.scales, type='grouped')
        self.figure.marks = self.figure.marks + [self.mark]

    def update_data(self, x, y, colors):
        self.mark.x = x
        self.mark.y = y
        self.mark.colors = colors

    def _reset_opacities(self):
        opacities = self.mark.y * 0 + self.opacity
        self.mark.opacities = opacities.T.ravel().tolist()

    def highlight(self, index):
        if index is None:
            self._reset_opacities()
        opacities = (self.mark.y * 0 + 0.2)
        if len(self.mark.y.shape) == 2:
            opacities[:, index] = self.opacity
        else:
            opacities[index] = self.opacity
        self.mark.opacities = opacities.T.ravel().tolist()


class PieChart(_BqplotMixin):
    opacity = 0.7

    def __init__(self, output, presenter, radius=100, **kwargs):
        self.output = output
        self.presenter = presenter
        self.radius = radius
        super().__init__(zoom_y=False, **kwargs)
        self.pie1 = self.mark = bqplot.Pie(sizes=[1, 2], radius=self.radius, inner_radius=0, stroke=blackish)
        self.figure.marks = self.figure.marks + [self.mark]

    def update_data(self, x, y, colors):
        # TODO: support groups
        self.pie1.sizes = y[0]

    def reset_opacities(self):
        opacities = self.mark.y * 0 + self.opacity
        self.state.x_slice = None
        self.mark.opacities = opacities.T.ravel().tolist()

    def highlight(self, index):
        opacities = (self.mark.y * 0 + 0.2)
        if len(self.mark.y.shape) == 2:
            opacities[:, index] = self.opacity
        else:
            opacities[index] = self.opacity
        self.mark.opacities = opacities.T.ravel().tolist()
        self.reset_opacities()


class Heatmap(_BqplotMixin):
    def __init__(self, output, presenter, **kwargs):
        self.output = output
        self.presenter = presenter
        super().__init__(**kwargs)
        self.heatmap_image = widgets.Image(format='png')
        self.heatmap_image_fix = widgets.Image(format='png')
        self.mark = bqplot.Image(scales=self.scales, image=self.heatmap_image)
        self.figure.marks = self.figure.marks + [self.mark]

    def set_rgb_image(self, rgb_image):
        with self.output:
            assert rgb_image.shape[-1] == 4, "last dimention is channel"
            rgb_image = (rgb_image * 255.).astype(np.uint8)
            pil_image = vaex.image.rgba_2_pil(rgb_image)
            data = vaex.image.pil_2_data(pil_image)
            self.heatmap_image.value = data
            # force update
            self.mark.image = self.heatmap_image_fix
            self.mark.image = self.heatmap_image
            # TODO: bqplot bug that this does not work?
            # with self.image.hold_sync():
            self.mark.x = (self.x_min, self.x_max)
            self.mark.y = (self.y_min, self.y_max)
