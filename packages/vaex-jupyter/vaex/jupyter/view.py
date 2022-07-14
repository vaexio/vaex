from __future__ import absolute_import
import sys
import vaex
import traitlets
import ipywidgets as widgets
import ipyvuetify as v
import numpy as np
from IPython.display import display

from . import widgets as vw
from . import model
from .traitlets import traitlet_fixes


import matplotlib.colors
colors_default = [f'C{i}' for i in range(10)]
colors_default = list(map(matplotlib.colors.to_hex, colors_default))
colors_default
C0, C1 = colors_default[:2]
C0, C1 = '#9ECBF5', '#E0732C'

DEBOUNCE_SLICE = 0.1
DEBOUNCE_LIMITS = 0.3
DEBOUNCE_HOVER_SLICED = 3
DEBOUNCE_SELECT = 0.5


ICON_HISTOGRAM = 'histogram'
ICON_HEATMAP = 'heatmap'


def _translate_selection(selection):
    if selection in [None, False]:
        return None
    if selection is True:
        return 'default'
    else:
        return selection


@traitlet_fixes
class ViewBase(v.Container):
    selection_interact = traitlets.Unicode('default', allow_none=True)
    selection_mode =  traitlets.Unicode('replace', allow_none=True)
    tool = traitlets.Unicode(None, allow_none=True)

    def __init__(self, **kwargs):
        super(ViewBase, self).__init__(**kwargs)
        self.df = self.model.df
        self.progress_indicator = vw.ProgressCircularNoAnimation(size=30, width=5, height=20, color=C0, value=10.4, text='')
        self.progress_text = vw.Status(value=self.model.status_text)
        traitlets.dlink((self.model, 'status_text'), (self.progress_text, 'value'))
        self.progress_widget = v.Container(children=[self.progress_indicator, self.progress_text])
        # self.progress_widget.layout.width = "95%"
        # self.progress_widget.layout.max_width = '500px'
        # self.progress_widget.description = "progress"
        self.model.signal_grid_progress.connect(self.on_grid_progress)

    def on_grid_progress(self, fraction):
        try:
            with self.output:
                self.progress_indicator.hidden = False
                vaex.jupyter.kernel_tick()
                self.progress_indicator.value = fraction * 100
                if fraction == 1:
                    self.hide_progress()
            return True
        except Exception as e:  # noqa
            with self.output:
                print("oops", e)
            return True

    @vaex.jupyter.debounced(0.3, skip_gather=True)
    def hide_progress(self):
        self.progress_indicator.hidden = True

    def select_nothing(self):
        with self.output:
            name = _translate_selection(self.selection_interact)
            self.df.select_nothing(name=name)

    def select_rectangle(self, x1, x2, y1, y2):
        with self.output:
            name = _translate_selection(self.selection_interact)
            self.df.select_rectangle(self.model.x.expression, self.model.y.expression, limits=[[x1, x2], [y1, y2]], mode=self.selection_mode, name=name)

    def select_x_range(self, x1, x2):
        with self.output:
            name = _translate_selection(self.selection_interact)
            self.df.select_box([self.model.x.expression], [[x1, x2]], mode=self.selection_mode, name=name)


@traitlet_fixes
class DataArray(ViewBase):
    """Will display a DataArray interactively, with an optional custom display_function.

    By default, it will simply display(...) the DataArray, using xarray's default display mechanism.
    """

    model = traitlets.Instance(model.DataArray)
    clear_output = traitlets.Bool(True, help="Clear output each time the data changes")
    display_function = traitlets.Any(display)
    matplotlib_autoshow = traitlets.Bool(True, help="Will call plt.show() inside output context if open figure handles exist")
    numpy_errstate = traitlets.Dict({'all': 'ignore'}, help="Default numpy errstate during display to avoid showing error messsages, see :class:`numpy.errstate`")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output = widgets.Output()
        self.output_data_array = widgets.Output()
        self.children = (self.progress_widget, self.output_data_array, self.output)
        self.model.observe(self.update_output, ['grid', 'grid_sliced'])
        self.update_output()

    def update_output(self, change=None):
        if self.clear_output:
            self.output_data_array.clear_output(wait=True)
        with self.output_data_array, np.errstate(**self.numpy_errstate):
            grid = self.model.grid_sliced
            if grid is None:
                grid = self.model.grid
            if grid is not None:
                self.display_function(grid)
                # make sure show is called inside the output widget
                if self.matplotlib_autoshow and 'matplotlib' in sys.modules:
                    import matplotlib.pyplot as plt
                    if plt.get_fignums():
                        plt.show()


class Heatmap(ViewBase):
    TOOLS_SUPPORTED = ['pan-zoom', 'select-rect', 'select-x']
    tool = traitlets.Unicode('pan-zoom', allow_none=True)
    model = traitlets.Instance(model.Heatmap)
    normalize = traitlets.Bool(False)
    colormap = traitlets.Unicode('afmhot')
    blend = traitlets.Unicode('selections')
    # TODO: should we expose this trait?
    tool = traitlets.Unicode(None, allow_none=True)
    transform = traitlets.Unicode("identity")

    dimension_fade = traitlets.Unicode('selections')
    dimension_facets = traitlets.Unicode('groupby1')
    dimension_alternative = traitlets.Unicode('slice')
    supports_transforms = True
    supports_normalize = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from . import bqplot
        self.output = widgets.Output()
        self.plot = bqplot.Heatmap(self.output, self,
                                   x_min=self.model.x.min, x_max=self.model.x.max,
                                   y_min=self.model.y.min, y_max=self.model.y.max)
        self.children = (self.progress_widget, self.plot.widget, self.output)

        grid = self.model.grid
        if self.model.grid_sliced is not None:
            grid = self.model.grid_sliced
        if self.normalize:
            grid = grid/grid.sum()

        traitlets.dlink((self, 'tool'), (self.plot, 'tool'))

        # first dlink our model to the plot
        traitlets.dlink((self.model.x, 'expression'), (self.plot, 'x_label'), transform=str)
        traitlets.dlink((self.model.y, 'expression'), (self.plot, 'y_label'), transform=str)

        # dlink the plot axis to the model
        traitlets.dlink((self.plot, 'x_min'), (self.model.x, 'min'))
        traitlets.dlink((self.plot, 'x_max'), (self.model.x, 'max'))
        traitlets.dlink((self.plot, 'y_min'), (self.model.y, 'min'))
        traitlets.dlink((self.plot, 'y_max'), (self.model.y, 'max'))

        self.model.observe(self.update_heatmap, ['grid', 'grid_sliced'])
        self.observe(self.update_heatmap, ['transform'])
        if self.model.grid is not None:
            self.update_heatmap()

    def update_heatmap(self, change=None):
        with self.output:
            selection_was_list, [selections] = vaex.utils.listify(self.model.selection)
            grid = self.model.grid
            if self.dimension_alternative == 'slice':
                if self.model.grid_sliced is not None:
                    grid = self.model.grid_sliced
            from vaex.utils import _parse_reduction, _parse_f, _normalize
            f = _parse_f(self.transform)
            with np.errstate(divide='ignore', invalid='ignore'):
                grid = f(grid)
            # if self.model.grid_sliced is not None:
            #     grid = self.model.grid_sliced
            # if self.normalize:
            grid = grid.astype(np.float64)
            grid, vmin, vmax = _normalize(grid)
            rgb_image = _parse_reduction("colormap", self.colormap, [])(grid)

            if rgb_image.shape[0] == 1:
                rgb_image = rgb_image[0]
            else:
                if self.blend == 'selections':
                    if selection_was_list:
                        rgb_image = vaex.image.fade(rgb_image[::-1])
                else:
                    raise ValueError('Unknown what to do with selection')
            assert rgb_image.ndim == 3  # including color channel
            rgb_image = np.transpose(rgb_image, (1, 0, 2))  # flip with/height
            rgb_image = rgb_image.copy()  # make contiguous
            assert rgb_image.shape[-1] == 4, "last dimention is channel"

            # TODO: we should pass the xarray to plot and let that take tare
            dims = self.model.grid.dims
            dim_x = dims[1 if selection_was_list else 0]
            dim_y = dims[2 if selection_was_list else 1]
            self.plot.x_min = self.model.grid.coords[dim_x].attrs['min']
            self.plot.x_max = self.model.grid.coords[dim_x].attrs['max']
            self.plot.y_min = self.model.grid.coords[dim_y].attrs['min']
            self.plot.y_max = self.model.grid.coords[dim_y].attrs['max']
            self.plot.set_rgb_image(rgb_image)


class Histogram(ViewBase):
    TOOLS_SUPPORTED = ['pan-zoom', 'select-x']
    model = traitlets.Instance(model.Histogram)
    normalize = traitlets.Bool(False)
    dimension_groups = traitlets.Unicode('selections')
    dimension_facets = traitlets.Unicode('group1')
    dimension_overplot = traitlets.Unicode('slice')
    transform = traitlets.Unicode("identity")
    supports_transforms = False
    supports_normalize = True

    def __init__(self, **kwargs):
        self._control = None
        super().__init__(**kwargs)
        self.output = widgets.Output()
        self.plot = self.create_plot()
        self.children = (self.progress_widget, self.plot.widget, self.output)

        widgets.dlink((self, 'tool'), (self.plot, 'tool'))

        # first dlink our model to the plot
        widgets.dlink((self.model.x, 'expression'), (self.plot, 'x_label'), transform=str)
        self.plot.y_label = "count"

        # set before we observe changes
        if self.model.x.min is not None:
            self.plot.x_min = self.model.x.min
        if self.model.x.max is not None:
            self.plot.x_max = self.model.x.max

        # then we sync the limits of the plot with a debouce to the model
        traitlets.dlink((self.plot, 'x_min'), (self.model.x, 'min'))
        traitlets.dlink((self.plot, 'x_max'), (self.model.x, 'max'))

        self.model.observe(self.update_data, ['grid', 'grid_sliced'])
        self.observe(self.update_data, ['normalize', 'dimension_groups'])

        @self.output.capture()
        @vaex.jupyter.debounced(DEBOUNCE_HOVER_SLICED)
        def unhighlight():
            self.plot.highlight(None)
            self.model.x_slice = None

        @self.output.capture()
        # @vaex.jupyter.debounced(DEBOUNCE_SLICE)
        def on_bar_hover(bar, event):
            self.model.x_slice = event['data']['index']
            self.plot.highlight(self.model.x_slice)
            unhighlight()

        self.plot.mark.on_hover(on_bar_hover)
        if self.model.grid is not None:
            self.update_data()

    def create_plot(self):
        from . import bqplot
        return bqplot.Histogram(self.output, self)

    def update_data(self, change=None):
        ylist = []
        colors = []
        selection_was_list, [selections] = vaex.utils.listify(self.model.selection)
        if self.dimension_groups == 'slice':
            y0 = self.model.grid
            if selection_was_list:
                y0 = y0[0]
            ylist.append(y0)
            colors.append(C0)
            if self.model.grid_sliced is not None:
                y1 = self.model.grid_sliced
                if selection_was_list:
                    y1 = y1[0]
                ylist.append(y1)
                colors.append(C1)
        elif self.dimension_groups == 'selections':
            ylist = self.model.grid
            if selection_was_list:
                colors = colors_default[:len(ylist)]
            else:
                colors = [colors_default[0]]
        else:
            raise ValueError(f'Unknown action {self.dimension_groups} for dimension_groups')

        if self.normalize:
            ylist = [y / np.sum(y) for y in ylist]
        x = self.model.x.bin_centers
        self.plot.x_min = self.model.x.min
        self.plot.x_max = self.model.x.max
        self.plot.update_data(x, np.array(ylist), colors)


class PieChart(Histogram):
    radius_split_fraction = 0.8

    def create_plot(self):
        from . import bqplot
        return bqplot.Piechart(self.output, self)
