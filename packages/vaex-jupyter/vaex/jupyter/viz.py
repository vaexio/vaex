from __future__ import absolute_import
import vaex.ml
import traitlets
from .state import *
import bqplot.pyplot as plt
import bqplot
import ipyleaflet
import ipywidgets as widgets
import ipyvuetify as v
from .widgets import PlotTemplate

colors = ['C0', 'C1']
import matplotlib.colors
colors = list(map(matplotlib.colors.to_hex, colors))
colors
C0, C1 = colors
C0, C1 = '#9ECBF5', '#E0732C'

blackish = '#777'
DEBOUNCE_SLICE = 0.1
DEBOUNCE_HOVER_SLICED = 3


ICON_HISTOGRAM = 'histogram'
ICON_HEATMAP = 'heatmap'


class VizBaseBqplot(vaex.ml.state.HasState):
    opacity = 0.7

    def __init__(self, show_drawer=False, **kwargs):
        super(VizBaseBqplot, self).__init__(**kwargs)
        self._control = None
        self.create_axes()



        self.fig = bqplot.Figure(axes=self.axes)#, scale_x=self.x_scale, scale_y=self.y_scale)
        self.fig.fig_margin = {'bottom': 60, 'left': 60, 'right': 10, 'top': 60}
        self.fig.background_style = {'fill': 'none'}

        self.output = widgets.Output()
        self.create_viz()
        # self.widget = widgets.HBox([self.control, widgets.VBox([self.fig, self.output])])
        self.widget = PlotTemplate(components={
                    'main-widget': self.fig,
                    'control-widget': self.control,
                    'output-widget': self.output
                },
                model=show_drawer
        )
        def output_changed(*ignore):
            self.widget.new_output = True
        self.output.observe(output_changed, 'outputs')

    def create_viz(self):
        pass

    def create_axes(self):
        self.x_scale = bqplot.LinearScale()
        self.y_scale = bqplot.LinearScale()
        self.x_axis = bqplot.Axis(scale=self.x_scale)
        self.y_axis = bqplot.Axis(scale=self.y_scale, orientation='vertical')
        if 1:
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

    @property
    def control(self):
        if self._control is None:
            self._control = type(self).Control(self)
        return self._control
        
    def _ipython_display_(self):
        display(self.widget)

class VizHeatmapBqplot(VizBaseBqplot):
    type_name = 'Heatmap'
    icon = ICON_HEATMAP

    state = traitlets.Instance(VizHeatmapState)
    normalize = traitlets.Bool(False)

    class Control(v.Layout):
        def __init__(self, viz):
            super(type(self), self).__init__(column=True)
            self.viz = viz
            self.ds = self.viz.state.ds
            column_names = self.ds.get_column_names()
            self.x = v.Select(items=column_names, v_model=self.viz.state.x_expression, label='x axis')
            widgets.link((self.viz.state, 'x_expression'), (self.x, 'v_model'))

            self.y = v.Select(items=column_names, v_model=self.viz.state.y_expression, label='y axis')
            widgets.link((self.viz.state, 'y_expression'), (self.y, 'v_model'))

            self.normalize = v.Checkbox(label='normalize', v_model=self.viz.normalize)
            widgets.link((self.viz, 'normalize'), (self.normalize, 'v_model'))
            

            self.children = [self.x, self.y, self.normalize]

    def create_axes(self):
        self.x_scale = bqplot.OrdinalScale()
        self.y_scale = bqplot.OrdinalScale(reverse=False)
        self.x_axis = bqplot.Axis(scale=self.x_scale)
        self.y_axis = bqplot.Axis(scale=self.y_scale, orientation='vertical')
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

    def __init__(self, **kwargs):
        super(VizHeatmapBqplot, self).__init__(**kwargs)
        self.fig.padding_y = 0
        self.fig.fig_margin = {'bottom': 80, 'left': 60, 'right': 10, 'top': 60}

        grid = self.state.grid
        if self.state.grid_sliced is not None:
            grid = self.state.grid_sliced
        if self.normalize:
            grid = grid/grid.sum()
        self.color_scale = bqplot.ColorScale(scheme='Blues')
        self.color_axis = bqplot.ColorAxis(scale=self.color_scale, label='counts', scheme='Blues')
        self.fig.axes = self.fig.axes + [self.color_axis]
        self.scales = {'row': self.y_scale, 'column': self.x_scale, 'color': self.color_scale}
        # self.scales = {'row': bqplot.OrdinalScale(), 'column': bqplot.OrdinalScale(), 'color': self.color_scale}
        
        self.heatmap = bqplot.GridHeatMap(color=grid.T[:,::], scales=self.scales)
        self.update_heatmap()
        self.fig.marks = self.fig.marks + [self.heatmap]
        self.state.observe(self.update_heatmap, 'grid')
        widgets.dlink((self.state, 'x_expression'), (self.x_axis, 'label'))
        widgets.dlink((self.state, 'y_expression'), (self.y_axis, 'label'))
        @self.output.capture()
        # @vaex.jupyter.debounced(DEBOUNCE_SLICE)
        def on_bar_hover(bar, event):
#             print(event)
            #set_transparancy(event['data']['sub_index'], event['data']['index'])
            #print(event['data']['index'])
            self.state.y_slice = event['data']['row']
            self.state.x_slice = event['data']['column']
            self.set_transparancy()
            #print(viz_state.grid.sum())
        self.heatmap.on_hover(on_bar_hover)

    @vaex.jupyter.debounced(DEBOUNCE_HOVER_SLICED, method=True)
    def reset_opacities(self):
        #opacities = self.state.grid * 0 + self.opacity
        self.state.x_slice = None
        self.state.y_slice = None
        #self.heatmap.opacities = opacities.T.ravel().tolist()

    def set_transparancy(self):
        #opacities = self.state.grid * 0 + 0.2
#         if self.state.groupby is None:
#             opacities[index2] = self.opacity
#         else:
        #opacities[self.state.x_slice, self.state.y_slice] = self.opacity
        #self.heatmap.opacities = opacities.T.ravel().tolist()
        self.reset_opacities()
        
    
    def update_heatmap(self, change=None):
        with self.output:
            grid = self.state.grid
            if self.state.grid_sliced is not None:
                grid = self.state.grid_sliced
            if self.normalize:
                grid = grid/grid.sum()
            need_update_hack = self.heatmap.color.shape != grid.T.shape
            self.heatmap.color = grid.T[:,::]
            self.heatmap.row = self.state.y_centers
            self.heatmap.column = self.state.x_centers
            if need_update_hack:
                # print('update hack')
                marks = self.fig.marks
                self.fig.marks = []
                self.fig.marks = marks

    # def _ipython_display_(self):
    #     display(widgets.VBox([self.fig, self.output]))


class VizHistogramBqplot(VizBaseBqplot):
    type_name = 'Histogram'
    icon = ICON_HISTOGRAM

    state = traitlets.Instance(VizHistogramState)
    normalize = traitlets.Bool(False)

    class Control(v.Layout):
        def __init__(self, viz):
            super(type(self), self).__init__(column=True)
            self.viz = viz
            self.ds = self.viz.state.ds
            column_names = self.ds.get_column_names()

            self.x = v.Select(items=column_names, v_model=self.viz.state.x_expression, label='x axis')
            widgets.link((self.viz.state, 'x_expression'), (self.x, 'v_model'))

            self.normalize = v.Checkbox(v_model=self.viz.normalize, label='normalize')
            widgets.link((self.viz, 'normalize'), (self.normalize, 'v_model'))

            self.min = widgets.FloatText(description='min')
            self.max = widgets.FloatText(description='max')
            widgets.link((self.viz.state, 'x_min'), (self.min, 'value'))
            widgets.link((self.viz.state, 'x_max'), (self.max, 'value'))
            
            self.shape = widgets.IntSlider(min=1, max=512, value=64, description='bins')
            widgets.link((self.viz.state, 'shape'), (self.shape, 'value'))

            self.bar_style = widgets.ToggleButtons(options=[('Stacked', 'stacked'), ('Grouped', 'grouped')], description='Bar style')
            widgets.link((self.viz. bar, 'type'), (self.bar_style, 'value'))

            self.children = [self.x, self.normalize, self.min, self.max, self.shape, self.bar_style]

    def create_viz(self):
        self.scales = {'x': self.x_scale, 'y': self.y_scale}
        self.bar = self.mark = bqplot.Bars(x=self.state.x_centers, y=self.state.grid, scales=self.scales)
        self.marks = [self.mark]

    def __init__(self, **kwargs):
        self._control = None
        super(VizHistogramBqplot, self).__init__(**kwargs)
        # using dlink allows us to change the label
        if len(self.axes) > 0:
            widgets.dlink((self.state, 'x_expression'), (self.x_axis, 'label'))
            self.y_axis.label = 'counts'
        else:
            widgets.dlink((self.state, 'x_expression'), (self.fig, 'title'))
        self.fig.marks = self.fig.marks + self.marks
        self.state.observe(self.update_data, ['grid', 'grid_sliced'])
        self.observe(self.update_data, ['normalize'])
        @self.output.capture()
        # @vaex.jupyter.debounced(DEBOUNCE_SLICE)
        def on_hover(bar, event):
            #set_transparancy(event['data']['sub_index'], event['data']['index'])
#             print(event['data']['index'])
            self.state.x_slice = event['data']['index']
            self.set_transparancy(self.state.x_slice)
            #print(viz_state.grid.sum())
        self.mark.on_hover(on_hover)
        self.reset_opacities()

    def update_data(self, change):
        self.mark.x = self.state.x_centers
        y0 = self.state.grid
        if self.normalize:
            y0 = y0 / np.sum(y0)
        
        if self.state.grid_sliced is not None:
            y1 = self.state.grid_sliced
            if self.normalize:
                y1 = y1 / np.sum(y1)
            self.mark.y = np.array([y0, y1])
            self.mark.colors = [C0, C1]
            self.mark.type = 'grouped'
        else:
            self.mark.y = y0
            self.mark.colors = [C0]

    @vaex.jupyter.debounced(DEBOUNCE_HOVER_SLICED, method=True)
    def reset_opacities(self):
        opacities = self.state.grid * 0 + self.opacity
        self.state.x_slice = None
        self.mark.opacities = opacities.T.ravel().tolist()

    def set_transparancy(self, index, index2=None):
        opacities = self.mark.y * 0 + 0.2
#         if self.state.groupby is None:
#             opacities[index2] = self.opacity
#         else:
        if len(self.mark.y.shape) == 2:
            opacities[:, index] = self.opacity
        else:
            opacities[index] = self.opacity
        self.mark.opacities = opacities.T.ravel().tolist()
        self.reset_opacities()

class VizPieChartBqplot(VizHistogramBqplot):
    radius_split_fraction = 0.8

    class Control(widgets.VBox):
        def __init__(self, viz):
            super(type(self), self).__init__()
            self.viz = viz
            self.ds = self.viz.state.ds
            column_names = self.ds.get_column_names()
            self.x = widgets.Dropdown(options=column_names, description='x axis')
            widgets.link((self.viz.state, 'x_expression'), (self.x, 'value'))

            self.normalize = widgets.Checkbox(description='normalize')
            widgets.link((self.viz, 'normalize'), (self.normalize, 'value'))


            self.min = widgets.FloatText(description='min')
            self.max = widgets.FloatText(description='max')
            widgets.link((self.viz.state, 'x_min'), (self.min, 'value'))
            widgets.link((self.viz.state, 'x_max'), (self.max, 'value'))
            
            self.shape = widgets.IntSlider(min=1, max=512, value=64, description='bins')
            widgets.link((self.viz.state, 'shape'), (self.shape, 'value'))

            # self.bar_style = widgets.ToggleButtons(options=[('Stacked', 'stacked'), ('Grouped', 'grouped')], description='Bar style')
            # widgets.link((self.viz. bar, 'type'), (self.bar_style, 'value'))

            self.children = [self.x, self.normalize, self.min, self.max, self.shape]#, self.bar_style]

    def __init__(self, radius=100, **kwargs):
        self.radius = radius
        super(VizPieChartBqplot, self).__init__(**kwargs)

    def create_axes(self):
        self.axes = []

    def create_viz(self):
        self.mark = self.pie1 = bqplot.Pie(sizes=self.state.grid, radius=self.radius, inner_radius=0, stroke=blackish)
        # pie 2 holds the sliced data
        self.pie2 = bqplot.Pie(sizes=self.state.grid, radius=self.radius, inner_radius=self.radius,
                               label_color='transparent', stroke=blackish)
        self.marks = [self.pie1, self.pie2]
        self.pie1.labels = self.state.ds.category_labels(self.state.x_expression)
        self.pie2.labels = self.pie1.labels
        self.pie2.label_color = 'transparent'

    def update_data(self, change):
        y0 = self.state.grid
        if self.state.grid_sliced is not None:
            y0 = self.state.grid_sliced
        if self.normalize:
            y0 = y0 / np.sum(y0)
        self.pie1.sizes = y0
        # no labels does not work (nothing gets displayed)
        # if self.state.grid_sliced is not None:
        #     y1 = self.state.grid_sliced
        #     if self.normalize:
        #         y1 = y1 / np.sum(y1)
        #     self.pie2.sizes = y1
        #     self.pie1.radius = self.pie2.inner_radius = self.radius * self.radius_split_fraction
        # else:
        #     self.pie1.radius = self.pie2.inner_radius = self.radius

    @vaex.jupyter.debounced(DEBOUNCE_HOVER_SLICED, method=True)
    def reset_opacities(self):
        opacities = self.state.grid * 0 + self.opacity
        self.state.x_slice = None
        self.pie1.opacities = opacities.T.ravel().tolist()
        self.pie2.opacities = opacities.T.ravel().tolist()

    def set_transparancy(self, index, index2=None):
        opacities = self.state.grid * 0 + 0.2
        opacities[index] = self.opacity

        self.pie1.opacities = opacities.T.ravel().tolist()
        self.pie2.opacities = opacities.T.ravel().tolist()
        self.reset_opacities()

class VizMapGeoJSONLeaflet(vaex.ml.state.HasState):
    type_name = 'GeoJSON'
    icon = ICON_HISTOGRAM

    state = traitlets.Instance(VizHistogramState)
    normalize = traitlets.Bool(False)

    class Control(widgets.VBox):
        def __init__(self, viz, column_names):
            super(type(self), self).__init__()
            self.viz = viz
            self.ds = self.viz.state.ds
            self.x = widgets.Dropdown(options=column_names, description='x axis')
            widgets.link((self.viz.state, 'x_expression'), (self.x, 'value'))

            # self.normalize = widgets.Checkbox(description='normalize')
            # widgets.link((self.viz, 'normalize'), (self.normalize, 'value'))


            # self.min = widgets.FloatText(description='min')
            # self.max = widgets.FloatText(description='max')
            # widgets.link((self.viz.state, 'x_min'), (self.min, 'value'))
            # widgets.link((self.viz.state, 'x_max'), (self.max, 'value'))
            
            # self.shape = widgets.IntSlider(min=1, max=512, value=64, description='bins')
            # widgets.link((self.viz.state, 'shape'), (self.shape, 'value'))

            # self.bar_style = widgets.ToggleButtons(options=[('Stacked', 'stacked'), ('Grouped', 'grouped')], description='Bar style')
            # widgets.link((self.viz. bar, 'type'), (self.bar_style, 'value'))

            # self.children = [self.x, self.normalize, self.min, self.max, self.shape, self.bar_style]
            self.children = [self.x]

    # def create_viz(self):
    #     self.scales = {'x': self.x_scale, 'y': self.y_scale}
    #     self.bar = bqplot.Bars(x=self.state.x_centers, y=self.state.grid, scales=self.scales)


    def __init__(self, geo_json, column_names,  **kwargs):
        # self._control = None
        super(VizMapGeoJSONLeaflet, self).__init__(**kwargs)
        self.map = ipyleaflet.Map(center=[40.72866940630964, -73.80228996276857], zoom=10)
        self.control = VizMapGeoJSONLeaflet.Control(self, column_names)
        self.regions_layer = ipyleaflet.LayerGroup()
        # self.index_mapping = {}
        self.output = widgets.Output()
        for i, feature in enumerate(geo_json["features"]):
            if feature["geometry"]["type"] == "Polygon":
                    feature["geometry"]["type"] = "MultiPolygon"
                    feature["geometry"]["coordinates"] = [feature["geometry"]["coordinates"]]
            polygon = ipyleaflet.GeoJSON(data=feature, hover_style={'fillColor': 'red', 'fillOpacity': 0.6})
            self.regions_layer.add_layer(polygon)
            # self.index_mapping[feature['properties'][index_key]] = i
            @self.output.capture()
            # @vaex.jupyter.debounced(DEBOUNCE_SLICE)
            def on_hover(index=i, **properties):
                # index_value = properties[index_key]
                # index = self.index_mapping[index_value]
                self.state.x_slice = index#event['data']['index']
                self.reset_slice()
                # self.set_transparancy(self.state.x_slice)
                #print(viz_state.grid.sum())
            polygon.on_hover(on_hover)
            # @self.output.capture()
            # def on_click(index=i, **properties):
            #     if self.state.x_slice :
            #         self.state.x_slice = None
            #     else:
            #         self.state.x_slice = None
            #     self.reset_slice()
            #     # self.set_transparancy(self.state.x_slice)
            #     #print(viz_state.grid.sum())
            # polygon.on_click(on_click)
        self.map.add_layer(self.regions_layer)

        # self.state.observe(self.update_bars, ['grid', 'grid_sliced'])
        # self.observe(self.update_bars, ['normalize'])

        # self.regions_layer.on_hover(on_hover)
        # self.reset_opacities()
        # self.create_viz()
        self.widget = widgets.HBox([self.control, widgets.VBox([self.map, self.output])])

    @vaex.jupyter.debounced(DEBOUNCE_HOVER_SLICED, method=True)
    def reset_slice(self):
        # opacities = self.state.grid * 0 + self.opacity
        self.state.x_slice = None
        # self.bar.opacities = opacities.T.ravel().tolist()
#     def update_bars(self, change):
#         self.bar.x = self.state.x_centers
#         y0 = self.state.grid
#         if self.normalize:
#             y0 = y0 / np.sum(y0)
        
#         if self.state.grid_sliced is not None:
#             y1 = self.state.grid_sliced
#             if self.normalize:
#                 y1 = y1 / np.sum(y1)
#             self.bar.y = np.array([y0, y1])
#             self.bar.colors = [C0, C1]
#             self.bar.type = 'grouped'
#         else:
#             self.bar.y = y0
#             self.bar.colors = [C0]

#     @vaex.jupyter.debounced(1.0, method=True)
#     def reset_opacities(self):
#         opacities = self.state.grid * 0 + self.opacity
#         self.state.x_slice = None
#         self.bar.opacities = opacities.T.ravel().tolist()

#     def set_transparancy(self, index, index2=None):
#         opacities = self.bar.y * 0 + 0.2
# #         if self.state.groupby is None:
# #             opacities[index2] = self.opacity
# #         else:
#         if len(self.bar.y.shape) == 2:
#             opacities[:, index] = self.opacity
#         else:
#             opacities[index] = self.opacity
#         self.bar.opacities = opacities.T.ravel().tolist()
#         self.reset_opacities()
        
# viz_hist = VizHistogramBqplot(state=hist_state)
# if dev:
#     display(viz_hist)

    