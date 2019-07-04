import vaex.jupyter.grid
import vaex.jupyter.state
import vaex.jupyter.viz
import ipywidgets as widgets

# viz_types = [vaex.jupyter.viz.VizHeatmapBqplot, vaex.jupyter.viz.VizHistogramBqplot]

class Creator:
    def __init__(self, ds):
        self.ds = ds
        self.grid = vaex.jupyter.grid.Grid(self.ds, [])
        
        self.widget_button_new_histogram = widgets.Button(description='New: ' + vaex.jupyter.viz.VizHistogramBqplot.type_name, icon=vaex.jupyter.viz.VizHistogramBqplot.icon)
        self.widget_button_new_histogram.on_click(self.on_new_histogram)

        self.widget_button_new_heatmap = widgets.Button(description='New: ' + vaex.jupyter.viz.VizHeatmapBqplot.type_name, icon=vaex.jupyter.viz.VizHeatmapBqplot.icon)
        self.widget_button_new_heatmap.on_click(self.on_new_heatmap)

        self.widget_container = widgets.VBox()
        self.viz = []
        self.widget_buttons_remove = []
        self.widget_buttons = widgets.HBox([self.widget_button_new_histogram, self.widget_button_new_heatmap])
        self.widget = widgets.VBox([self.widget_buttons, self.widget_container])

    @property
    def column_names(self):
        return list(map(str, self.ds.get_column_names(virtual=True)))

    def on_new_histogram(self, button=None):
        state = vaex.jupyter.state.VizHistogramState(self.ds, x_expression=self.column_names[0])
        self.grid.state_add(state)
        viz = vaex.jupyter.viz.VizHistogramBqplot(state=state)
        self.viz_add(viz)

    def viz_add(self, viz):
        N = len(self.viz)
        self.viz.append(viz)
        self.widget_container.children = self.widget_container.children + (viz.widget,)
        # self.widget_container.set_title(N, 'Viz: ' + viz.type_name)
        # self.widget_container.selected_index = N

        def on_button_remove(button, viz=viz):
            self.grid.state_remove(viz.state)
            self.viz.remove(viz)
            self.widget_buttons_remove.remove(button)
            children = list(self.widget_container.children)
            children.remove(viz.widget)
            self.widget_container.children = children
            # if self.widget_container.selected_index >= len(self.widget_container.children):
            #     self.widget_container.selected_index = len(self.widget_container.children) - 1


        button_remove = widgets.Button(description='Remove', icon='trash')
        button_remove.on_click(on_button_remove)
        self.widget_buttons_remove.append(button_remove)

    def on_new_heatmap(self, button=None):
        state = vaex.jupyter.state.VizHeatmapState(self.ds, x_expression=self.column_names[0], y_expression=self.column_names[1])
        self.grid.state_add(state)
        viz = vaex.jupyter.viz.VizHeatmapBqplot(state=state)
        self.viz_add(viz)
        # N = len(self.viz)
        # self.viz.append(viz)
        # self.widget_container.children = self.widget_container.children + (viz.widget,)
        # self.widget_container.set_title(N, 'Viz: ' + viz.type_name)
        # self.widget_container.selected_index = N
