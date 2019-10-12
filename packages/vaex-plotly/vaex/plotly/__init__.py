import vaex
from vaex.utils import (_ensure_string_from_expression,
                        _ensure_strings_from_expressions,
                        _parse_f,
                        _parse_n,
                        _ensure_list)
from .widgets import PlotTemplatePlotly

import numpy as np
import ipywidgets as widgets
import ipyvuetify as vue


class DataFrameAccessorPlotly(object):
    def __init__(self, df):
        self.df = df

    def scatter(self, x, y, xerr=None, yerr=None, selection=None,
                size=None, color=None, symbol=None,
                label=None, xlabel=None, ylabel=None, title=None,
                colorbar=None, colorbar_label=None, colormap=None,
                figure_height=None, figure_width=None,
                tooltip_title=None, tooltip_data=None,
                length_limit=50000, length_check=True):
        """Scatter plot using plotly.
        Convenience wrapper around plotly.graph_objs.Scatter when for working with small DataFrames or selections.

        :param x: Expression to plot on the x axis. Can be a list to plot multiple sets
        :param y: Expression to plot on the y axis. Can be a list to plot multiple sets
        :param xerr: Expression or a list of expressions for x-error bars
        :param xerr: Expression or a list of expressions for y-error bars
        :param selection: Selection, or None. Can be a list to plot multiple sets
        :param size: The size of the markers. Can be an Expression or a list, if multiple sets are plotted
        :param color: The color of the markers. Can be an Expression or a list if multiple sets are plotted
        :param symbol: Plotly symbols for the markers. Can be a list if multiple sets are plotted
        :param label: Label for the legend
        :param xlabel: label for x axis, if None .label(x) is used
        :param ylabel: label for y axis, if None .label(y) is used
        :param title: the plot title
        :param colorbar: if True, display a colorbar
        :param colorbar_label: A label for the colorbar
        :param colormap: A name of a colormap/colorscale supported by plotly
        :param figure_height: The figure height in pix
        :param figure_width: The figure width in pix
        :param tooltip_title: Expression for the tooltip title
        :param tooltip_data: A list of expressions for the extra tooltip data
        :param length_limit: maximum number of rows it will plot
        :param length_check: should we do the maximum row check or not?
        :return plotly.graph_objs._figurewidget.FigureWidget fig: a plotly FigureWidget
        """

        import plotly.graph_objs as go
        import plotly.callbacks

        x = _ensure_list(x)
        y = _ensure_list(y)

        x = _ensure_strings_from_expressions(x)
        y = _ensure_strings_from_expressions(y)
        assert len(x) == len(y), 'x and y should have the same number of Expressions.'
        num_traces = len(x)

        args = self._arg_len_check(num_traces, xerr=xerr, yerr=yerr, size=size,
                                   color=color, symbol=symbol, label=label,
                                   selection=selection, tooltip_title=tooltip_title)
        xerr, yerr, size, color, symbol, label, selection, tooltip_title = args

        if length_check:
            count = np.sum([self.df.count(selection=selection)])
            if count > length_limit:
                raise ValueError("the number of rows (%d) is above the limit (%d), pass length_check=False, \
                                 or increase length_limit" % (count, length_limit))

        traces = []
        for i in range(num_traces):
            symbol_value = symbol[i]
            label_value = label[i]
            selection_value = selection[i]

            x_values = self.df.evaluate(x[i], selection=selection_value)
            y_values = self.df.evaluate(y[i], selection=selection_value)
            if xerr[i] is not None:
                xerr_values = self.df.evaluate(xerr[i], selection=selection_value)
                xerr_object = go.scatter.ErrorX(array=xerr_values, thickness=0.5)
            else:
                xerr_object = None
            if yerr[i] is not None:
                yerr_values = self.df.evaluate(yerr[i], selection=selection_value)
                yerr_object = go.scatter.ErrorY(array=yerr_values, thickness=0.5)
            else:
                yerr_object = None
            if size[i] is not None:
                if isinstance(size[i], vaex.expression.Expression):
                    size_values = self.df.evaluate(size[i], selection=selection_value)
                else:
                    size_values = size[i]
            else:
                size_values = size[i]
            if color[i] is not None:
                if isinstance(color[i], vaex.expression.Expression):
                    color_values = self.df.evaluate(color[i], selection=selection_value)
                    cbar = go.scatter.marker.ColorBar(title=colorbar_label)
                else:
                    cbar = None
                    color_values = color[i]
            else:
                cbar = None
                color_values = color[i]

            # This builds the data needed for the tooltip display, including the template
            hovertemplate = ''
            if tooltip_title[i] is not None:
                hover_title = self.df.evaluate(tooltip_title[i])
                hovertemplate += '<b>%{hovertext}</b><br>'
            else:
                hover_title = None

            hovertemplate += '<br>' + x[i] + '=%{x}'
            hovertemplate += '<br>' + y[i] + '=%{y}'

            if tooltip_data is not None:
                tooltip_data = _ensure_strings_from_expressions(tooltip_data)
                customdata = np.array(self.df.evaluate(', '.join(tooltip_data), selection=selection_value)).T
                for j, expr in enumerate(tooltip_data):
                    hovertemplate += '<br>' + expr + '=%{customdata[' + str(j) + ']}'
            else:
                customdata = None
            hovertemplate += '<extra></extra>'

            # the plotting starts here
            marker = go.scatter.Marker(color=color_values, size=size_values, showscale=colorbar,
                                       colorscale=colormap, symbol=symbol_value, colorbar=cbar)

            trace = go.Scatter(x=x_values, y=y_values, error_x=xerr_object, error_y=yerr_object,
                               mode='markers',
                               marker=marker,
                               hovertemplate=hovertemplate,
                               customdata=customdata,
                               hovertext=hover_title,
                               name=label_value)
            traces.append(trace)

        legend = go.layout.Legend(orientation='h')
        title = go.layout.Title(text=title, xanchor='center', x=0.5, yanchor='top')
        layout = go.Layout(height=figure_height,
                           width=figure_width,
                           legend=legend,
                           title=title,
                           xaxis=go.layout.XAxis(title=xlabel or x[0]),
                           yaxis=go.layout.YAxis(title=ylabel or y[0],
                                                 scaleanchor='x',
                                                 scaleratio=1))

        fig = go.FigureWidget(data=traces, layout=layout)

        # Define the widget components
        _widget_selection = widgets.ToggleButtons(options=['default'], description='selection')
        _items = [{'text': xexpr + ' -vs-  ' + yexpr, 'value': i} for i, (xexpr, yexpr) in enumerate(zip(x, y))]
        _widget_selection_space = vue.Select(items=_items, v_model=0, label='selection space')
        _widget_selection_mode = widgets.ToggleButtons(options=['replace', 'and', 'or', 'xor', 'subtract'],
                                                       value='replace',
                                                       description='mode')
        _widget_selection_undo = widgets.Button(description='undo', icon='arrow-left')
        _widget_selection_redo = widgets.Button(description='redo', icon='arrow-right')
        _widget_history_box = widgets.HBox(children=[widgets.Label('history', layout={'width': '80px'}),
                                                     _widget_selection_undo,
                                                     _widget_selection_redo])
        _widget_clear_history = vue.Btn(children=[vue.Icon(children=['menu']), 'clear selections'])
        # Put them together in the control-widget: this is what is contained within the navigation drawer
        control_widget = vue.Layout(pa_1=True, column=True, children=[_widget_selection_space,
                                                                      _widget_selection,
                                                                      _widget_selection_mode,
                                                                      _widget_history_box])
        # The output widget
        _widget_output = widgets.Output()

        # Set up all the links and interactive actions
        @_widget_output.capture(clear_output=True)
        def _selection(trace, points, selector):
            i = _widget_selection_space.v_model
            if isinstance(selector, plotly.callbacks.BoxSelector):
                limits = [selector.xrange, selector.yrange]
                print(i, x, y, limits)
                self.df.select_rectangle(x=x[i], y=y[i], limits=limits, mode=_widget_selection_mode.value)
            elif isinstance(selector, plotly.callbacks.LassoSelector):
                self.df.select_lasso(expression_x=x[i], expression_y=y[i],
                                     xsequence=selector.xs, ysequence=selector.ys,
                                     mode=_widget_selection_mode.value)
            else:
                raise ValueError('Unsupported selection: please use a Box or a Lasso selection.')

        fig.data[0].on_selection(_selection)
        fig.data[0].on_deselect(self._selection_clear)
        _widget_selection_undo.on_click(self._selection_undo)
        _widget_selection_redo.on_click(self._selection_redo)

        # create the complete widget
        figure_widget = PlotTemplatePlotly(components={'main-widget': fig,
                                                       'control-widget': control_widget,
                                                       'output-widget': _widget_output
                                                       })

        return figure_widget


    def histogram(self, x, what='count(*)', grid=None, shape=64, limits=None, f='identity', n=None,
                  lw=None, ls=None, color=None, figure_height=None, figure_width=None,
                  xlabel=None, ylabel=None, label=None, title=None, selection=None, progress=None):
        """Create a histogram using plotly.

        Example

        >>> df.plotly.histogram(df.x)
        >>> df.plotly.histogram(df.x, limits=[0, 100], shape=100)
        >>> df.plotly.histogram(df.x, what='mean(y)', limits=[0, 100], shape=100)

        If you want to do a computation yourself, pass the grid argument, but you are responsible for passing the
        same limits arguments:

        >>> counts = df.mean(df.y, binby=df.x, limits=[0, 100], shape=100)/100.
        >>> df.plot1d(df.x, limits=[0, 100], shape=100, grid=means, label='mean(y)/100')

        :param x: Expression or a list of expressions to bin in the x direction
        :param what: What to plot, count(*) will show a N-d histogram, mean('x'), the mean of the x column, sum('x') the sum
        :param grid: If the binning is done before by yourself, you can pass it
        :param shape: Int or a list of ints describing the grid on which to bin the data
        :param limits: list of [xmin, xmax], or a description such as 'minmax', '99%'
        :param f: transform values by: 'identity' does nothing 'log' or 'log10' will show the log of the value
        :param n: normalization function, currently only 'normalize' is supported, or None for no normalization
        :param lw: width or a list of widths of the lines for each of the histograms
        :param ls: line style or a line of line style for each of the histograms
        :param color: color or a list of colors for each of the histograms
        :param figure_height: The figure height in pix
        :param figure_width: The figure width in pix
        :param xlabel: String for label on x axis
        :param ylabel: Same for y axis
        :param label: labels or names for the data being plotted
        :param title: the plot title
        :param selection: Name of selection to use (or True for the 'default'), or a selection-like expresson
        :param progress: If True, display a progress bar of the binning process
        :return plotly.graph_objs._figurewidget.FigureWidget fig: a plotly FigureWidget
        """

        import plotly.graph_objs as go
        import plotly.callbacks

        _widget_f = vue.Select(items=['identity', 'log', 'log10', 'log1p'], v_model=f or 'identity', label='Transform')


        if isinstance(x, list) is False:
            x = [x]
        x = _ensure_strings_from_expressions(x)
        num_traces = len(x)
        # make consistency checks
        args = self._arg_len_check(num_traces, shape=shape, color=color, lw=lw, ls=ls,
                                   label=label, selection=selection)
        shape, color, lw, ls, label, selection = args

        traces = []
        for i in range(num_traces):

            xar, counts = self._grid(expr=x[i], what=what, shape=shape[i], limits=limits,
                                     f=f, n=n, selection=selection[i], progress=progress)

            line = go.scatter.Line(color=color[i], width=lw[i], dash=ls[i])
            traces.append(go.Scatter(x=xar, y=counts, mode='lines', line_shape='hv', line=line, name=label[i]))

        legend = go.layout.Legend(orientation='h')
        title = go.layout.Title(text=title, xanchor='center', x=0.5, yanchor='top')
        layout = go.Layout(height=figure_height,
                           width=figure_width,
                           legend=legend,
                           title=title,
                           xaxis=go.layout.XAxis(title=xlabel or x[0]),
                           yaxis=go.layout.YAxis(title=ylabel or what))
        fig = go.FigureWidget(data=traces, layout=layout)

        return fig

    def heatmap(self, x, y, what="count(*)", shape=128, limits=None, selection=None, f=None, n=None,
                colorbar=None, colorbar_label=None, colormap=None, vmin=None, vmax=None,
                xlabel=None, ylabel=None, title=None, figure_height=None, figure_width=None,
                equal_aspect=None, progress=None):
        """Create a heatmap using plotly.

        :param x: Expression to bin in the x direction
        :param y: Expression to bin in the y direction
        :param what: What to plot, count(*) will show a N-d histogram, mean('x'), the mean of the x column, sum('x') the sum
        :param shape: shape of the 2D histogram grid
        :param limits: list of [[xmin, xmax], [ymin, ymax]], or a description such as 'minmax', '99%'
        :param f: transform values by: 'identity' does nothing 'log' or 'log10' will show the log of the value
        :param n: normalization function, currently only 'normalize' is supported, or None for no normalization
        :param colorbar: if True, display a colorbar
        :param colorbar_label: A label for the colorbar
        :param colormap: A name of a colormap/colorscale supported by plotly
        :param vmin: The lower limit of the color range (vmax must be set as well)
        :param vmax: The upper limit of the color range (vmin must be set as well)
        :param xlabel: label for x axis, if None .label(x) is used
        :param ylabel: label for y axis, if None .label(y) is used
        :param title: the plot title
        :param figure_height: The figure height in pix
        :param figure_width: The figure width in pix
        :param equal_aspect: If True, the axis will have a scale ratio of 1 (equal aspect)
        :param progress: If True, display a progress bar of the binning process
        :return plotly.graph_objs._figurewidget.FigureWidget fig: a plotly FigureWidget
        """

        import plotly.graph_objs as go
        import plotly.callbacks

        # Degine the widget components
        _widget_progress = widgets.FloatProgress(value=0.0, min=0.0, max=1.0, step=0.01,
                                                 layout={'width': '95%', 'max_width': '500pix'},
                                                 description='progress')

        _widget_f = vue.Select(items=['identity', 'log', 'log10', 'log1p'], v_model=f or 'identity', label='Transform')
        _widget_vmin = widgets.FloatSlider(value=0, min=0, max=100, step=0.1, description='vmin%')
        _widget_vmax = widgets.FloatSlider(value=100, min=0, max=100, step=0.1, description='vmax%')
        _widget_selection = widgets.ToggleButtons(options=['default'], description='selection')
        _widget_selection_mode = widgets.ToggleButtons(options=['replace', 'and', 'or', 'xor', 'subtract'],
                                                       value='replace',
                                                       description='mode')
        _widget_selection_undo = widgets.Button(description='undo', icon='arrow-left')
        _widget_selection_redo = widgets.Button(description='redo', icon='arrow-right')
        _widget_history_box = widgets.HBox(children=[widgets.Label('history', layout={'width': '80px'}),
                                                     _widget_selection_undo,
                                                     _widget_selection_redo])
        _widget_clear_history = vue.Btn(children=[vue.Icon(children=['menu']), 'clear selections'])
        # Put them together in the control-widget: this is what is contained within the navigation drawer
        control_widget = vue.Layout(pa_1=True, column=True, children=[_widget_f,
                                                                      _widget_vmin,
                                                                      _widget_vmax,
                                                                      _widget_selection,
                                                                      _widget_selection_mode,
                                                                      _widget_history_box,
                                                                      _widget_clear_history])
        # The output widget
        _widget_output = widgets.Output()
        # The widget for the temporary output of the progressbar
        _widget_progress_output = widgets.Output()

        #  Creating the plotly figure, which is also a widget
        x = _ensure_string_from_expression(x)
        y = _ensure_string_from_expression(y)

        binby = []
        for expression in [y, x]:
            if expression is not None:
                binby = [expression] + binby
        limits = self.df.limits(binby, limits)

        extent, counts = self._grid(expr=binby, what=what, shape=shape, limits=limits,
                                    f=_widget_f.v_model, n=n, selection=selection, progress=progress)

        cbar = go.heatmap.ColorBar(title=colorbar_label)
        heatmap = go.Heatmap(z=counts, colorscale=colormap, zmin=vmin, zmax=vmax,
                             x0=extent[0], dx=np.abs((extent[1]-extent[0])/shape),
                             y0=extent[2], dy=np.abs((extent[3]-extent[2])/shape),
                             colorbar=cbar, showscale=colorbar,
                             hoverinfo=['x', 'y', 'z'])

        dummy_scatter = go.Scatter(y=[None])

        title = go.layout.Title(text=title, xanchor='center', x=0.5, yanchor='top')
        layout = go.Layout(height=figure_height,
                           width=figure_width,
                           title=title,
                           xaxis=go.layout.XAxis(title='x', range=limits[0]),
                           yaxis=go.layout.YAxis(title='y', range=limits[1], scaleanchor='x', scaleratio=1))
        if equal_aspect:
            layout['yaxis']['scaleanchor'] = 'x'
            layout['yaxis']['scaleratio'] = 1

        fig = go.FigureWidget(data=[dummy_scatter, heatmap], layout=layout)

        @_widget_progress_output.capture(clear_output=True)
        def _pan_and_zoom(layout, _xrange, _yrange):
            limits = [_yrange, _xrange]
            extent, counts = self._grid(expr=binby, what=what, shape=shape, limits=limits, f=_widget_f.v_model, progress=True)
            with fig.batch_update():
                fig.data[1]['z'] = counts
                fig.data[1]['x0'] = extent[0]
                fig.data[1]['dx'] = np.abs((extent[1]-extent[0])/shape)
                fig.data[1]['y0'] = extent[2]
                fig.data[1]['dy'] = np.abs((extent[3]-extent[2])/shape)
                fig.data[1]['zmin'] = 0
                fig.data[1]['zmax'] = 0
                fig.data[1]['zauto'] = True

        @_widget_output.capture(clear_output=True)
        def _selection(trace, points, selector):
            if isinstance(selector, plotly.callbacks.BoxSelector):
                limits = [selector.xrange, selector.yrange]
                self.df.select_rectangle(x=x, y=y, limits=limits, mode=_widget_selection_mode.value)
            elif isinstance(selector, plotly.callbacks.LassoSelector):
                self.df.select_lasso(expression_x=x, expression_y=y,
                                     xsequence=selector.xs, ysequence=selector.ys,
                                     mode=_widget_selection_mode.value)
            else:
                raise ValueError('Unsupported selection: please complain to Jovan.')

        @_widget_progress_output.capture(clear_output=True)
        def _transform_f(change=None, *args, **kwargs):
            extent, counts = self._grid(expr=binby, what=what, shape=shape, limits=limits, f=_widget_f.v_model, progress=True)
            with fig.batch_update():
                fig.data[1]['z'] = counts
                fig.data[1]['x0'] = extent[0]
                fig.data[1]['dx'] = np.abs((extent[1]-extent[0])/shape)
                fig.data[1]['y0'] = extent[2]
                fig.data[1]['dy'] = np.abs((extent[3]-extent[2])/shape)
                fig.data[1]['zmin'] = 0
                fig.data[1]['zmax'] = 0
                fig.data[1]['zauto'] = True

        @_widget_progress_output.capture(clear_output=True)
        def _update_colorbar_range(change=None, *args, **kwargs):
            _vmin, _vmax = np.percentile(fig.data[1]['z'], q=[_widget_vmin.value, _widget_vmax.value])
            with fig.batch_update():
                fig.data[1]['zmin'] = _vmin
                fig.data[1]['zmax'] = _vmax

        # Enable the dynamic zooming, panning and selections
        fig.layout.on_change(_pan_and_zoom, 'xaxis.range', 'yaxis.range')
        fig.data[0].on_selection(_selection)

        # link the buttons and sliders
        _widget_clear_history.on_event('click', self._selection_clear)
        _widget_selection_undo.on_click(self._selection_undo)
        _widget_selection_redo.on_click(self._selection_redo)
        widgets.Widget.observe(_widget_f, _transform_f, names='v_model')
        widgets.Widget.observe(_widget_vmin, _update_colorbar_range, names='value')
        widgets.Widget.observe(_widget_vmax, _update_colorbar_range, names='value')

        figure_widget = PlotTemplatePlotly(components={'main-widget': widgets.VBox(children=[fig, _widget_progress_output]),
                                                       'control-widget': control_widget,
                                                       'output-widget': _widget_output
                                                       })

        return figure_widget

    def _arg_len_check(self, num_traces, **kwargs):
        """Check if list arguments have the expected number of elements.
        If the arguments are not of type list, convert them to a list with a single element
        """
        result = []
        for kw, value in kwargs.items():
            if isinstance(value, list) is False:
                result.append([value] * num_traces)
            else:
                assert len(value) == num_traces, '%s must have the same length as x, or have an appropriate value.' % (kw)
                result.append(value)
        return result

    def _grid(self, expr, what=None, shape=64, limits=None, f='identity', n=None, selection=None, progress=None):

        import re

        f = _parse_f(f)
        n = _parse_n(n)

        # if type(shape) == int:
        #     shape = (shape,)
        binby = []
        expr = _ensure_strings_from_expressions(expr)
        expr = _ensure_list(expr)
        for expression in expr:
            if expression is not None:
                binby = [expression] + binby
        limits = self.df.limits(binby, limits)

        if type(shape) == int:
            shape = [shape] * len(expr)

        if isinstance(what, (vaex.stat.Expression)):
            grid = what.calculate(self.df, binby=binby, limits=limits, shape=shape, selection=selection)
        else:
            what = what.strip()
            groups = re.match("(.*)\\((.*)\\)", what).groups()
            if groups and len(groups) == 2:
                function = groups[0]
                arguments = groups[1].strip()
                functions = ["mean", "sum", "std", "count"]
                if function in functions:
                    grid = getattr(vaex.stat, function)(arguments).calculate(self.df, binby=binby, limits=limits,
                                                                             shape=shape, selection=selection, progress=progress)
                elif function == "count" and arguments == "*":
                    grid = self.df.count(binby=binby, shape=shape, limits=limits, selection=selection, progress=progress, edges=True)
                elif function == "cumulative" and arguments == "*":
                    grid = self.df.count(binby=binby, shape=shape, limits=limits, selection=selection, progress=progress)
                    grid = np.cumsum(grid)
                else:
                    raise ValueError("Could not understand method: %s, expected one of %r'" % (function, functions))
            else:
                raise ValueError("Could not understand 'what' argument %r, expected something in form: 'count(*)', 'mean(x)'" % what)

        # Transformations and normalisaions
        fgrid = f(grid)
        if n is not None:
            ngrid = fgrid / fgrid.sum()
        else:
            ngrid = fgrid
        if len(expr) == 1:
            limits = np.array(limits)
            xmin = limits.min()
            xmax = limits.max()
            N = len(grid)
            extent = np.arange(N + 1) / (N - 0.) * (xmax - xmin) + xmin
        elif len(expr) == 2:
            extent = np.array(limits[::-1]).flatten()
        # The y axis values
        counts = np.concatenate([ngrid[0:1], ngrid])
        # Done!
        return extent, counts


    def _selection_clear(self, change=None, *args, **kwargs):
        self.df.select_nothing()

    def _selection_undo(self, change=None, *args, **kwargs):
        if self.df.selection_can_undo():
            self.df.selection_undo()

    def _selection_redo(self, change=None, *args, **kwargs):
        if self.df.selection_can_redo():
            self.df.selection_redo()
