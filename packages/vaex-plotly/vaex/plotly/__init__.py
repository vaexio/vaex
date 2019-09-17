import vaex
from vaex.utils import _ensure_strings_from_expressions, _parse_f, _parse_n

import numpy as np


class DataFrameAccessorPlotly(object):
    def __init__(self, df):
        self.df = df

    def scatter(self, x, y, xerr=None, yerr=None, size=None, color=None, symbol=None,
                label=None, xlabel=None, ylabel=None,
                selection=None, length_limit=50_000, length_check=True, colorbar_label=None,
                tooltip_title=None, tooltip_data=None):
        """Scatter plot using plotly.
        Convenience wrapper around plotly.graph_objs.Scatter when for working with small DataFrames or selections.
        """

        # import plotly.graph_objs as go

        # x = _ensure_strings_from_expressions(x)
        # y = _ensure_strings_from_expressions(y)
        # this should be done later, depending on number of datasets
        # label = str(label or selection)
        # selection = _ensure_strings_from_expressions(selection)

        # if length_check:
        # count = self.df.count(selection=selection)
        # if count > length_limit:
        #     raise ValueError("the number of rows (%d) is above the limit (%d), pass length_check=False, or increase length_limit" % (count, length_limit))

        # x_values = self.df.evaluate(x, selection=selection)
        # y_values = self.df.evaluate(y, selection=selection)

        # if isinstance(color, vaex.expression.Expression):
        #     color = self.df.evaluate(color, selection=selection)
        # if isinstance(size, vaex.expression.Expression):
        #     size = self.df.evaluate(size, selection=selection)
        # if isinstance()





    def histogram(self, x, what='count(*)', grid=None, shape=64, limits=None, f='identity', n=None,
                  lw=None, ls=None, color=None,
                  xlabel=None, ylabel=None, label=None, selection=None, progress=None):
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
        :param xlabel: String for label on x axis
        :param ylabel: Same for y axis
        :param label: labels or names for the data being plotted
        :param selection: Name of selection to use (or True for the 'default'), or a selection-like expresson
        :param bool progress: If True, display a progress bar of the binning process
        :return plotly.graph_objs._figurewidget.FigureWidget fig: a plotly FigureWidget
        """

        import plotly.graph_objs as go
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

            xar, counts = self._grid1d(x=x[i], what=what, shape=shape[i], limits=limits,
                                       f=f, n=n, selection=selection[i], progress=progress)

            line = go.scatter.Line(color=color[i], width=lw[i], dash=ls[i])
            traces.append(go.Scatter(x=xar, y=counts, mode='lines', line_shape='hv', line=line, name=label[i]))

        layout = go.Layout(xaxis=go.layout.XAxis(title=xlabel or x[0]),
                           yaxis=go.layout.YAxis(title=ylabel or what))
        fig = go.FigureWidget(data=traces, layout=layout)

        return fig

    def _arg_len_check(self, num_traces, **kwargs):
        result = []
        for kw, value in kwargs.items():
            if isinstance(value, list) is False:
                result.append([value] * num_traces)
            else:
                assert len(value) == num_traces, '%s must have the same length as x, or have an appropriate value.' % (kw)
                result.append(value)
        return result

    def _grid1d(self, x, what=None, shape=64, limits=None, f='identity', n=None, selection=None, progress=None):

        import re

        f = _parse_f(f)
        n = _parse_n(n)

        if type(shape) == int:
            shape = (shape,)
        binby = []
        x = _ensure_strings_from_expressions(x)
        for expression in [x]:
            if expression is not None:
                binby = [expression] + binby
        limits = self.df.limits(binby, limits)

        if what:
            if isinstance(what, (vaex.stat.Expression)):
                grid = what.calculate(self.df, binby=binby, limits=limits, shape=shape, selection=selection)
            else:
                what = what.strip()
                index = what.index("(")
                groups = re.match("(.*)\\((.*)\\)", what).groups()
                if groups and len(groups) == 2:
                    function = groups[0]
                    arguments = groups[1].strip()
                    functions = ["mean", "sum", "std", "count"]
                    if function in functions:
                        grid = getattr(vaex.stat, function)(arguments).calculate(self.df, binby=binby, limits=limits,
                                                                                 shape=shape, selection=selection, progress=progress)
                    elif function == "count" and arguments == "*":
                        grid = self.df.count(binby=binby, shape=shape, limits=limits, selection=selection, progress=progress)
                    elif function == "cumulative" and arguments == "*":
                        grid = self.df.count(binby=binby, shape=shape, limits=limits, selection=selection, progress=progress)
                        grid = np.cumsum(grid)
                    else:
                        raise ValueError("Could not understand method: %s, expected one of %r'" % (function, functions))
                else:
                    raise ValueError("Could not understand 'what' argument %r, expected something in form: 'count(*)', 'mean(x)'" % what)
        else:
            grid = self.df.histogram(binby, size=shape, limits=limits, selection=selection)

        # Transformations and normalisaions
        fgrid = f(grid)
        if n is not None:
            ngrid = fgrid / fgrid.sum()
        else:
            ngrid = fgrid
        # The x-axis values
        xmin, xmax = limits[-1]
        N = len(grid)
        xar = np.arange(N + 1) / (N - 0.) * (xmax - xmin) + xmin
        # The y axis values
        counts = np.concatenate([ngrid[0:1], ngrid])
        # Done!
        return xar, counts
