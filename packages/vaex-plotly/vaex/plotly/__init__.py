import vaex
from vaex.utils import _ensure_strings_from_expressions, _parse_f, _parse_n

import numpy as np


class DataFrameAccessorPlotly(object):
    def __init__(self, df):
        self.df = df

    def histogram(self, x, what='count(*)', grid=None, shape=64, limits=None, f='identity', n=None,
                  lw=None, ls=None, color=None,
                  xlabel=None, ylabel=None, label=None, selection=None, progress=None):
        """Create a histogram.
        """

        import plotly.graph_objs as go

        x = _ensure_strings_from_expressions(x)

        # If there is a list of expressions
        if isinstance(x, list):

            num_traces = len(x)

            if isinstance(shape, list) is False:
                shape = [shape] * num_traces
            else:
                assert len(shape) == num_traces, 'shape arg must have the same legth as `x` or be an int.'
            if isinstance(color, list) is False:
                color = [color] * num_traces
            if isinstance(lw, list) is False:
                lw = [lw] * num_traces
            if isinstance(ls, list) is False:
                ls = [ls] * num_traces
            if isinstance(label, list) is False:
                label = [label] * num_traces

            traces = []
            for i in range(num_traces):

                xar, counts = self._grid1d(x=x[i], what=what, shape=shape[i], limits=limits,
                                           f=f, n=n, selection=selection, progress=progress)

                line = go.scatter.Line(color=color[i], width=lw[i], dash=ls[i])
                traces.append(go.Scatter(x=xar, y=counts, mode='lines', line_shape='hv', line=line, name=label[i]))

            layout = go.Layout(xaxis=go.layout.XAxis(title=xlabel or x[0]),
                               yaxis=go.layout.YAxis(title=ylabel or what))
            fig = go.Figure(data=traces, layout=layout)

        # The standard case
        else:
            xar, counts = self._grid1d(x=x, what=what, shape=shape, limits=limits,
                                       f=f, n=n, selection=selection, progress=progress)

            line = go.scatter.Line(color=color, width=lw, dash=ls)
            trace = go.Scatter(x=xar, y=counts, mode='lines', line_shape='hv', line=line, name=label)
            layout = go.Layout(xaxis=go.layout.XAxis(title=xlabel or x),
                               yaxis=go.layout.YAxis(title=ylabel or what))
            fig = go.Figure(data=trace, layout=layout)

        return fig

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
