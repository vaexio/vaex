
class DataFrameAccessorViz(object):
    def __init__(self, df):
        self.df = df

# this will add methods to the above class
from . import mpl


class ExpressionAccessorViz:
    def __init__(self, expression):
        self.expression = expression
        self.df = self.expression.df

    def histogram(self, what="count(*)", grid=None, shape=64, facet=None, limits=None, figsize=None, f="identity", n=None, normalize_axis=None,
                    xlabel=None, ylabel=None, label=None,
                    selection=None, show=False, tight_layout=True, hardcopy=None,
                    progress=None,
                    **kwargs):
            """ Plot a histogram of the expression. This is a convenience method for `df.histogram(...)`

            Example:

            >>> df.x.histogram()
            >>> df.x.histogram(limits=[0, 100], shape=100)
            >>> df.x.histogram(what='mean(y)', limits=[0, 100], shape=100)

            If you want to do a computation yourself, pass the grid argument, but you are responsible for passing the
            same limits arguments:

            >>> counts = df.mean(df.y, binby=df.x, limits=[0, 100], shape=100)/100.
            >>> df.plot1d(df.x, limits=[0, 100], shape=100, grid=means, label='mean(y)/100')

            :param x: Expression to bin in the x direction
            :param what: What to plot, count(*) will show a N-d histogram, mean('x'), the mean of the x column, sum('x') the sum
            :param grid: If the binning is done before by yourself, you can pass it
            :param facet: Expression to produce facetted plots ( facet='x:0,1,12' will produce 12 plots with x in a range between 0 and 1)
            :param limits: list of [xmin, xmax], or a description such as 'minmax', '99%'
            :param figsize: (x, y) tuple passed to plt.figure for setting the figure size
            :param f: transform values by: 'identity' does nothing 'log' or 'log10' will show the log of the value
            :param n: normalization function, currently only 'normalize' is supported, or None for no normalization
            :param normalize_axis: which axes to normalize on, None means normalize by the global maximum.
            :param normalize_axis:
            :param xlabel: String for label on x axis (may contain latex)
            :param ylabel: Same for y axis
            :param: tight_layout: call plt.tight_layout or not
            :param kwargs: extra argument passed to plt.plot
            :return:
            """
            return self.df.viz.histogram(self.expression, what=what, grid=grid, shape=shape, facet=facet, limits=limits, figsize=figsize, f=f, n=n, normalize_axis=normalize_axis,
                                         xlabel=xlabel, ylabel=ylabel, label=label, selection=selection, show=show, tight_layout=tight_layout, hardcopy=hardcopy,
                                         progress=progress, **kwargs)
