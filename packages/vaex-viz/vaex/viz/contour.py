import re
import vaex
import numpy as np
import matplotlib.pyplot as plt

def plot2d_contour(self, x=None, y=None, what="count(*)", limits=None, shape=256,
                   selection=None, f="identity", figsize=None,
                   xlabel=None, ylabel=None,
                   aspect="auto", levels=None, fill=False,
                   colorbar=False, colorbar_label=None,
                   colormap=None, colors=None, linewidths=None, linestyles=None,
                   vmin=None, vmax=None,
                   grid=None, show=None, **kwargs):
    """
    Plot conting contours on 2D grid.

    :param x: {expression}
    :param y: {expression}
    :param what: What to plot, count(*) will show a N-d histogram, mean('x'), the mean of the x column, sum('x') the sum, std('x') the standard deviation, correlation('vx', 'vy') the correlation coefficient. Can also be a list of values, like ['count(x)', std('vx')], (by default maps to column)
    :param limits: {limits}
    :param shape: {shape}
    :param selection: {selection}
    :param f: transform values by: 'identity' does nothing 'log' or 'log10' will show the log of the value
    :param figsize: (x, y) tuple passed to plt.figure for setting the figure size
    :param xlabel: label of the x-axis (defaults to param x)
    :param ylabel: label of the y-axis (defaults to param y)
    :param aspect: the aspect ratio of the figure
    :param levels: the contour levels to be passed on plt.contour or plt.contourf
    :param colorbar: plot a colorbar or not
    :param colorbar_label: the label of the colourbar (defaults to param what)
    :param colormap: matplotlib colormap to pass on to plt.contour or plt.contourf
    :param colors: the colours of the contours
    :param linewidths: the widths of the contours
    :param linestyles: the style of the contour lines
    :param vmin: instead of automatic normalization, scale the data between vmin and vmax
    :param vmax: see vmin
    :param grid: {grid}
    :param show:
    """


    # Get the function out of the string
    f = vaex.dataset._parse_f(f)

    # Internals on what to bin
    binby = []
    x = vaex.dataset._ensure_strings_from_expressions(x)
    y = vaex.dataset._ensure_strings_from_expressions(y)
    for expression in [y, x]:
        if expression is not None:
            binby = [expression] + binby

    # The shape
    shape = vaex.dataset._expand_shape(shape, 2)

    # The limits and
    limits = self.limits(binby, limits)

    # Constructing the 2d histogram
    if grid is None:
        if what:
            if isinstance(what, (vaex.stat.Expression)):
                grid = what.calculate(self, binby=binby, limits=limits, shape=shape, selection=selection)
            else:
                what = what.strip()
                index = what.index("(")
                groups = re.match("(.*)\((.*)\)", what).groups()
                if groups and len(groups) == 2:
                    function = groups[0]
                    arguments = groups[1].strip()
                    functions = ["mean", "sum", "std", "count"]
                    if function in functions:
                        # grid = getattr(self, function)(arguments, binby, limits=limits, shape=shape, selection=selection)
                        grid = getattr(vaex.stat, function)(arguments).calculate(self, binby=binby, limits=limits, shape=shape, selection=selection)
                    elif function == "count" and arguments == "*":
                        grid = self.count(binby=binby, shape=shape, limits=limits, selection=selection)
                    elif function == "cumulative" and arguments == "*":
                        # TODO: comulative should also include the tails outside limits
                        grid = self.count(binby=binby, shape=shape, limits=limits, selection=selection)
                        grid = np.cumsum(grid)
                    else:
                        raise ValueError("Could not understand method: %s, expected one of %r'" % (function, functions))
                else:
                    raise ValueError("Could not understand 'what' argument %r, expected something in form: 'count(*)', 'mean(x)'" % what)
        else:
            grid = self.histogram(binby, size=shape, limits=limits, selection=selection)

    # Apply the function on the grid
    fgrid = f(grid)

    # Figure creation
    if figsize is not None:
        fig = plt.figure(num=None, figsize=figsize, dpi=80, facecolor='w', edgecolor='k')
    fig = plt.gcf()

    # labels
    plt.xlabel(xlabel or x)
    plt.ylabel(ylabel or y)

    # The master contour plot
    if fill == False:
        value = plt.contour(fgrid.T, origin="lower", extent=np.array(limits).ravel().tolist(),
                            linestyles=linestyles, linewidths=linewidths, levels=levels,
                            colors=colors, cmap=colormap, vmin=vmin, vmax=vmax, **kwargs)
    else:
        value = plt.contourf(fgrid.T, origin="lower", extent=np.array(limits).ravel().tolist(),
                             linestyles=linestyles, levels=levels, colors=colors,
                             cmap=colormap, vmin=vmin, vmax=vmax, **kwargs)
    if colorbar:
        plt.colorbar(label=colorbar_label or what)

    # Wrap things up
    if show:
        plt.show()
    return value
