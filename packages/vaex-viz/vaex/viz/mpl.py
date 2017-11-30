import ast
import numpy as np
import logging
from vaex.dataset import Dataset, _parse_n, _parse_f, _ensure_string_from_expression, \
    _ensure_strings_from_expressions, _ensure_list,\
    _expand_limits, _expand_shape, _expand, _parse_reduction
import vaex.utils
import vaex.image

logger = logging.getLogger("vaex.viz")


def add_plugin():
    pass  # importing this module already does the job


def patch(f):
    '''Adds method f to the Dataset class'''
    name = f.__name__
    setattr(Dataset, name, f)
    return f

from .vector import plot2d_vector
patch(plot2d_vector)

from .tensor import plot2d_tensor
patch(plot2d_tensor)

@patch
def plot1d(self, x=None, what="count(*)", grid=None, shape=64, facet=None, limits=None, figsize=None, f="identity", n=None, normalize_axis=None,
    xlabel=None, ylabel=None, label=None,
    selection=None, show=False, tight_layout=True, hardcopy=None,
           **kwargs):
    """

    :param x: Expression to bin in the x direction
    :param what: What to plot, count(*) will show a N-d histogram, mean('x'), the mean of the x column, sum('x') the sum
    :param grid:
    :param grid: if the binning is done before by yourself, you can pass it
    :param facet: Expression to produce facetted plots ( facet='x:0,1,12' will produce 12 plots with x in a range between 0 and 1)
    :param limits: list of [xmin, xmax], or a description such as 'minmax', '99%'
    :param figsize: (x, y) tuple passed to pylab.figure for setting the figure size
    :param f: transform values by: 'identity' does nothing 'log' or 'log10' will show the log of the value
    :param n: normalization function, currently only 'normalize' is supported, or None for no normalization
    :param normalize_axis: which axes to normalize on, None means normalize by the global maximum.
    :param normalize_axis:
    :param xlabel: String for label on x axis (may contain latex)
    :param ylabel: Same for y axis
    :param: tight_layout: call pylab.tight_layout or not
    :param kwargs: extra argument passed to pylab.plot
    :return:
    """



    import pylab
    f = _parse_f(f)
    n = _parse_n(n)
    if type(shape) == int:
        shape = (shape,)
    binby = []
    x = _ensure_strings_from_expressions(x)
    for expression in [x]:
        if expression is not None:
            binby = [expression] + binby
    limits = self.limits(binby, limits)
    if figsize is not None:
        pylab.figure(num=None, figsize=figsize, dpi=80, facecolor='w', edgecolor='k')
    fig = pylab.gcf()
    import re
    if facet is not None:
        match = re.match("(.*):(.*),(.*),(.*)", facet)
        if match:
            groups = match.groups()
            import ast
            facet_expression = groups[0]
            facet_limits = [ast.literal_eval(groups[1]), ast.literal_eval(groups[2])]
            facet_count = ast.literal_eval(groups[3])
            limits.append(facet_limits)
            binby.append(facet_expression)
            shape = (facet_count,) + shape
        else:
            raise ValueError("Could not understand 'facet' argument %r, expected something in form: 'column:-1,10:5'" % facet)

    if grid is None:
        if what:
            if isinstance(what, (vaex.stat.Expression)):
                grid = what.calculate(self, binby=binby, limits=limits, shape=shape, selection=selection)
            else:
                what = what.strip()
                index = what.index("(")
                import re
                groups = re.match("(.*)\((.*)\)", what).groups()
                if groups and len(groups) == 2:
                    function = groups[0]
                    arguments = groups[1].strip()
                    functions = ["mean", "sum", "std", "count"]
                    if function in functions:
                        #grid = getattr(self, function)(arguments, binby, limits=limits, shape=shape, selection=selection)
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
    fgrid = f(grid)
    if n is not None:
        #ngrid = n(fgrid, axis=normalize_axis)
        ngrid = fgrid / fgrid.sum()
    else:
        ngrid = fgrid
        #reductions = [_parse_reduction(r, colormap, colors) for r in reduce]
        #rgrid = ngrid * 1.
        #for r in reduce:
        #   r = _parse_reduction(r, colormap, colors)
        #   rgrid = r(rgrid)
        #grid = self.reduce(grid, )
    xmin, xmax = limits[-1]
    if facet:
        N = len(grid[-1])
    else:
        N = len(grid)
    xar = np.arange(N+1) / (N-0.) * (xmax-xmin) + xmin
    if facet:
        import math
        rows, columns = int(math.ceil(facet_count / 4.)), 4
        values = np.linspace(facet_limits[0], facet_limits[1], facet_count+1)
        for i in range(facet_count):
            ax = pylab.subplot(rows, columns, i+1)
            value = ax.plot(xar, ngrid[i], drawstyle="steps-mid", label=label or x, **kwargs)
            v1, v2 = values[i], values[i+1]
            pylab.xlabel(xlabel or x)
            pylab.ylabel(ylabel or what)
            ax.set_title("%3f <= %s < %3f" % (v1, facet_expression, v2))
            #pylab.show()
    else:
        #im = pylab.imshow(rgrid, extent=np.array(limits[:2]).flatten(), origin="lower", aspect=aspect)
        pylab.xlabel(xlabel or self.label(x))
        pylab.ylabel(ylabel or what)
        #print(xar, ngrid)
        # repeat the first element, that's how plot/steps likes it..
        g = np.concatenate([ngrid[0:1], ngrid])
        value = pylab.plot(xar, g, drawstyle="steps-pre", label=label or x, **kwargs)
    if tight_layout:
        pylab.tight_layout()
    if hardcopy:
        pylab.savefig(hardcopy)
    if show:
        pylab.show()
    return value
    #N = len(grid)
    #xmin, xmax = limits[0]
    #return pylab.plot(np.arange(N) / (N-1.0) * (xmax-xmin) + xmin, f(grid,), drawstyle="steps", **kwargs)
    #pylab.ylim(-1, 6)

@patch
def scatter(self, x, y, xerr=None, yerr=None, s_expr=None, c_expr=None, selection=None, length_limit=50000, length_check=True, label=None, xlabel=None, ylabel=None, errorbar_kwargs={}, **kwargs):
    """Convenience wrapper around pylab.scatter when for working with small datasets or selections

    :param x: Expression for x axis
    :param y: Idem for y
    :param s_expr: When given, use if for the s (size) argument of pylab.scatter
    :param c_expr: When given, use if for the c (color) argument of pylab.scatter
    :param selection: Single selection expression, or None
    :param length_limit: maximum number of rows it will plot
    :param length_check: should we do the maximum row check or not?
    :param xlabel: label for x axis, if None .label(x) is used
    :param ylabel: label for y axis, if None .label(y) is used
    :param errorbar_kwargs: extra dict with arguments passed to plt.errorbar
    :param kwargs: extra arguments passed to pylab.scatter
    :return:
    """
    import pylab as plt
    x = _ensure_strings_from_expressions(x)
    y = _ensure_strings_from_expressions(y)
    selection = _ensure_strings_from_expressions(selection)
    if length_check:
        count = self.count(selection=selection)
        if count > length_limit:
            raise ValueError("the number of rows (%d) is above the limit (%d), pass length_check=False, or increase length_limit" % (count, length_limit))
    x_values = self.evaluate(x, selection=selection)
    y_values = self.evaluate(y, selection=selection)
    if s_expr:
        kwargs["s"] = self.evaluate(s_expr, selection=selection)
    if c_expr:
        kwargs["c"] = self.evaluate(c_expr, selection=selection)
    plt.xlabel(xlabel or self.label(x))
    plt.ylabel(ylabel or self.label(y))
    s = plt.scatter(x_values, y_values, **kwargs)
    if label:
        label_values = self.evaluate(label)
        for i, label_value in enumerate(label_values):
            plt.annotate(label_value, (x_values[i],y_values[i]))
    xerr_values = None
    yerr_values = None
    if xerr is not None:
        if _issequence(xerr):
            assert len(xerr) == 2, "if xerr is a sequence it should be of length 2"
            xerr_values = [self.evaluate(xerr[0], selection=selection), self.evaluate(xerr[1], selection=selection)]
        else:
            xerr_values = self.evaluate(xerr, selection=selection)
    if yerr is not None:
        if _issequence(yerr):
            assert len(yerr) == 2, "if yerr is a sequence it should be of length 2"
            yerr_values = [self.evaluate(yerr[0], selection=selection), self.evaluate(yerr[1], selection=selection)]
        else:
            yerr_values = self.evaluate(yerr, selection=selection)
    if xerr_values is not None or yerr_values is not None:
        plt.errorbar(x_values, y_values, yerr=yerr_values, xerr=xerr_values, **errorbar_kwargs)
    return s

#def plot(self, x=None, y=None, z=None, axes=[], row=None, agg=None, extra=["selection:none,default"], reduce=["colormap", "stack.fade"], f="log", n="normalize", naxis=None,

@patch
def plot(self, x=None, y=None, z=None, what="count(*)", vwhat=None, reduce=["colormap"], f=None,
        normalize="normalize", normalize_axis="what",
        vmin=None, vmax=None,
        shape=256, vshape=32, limits=None, grid=None, colormap="afmhot", # colors=["red", "green", "blue"],
        figsize=None, xlabel=None, ylabel=None, aspect="auto", tight_layout=True, interpolation="nearest", show=False,
        colorbar=True,
        selection=None, selection_labels=None, title=None,
        background_color="white", pre_blend=False, background_alpha=1.,
        visual=dict(x="x", y="y", layer="z", fade="selection", row="subspace", column="what"),
        smooth_pre=None, smooth_post=None,
        wrap=True, wrap_columns=4,
        return_extra=False, hardcopy=None):
    """Declarative plotting of statistical plots using matplotlib, supports subplots, selections, layers

    Instead of passing x and y, pass a list as x argument for multiple panels. Give what a list of options to have multiple
    panels. When both are present then will be origanized in a column/row order.

    This methods creates a 6 dimensional 'grid', where each dimension can map the a visual dimension.
    The grid dimensions are:

     * x: shape determined by shape, content by x argument or the first dimension of each space
     * y:   ,,
     * z:  related to the z argument
     * selection: shape equals length of selection argument
     * what: shape equals length of what argument
     * space: shape equals length of x argument if multiple values are given

     By default, this its shape is (1, 1, 1, 1, shape, shape) (where x is the last dimension)

    The visual dimensions are

     * x: x coordinate on a plot / image (default maps to grid's x)
     * y: y   ,,                         (default maps to grid's y)
     * layer: each image in this dimension is blended togeher to one image (default maps to z)
     * fade: each image is shown faded after the next image (default mapt to selection)
     * row: rows of subplots (default maps to space)
     * columns: columns of subplot (default maps to what)

    All these mappings can be changes by the visual argument, some examples:

    >>> ds.plot('x', 'y', what=['mean(x)', 'correlation(vx, vy)'])

    Will plot each 'what' as a column

    >>> ds.plot('x', 'y', selection=['FeH < -3', '(FeH >= -3) & (FeH < -2)'], visual=dict(column='selection'))

    Will plot each selection as a column, instead of a faded on top of each other.





    :param x: Expression to bin in the x direction (by default maps to x), or list of pairs, like [['x', 'y'], ['x', 'z']], if multiple pairs are given, this dimension maps to rows by default
    :param y:                          y           (by default maps to y)
    :param z: Expression to bin in the z direction, followed by a :start,end,shape  signature, like 'FeH:-3,1:5' will produce 5 layers between -10 and 10 (by default maps to layer)
    :param what: What to plot, count(*) will show a N-d histogram, mean('x'), the mean of the x column, sum('x') the sum, std('x') the standard deviation, correlation('vx', 'vy') the correlation coefficient. Can also be a list of values, like ['count(x)', std('vx')], (by default maps to column)
    :param reduce:
    :param f: transform values by: 'identity' does nothing 'log' or 'log10' will show the log of the value
    :param normalize: normalization function, currently only 'normalize' is supported
    :param normalize_axis: which axes to normalize on, None means normalize by the global maximum.
    :param vmin: instead of automatic normalization, (using normalize and normalization_axis) scale the data between vmin and vmax to [0, 1]
    :param vmax: see vmin
    :param shape: shape/size of the n-D histogram grid
    :param limits: list of [[xmin, xmax], [ymin, ymax]], or a description such as 'minmax', '99%'
    :param grid: if the binning is done before by yourself, you can pass it
    :param colormap: matplotlib colormap to use
    :param figsize: (x, y) tuple passed to pylab.figure for setting the figure size
    :param xlabel:
    :param ylabel:
    :param aspect:
    :param tight_layout: call pylab.tight_layout or not
    :param colorbar: plot a colorbar or not
    :param interpolation: interpolation for imshow, possible options are: 'nearest', 'bilinear', 'bicubic', see matplotlib for more
    :param return_extra:
    :return:
    """
    import pylab
    import matplotlib
    n = _parse_n(normalize)
    if type(shape) == int:
        shape = (shape,) * 2
    binby = []
    x = _ensure_strings_from_expressions(x)
    y = _ensure_strings_from_expressions(y)
    for expression in [y,x]:
        if expression is not None:
            binby = [expression] + binby
    fig = pylab.gcf()
    if figsize is not None:
        fig.set_size_inches(*figsize)
    import re

    what_units = None
    whats = _ensure_list(what)
    selections = _ensure_list(selection)
    selections = _ensure_strings_from_expressions(selections)

    if y is None:
        waslist, [x,] = vaex.utils.listify(x)
    else:
        waslist, [x,y] = vaex.utils.listify(x, y)
        x = list(zip(x, y))
        limits = [limits]

    # every plot has its own vwhat for now
    vwhats = _expand_limits(vwhat, len(x)) # TODO: we're abusing this function..
    logger.debug("x: %s", x)
    limits = self.limits(x, limits)
    logger.debug("limits: %r", limits)

    labels = {}
    shape = _expand_shape(shape, 2)
    vshape = _expand_shape(shape, 2)
    if z is not None:
        match = re.match("(.*):(.*),(.*),(.*)", z)
        if match:
            groups = match.groups()
            import ast
            z_expression = groups[0]
            logger.debug("found groups: %r", list(groups))
            z_limits = [ast.literal_eval(groups[1]), ast.literal_eval(groups[2])]
            z_shape = ast.literal_eval(groups[3])
            #for pair in x:
            x = [[z_expression] + list(k) for k in x]
            limits = np.array([[z_limits]  + list(k) for k in limits])
            shape =  (z_shape,)+ shape
            vshape =  (z_shape,)+ vshape
            logger.debug("x = %r", x)
            values = np.linspace(z_limits[0], z_limits[1], num=z_shape+1)
            labels["z"] = list(["%s <= %s < %s" % (v1, z_expression, v2) for v1, v2 in zip(values[:-1], values[1:])])
        else:
            raise ValueError("Could not understand 'z' argument %r, expected something in form: 'column:-1,10:5'" % facet)
    else:
        z_shape = 1


    # z == 1
    if z is None:
        total_grid = np.zeros( (len(x), len(whats), len(selections), 1) + shape, dtype=float)
        total_vgrid = np.zeros( (len(x), len(whats), len(selections), 1) + vshape, dtype=float)
    else:
        total_grid = np.zeros( (len(x), len(whats), len(selections)) + shape, dtype=float)
        total_vgrid = np.zeros( (len(x), len(whats), len(selections)) + vshape, dtype=float)
    logger.debug("shape of total grid: %r", total_grid.shape)
    axis = dict(plot=0, what=1, selection=2)
    xlimits = limits
    if xlabel is None:
        xlabels = []
        ylabels = []
        for i, (binby, limits) in enumerate(zip(x, xlimits)):
            xlabels.append(self.label(binby[0]))
            ylabels.append(self.label(binby[1]))
    else:
        xlabels = _expand(xlabel, len(x))
        ylabels = _expand(ylabel, len(x))
    labels["subspace"] = (xlabels, ylabels)

    grid_axes = dict(x=-1, y=-2, z=-3, selection=-4, what=-5, subspace=-6)
    visual_axes = dict(x=-1, y=-2, layer=-3, fade=-4, column=-5, row=-6)
    #visual_default=dict(x="x", y="y", z="layer", selection="fade", subspace="row", what="column")
    visual_default=dict(x="x", y="y", layer="z", fade="selection", row="subspace", column="what")
    invert = lambda x: dict((v, k) for k, v in x.items())
    #visual_default_reverse = invert(visual_default)
    #visual_ = visual_default
    #visual = dict(visual) # copy for modification
    # add entries to avoid mapping multiple times to the same axis
    free_visual_axes = list(visual_default.keys())
    #visual_reverse = invert(visual)
    logger.debug("1: %r %r", visual, free_visual_axes)
    for visual_name, grid_name in visual.items():
        if visual_name in free_visual_axes:
            free_visual_axes.remove(visual_name)
        else:
            raise ValueError("visual axes %s used multiple times" % visual_name)
    logger.debug("2: %r %r", visual, free_visual_axes)
    for visual_name, grid_name in visual_default.items():
        if visual_name in free_visual_axes and grid_name not in visual.values():
            free_visual_axes.remove(visual_name)
            visual[visual_name] = grid_name
    logger.debug("3: %r %r", visual, free_visual_axes)
    for visual_name, grid_name in visual_default.items():
        if visual_name not in free_visual_axes and grid_name not in visual.values():
            visual[free_visual_axes.pop(0)] = grid_name

    logger.debug("4: %r %r", visual, free_visual_axes)


    visual_reverse = invert(visual)
    # TODO: the meaning of visual and visual_reverse is changed below this line, super confusing
    visual, visual_reverse = visual_reverse, visual
    move = {}
    for grid_name, visual_name in visual.items():
        if visual_axes[visual_name] in visual.values():
            index = visual.values().find(visual_name)
            key = visual.keys()[index]
            raise ValueError("trying to map %s to %s while, it is already mapped by %s" % (grid_name, visual_name, key))
        move[grid_axes[grid_name]] = visual_axes[visual_name]

    #normalize_axis = _ensure_list(normalize_axis)

    fs = _expand(f, total_grid.shape[grid_axes[normalize_axis]])
    #assert len(vwhat)
    #labels["y"] = ylabels
    what_labels = []
    if grid is None:
        grid_of_grids = []
        for i, (binby, limits) in enumerate(zip(x, xlimits)):
            grid_of_grids.append([])
            for j, what in enumerate(whats):
                if isinstance(what, vaex.stat.Expression):
                    grid = what.calculate(self, binby=binby, shape=shape, limits=limits, selection=selections, delay=True)
                else:
                    what = what.strip()
                    index = what.index("(")
                    import re
                    groups = re.match("(.*)\((.*)\)", what).groups()
                    if groups and len(groups) == 2:
                        function = groups[0]
                        arguments = groups[1].strip()
                        if "," in arguments:
                            arguments = arguments.split(",")
                        functions = ["mean", "sum", "std", "var", "correlation", "covar", "min", "max", "median_approx"]
                        unit_expression = None
                        if function in ["mean", "sum", "std", "min", "max", "median"]:
                            unit_expression = arguments
                        if function in ["var"]:
                            unit_expression = "(%s) * (%s)" % (arguments, arguments)
                        if function in ["covar"]:
                            unit_expression = "(%s) * (%s)" % arguments
                        if unit_expression:
                            unit = self.unit(unit_expression)
                            if unit:
                                what_units = unit.to_string('latex_inline')
                        if function in functions:
                            grid = getattr(self, function)(arguments, binby=binby, limits=limits, shape=shape, selection=selections, delay=True)
                        elif function == "count":
                            grid = self.count(arguments, binby, shape=shape, limits=limits, selection=selections, delay=True)
                        else:
                            raise ValueError("Could not understand method: %s, expected one of %r'" % (function, functions))
                    else:
                        raise ValueError("Could not understand 'what' argument %r, expected something in form: 'count(*)', 'mean(x)'" % what)
                if i == 0:# and j == 0:
                    what_label = str(whats[j])
                    if what_units:
                        what_label += " (%s)" % what_units
                    if fs[j]:
                        what_label = fs[j] + " " + what_label
                    what_labels.append(what_label)
                grid_of_grids[-1].append(grid)
        self.executor.execute()
        for i, (binby, limits) in enumerate(zip(x, xlimits)):
            for j, what in enumerate(whats):
                grid = grid_of_grids[i][j].get()
                total_grid[i,j,:,:] = grid[:,None,...]
        labels["what"] = what_labels
    else:
        dims_left = 6-len(grid.shape)
        total_grid = np.broadcast_to(grid, (1,) * dims_left + grid.shape)

    #           visual=dict(x="x", y="y", selection="fade", subspace="facet1", what="facet2",)
    def _selection_name(name):
        if name in [None, False]:
            return "selection: all"
        elif name in ["default", True]:
            return "selection: default"
        else:
            return "selection: %s" % name
    if selection_labels is None:
        labels["selection"] = list([_selection_name(k) for k in selections])
    else:
        labels["selection"] = selection_labels

    #visual_grid = np.moveaxis(total_grid, move.keys(), move.values())
    # np.moveaxis is in np 1.11 only?, use transpose
    axes = [None] * len(move)
    for key, value in move.items():
        axes[value] = key
    visual_grid = np.transpose(total_grid, axes)

    logger.debug("grid shape: %r", total_grid.shape)
    logger.debug("visual: %r", visual.items())
    logger.debug("move: %r", move)
    logger.debug("visual grid shape: %r", visual_grid.shape)
    #grid = total_grid
    #print(grid.shape)
    #grid = self.reduce(grid, )
    axes = []
    #cax = pylab.subplot(1,1,1)

    background_color = np.array(matplotlib.colors.colorConverter.to_rgb(background_color))


    #if grid.shape[axis["selection"]] > 1:#  and not facet:
    #   rgrid = vaex.image.fade(rgrid)
    #   finite_mask = np.any(finite_mask, axis=0) # do we really need this
    #   print(rgrid.shape)
    #facet_row_axis = axis["what"]
    import math
    facet_columns = None
    facets = visual_grid.shape[visual_axes["row"]] * visual_grid.shape[visual_axes["column"]]
    if visual_grid.shape[visual_axes["column"]] ==  1 and wrap:
        facet_columns = min(wrap_columns, visual_grid.shape[visual_axes["row"]])
        wrapped = True
    elif visual_grid.shape[visual_axes["row"]] ==  1 and wrap:
        facet_columns = min(wrap_columns, visual_grid.shape[visual_axes["column"]])
        wrapped = True
    else:
        wrapped = False
        facet_columns = visual_grid.shape[visual_axes["column"]]
    facet_rows = int(math.ceil(facets/facet_columns))
    logger.debug("facet_rows: %r", facet_rows)
    logger.debug("facet_columns: %r", facet_columns)
        #if visual_grid.shape[visual_axes["row"]] > 1: # and not wrap:
        #   #facet_row_axis = axis["what"]
        #   facet_columns = visual_grid.shape[visual_axes["column"]]
        #else:
        #   facet_columns = min(wrap_columns, facets)
    #if grid.shape[axis["plot"]] > 1:#  and not facet:

    # this loop could be done using axis arguments everywhere
    #assert len(normalize_axis) == 1, "currently only 1 normalization axis supported"
    grid = visual_grid * 1.
    fgrid = visual_grid * 1.
    ngrid = visual_grid * 1.
    #colorgrid = np.zeros(ngrid.shape + (4,), float)
    #print "norma", normalize_axis, visual_grid.shape[visual_axes[visual[normalize_axis]]]
    vmins = _expand(vmin, visual_grid.shape[visual_axes[visual[normalize_axis]]], type=list)
    vmaxs = _expand(vmax, visual_grid.shape[visual_axes[visual[normalize_axis]]], type=list)
    #for name in normalize_axis:
    visual_grid
    if smooth_pre:
        grid = vaex.grids.gf(grid, smooth_pre)
    if 1:
        axis = visual_axes[visual[normalize_axis]]
        for i in range(visual_grid.shape[axis]):
            item = [slice(None, None, None), ] * len(visual_grid.shape)
            item[axis] = i
            item = tuple(item)
            f = _parse_f(fs[i])
            with np.errstate(divide='ignore', invalid='ignore'): # these are fine, we are ok with nan's in vaex
                fgrid.__setitem__(item, f(grid.__getitem__(item)))
            #print vmins[i], vmaxs[i]
            if vmins[i] is not None and vmaxs[i] is not None:
                nsubgrid = fgrid.__getitem__(item) * 1
                nsubgrid -= vmins[i]
                nsubgrid /= (vmaxs[i]-vmins[i])
                nsubgrid = np.clip(nsubgrid, 0, 1)
            else:
                nsubgrid, vmin, vmax = n(fgrid.__getitem__(item))
                vmins[i] = vmin
                vmaxs[i] = vmax
            #print "    ", vmins[i], vmaxs[i]
            ngrid.__setitem__(item, nsubgrid)

    if 0: # TODO: above should be like the code below, with custom vmin and vmax
        grid = visual_grid[i]
        f = _parse_f(fs[i])
        fgrid = f(grid)
        finite_mask = np.isfinite(grid)
        finite_mask = np.any(finite_mask, axis=0)
        if vmin is not None and vmax is not None:
            ngrid = fgrid * 1
            ngrid -= vmin
            ngrid /= (vmax-vmin)
            ngrid = np.clip(ngrid, 0, 1)
        else:
            ngrid, vmin, vmax = n(fgrid)
            #vmin, vmax = np.nanmin(fgrid), np.nanmax(fgrid)
    # every 'what', should have its own colorbar, check if what corresponds to
    # rows or columns in facets, if so, do a colorbar per row or per column


    rows, columns = int(math.ceil(facets / float(facet_columns))), facet_columns
    colorbar_location = "individual"
    if visual["what"] == "row" and visual_grid.shape[1] == facet_columns:
        colorbar_location = "per_row"
    if visual["what"] == "column" and visual_grid.shape[0] == facet_rows:
        colorbar_location = "per_column"
    #values = np.linspace(facet_limits[0], facet_limits[1], facet_count+1)
    logger.debug("rows: %r, columns: %r", rows, columns)
    import matplotlib.gridspec as gridspec
    column_scale = 1
    row_scale = 1
    row_offset = 0
    if facets > 1:
        if colorbar_location == "per_row":
            column_scale = 4
            gs = gridspec.GridSpec(rows, columns*column_scale+1)
        elif colorbar_location == "per_column":
            row_offset = 1
            row_scale = 4
            gs = gridspec.GridSpec(rows*row_scale+1, columns)
        else:
            gs = gridspec.GridSpec(rows, columns)
    facet_index = 0
    fs = _expand(f, len(whats))
    colormaps = _expand(colormap, len(whats))

    # row
    for i in range(visual_grid.shape[0]):
        # column
        for j in range(visual_grid.shape[1]):
            if colorbar and colorbar_location == "per_column" and i == 0:
                norm = matplotlib.colors.Normalize(vmins[j], vmaxs[j])
                sm = matplotlib.cm.ScalarMappable(norm, colormaps[j])
                sm.set_array(1) # make matplotlib happy (strange behavious)
                if facets > 1:
                    ax = pylab.subplot(gs[0, j])
                    colorbar = fig.colorbar(sm, cax=ax, orientation="horizontal")
                else:
                    colorbar = fig.colorbar(sm)
                if "what" in labels:
                    label = labels["what"][j]
                    if facets > 1:
                        colorbar.ax.set_title(label)
                    else:
                        colorbar.ax.set_ylabel(label)

            if colorbar and colorbar_location == "per_row" and j == 0:
                norm = matplotlib.colors.Normalize(vmins[i], vmaxs[i])
                sm = matplotlib.cm.ScalarMappable(norm, colormaps[i])
                sm.set_array(1) # make matplotlib happy (strange behavious)
                if facets > 1:
                    ax = pylab.subplot(gs[i, -1])
                    colorbar = fig.colorbar(sm, cax=ax)
                else:
                    colorbar = fig.colorbar(sm)
                label = labels["what"][i]
                colorbar.ax.set_ylabel(label)

            rgrid = ngrid[i,j] * 1.
            #print rgrid.shape
            for k in range(rgrid.shape[0]):
                for l in range(rgrid.shape[0]):
                    if smooth_post is not None:
                        rgrid[k,l] = vaex.grids.gf(rgrid, smooth_post)
            if visual["what"] == "column":
                what_index = j
            elif visual["what"] == "row":
                what_index = i
            else:
                what_index = 0

            if visual[normalize_axis] == "column":
                normalize_index = j
            elif visual[normalize_axis] == "row":
                normalize_index = i
            else:
                normalize_index = 0
            for r in reduce:
                r = _parse_reduction(r, colormaps[what_index], [])
                rgrid = r(rgrid)


            row = facet_index // facet_columns
            column = facet_index % facet_columns

            if colorbar and colorbar_location == "individual":
                #visual_grid.shape[visual_axes[visual[normalize_axis]]]
                norm = matplotlib.colors.Normalize(vmins[normalize_index], vmaxs[normalize_index])
                sm = matplotlib.cm.ScalarMappable(norm, colormaps[what_index])
                sm.set_array(1) # make matplotlib happy (strange behavious)
                if facets > 1:
                    ax = pylab.subplot(gs[row, column])
                    colorbar = fig.colorbar(sm, ax=ax)
                else:
                    colorbar = fig.colorbar(sm)
                label = labels["what"][what_index]
                colorbar.ax.set_ylabel(label)


            if facets > 1:
                ax = pylab.subplot(gs[row_offset + row * row_scale:row_offset + (row+1) * row_scale, column*column_scale:(column+1)*column_scale])
            else:
                ax = pylab.gca()
            axes.append(ax)
            logger.debug("rgrid: %r", rgrid.shape)
            plot_rgrid = rgrid
            assert plot_rgrid.shape[1] == 1, "no layers supported yet"
            plot_rgrid = plot_rgrid[:,0]
            if plot_rgrid.shape[0] > 1:
                plot_rgrid = vaex.image.fade(plot_rgrid[::-1])
            else:
                plot_rgrid = plot_rgrid[0]
            extend = None
            if visual["subspace"] == "row":
                subplot_index = i
            elif visual["subspace"] == "column":
                subplot_index = j
            else:
                subplot_index = 0
            extend = np.array(xlimits[subplot_index][-2:]).flatten()
            #   extend = np.array(xlimits[i]).flatten()
            logger.debug("plot rgrid: %r", plot_rgrid.shape)
            plot_rgrid = np.transpose(plot_rgrid, (1,0,2))
            im = ax.imshow(plot_rgrid, extent=extend.tolist(), origin="lower", aspect=aspect, interpolation=interpolation)
            #v1, v2 = values[i], values[i+1]
            def label(index, label, expression):
                if label and _issequence(label):
                    return label[i]
                else:
                    return self.label(expression)
            # we don't need titles when we have a colorbar
            if (visual_reverse["row"] != "what") or not colorbar:
                labelsxy = labels.get(visual_reverse["row"])
                has_title = False
                if isinstance(labelsxy, tuple):
                    labelsx, labelsy = labelsxy
                    pylab.xlabel(labelsx[i])
                    pylab.ylabel(labelsy[i])
                elif labelsxy is not None:
                    ax.set_title(labelsxy[i])
                    has_title = True
                #print visual_reverse["row"], visual_reverse["column"], labels.get(visual_reverse["row"]), labels.get(visual_reverse["column"])
            if (visual_reverse["column"] != "what")  or not colorbar:
                labelsxy = labels.get(visual_reverse["column"])
                if isinstance(labelsxy, tuple):
                    labelsx, labelsy = labelsxy
                    pylab.xlabel(labelsx[j])
                    pylab.ylabel(labelsy[j])
                elif labelsxy is not None and not has_title:
                    ax.set_title(labelsxy[j])
                    pass
            facet_index += 1
    if title:
        fig.suptitle(title, fontsize="x-large")
    if tight_layout:
        if title:
            pylab.tight_layout(rect=[0, 0.03, 1, 0.95])
        else:
            pylab.tight_layout()
    if hardcopy:
        pylab.savefig(hardcopy)
    if show:
        pylab.show()
    if return_extra:
        return im, grid, fgrid, ngrid, rgrid, rgba8
    else:
        return im
    #colorbar = None
    #return im, colorbar
