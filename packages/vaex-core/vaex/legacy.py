# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import vaex
from .tasks import Task, TaskMapReduce
from .utils import _parse_f
import six


def _asfloat(a):
    if a.dtype.type == np.float64 and a.strides[0] == 8:
        return a
    else:
        return a.astype(np.float64, copy=False)

class TaskMapReduceLegacy(TaskMapReduce):
    def __init__(self, *args, **kwargs):
        kwargs = kwargs.copy()
        kwargs['ignore_filter'] = True
        TaskMapReduce.__init__(self, *args, **kwargs)

class TaskHistogram(Task):
    def __init__(self, df, subspace, expressions, size, limits, masked=False, weight=None):
        self.size = size
        self.limits = limits
        Task.__init__(self, df, expressions, name="histogram")
        self.subspace = subspace
        self.dtype = np.float64
        self.masked = masked
        self.weight = weight
        # self.grids = vaex.grids.Grids(self.df, self.df.executor.thread_pool, *expressions)
        # self.grids.ranges = limits
        # self.grids.grids["counts"] = vaex.grids.Grid(self.grids, size, self.dimension, None)
        shape1 = (self.size,) * self.dimension
        try:
            self.size[0]
            shape1 = tuple(self.size)
        except:
            pass
        shape = (self.subspace.executor.thread_pool.nthreads,) + shape1
        self.data = np.zeros(shape, dtype=self.dtype)
        self.ranges_flat = []
        self.minima = []
        self.maxima = []
        for limit in self.limits:
            self.ranges_flat.extend(limit)
            vmin, vmax = limit
            self.minima.append(vmin)
            self.maxima.append(vmax)
        if self.weight is not None:
            self.expressions_all.append(weight)
        # print self.ranges_flat

    def __repr__(self):
        name = self.__class__.__module__ + "." + self.__class__.__name__
        return "<%s(df=%r, expressions=%r, size=%r, limits=%r)> instance at 0x%x" % (name, self.df, self.expressions, self.size, self.limits, id(self))

    def map(self, thread_index, i1, i2, *blocks):
        class Info(object):
            pass
        info = Info()
        info.i1 = i1
        info.i2 = i2
        info.first = i1 == 0
        info.last = i2 == self.df.length_unfiltered()
        info.size = i2 - i1
        # print "bin", i1, i2, info.last
        # self.grids["counts"].bin_block(info, *blocks)
        # mask = self.df.mask
        data = self.data[thread_index]

        blocks = [_asfloat(block) for block in blocks]

        if self.masked or self.df.filtered:
            mask = self.df.evaluate_selection_mask("default" if self.masked else None, i1=i1, i2=i2)
            blocks = [block[mask] for block in blocks]

        subblock_weight = None
        if len(blocks) == len(self.expressions) + 1:
            subblock_weight = blocks[-1]
            blocks = list(blocks[:-1])
        # print subblocks[0]
        # print subblocks[1]

        if self.dimension == 1:
            vaex.vaexfast.histogram1d(blocks[0], subblock_weight, data, *self.ranges_flat)
        elif self.dimension == 2:
            # if subblock_weight is None:
            # #print "speedup?"
            # histogram_numba(blocks[0], blocks[1], subblock_weight, data, *self.ranges_flat)
            # else:
            vaex.vaexfast.histogram2d(blocks[0], blocks[1], subblock_weight, data, *self.ranges_flat)
            # vaex.vaexfast.statisticNd([blocks[0], blocks[1]], subblock_weight, data, self.minima, self.maxima, 0)
        elif self.dimension == 3:
            vaex.vaexfast.histogram3d(blocks[0], blocks[1], blocks[2], subblock_weight, data, *self.ranges_flat)
        else:
            blocks = list(blocks)  # histogramNd wants blocks to be a list
            vaex.vaexfast.histogramNd(blocks, subblock_weight, data, self.minima, self.maxima)

        return i1
        # return map(self._map, blocks)#[self.map(block) for block in blocks]

    def reduce(self, results):
        for i in range(1, self.subspace.executor.thread_pool.nthreads):
            self.data[0] += self.data[i]
        return self.data[0]
        # return self.data


class SubspaceGridded(object):
    def __init__(self, subspace_bounded, grid, vx=None, vy=None, vcounts=None):
        self.subspace_bounded = subspace_bounded
        self.grid = grid
        self.vx = vx
        self.vy = vy
        self.vcounts = vcounts

    def vector(self, weightx, weighty, size=32):
        counts = self.subspace_bounded.gridded_by_histogram(size=size)
        vx = self.subspace_bounded.gridded_by_histogram(size=size, weight=weightx)
        vy = self.subspace_bounded.gridded_by_histogram(size=size, weight=weighty)
        return SubspaceGridded(self.subspace_bounded, self.grid, vx=vx, vy=vy, vcounts=counts)

    def filter_gaussian(self, sigmas=1):
        import scipy.ndimage
        return SubspaceGridded(self.subspace_bounded, scipy.ndimage.filters.gaussian_filter(self.grid, sigmas))

    def clip_relative(self, v1, v2):
        vmin = self.grid.min()
        vmax = self.grid.max()
        width = vmax - vmin
        return SubspaceGridded(self.subspace_bounded, np.clip(self.grid, vmin + v1 * width, vmin + v2 * width))

    def volr(self, **kwargs):
        import vaex.notebook
        return vaex.notebook.volr(subspace_gridded=self, **kwargs)

    def plot(self, axes=None, **kwargs):
        self.subspace_bounded.subspace.plot(np.log1p(self.grid), limits=self.subspace_bounded.bounds, axes=axes, **kwargs)

    def mean_line(self, axis=0, **kwargs):
        from matplotlib import pylab
        assert axis in [0, 1]
        other_axis = 0 if axis == 1 else 1
        xmin, xmax = self.subspace_bounded.bounds[axis]
        ymin, ymax = self.subspace_bounded.bounds[other_axis]
        x = vaex.utils.linspace_centers(xmin, xmax, self.grid.shape[axis])
        y = vaex.utils.linspace_centers(ymin, ymax, self.grid.shape[other_axis])
        print(y)
        if axis == 0:
            counts = np.sum(self.grid, axis=axis)
            means = np.sum(self.grid * y[np.newaxis, :].T, axis=axis) / counts
        else:
            counts = np.sum(self.grid, axis=axis)
            means = np.sum(self.grid * y[:, np.newaxis].T, axis=axis) / counts
        if axis == 0:
            result = pylab.plot(x, means, **kwargs)
        else:
            result = pylab.plot(means, x, **kwargs)

        self.subspace_bounded.lim()
        return result, x, means

    def _repr_png_(self):
        from matplotlib import pylab
        fig, ax = pylab.subplots()
        self.plot(axes=ax, f=np.log1p)
        import vaex.utils
        if all([k is not None for k in [self.vx, self.vy, self.vcounts]]):
            N = self.vx.grid.shape[0]
            bounds = self.subspace_bounded.bounds
            print(bounds)
            positions = [vaex.utils.linspace_centers(bounds[i][0], bounds[i][1], N) for i in range(self.subspace_bounded.subspace.dimension)]
            print(positions)
            mask = self.vcounts.grid > 0
            vx = np.zeros_like(self.vx.grid)
            vy = np.zeros_like(self.vy.grid)
            vx[mask] = self.vx.grid[mask] / self.vcounts.grid[mask]
            vy[mask] = self.vy.grid[mask] / self.vcounts.grid[mask]
            # vx = self.vx.grid / self.vcounts.grid
            # vy = self.vy.grid / self.vcounts.grid
            x2d, y2d = np.meshgrid(positions[0], positions[1])
            ax.quiver(x2d[mask], y2d[mask], vx[mask], vy[mask])
            # print x2d
            # print y2d
            # print vx
            # print vy
            # ax.quiver(x2d, y2d, vx, vy)
        ax.title.set_text(r"$\log(1+counts)$")
        ax.set_xlabel(self.subspace_bounded.subspace.expressions[0])
        ax.set_ylabel(self.subspace_bounded.subspace.expressions[1])
        # pylab.savefig
        # from .io import StringIO
        from six import StringIO
        file_object = StringIO()
        fig.canvas.print_png(file_object)
        pylab.close(fig)
        return file_object.getvalue()

    def cube_png(self, f=np.log1p, colormap="afmhot", file="cube.png"):
        if self.grid.shape != ((128,) * 3):
            logger.error("only 128**3 cubes are supported")
            return None
        colormap_name = "afmhot"
        import matplotlib.cm
        colormap = matplotlib.cm.get_cmap(colormap_name)
        mapping = matplotlib.cm.ScalarMappable(cmap=colormap)
        # pixmap = QtGui.QPixmap(32*2, 32)
        data = np.zeros((128 * 8, 128 * 16, 4), dtype=np.uint8)

        # mi, ma = 1*10**self.mod1, self.data3d.max()*10**self.mod2
        grid = f(self.grid)
        vmin, vmax = grid.min(), grid.max()
        grid_normalized = (grid - vmin) / (vmax - vmin)
        # intensity_normalized = (np.log(self.data3d + 1.) - np.log(mi)) / (np.log(ma) - np.log(mi));
        import PIL.Image
        for y2d in range(8):
            for x2d in range(16):
                zindex = x2d + y2d * 16
                I = grid_normalized[zindex]
                rgba = mapping.to_rgba(I, bytes=True)  # .reshape(Nx, 4)
                # print rgba.shape
                subdata = data[y2d * 128:(y2d + 1) * 128, x2d * 128:(x2d + 1) * 128]
                for i in range(3):
                    subdata[:, :, i] = rgba[:, :, i]
                subdata[:, :, 3] = (grid_normalized[zindex] * 255).astype(np.uint8)  # * 0 + 255
                if 0:
                    filename = "cube%03d.png" % zindex
                    img = PIL.Image.frombuffer("RGB", (128, 128), subdata[:, :, 0:3] * 1)
                    print(("saving to", filename))
                    img.save(filename)
        img = PIL.Image.frombuffer("RGBA", (128 * 16, 128 * 8), data, 'raw')  # , "RGBA", 0, -1)
        # filename = "cube.png"
        # print "saving to", file
        img.save(file, "png")

        if 0:
            filename = "colormap.png"
            print(("saving to", filename))
            height, width = self.colormap_data.shape[:2]
            img = PIL.Image.frombuffer("RGB", (width, height), self.colormap_data)
            img.save(filename)


class SubspaceBounded(object):
    def __init__(self, subspace, bounds):
        self.subspace = subspace
        self.bounds = bounds

    def histogram(self, size=256, weight=None):
        return self.subspace.histogram(limits=self.bounds, size=size, weight=weight)

    def gridded(self, size=256, weight=None):
        return self.gridded_by_histogram(size=size, weight=weight)

    def gridded_by_histogram(self, size=256, weight=None):
        grid = self.histogram(size=size, weight=weight)
        return SubspaceGridded(self, grid)

    def lim(self):
        from matplotlib import pylab
        xmin, xmax = self.bounds[0]
        ymin, ymax = self.bounds[1]
        pylab.xlim(xmin, xmax)
        pylab.ylim(ymin, ymax)


class Subspaces(object):
    """
    :type: subspaces: list[Subspace]

    """

    def __init__(self, subspaces):
        self.subspaces = subspaces
        self.expressions = set()
        first_subspace = self.subspaces[0]
        self.delay = first_subspace.delay
        self.dimension = first_subspace.dimension
        self.df = self.subspaces[0].df
        for subspace in self.subspaces:
            assert subspace.df == self.subspaces[0].df
            assert subspace.delay == self.subspaces[0].delay
            assert subspace.dimension == self.subspaces[0].dimension, "subspace is of dimension %s, while first subspace if of dimension %s" % (subspace.dimension, self.subspaces[0].dimension)
            # assert subspace.sele== self.subspaces[0].delay
            self.expressions.update(subspace.expressions)
        self.expressions = list(self.expressions)
        self.subspace = self.df(*list(self.expressions), delay=self.delay, executor=first_subspace.executor)

    # def _repr_html_(self):

    def __len__(self):
        return len(self.subspaces)

    def names(self, seperator=" "):
        return [seperator.join(subspace.expressions) for subspace in self.subspaces]

    def expressions_list(self):
        return [subspace.expressions for subspace in self.subspaces]

    def selected(self):
        return Subspaces([subspace.selected() for subspace in self.subspaces])

    def _unpack(self, values):
        value_map = dict(zip(self.expressions, values))
        return [[value_map[ex] for ex in subspace.expressions] for subspace in self.subspaces]

    def _pack(self, values):
        value_map = {}
        for subspace_values, subspace in zip(values, self.subspaces):
            for value, expression in zip(subspace_values, subspace.expressions):
                if expression in value_map:
                    if isinstance(value, np.ndarray):
                        assert np.all(value_map[expression] == value), "inconsistency in subspaces, value for expression %r is %r in one case, and %r in the other" % (expression, value, value_map[expression])
                    else:
                        assert value_map[expression] == value, "inconsistency in subspaces, value for expression %r is %r in one case, and %r in the other" % (expression, value, value_map[expression])
                else:
                    value_map[expression] = value
        return [value_map[expression] for expression in self.expressions]

    def minmax(self):
        if self.delay:
            return self.subspace.minmax().then(self._unpack)
        else:
            return self._unpack(self.subspace.minmax())

    def limits_sigma(self, sigmas=3, square=False):
        if self.delay:
            return self.subspace.limits_sigma(sigmas=sigmas, square=square).then(self._unpack)
        else:
            return self._unpack(self.subspace.limits_sigma(sigmas=sigmas, square=square))

    def mutual_information(self, limits=None, size=256):
        if limits is not None:
            limits = self._pack(limits)

        def mutual_information(limits):
            return vaex.promise.listPromise([vaex.promise.Promise.fulfilled(subspace.mutual_information(subspace_limits, size=size)) for subspace_limits, subspace in zip(limits, self.subspaces)])
            # return histograms
        if limits is None:
            limits_promise = vaex.promise.Promise.fulfilled(self.subspace.minmax())
        else:
            limits_promise = vaex.promise.Promise.fulfilled(limits)
        limits_promise = limits_promise.then(self._unpack)
        promise = limits_promise.then(mutual_information)
        return promise if self.delay else promise.get()

    def mean(self):
        if self.delay:
            return self.subspace.mean().then(self._unpack)
        else:
            means = self.subspace.mean()
            return self._unpack(means)

    def var(self, means=None):
        # 'pack' means, and check if it makes sence
        if means is not None:
            means = self._pack(means)

        def var(means):
            return self.subspace.var(means=means)
        if self.delay:
            # if means is None:
            # return self.subspace.mean().then(var).then(self._unpack)
            # else:
            return var(means).then(self._unpack)
        else:
            # if means is None:
            # means = self.subspace.mean()
            # logger.debug("means: %r", means)
            return self._unpack(var(means=means))

    def correlation(self, means=None, vars=None):
        def var(means):
            return self.subspace.var(means=means)

        def correlation(means_and_vars):
            means, vars = means_and_vars
            means, vars = self._unpack(means), self._unpack(vars)
            # return self.subspace.correlation(means=means, vars=vars)
            return vaex.promise.listPromise([subspace.correlation(means=subspace_mean, vars=subspace_var) for subspace_mean, subspace_var, subspace in zip(means, vars, self.subspaces)])
        if means is not None:
            means = self._pack(means)
        if vars is not None:
            vars = self._pack(vars)
        if self.delay:
            if means is None:
                mean_promise = self.subspace.mean()
            else:
                mean_promise = vaex.promise.Promise.fulfilled(means)
            if vars is None:
                var_promise = mean_promise.then(var)
            else:
                var_promise = vaex.promise.Promise.fulfilled(vars)
            mean_and_var_calculated = vaex.promise.listPromise(mean_promise, var_promise)
            return mean_and_var_calculated.then(correlation)
        else:
            if means is None:
                means = self.subspace.mean()
            if vars is None:
                vars = self.subspace.var(means=means)
            means = self._unpack(means)
            vars = self._unpack(vars)
            return [subspace.correlation(means=subspace_mean, vars=subspace_var) for subspace_mean, subspace_var, subspace in zip(means, vars, self.subspaces)]
            # return correlation((means, vars))

    # def bounded_by(self, limits_list):
    # return SubspacesBounded(SubspaceBounded(subspace, limits) for subspace, limit in zip(self.subspaces, limits_list))


class Subspace(object):
    """A Subspace represent a subset of columns or expressions from a df.

    subspace are not instantiated directly, but by 'calling' the df like this:

    >>> subspace_xy = some_df("x", "y")
    >>> subspace_r = some_df("sqrt(x**2+y**2)")

    See `vaex.df.Dataset` for more documentation.

    """

    def __init__(self, df, expressions, executor, delay, masked=False):
        """

        :param Dataset df: the df the subspace refers to
        :param list[str] expressions: list of expressions that forms the subspace
        :param Executor executor: responsible for executing the tasks
        :param bool delay: return answers directly, or as a promise
        :param bool masked: work on the selection or not
        :return:
        """
        self.df = df
        self.expressions = expressions
        self.executor = executor
        self.delay = delay
        self.is_masked = masked

    def __repr__(self):
        name = self.__class__.__module__ + "." + self.__class__.__name__
        return "<%s(df=%r, expressions=%r, delay=%r, is_masked=%r)> instance at 0x%x" % (name, self.df, self.expressions, self.delay, self.is_masked, id(self))

    @property
    def dimension(self):
        return len(self.expressions)

    def get_selection(self):
        return self.df.get_selection("default") if self.is_masked else None

    def is_selected(self):
        return self.is_masked

    def selected(self):
        return self.__class__(self.df, expressions=self.expressions, executor=self.executor, delay=self.delay, masked=True)

    def delayhronous(self):
        return self.__class__(self.df, expressions=self.expressions, executor=self.executor, delay=True, masked=self.is_masked)

    def image_rgba_save(self, filename, data=None, rgba8=None, **kwargs):
        if rgba8 is not None:
            data = self.image_rgba_data(rgba8=rgba8, **kwargs)
        if data is None:
            data = self.image_rgba_data(**kwargs)
        with open(filename, "wb") as f:
            f.write(data)

    def image_rgba_notebook(self, data=None, rgba8=None, **kwargs):
        if rgba8 is not None:
            data = self.image_rgba_data(rgba8=rgba8, **kwargs)
        if data is None:
            data = self.image_rgba_data(**kwargs)
        from IPython.display import display, Image
        return Image(data=data)

    def image_rgba_data(self, rgba8=None, format="png", pil_draw=False, **kwargs):
        import PIL.Image
        import PIL.ImageDraw
        from six import StringIO
        if rgba8 is None:
            rgba8 = self.image_rgba(**kwargs)
        img = PIL.Image.frombuffer("RGBA", rgba8.shape[:2], rgba8, 'raw')  # , "RGBA", 0, -1)
        if pil_draw:
            draw = PIL.ImageDraw.Draw(img)
            pil_draw(draw)

        f = StringIO()
        img.save(f, format)
        return f.getvalue()

    def image_rgba_url(self, rgba8=None, **kwargs):
        if rgba8 is None:
            rgba8 = self.image_rgba(**kwargs)
        import PIL.Image
        img = PIL.Image.frombuffer("RGBA", rgba8.shape[:2], rgba8, 'raw')  # , "RGBA", 0, -1)
        from six import StringIO
        f = StringIO()
        img.save(f, "png")
        from base64 import b64encode
        imgurl = "data:image/png;base64," + b64encode(f.getvalue()) + ""
        return imgurl

    def normalize_grid(self, grid):
        grid = grid * 1  # copy
        mask = (grid > 0) & np.isfinite(grid)
        if grid.sum():
            grid -= grid[mask].min()
            grid /= grid[mask].max()
        else:
            grid[:] = 0
        return grid

    def limits(self, value, square=False):
        """TODO: doc + server side implementation"""
        if isinstance(value, six.string_types):
            import re
            match = re.match(r"(\d*)(\D*)", value)
            if match is None:
                raise ValueError("do not understand limit specifier %r, examples are 90%, 3sigma")
            else:
                value, type = match.groups()
                import ast
                value = ast.literal_eval(value)
                type = type.strip()
                if type in ["s", "sigma"]:
                    return self.limits_sigma(value)
                elif type in ["ss", "sigmasquare"]:
                    return self.limits_sigma(value, square=True)
                elif type in ["%", "percent"]:
                    return self.limits_percentage(value)
                elif type in ["%s", "%square", "percentsquare"]:
                    return self.limits_percentage(value, square=True)
        if value is None:
            return self.limits_percentage(square=square)
        else:
            return value

    def image_rgba(self, grid=None, size=256, limits=None, square=False, center=None, weight=None, weight_stat="mean", figsize=None,
                   aspect="auto", f=lambda x: x, axes=None, xlabel=None, ylabel=None,
                   group_by=None, group_limits=None, group_colors='jet', group_labels=None, group_count=10, cmap="afmhot",
                   vmin=None, vmax=None,
                   pre_blend=False, background_color="white", background_alpha=1., normalize=True, color=None):
        f = _parse_f(f)
        if grid is None:
            limits = self.limits(limits)
            if limits is None:
                limits = self.limits_sigma()
            if group_limits is None and group_by:
                group_limits = tuple(self.df(group_by).minmax()[0]) + (group_count,)
            if weight_stat == "mean" and weight is not None:
                grid = self.bin_mean(weight, limits=limits, size=size, group_limits=group_limits, group_by=group_by)
            else:
                grid = self.histogram(limits=limits, size=size, weight=weight, group_limits=group_limits, group_by=group_by)
            if grid is None:  # cancel occured
                return
        import matplotlib.cm
        background_color = np.array(matplotlib.colors.colorConverter.to_rgb(background_color))
        if group_by:
            gmin, gmax, group_count = group_limits
            if isinstance(group_colors, six.string_types):
                group_colors = matplotlib.cm.get_cmap(group_colors)
            if isinstance(group_colors, matplotlib.colors.Colormap):
                group_count = group_limits[2]
                colors = [group_colors(k / float(group_count - 1.)) for k in range(group_count)]
            else:
                colors = [matplotlib.colors.colorConverter.to_rgba(k) for k in group_colors]
            total = np.sum(grid, axis=0).T
            # grid /= total
            mask = total > 0
            alpha = total - total[mask].min()
            alpha[~mask] = 0
            alpha = total / alpha.max()
            rgba = grid.T.dot(colors)

            def _norm(data):
                mask = np.isfinite(data)
                data = data - data[mask].min()
                data /= data[mask].max()
                return data
            rgba[..., 3] = (f(alpha))
            # rgba[...,3] = 1
            rgba[total == 0, 3] = 0.
            mask = alpha > 0
            if 1:
                for i in range(3):
                    rgba[..., i] /= total
                    # rgba[...,i] /= rgba[...,0:3].max()
                    rgba[~mask, i] = background_color[i]
            rgba = (np.swapaxes(rgba, 0, 1))
        else:
            if color:
                color = np.array(matplotlib.colors.colorConverter.to_rgba(color))
                rgba = np.zeros(grid.shape + (4,))
                rgba[..., 0:4] = color
                data = f(grid)
                mask = (grid > 0) & np.isfinite(data)
                if vmin is None:
                    vmin = data[mask].min()
                if vmax is None:
                    vmax = data[mask].max()
                if mask.sum():
                    data -= vmin
                    data /= vmax
                    data[~mask] = 0
                else:
                    data[:] = 0
                rgba[..., 3] = data
            else:
                cmap = matplotlib.cm.get_cmap(cmap)
                data = f(grid)
                if normalize:
                    mask = (data > 0) & np.isfinite(data)
                    if vmin is None:
                        vmin = data[mask].min()
                    if vmax is None:
                        vmax = data[mask].max()
                    if mask.sum():
                        data -= vmin
                        data /= vmax
                    else:
                        data[:] = 0
                    data[~mask] = 0
                data = np.clip(data, 0, 1)
                rgba = cmap(data)
                if normalize:
                    rgba[~mask, 3] = 0
                rgba[..., 3] = 1  # data
            # rgba8 = np.swapaxes(rgba8, 0, 1)
        # white = np.ones_like(rgba[...,0:3])
        if pre_blend:
            # rgba[...,3] = background_alpha
            rgb = rgba[..., :3].T
            alpha = rgba[..., 3].T
            rgb[:] = rgb * alpha + background_color[:3].reshape(3, 1, 1) * (1 - alpha)
            alpha[:] = alpha + background_alpha * (1 - alpha)
        rgba = np.clip(rgba, 0, 1)
        rgba8 = (rgba * 255).astype(np.uint8)
        return rgba8

    def plot_vectors(self, expression_x, expression_y, limits, wx=None, wy=None, counts=None, size=32, axes=None, **kwargs):
        import pylab
        # refactor: should go to bin_means_xy
        if counts is None:
            counts = self.histogram(size=size, limits=limits)
        if wx is None:
            wx = self.histogram(size=size, weight=expression_x, limits=limits)
        if wy is None:
            wy = self.histogram(size=size, weight=expression_y, limits=limits)
        N = size
        positions = [vaex.utils.linspace_centers(limits[i][0], limits[i][1], N) for i in range(self.dimension)]
        # print(positions)
        mask = counts > 0
        vx = wx / counts
        vy = wy / counts
        vx[counts == 0] = 0
        vy[counts == 0] = 0
        # vx = self.vx.grid / self.vcounts.grid
        # vy = self.vy.grid / self.vcounts.grid
        x2d, y2d = np.meshgrid(positions[0], positions[1])
        if axes is None:
            axes = pylab.gca()
        axes.quiver(x2d[mask], y2d[mask], vx[mask], vy[mask], **kwargs)

    def plot(self, grid=None, size=256, limits=None, square=False, center=None, weight=None, weight_stat="mean", figsize=None,
             aspect="auto", f="identity", axes=None, xlabel=None, ylabel=None,
             group_by=None, group_limits=None, group_colors='jet', group_labels=None, group_count=None,
             vmin=None, vmax=None,
             cmap="afmhot",
             **kwargs):
        """Plot the subspace using sane defaults to get a quick look at the data.

        :param grid: A 2d numpy array with the counts, if None it will be calculated using limits provided and Subspace.histogram
        :param size: Passed to Subspace.histogram
        :param limits: Limits for the subspace in the form [[xmin, xmax], [ymin, ymax]], if None it will be calculated using Subspace.limits_sigma
        :param square: argument passed to Subspace.limits_sigma
        :param Executor executor: responsible for executing the tasks
        :param figsize: (x, y) tuple passed to pylab.figure for setting the figure size
        :param aspect: Passed to matplotlib's axes.set_aspect
        :param xlabel: String for label on x axis (may contain latex)
        :param ylabel: Same for y axis
        :param kwargs: extra argument passed to axes.imshow, useful for setting the colormap for instance, e.g. cmap='afmhot'
        :return: matplotlib.image.AxesImage

         """
        import pylab
        f = _parse_f(f)
        limits = self.limits(limits)
        if limits is None:
            limits = self.limits_sigma()
        # if grid is None:
        if group_limits is None and group_by:
            group_limits = tuple(self.df(group_by).minmax()[0]) + (group_count,)
        # grid = self.histogram(limits=limits, size=size, weight=weight, group_limits=group_limits, group_by=group_by)
        if figsize is not None:
            pylab.figure(num=None, figsize=figsize, dpi=80, facecolor='w', edgecolor='k')
        if axes is None:
            axes = pylab.gca()
        fig = pylab.gcf()
        # if xlabel:
        pylab.xlabel(xlabel or self.expressions[0])
        # if ylabel:
        pylab.ylabel(ylabel or self.expressions[1])
        # axes.set_aspect(aspect)
        rgba8 = self.image_rgba(grid=grid, size=size, limits=limits, square=square, center=center, weight=weight, weight_stat=weight_stat,
                                f=f, axes=axes,
                                group_by=group_by, group_limits=group_limits, group_colors=group_colors, group_count=group_count,
                                vmin=vmin, vmax=vmax,
                                cmap=cmap)
        import matplotlib
        if group_by:
            if isinstance(group_colors, six.string_types):
                group_colors = matplotlib.cm.get_cmap(group_colors)
            if isinstance(group_colors, matplotlib.colors.Colormap):
                group_count = group_limits[2]
                colors = [group_colors(k / float(group_count - 1.)) for k in range(group_count)]
            else:
                colors = [matplotlib.colors.colorConverter.to_rgba(k) for k in group_colors]
            colormap = matplotlib.colors.ListedColormap(colors)
            gmin, gmax, group_count = group_limits  # [:2]
            delta = (gmax - gmin) / (group_count - 1.)
            norm = matplotlib.colors.Normalize(gmin - delta / 2, gmax + delta / 2)
            sm = matplotlib.cm.ScalarMappable(norm, colormap)
            sm.set_array(1)  # make matplotlib happy (strange behavious)
            colorbar = fig.colorbar(sm)
            if group_labels:
                colorbar.set_ticks(np.arange(gmin, gmax + delta / 2, delta))
                colorbar.set_ticklabels(group_labels)
            else:
                colorbar.set_ticks(np.arange(gmin, gmax + delta / 2, delta))
                colorbar.set_ticklabels(map(lambda x: "%f" % x, np.arange(gmin, gmax + delta / 2, delta)))
            colorbar.ax.set_ylabel(group_by)
            # matplotlib.colorbar.ColorbarBase(axes, norm=norm, cmap=colormap)
            im = axes.imshow(rgba8, extent=np.array(limits).flatten(), origin="lower", aspect=aspect, **kwargs)
        else:
            norm = matplotlib.colors.Normalize(0, 23)
            sm = matplotlib.cm.ScalarMappable(norm, cmap)
            sm.set_array(1)  # make matplotlib happy (strange behavious)
            colorbar = fig.colorbar(sm)
            im = axes.imshow(rgba8, extent=np.array(limits).flatten(), origin="lower", aspect=aspect, **kwargs)
            colorbar = None
        return im, colorbar

    def plot1d(self, grid=None, size=64, limits=None, weight=None, figsize=None, f="identity", axes=None, xlabel=None, ylabel=None, **kwargs):
        """Plot the subspace using sane defaults to get a quick look at the data.

        :param grid: A 2d numpy array with the counts, if None it will be calculated using limits provided and Subspace.histogram
        :param size: Passed to Subspace.histogram
        :param limits: Limits for the subspace in the form [[xmin, xmax], [ymin, ymax]], if None it will be calculated using Subspace.limits_sigma
        :param figsize: (x, y) tuple passed to pylab.figure for setting the figure size
        :param xlabel: String for label on x axis (may contain latex)
        :param ylabel: Same for y axis
        :param kwargs: extra argument passed to ...,

         """
        import pylab
        f = _parse_f(f)
        limits = self.limits(limits)
        assert self.dimension == 1, "can only plot 1d, not %s" % self.dimension
        if limits is None:
            limits = self.limits_sigma()
        if grid is None:
            grid = self.histogram(limits=limits, size=size, weight=weight)
        if figsize is not None:
            pylab.figure(num=None, figsize=figsize, dpi=80, facecolor='w', edgecolor='k')
        if axes is None:
            axes = pylab.gca()
        # if xlabel:
        pylab.xlabel(xlabel or self.expressions[0])
        # if ylabel:
        # pylab.ylabel(ylabel or self.expressions[1])
        pylab.ylabel("counts" or ylabel)
        # axes.set_aspect(aspect)
        N = len(grid)
        xmin, xmax = limits[0]
        return pylab.plot(np.arange(N) / (N - 1.0) * (xmax - xmin) + xmin, f(grid,), drawstyle="steps", **kwargs)
        # pylab.ylim(-1, 6)

    def plot_histogram_bq(self, f="identity", size=64, limits=None, color="red", bq_cleanup=True):
        import vaex.ext.bqplot
        limits = self.limits(limits)
        plot = vaex.ext.bqplot.BqplotHistogram(self, color, size, limits)
        if not hasattr(self, "_bqplot"):
            self._bqplot = {}
            self._bqplot["cleanups"] = []
        else:
            if bq_cleanup:
                for cleanup in self._bqplot["cleanups"]:
                    cleanup()
            self._bqplot["cleanups"] = []

        def cleanup(callback=plot.callback):
            self.df.signal_selection_changed.disconnect(callback=callback)
        self._bqplot["cleanups"].append(cleanup)

        return plot

    def plot_bq(self, grid=None, size=256, limits=None, square=False, center=None, weight=None, figsize=None,
                aspect="auto", f="identity", fig=None, axes=None, xlabel=None, ylabel=None, title=None,
                group_by=None, group_limits=None, group_colors='jet', group_labels=None, group_count=None,
                cmap="afmhot", scales=None, tool_select=False, bq_cleanup=True,
                **kwargs):
        import vaex.ext.bqplot
        import bqplot.interacts
        import bqplot.pyplot as p
        import ipywidgets as widgets
        import bqplot as bq
        f = _parse_f(f)
        limits = self.limits(limits)
        import vaex.ext.bqplot
        vaex.ext.bqplot.patch()
        if not hasattr(self, "_bqplot"):
            self._bqplot = {}
            self._bqplot["cleanups"] = []
        else:
            if bq_cleanup:
                for cleanup in self._bqplot["cleanups"]:
                    cleanup()
            self._bqplot["cleanups"] = []
        if limits is None:
            limits = self.limits_sigma()
        # if fig is None:
        if scales is None:
            x_scale = bq.LinearScale(min=limits[0][0], max=limits[0][1])
            y_scale = bq.LinearScale(min=limits[1][0], max=limits[1][1])
            scales = {'x': x_scale, 'y': y_scale}
        else:
            x_scale = scales["x"]
            y_scale = scales["y"]
        if 1:
            fig = p.figure()  # actually, bqplot doesn't return it
            fig = p.current_figure()
            fig.fig_color = "black"  # TODO, take the color from the colormap
            fig.padding_y = 0
            # if we don't do this, bqplot may flip some axes... report this bug
            x = np.arange(10)
            y = x**2
            p.plot(x, y, scales=scales)
            # p.xlim(*limits[0])
            # p.ylim(*limits[1])
            # if grid is None:
        if group_limits is None and group_by:
            group_limits = tuple(self.df(group_by).minmax()[0]) + (group_count,)
        # fig = p.
        # if xlabel:
        fig.axes[0].label = xlabel or self.expressions[0]
        # if ylabel:
        fig.axes[1].label = ylabel or self.expressions[1]
        if title:
            fig.title = title
        # axes.set_aspect(aspect)
        rgba8 = self.image_rgba(grid=grid, size=size, limits=limits, square=square, center=center, weight=weight,
                                f=f, axes=axes,
                                group_by=group_by, group_limits=group_limits, group_colors=group_colors, group_count=group_count,
                                cmap=cmap)
        # x_scale = p._context["scales"]["x"]
        # y_scale = p._context["scales"]["y"]
        src = "http://localhost:8888/kernelspecs/python2/logo-64x64.png"
        import bqplot.marks
        im = vaex.ext.bqplot.Image(src=src, scales=scales, x=0, y=0, width=1, height=1)
        if 0:
            size = 20
            x_data = np.arange(size)
            line = bq.Lines(x=x_data, y=np.random.randn(size), scales={'x': x_scale, 'y': y_scale},
                            stroke_width=3, colors=['red'])

            ax_x = bq.Axis(scale=x_scale, tick_format='0.2f', grid_lines='solid')
            ax_y = bq.Axis(scale=y_scale, orientation='vertical', tick_format='0.2f', grid_lines='solid')
            panzoom = bq.PanZoom(scales={'x': [x_scale], 'y': [y_scale]})
            lasso = bqplot.interacts.LassoSelector()
            brush = bqplot.interacts.BrushSelector(x_scale=x_scale, y_scale=y_scale, color="green")
            fig = bq.Figure(marks=[line, im], axes=[ax_x, ax_y], min_width=100, min_height=100, interaction=panzoom)
        else:
            fig.marks = list(fig.marks) + [im]

        def make_image(executor, limits):
            # print "make image" * 100
            self.executor = executor
            if self.df.has_selection():
                sub = self.selected()
            else:
                sub = self
            return sub.image_rgba(limits=limits, size=size, f=f)
        progress = widgets.FloatProgress(value=0.0, min=0.0, max=1.0, step=0.01)
        updater = vaex.ext.bqplot.DebouncedThreadedUpdater(self, size, im, make_image, progress_widget=progress)

        def update_image():
            limits = [x_scale.min, x_scale.max], [y_scale.min, y_scale.max]
            # print limits
            # print "update...", limits
            # vxbq.debounced_threaded_update(self.df, im, make_image2, limits=limits)
            updater.update(limits)

        def update(*args):
            update_image()
        y_scale.observe(update, "min")
        y_scale.observe(update, "max")
        x_scale.observe(update, "min")
        x_scale.observe(update, "max")
        update_image()
        # fig = kwargs.pop('figure', p.current_figure())
        tools = []
        tool_actions = []
        panzoom = bq.PanZoom(scales={'x': [x_scale], 'y': [y_scale]})
        tool_actions_map = {u"m": panzoom}
        tool_actions.append(u"m")

        fig.interaction = panzoom
        if tool_select:
            brush = bqplot.interacts.BrushSelector(x_scale=x_scale, y_scale=y_scale, color="green")
            tool_actions_map["b"] = brush
            tool_actions.append("b")

            def update_selection(*args):
                def f():
                    if brush.selected:
                        (x1, y1), (x2, y2) = brush.selected
                        ex1, ex2 = self.expressions
                        mode = modes_names[modes_labels.index(button_selection_mode.value)]
                        self.df.select_rectangle(ex1, ex2, limits=[[x1, x2], [y1, y2]], mode=mode)
                    else:
                        self.df.select_nothing()
                updater.update_select(f)
            brush.observe(update_selection, "selected")
            # fig.interaction = brush
            # callback = self.df.signal_selection_changed.connect(lambda df: update_image())
            callback = self.df.signal_selection_changed.connect(lambda df: updater.update_direct_safe())

            def cleanup(callback=callback):
                self.df.signal_selection_changed.disconnect(callback=callback)
            self._bqplot["cleanups"].append(cleanup)

            button_select_nothing = widgets.Button(icon="fa-trash-o")

            def select_nothing(button):
                self.df.select_nothing()
            button_select_nothing.on_click(select_nothing)
            tools.append(button_select_nothing)
            modes_names = "replace and or xor subtract".split()
            modes_labels = "= & | ^ -".split()
            button_selection_mode = widgets.ToggleButtons(description='', options=modes_labels)
            tools.append(button_selection_mode)

        def change_interact(*args):
            # print "change", args
            fig.interaction = tool_actions_map[button_action.value]
        # tool_actions = ["m", "b"]
        # tool_actions = [("m", "m"), ("b", "b")]
        button_action = widgets.ToggleButtons(description='', options=tool_actions, icons=["fa-arrows", "fa-pencil-square-o"])
        button_action.observe(change_interact, "value")
        tools.insert(0, button_action)
        button_action.value = "m"  # tool_actions[-1]
        if len(tools) == 1:
            tools = []
        tools = widgets.HBox(tools)

        box_layout = widgets.Layout(display='flex',
                                    flex_flow='column',
                                    # border='solid',
                                    width='100%', height="100%")
        fig.fig_margin = {'bottom': 40, 'left': 60, 'right': 10, 'top': 40}
        # fig.min_height = 700
        # fig.min_width = 400
        fig.layout = box_layout
        return widgets.VBox([fig, progress, tools])

    def figlarge(self, size=(10, 10)):
        import pylab
        pylab.figure(num=None, figsize=size, dpi=80, facecolor='w', edgecolor='k')

    # def bounded(self):
    # return self.bounded_by_minmax()

    def bounded_by(self, limits):
        """Returns a bounded subspace (SubspaceBounded) with limits as given by limits

        :param limits: sequence of [(min, max), ..., (min, max)] values
        :rtype: SubspaceBounded
        """
        return SubspaceBounded(self, np.array(limits))

    def bounded_by_minmax(self):
        """Returns a bounded subspace (SubspaceBounded) with limits given by Subspace.minmax()

        :rtype: SubspaceBounded
        """
        bounds = self.minmax()
        return SubspaceBounded(self, bounds)

    bounded = bounded_by_minmax

    def bounded_by_sigmas(self, sigmas=3, square=False):
        """Returns a bounded subspace (SubspaceBounded) with limits given by Subspace.limits_sigma()

        :rtype: SubspaceBounded
        """
        bounds = self.limits_sigma(sigmas=sigmas, square=square)
        return SubspaceBounded(self, bounds)

    def minmax(self):
        """Return a sequence of [(min, max), ..., (min, max)] corresponding to each expression in this subspace ignoring NaN.
        """
        raise NotImplementedError

    def mean(self):
        """Return a sequence of [mean, ... , mean] corresponding to the mean of each expression in this subspace ignoring NaN.
        """
        raise NotImplementedError

    def var(self, means=None):
        """Return a sequence of [var, ... , var] corresponding to the variance of each expression in this subspace ignoring NaN.
        """
        raise NotImplementedError

    def sum(self):
        """Return a sequence of [sum, ... , sum] corresponding to the sum of values of each expression in this subspace ignoring NaN."""
        raise NotImplementedError

    def histogram(self, limits, size=256, weight=None):
        """Return a grid of shape (size, ..., size) corresponding to the dimensionality of this subspace containing the counts in each element

        The type of the grid of np.float64

        """
        raise NotImplementedError

    def limits_sigma(self, sigmas=3, square=False):
        raise NotImplementedError

    def row(self, index):
        return np.array([self.df.evaluate(expression, i1=index, i2=index + 1)[0] for expression in self.expressions])


class SubspaceLocal(Subspace):
    """Subclass of subspace which implemented methods that can be run locally.
    """

    def _toarray(self, list):
        return np.array(list)

    @property
    def pre(self):
        self.executor.pre

    @property
    def post(self):
        self.executor.post

    def _task(self, task, progressbar=False):
        """Helper function for returning tasks results, result when immediate is True, otherwise the task itself, which is a promise"""
        if self.delay:
            # should return a task or a promise nesting it
            return self.executor.schedule(task)
        else:
            import vaex.utils
            callback = None
            try:
                if progressbar == True:
                    def update(fraction):
                        bar.update(fraction)
                        return True
                    bar = vaex.utils.progressbar(task.name)
                    callback = self.executor.signal_progress.connect(update)
                elif progressbar:
                    callback = self.executor.signal_progress.connect(progressbar)
                result = self.executor.run(task)
                if progressbar == True:
                    bar.finish()
                    sys.stdout.write('\n')
                return result
            finally:
                if callback:
                    self.executor.signal_progress.disconnect(callback)

    def minmax(self, progressbar=False):
        def min_max_reduce(minmax1, minmax2):
            if minmax1 is None:
                return minmax2
            if minmax2 is None:
                return minmax1
            result = []
            for d in range(self.dimension):
                min1, max1 = minmax1[d]
                min2, max2 = minmax2[d]
                result.append((min(min1, min2), max(max1, max2)))
            return result

        def min_max_map(thread_index, i1, i2, *blocks):
            if self.is_masked or self.df.filtered:
                mask = self.df.evaluate_selection_mask("default" if self.is_masked else None, i1=i1, i2=i2)
                blocks = [block[mask] for block in blocks]
                is_empty = all(~mask)
                if is_empty:
                    return None
            # with lock:
            # print blocks
            # with lock:
            # print thread_index, i1, i2, blocks
            blocks = [_asfloat(block) for block in blocks]
            return [vaex.vaexfast.find_nan_min_max(block) for block in blocks]
            if 0:  # TODO: implement using statisticNd and benchmark
                minmaxes = np.zeros((len(blocks), 2), dtype=float)
                minmaxes[:, 0] = np.inf
                minmaxes[:, 1] = -np.inf
                for i, block in enumerate(blocks):
                    vaex.vaexfast.statisticNd([], block, minmaxes[i, :], [], [], 2)
                # minmaxes[~np.isfinite(minmaxes)] = np.nan
                return minmaxes
        task = TaskMapReduceLegacy(self.df, self.expressions, min_max_map, min_max_reduce, self._toarray, info=True, name="minmax")
        return self._task(task, progressbar=progressbar)

    def mean(self):
        return self._moment(1)

    def _moment(self, moment=1):
        def mean_reduce(means_and_counts1, means_and_counts2):
            means_and_counts = []
            for (mean1, count1), (mean2, count2) in zip(means_and_counts1, means_and_counts2):
                means_and_counts.append([np.nansum([mean1 * count1, mean2 * count2]) / (count1 + count2), count1 + count2])
            return means_and_counts

        def remove_counts(means_and_counts):
            return self._toarray(means_and_counts)[:, 0]

        def mean_map(thread_index, i1, i2, *blocks):
            if self.is_masked or self.df.filtered:
                mask = self.df.evaluate_selection_mask("default" if self.is_masked else None, i1=i1, i2=i2)
                return [(np.nanmean(block[mask]**moment), np.count_nonzero(~np.isnan(block[mask]))) for block in blocks]
            else:
                return [(np.nanmean(block**moment), np.count_nonzero(~np.isnan(block))) for block in blocks]
        task = TaskMapReduceLegacy(self.df, self.expressions, mean_map, mean_reduce, remove_counts, info=True)
        return self._task(task)

    def var(self, means=None):
        # variances are linear, use the mean to reduce
        def vars_reduce(vars_and_counts1, vars_and_counts2):
            vars_and_counts = []
            for (var1, count1), (var2, count2) in zip(vars_and_counts1, vars_and_counts2):
                vars_and_counts.append([np.nansum([var1 * count1, var2 * count2]) / (count1 + count2), count1 + count2])
            return vars_and_counts

        def remove_counts(vars_and_counts):
            return self._toarray(vars_and_counts)[:, 0]
        if self.is_masked or self.df.filtered:
            def var_map(thread_index, i1, i2, *blocks):
                mask = self.df.evaluate_selection_mask("default" if self.is_masked else None, i1=i1, i2=i2)
                if means is not None:
                    return [(np.nanmean((block[mask] - mean)**2), np.count_nonzero(~np.isnan(block[mask]))) for block, mean in zip(blocks, means)]
                else:
                    return [(np.nanmean(block[mask]**2), np.count_nonzero(~np.isnan(block[mask]))) for block in blocks]
            task = TaskMapReduceLegacy(self.df, self.expressions, var_map, vars_reduce, remove_counts, info=True)
        else:
            def var_map(*blocks):
                if means is not None:
                    return [(np.nanmean((block - mean)**2), np.count_nonzero(~np.isnan(block))) for block, mean in zip(blocks, means)]
                else:
                    return [(np.nanmean(block**2), np.count_nonzero(~np.isnan(block))) for block in blocks]
            task = TaskMapReduceLegacy(self.df, self.expressions, var_map, vars_reduce, remove_counts)
        return self._task(task)

    def correlation(self, means=None, vars=None):
        if self.dimension != 2:
            raise ValueError("correlation is only defined for 2d subspaces, not %dd" % self.dimension)

        def do_correlation(means, vars):
            meanx, meany = means
            sigmax, sigmay = vars[0]**0.5, vars[1]**0.5

            def remove_counts_and_normalize(covar_and_count):
                covar, counts = covar_and_count
                return covar / counts / (sigmax * sigmay)

            def covars_reduce(covar_and_count1, covar_and_count2):
                if covar_and_count1 is None:
                    return covar_and_count2
                if covar_and_count2 is None:
                    return covar_and_count1
                else:
                    covar1, count1 = covar_and_count1
                    covar2, count2 = covar_and_count2
                    return [np.nansum([covar1, covar2]), count1 + count2]

            mask = self.df.mask

            def covar_map(thread_index, i1, i2, *blocks):
                # return [(np.nanmean((block[mask[i1:i2]]-mean)**2), np.count_nonzero(~np.isnan(block[mask[i1:i2]]))) for block, mean in zip(blocks, means)]
                blockx, blocky = blocks
                if self.is_masked:
                    blockx, blocky = blockx[mask[i1:i2]], blocky[mask[i1:i2]]
                counts = np.count_nonzero(~(np.isnan(blockx) | np.isnan(blocky)))
                if counts == 0:
                    return None
                else:
                    return np.nansum((blockx - meanx) * (blocky - meany)), counts

            task = TaskMapReduceLegacy(self.df, self.expressions, covar_map, covars_reduce, remove_counts_and_normalize, info=True)
            return self._task(task)
        if means is None:
            if self.delay:
                means_wrapper = [None]

                def do_vars(means):
                    means_wrapper[0] = means
                    return self.var(means)

                def do_correlation_wrapper(vars):
                    return do_correlation(means_wrapper[0], vars)
                return self.mean().then(do_vars).then(do_correlation_wrapper)
            else:
                means = self.mean()
                vars = self.var(means=means)
                return do_correlation(means, vars)
        else:
            if vars is None:
                if self.delay:
                    def do_correlation_wrapper(vars):
                        return do_correlation(means, vars)
                    return self.vars(means=means).then(do_correlation_wrapper)
                else:
                    vars = self.var(means)
                    return do_correlation(means, vars)
            else:
                if means is None:
                    means = self.mean()
                if vars is None:
                    vars = self.var(means=means)
                return do_correlation(means, vars)

    def sum(self):
        def nansum(x): return np.nansum(x, dtype=np.float64)
        # TODO: we can speed up significantly using our own nansum, probably the same for var and mean
        nansum = vaex.vaexfast.nansum
        if self.is_masked or self.df.filtered:
            task = TaskMapReduceLegacy(self.df,
                                 self.expressions, lambda thread_index, i1, i2, *blocks: [nansum(block[self.df.evaluate_selection_mask("default" if self.is_masked else None, i1=i1, i2=i2)])
                                                                                          for block in blocks],
                                 lambda a, b: np.array(a) + np.array(b), self._toarray, info=True)
        else:
            task = TaskMapReduceLegacy(self.df, self.expressions, lambda *blocks: [nansum(block) for block in blocks], lambda a, b: np.array(a) + np.array(b), self._toarray)
        return self._task(task)

    def histogram(self, limits, size=256, weight=None, progressbar=False, group_by=None, group_limits=None):
        expressions = self.expressions
        if group_by:
            expressions = list(expressions) + [group_by]
            limits = list(limits) + [group_limits[:2]]  # [[group_limits[0] - 0,5, group_limits[1]+0.5]]
            # assert group_limits[2] == 1
            size = (group_limits[2],) + (size,) * (len(expressions) - 1)
        task = TaskHistogram(self.df, self, expressions, size, limits, masked=self.is_masked, weight=weight)
        return self._task(task, progressbar=progressbar)

    def bin_mean(self, expression, limits, size=256, progressbar=False, group_by=None, group_limits=None):
        # todo, fix progressbar into two...
        counts = self.histogram(limits=limits, size=size, progressbar=progressbar, group_by=group_by, group_limits=group_limits)
        weighted = self.histogram(limits=limits, size=size, progressbar=progressbar, group_by=group_by, group_limits=group_limits,
                                  weight=expression)
        mean = weighted / counts
        mean[counts == 0] = np.nan
        return mean

    def bin_mean_cyclic(self, expression, max_value, limits, size=256, progressbar=False, group_by=None, group_limits=None):
        # todo, fix progressbar into two...
        meanx = self.bin_mean(limits=limits, size=size, progressbar=progressbar, group_by=group_by, group_limits=group_limits,
                              expression="cos((%s)/%r*2*pi)" % (expression, max_value))
        meany = self.bin_mean(limits=limits, size=size, progressbar=progressbar, group_by=group_by, group_limits=group_limits,
                              expression="sin((%s)/%r*2*pi)" % (expression, max_value))
        angles = np.arctan2(meany, meanx)
        values = ((angles + 2 * np.pi) % (2 * np.pi)) / (2 * np.pi) * max_value
        length = np.sqrt(meanx**2 + meany**2)
        length[~np.isfinite(meanx)] = np.nan
        return values, length

    def mutual_information(self, limits=None, grid=None, size=256):
        if limits is None:
            limits_done = Task.fulfilled(self.minmax())
        else:
            limits_done = Task.fulfilled(limits)
        if grid is None:
            if limits is None:
                histogram_done = limits_done.then(lambda limits: self.histogram(limits, size=size))
            else:
                histogram_done = Task.fulfilled(self.histogram(limits, size=size))
        else:
            histogram_done = Task.fulfilled(grid)
        mutual_information_promise = histogram_done.then(vaex.kld.mutual_information)
        return mutual_information_promise if self.delay else mutual_information_promise.get()

    def limits_percentage(self, percentage=99.73, square=False):
        import scipy.ndimage
        limits = []
        for expr in self.expressions:
            subspace = self.df(expr)
            if self.is_selected():
                subspace = subspace.selected()
            limits_minmax = subspace.minmax()
            vmin, vmax = limits_minmax[0]
            size = 1024 * 16
            counts = subspace.histogram(size=size, limits=limits_minmax)
            cumcounts = np.concatenate([[0], np.cumsum(counts)])
            cumcounts /= cumcounts.max()
            # TODO: this is crude.. see the details!
            f = (1 - percentage / 100.) / 2
            x = np.linspace(vmin, vmax, size + 1)
            l = scipy.interp([f, 1 - f], cumcounts, x)
            limits.append(l)
        return limits

    def limits_sigma(self, sigmas=3, square=False):
        if self.delay:
            means_wrapper = [None]

            def do_vars(means):
                means_wrapper[0] = means
                return self.var(means)

            def do_limits(vars):
                stds = vars**0.5
                means = means_wrapper[0]
                if square:
                    stds = np.repeat(stds.mean(), len(stds))
                return np.array(list(zip(means - sigmas * stds, means + sigmas * stds)))
            return self.mean().then(do_vars).then(do_limits)
        else:
            means = self.mean()
            stds = self.var(means=means)**0.5
            if square:
                stds = np.repeat(stds.mean(), len(stds))
            return np.array(list(zip(means - sigmas * stds, means + sigmas * stds)))

    def _not_needed_current(self):
        index = self.df.get_current_row()

        def find(thread_index, i1, i2, *blocks):
            if (index >= i1) and (index < i2):
                return [block[index - i1] for block in blocks]
            else:
                return None
        task = TaskMapReduceLegacy(self.df, self.expressions, find, lambda a, b: a if b is None else b, info=True)
        return self._task(task)

    def nearest(self, point, metric=None):
        metric = metric or [1.] * len(point)

        def nearest_in_block(thread_index, i1, i2, *blocks):
            if self.is_masked:
                mask = self.df.evaluate_selection_mask("default", i1=i1, i2=i2)
                if mask.sum() == 0:
                    return None
                blocks = [block[mask] for block in blocks]
            distance_squared = np.sum([(blocks[i] - point[i])**2. * metric[i] for i in range(self.dimension)], axis=0)
            min_index_global = min_index = np.argmin(distance_squared)
            if self.is_masked:  # we skipped some indices, so correct for that
                min_index_global = np.argmin((np.cumsum(mask) - 1 - min_index)**2)
            # with lock:
            # print i1, i2, min_index, distance_squared, [block[min_index] for block in blocks]
            return min_index_global.item() + i1, distance_squared[min_index].item()**0.5, [block[min_index].item() for block in blocks]

        def nearest_reduce(a, b):
            if a is None:
                return b
            if b is None:
                return a
            if a[1] < b[1]:
                return a
            else:
                return b
        if self.is_masked:
            pass
        task = TaskMapReduceLegacy(self.df,
                             self.expressions,
                             nearest_in_block,
                             nearest_reduce, info=True)
        return self._task(task)
