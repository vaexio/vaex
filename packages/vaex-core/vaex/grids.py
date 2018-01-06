__author__ = 'maartenbreddels'
import numpy as np
import vaex.vaexfast
from vaex.utils import filesize_format
import logging
import vaex.utils
total_bytes = 0

logger = logging.getLogger("vaex.grids")


def dog(grid, sigma1, sigma2):
    return gf(grid, sigma1) - gf(grid, sigma2)


def gf(grid, sigma, **kwargs):
    import scipy.ndimage
    return scipy.ndimage.gaussian_filter(grid, sigma=sigma, **kwargs)


functions = {}
functions["dog"] = dog
functions["gf"] = gf


def grid_average(scope, counts_name="counts", weighted_name="weighted"):

    counts = scope.evaluate(counts_name)
    weighted = scope.evaluate(weighted_name)
    logger.debug("evaluating average: counts=%r weighted=%r" % (counts, weighted))
    mask = counts == 0
    average = np.zeros(counts.shape, dtype=np.float64)
    average[~mask] = weighted[~mask] / counts[~mask]
    average[mask] = np.nan
    return average


class GridScope(object):
    def __init__(self, locals=None, globals=None):
        # self.locals = locals or {}
        # if locals:
        # self.__dict__.update(locals)
        self.globals = globals or {}
        self.lazy = {}
        self.lazy["average"] = grid_average
        self.globals["cumulative"] = self.cumulative
        self.globals["normalize"] = self.normalize
        self.globals.update(functions)
        self.user_added = set()

    def cumulative(self, array, normalize=True):
        mask = (np.isnan(array) | np.isinf(array))
        values = array * 1
        values[mask] = 0
        c = np.cumsum(values)
        if normalize:
            return c / c[-1]
        else:
            return c

    def normalize(self, array):
        mask = (np.isnan(array) | np.isinf(array))
        values = array * 1
        total = np.sum(values[~mask])
        return values / total

    def setter(self, key):
        def apply(value, key=key):
            self[key] = value
            return value
        return apply

    def add_lazy(self, key, f):
        self.lazy[key] = f

    def __contains__(self, item):
        return (item in self.__dict__) or (item in self.lazy)

    def __setitem__(self, key, value):
        # logger.debug("%r.__setitem__(%r, %r)" % (self, key, value))
        self.__dict__[key] = value
        self.user_added.add(key)

    def __getitem__(self, key):
        logger.debug("%r.__getitem__(%r)" % (self, key))
        if key in self.lazy and key not in self.__dict__:
            logger.debug("%r lazy item %r" % (self, key))
            f = self.lazy[key]
            self.__dict__[key] = f(self)

        return self.__dict__[key]

    def evaluate(self, expression):
        if 0:
            locals = dict(self.__dict__)
            del locals["globals"]
            logger.debug("evaluating: %r locals=%r", expression, locals)
        return eval(expression, self.globals, self)

    def slice(self, slice):
        gridscope = GridScope(globals=self.globals)
        for key in self.user_added:
            value = self[key]
            if isinstance(value, np.ndarray):
                grid = value
                sliced = np.sum(grid[slice, ...], axis=0)
                logger.debug("sliced %s from %r to %r", key, grid.shape, sliced.shape)
                gridscope[key] = sliced
            else:
                gridscope[key] = value
        return gridscope

    def disjoined(self):
        gridscope = GridScope(globals=self.globals)
        for key in self.user_added:
            value = self[key]
            if isinstance(value, np.ndarray):
                grid = vaex.utils.disjoined(value)
                gridscope[key] = grid
            else:
                gridscope[key] = value
        return gridscope

    def marginal2d(self, i, j):
        gridscope = GridScope(globals=self.globals)
        for key in self.user_added:
            value = self[key]
            if isinstance(value, np.ndarray):
                dimension = len(value.shape)
                axes = list(range(dimension))
                axes.remove(i)
                axes.remove(j)
                grid = vaex.utils.multisum(value, axes)
                gridscope[key] = grid
            else:
                gridscope[key] = value
        return gridscope


def add_mem(bytes, *info):
    global total_bytes
    total_bytes += bytes
    added = filesize_format(bytes)
    total = filesize_format(total_bytes)
    print(("MEMORY USAGE: added %s, total %s (%r)" % (added, total, info)))
