# -*- coding: utf-8 -*-
import logging as logging
import numpy as np
import math
import functools
import vaex.vaexfast
import vaex.utils
from vaex.utils import Timer

logger = logging.getLogger("vaex.kld")


def mutual_information(data):
    Q = vaex.utils.disjoined(data)
    P = data

    P = P / P.sum()
    Q = Q / Q.sum()
    mask = (P > 0) & (Q > 0)
    information = np.sum(P[mask] * np.log(P[mask] / Q[mask]))  # * np.sum(dx)
    return information


def kl_divergence(P, Q, axis=None):
    P = P / P.sum(axis=axis)
    Q = Q / Q.sum(axis=axis)
    # mask = (P > 0) & (Q>0)
    # information = np.sum(P[mask] * np.log(P[mask]/Q[mask]), axis=axis)# * np.sum(dx)
    # mask = (P > 0) & (Q>0)
    information = np.sum(P * np.log(P / Q), axis=axis)  # * np.sum(dx)
    return information


class KlDivergenceShuffle(object):
    def __init__(self, dataset, pairs, gridsize=128):
        self.dataset = dataset
        self.pairs = pairs
        self.dimension = len(self.pairs[0])
        self.logger = logger.getLogger('kld')
        self.gridsize = gridsize
        logger.debug("dimension: %d, pairs: %s" % (self.dimension, self.pairs))

    def get_jobs(self):
        def job(pair):
            pass


def to_disjoined(counts):
    shape = counts.shape
    assert len(counts.shape) == 2
    counts_0 = counts.sum(axis=1).reshape((shape[0], 1))
    counts_1 = counts.sum(axis=0).reshape((1, shape[1]))
    counts_disjoined = counts_0 * counts_1
    return counts_disjoined


def kld_shuffled(columns, Ngrid=128, datamins=None, datamaxes=None, offset=1):
    if datamins is None:
        datamins = np.array([np.nanmin(column) for column in columns])
    if datamaxes is None:
        datamaxes = np.array([np.nanmax(column) for column in columns])
    dim = len(columns)
    counts = np.zeros((Ngrid, ) * dim, dtype=np.float64)
    counts_shuffled = np.zeros((Ngrid, ) * dim, dtype=np.float64)
    D_kl = -1
    if len(columns) == 2:
        x, y = columns
        # print x
        # print y
        print((x, y, counts, counts_shuffled, datamins[0], datamaxes[0], datamins[1], datamaxes[1], offset))
        try:
            vaex.histogram.hist2d_and_shuffled(x, y, counts, counts_shuffled, datamins[0], datamaxes[0], datamins[1], datamaxes[1], offset)
        except:
            args = [x, y, counts, counts_shuffled, datamins[0], datamaxes[0], datamins[1], datamaxes[1], offset]
            sig = [numba.dispatcher.typeof_pyval(a) for a in args]
            print(sig)
            raise

        print(("counts", sum(counts)))
        deltax = [float(datamaxes[i] - datamins[i]) for i in range(dim)]
        dx = np.array([deltax[d] / counts.shape[d] for d in range(dim)])
        density = counts / np.sum(counts)  # * np.sum(dx)
        density_shuffled = counts_shuffled / np.sum(counts_shuffled)  # * np.sum(dx)
        mask = (density_shuffled > 0) & (density > 0)
        # print density
        D_kl = np.sum(density[mask] * np.log(density[mask] / density_shuffled[mask]))  # * np.sum(dx)
        # if D_kl < 0:
        # import pdb
        # pdb.set_trace()
    return D_kl


def kld_shuffled_grouped(dataset, range_map, pairs, feedback=None, size_grid=32, use_mask=True, bytes_max=int(1024**3 / 2)):
    dimension = len(pairs[0])
    bytes_per_grid = size_grid ** dimension * 8  # 8==sizeof (double)
    grids_per_iteration = min(128, int(bytes_max / bytes_per_grid))
    iterations = int(math.ceil(len(pairs) * 1. / grids_per_iteration))
    jobsManager = vaex.dataset.JobsManager()
    ranges = [None] * len(pairs)
    D_kls = []

    class Wrapper(object):
        pass
    wrapper = Wrapper()
    wrapper.N_done = 0

    counts = np.zeros((grids_per_iteration,) + (size_grid,) * dimension, dtype=np.float64)
    # counts_shuffled = np.zeros((grids_per_iteration,) + (size_grid,) * dimension, dtype=np.float64)

    N_total = len(pairs) * len(dataset) * dimension + len(pairs) * counts.size

    logger.debug("{iterations} iterations with {grids_per_iteration} grids per iteration".format(**locals()))
    for part in range(iterations):
        if part > 0:  # next iterations reset the counts
            counts.reshape(-1)[:] = 0
            # counts_shuffled.reshape(-1)[:] = 0
        i1, i2 = part * grids_per_iteration, (part + 1) * grids_per_iteration
        if i2 > len(pairs):
            i2 = len(pairs)
        n_grids = i2 - i1
        logger.debug("part {part} of {iterations}, from {i1} to {i2}".format(**locals()))

        def grid(info, *blocks, **kwargs):
            index = kwargs["index"]
            if use_mask and dataset.mask is not None:
                mask = dataset.mask[info.i1:info.i2]
                blocks = [block[mask] for block in blocks]
            else:
                mask = None
            ranges = []
            minima = []
            maxima = []
            for dim in range(dimension):
                ranges += list(range_map[pairs[index][dim]])
                mi, ma = range_map[pairs[index][dim]]
                minima.append(mi)
                maxima.append(ma)
            if len(blocks) == 2:
                print(("mask", mask))
                vaex.vaexfast.histogram2d(blocks[0], blocks[1], None, counts[index], *(ranges + [0, 0]))
                # vaex.vaexfast.histogram2d(blocks[0], blocks[1], None, counts_shuffled[index], *(ranges + [1, 0]))
            if len(blocks) == 3:
                vaex.vaexfast.histogram3d(blocks[0], blocks[1], blocks[2], None, counts[index], *(ranges + [0, 0, 0]))
                # vaex.vaexfast.histogramNd(list(blocks), None, counts[index], minima, maxima)
                # for i in range(5):
                # vaex.vaexfast.histogram3d(blocks[0], blocks[1], blocks[2], None, counts_shuffled[index], *(ranges + [2+i,1+i,0]))
            if len(blocks) > 3:
                vaex.vaexfast.histogramNd(list(blocks), None, counts[index], minima, maxima)
                # for i in range(5):
                # vaex.vaexfast.histogram3d(blocks[0], blocks[1], blocks[2], None, counts_shuffled[index], *(ranges + [2+i,1+i,0]))
            if feedback:
                wrapper.N_done += len(blocks[0]) * dimension
                if feedback:
                    cancel = feedback(wrapper.N_done * 100. / N_total)
                    if cancel:
                        raise Exception("cancelled")

        for index, pair in zip(list(range(i1, i2)), pairs[i1:i2]):
            logger.debug("add job %r %r" % (index, pair))
            jobsManager.addJob(0, functools.partial(grid, index=index - i1), dataset, *pair)
        jobsManager.execute()

        with Timer("D_kl"):
            for i in range(n_grids):
                if 0:
                    deltax = [float(range_map[pairs[i][d]][1] - range_map[pairs[i][d]][0]) for d in range(dimension)]
                    dx = np.array([deltax[d] / counts[i].shape[d] for d in range(dimension)])
                    density = counts[i] / np.sum(counts[i])  # * np.sum(dx)
                    counts_shuffled = to_disjoined(counts[i])
                    density_shuffled = counts_shuffled / np.sum(counts_shuffled)  # * np.sum(dx)
                    mask = (density_shuffled > 0) & (density > 0)
                    print(("mask sum", np.sum(mask)))
                    print(("mask sum", np.sum((counts_shuffled > 0) & (counts[i] > 0))))
                    # print density
                    D_kl = np.sum(density[mask] * np.log(density[mask] / density_shuffled[mask]))  # * np.sum(dx)
                else:
                    D_kl = mutual_information(counts[i])
                D_kls.append(D_kl)
                if feedback:
                    wrapper.N_done += counts.size
                    cancel = feedback(wrapper.N_done * 100. / N_total)
                    if cancel:
                        return None
                        # raise Exception, "cancelled"
    return D_kls


if __name__ == "__main__":
    import vaex.dataset
    import vaex.files
    from optparse import OptionParser
    parser = OptionParser()  # usage="")

    # parser.add_option("-n", "--name",
    #                 help="dataset name [default=%default]", default="data", type=str)
    # parser.add_option("-o", "--output",
    #                 help="dataset output filename [by default the suffix of input filename will be replaced by hdf5]", default=None, type=str)
    (options, args) = parser.parse_args()
    if len(args) > 0:
        filename = args[0]
    else:
        filename = "gaussian4d-1e7.hdf5"

    path = vaex.files.get_datafile(filename)
    dataset = vaex.dataset.Hdf5MemoryMapped(path)
    # for column_name in dataset.column_names:
    # print column_name

    subspace = dataset.subspace(1)[:3] * dataset.subspace(1)[:3]
    # subspace = Subspace(dataset, [("x",), ("y",), ("z",)]
