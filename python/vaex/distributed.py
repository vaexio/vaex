import vaex.dataset
from vaex.remote import ServerExecutor
import numpy as np
import logging
import aplus

logger = logging.getLogger("vaex.distributed")

class SubspaceDistributed(vaex.dataset.Subspace):
    def toarray(self, list):
        return np.array(list)

    @property
    def dimension(self):
        return len(self.expressions)

    def _task(self, promise):
        """Helper function for returning tasks results, result when immediate is True, otherwise the task itself, which is a promise"""
        if self.async:
            return promise
        else:
            return promise

    def sleep(self, seconds, async=False):
        return self.dataset.server.call("sleep", seconds, async=async)

    def minmax(self):
        return self._task(self.dataset.server._call_subspace("minmax", self))
        #return self._task(task)

    def histogram(self, limits, size=256, weight=None, progressbar=False, group_by=None, group_limits=None):
        promises = []
        for dataset in self.dataset.datasets:
            logger.debug("calling histogram for %r", dataset)
            subspace = dataset(*self.expressions, async=True)
            promises.append(subspace.histogram(limits, size=size, weight=weight))#, group_by=group_by, group_limits=group_limits))
        def sum(grids):
            return np.sum(grids, axis=0)
        alldone = aplus.listPromise(promises)
        task = alldone.then(sum).get()
        return self._task(task)#, progressbar=progressbar)



    def nearest(self, point, metric=None):
        point = vaex.utils.make_list(point)
        result = self.dataset.server._call_subspace("nearest", self, point=point, metric=metric)
        return self._task(result)

    def mean(self):
        return self.dataset.server._call_subspace("mean", self)

    def correlation(self, means=None, vars=None):
        return self.dataset.server._call_subspace("correlation", self, means=means, vars=vars)

    def var(self, means=None):
        return self.dataset.server._call_subspace("var", self, means=means)

    def sum(self):
        return self.dataset.server._call_subspace("sum", self)

    def limits_sigma(self, sigmas=3, square=False):
        return self.dataset.server._call_subspace("limits_sigma", self, sigmas=sigmas, square=square)

    def mutual_information(self, limits=None, size=256):
        return self.dataset.server._call_subspace("mutual_information", self, limits=limits, size=size)

class DatasetDistributed(vaex.dataset.Dataset):
    def __init__(self, datasets):
        super(DatasetDistributed, self).__init__(datasets[0].name, datasets[0].column_names)
        self.datasets = datasets
        self.executor = ServerExecutor()
        #self.name = self.datasets[0].name
        #self.column_names = self.datasets[0].column_names
        self.dtypes = self.datasets[0].dtypes
        self.units = self.datasets[0].units
        self.virtual_columns.update(self.datasets[0].units)
        self.ucds = self.datasets[0].ucds
        self.descriptions = self.datasets[0].descriptions
        self.description = self.datasets[0].description
        self._full_length = self.datasets[0].full_length()
        self._length = self._full_length
        self.path = self.datasets[0].path # may we should use some cluster name oroso
        parts = np.linspace(0, self._length, len(self.datasets)+1, dtype=int)
        for dataset, i1, i2 in zip(self.datasets, parts[0:-1], parts[1:]):
            dataset.set_active_range(i1, i2)


    def __call__(self, *expressions, **kwargs):
        return SubspaceDistributed(self, expressions, kwargs.get("executor") or self.executor, async=kwargs.get("async", False))
