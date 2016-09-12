import vaex.dataset
from vaex.remote import ServerExecutor
import numpy as np
import logging
import aplus
from .delayed import delayed
logger = logging.getLogger("vaex.distributed")

"""
class sum:
    @classmethod
    def map(cls, x):
        return np.nansum(x)
    @classmethod
    def reduce_one(cls, a, b):
        return a + b
    @classmethod
    def reduce(cls, x, initial=1):
        reduce(cls.reduce_one, x initial)
def multi_map_reduce(self):
"""
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
            return promise.get()

    def sleep(self, seconds, async=False):
        return self.dataset.server.call("sleep", seconds, async=async)

    def _apply_all(self, name, *args, **kwargs):
        promises = []
        print("calling %s (selection: %s)" % (name, self.is_masked))
        selection_name = "default"
        selection = self.dataset.get_selection(name=selection_name)
        if selection:
            print(selection, selection.to_dict())
        for dataset in self.dataset.datasets:
            dataset.set_selection(selection)#, selection_name=selection_name)
            subspace = dataset(*self.expressions, async=True)
            if self.is_masked:
                subspace = subspace.selected()
            print(subspace.get_selection(), dataset.get_selection("default"), subspace.is_masked, self.is_masked)
            import time
            t0 = time.time()
            def timit(o, dataset=dataset, t0=t0):
                print("took %s %f" % (dataset.server.hostname, time.time() - t0))
                return o
            def error(e, dataset=dataset):
                print("issues with %s (%r)" % (dataset.server.hostname, e))
                try:
                    raise e
                except:
                    logger.exception("error in error handler")
            #subspace.histogram(limits, size=size, weight=weight).then(timit, error)
            f = getattr(subspace, name)
            promise = f(*args, **kwargs).then(timit, error)
            promises.append(promise)
        return aplus.listPromise(promises)

    def minmax(self):
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
        @delayed
        def reduce_minmaxes(minmaxes):
            if None in minmaxes:
                raise ValueError("one of the results are invalid")
            return reduce(min_max_reduce, minmaxes)
        minmaxes = self._apply_all("minmax")
        promise = reduce_minmaxes(minmaxes)
        task = vaex.dataset.Task()
        promise.then(task.fulfill)
        return self._task(task)#, progressbar=progressbar)

    def histogram(self, limits, size=256, weight=None, progressbar=False, group_by=None, group_limits=None, delay=None):
        @delayed
        def sum(grids):
            if None in grids:
                raise ValueError("one of the results are invalid")
            return np.sum(grids, axis=0)
        promise = sum(self._apply_all("histogram", limits=limits, size=size, weight=weight))
        task = vaex.dataset.Task()
        promise.then(task.fulfill)
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

    def dtype(self, expression):
        if expression in self.dtypes:
            return self.dtypes[expression]
        else:
            return np.zeros(1, dtype=np.float64).dtype
    def is_local(self): return False
    def __call__(self, *expressions, **kwargs):
        return SubspaceDistributed(self, expressions, kwargs.get("executor") or self.executor, async=kwargs.get("async", False))

if __name__ == "__main__":
	sys.exit(main(sys.argv))

import vaex.settings
import vaex as vx
import socket

try:
	from urllib.parse import urlparse
except ImportError:
	from urlparse import urlparse

def open(url, thread_mover=None):
    url = urlparse(url)
    assert url.scheme in ["cluster"]
    port = url.port
    base_path = url.path
    if base_path.startswith("/"):
        base_path = base_path[1:]
    clustername = url.hostname
    clusterlist = vaex.settings.cluster.get("clusters." + clustername, None)
    if clusterlist:
        datasets = []
        for hostname in clusterlist:
            try:
                server = vx.server(hostname, thread_mover=thread_mover)
                datasets_dict = server.datasets(as_dict=True)
            except socket.error as e:
                logger.info("could not connect to %s, skipping", hostname)
            else:
                dataset = datasets_dict[base_path]
                datasets.append(dataset)
            #datasets.append(vx.server(url).datasets()[0])
        dsd=DatasetDistributed(datasets=datasets)
        return dsd

	#return vaex.remote.ServerRest(hostname, base_path=base_path, port=port, websocket=websocket, **kwargs)


def main(argv):
    import argparse
    parser = argparse.ArgumentParser(argv[0])
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument('--quiet', '-q', default=False, action='store_true', help="do not output anything")

    subparsers = parser.add_subparsers(help='type of task', dest="task")

    parser_add = subparsers.add_parser('add', help='add hosts to cluser')
    parser_add.add_argument("name", help="name of cluster")
    parser_add.add_argument("hostnames", help="hostnames", nargs="*")
    parser_add.add_argument('--reset', '-r', default=False, action='store_true', help="clear previous hosts")

    parser_check = subparsers.add_parser('check', help='check if hosts exists')
    parser_check.add_argument("name", help="name of cluster")
    parser_check.add_argument('--clean', '-c', default=False, action='store_true', help="remove hosts that are not up")

    args = parser.parse_args(argv[1:])

    verbosity = ["ERROR", "WARNING", "INFO", "DEBUG"]
    logging.getLogger("vaex").setLevel(verbosity[min(3, args.verbose)])
    quiet = args.quiet
    if args.task == "check":
        name = args.name
        clusterlist = vaex.settings.cluster.get("clusters." + name, None)
        if clusterlist is None:
            if not quiet:
                print("cluster does not exist: %s" % name)
        else:
            common = None
            for hostname in clusterlist:
                print(hostname)
                try:
                    server = vx.server(hostname)
                    datasets = server.datasets()
                except socket.error as e:
                    print("\t" + str(e))
                    if args.clean:
                        clusterlist.remove(hostname)
                else:
                    for dataset in datasets:
                        print("\t" +dataset.name)
                        #if common is None:
                    names = set([k.name for k in datasets])
                    common = names if common is None else common.union(names)
            print("Cluster: " + name + " has %d hosts connected, to connect to a dataset, use the following urls:" % (len(clusterlist))  )
            for dsname in common:
                print("\tcluster://%s/%s" % (name, dsname))
            if args.clean:
                vaex.settings.cluster.store("clusters." + name, clusterlist)
    if args.task == "add":
        name = args.name
        clusterlist = vaex.settings.cluster.get("clusters." + name, [])
        if args.reset:
            clusterlist = []
        for hostname in args.hostnames:
            if hostname not in clusterlist:
                clusterlist.append(hostname)
        vaex.settings.cluster.store("clusters." + name, clusterlist)
        if not args.quiet:
            print("hosts in cluster: %s" % name)
            for hostname in clusterlist:
                print("\t%s" % (hostname))
