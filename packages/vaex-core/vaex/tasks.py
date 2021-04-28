import logging
import numpy as np

import vaex.promise
import vaex.encoding
from .utils import _expand_shape
from .datatype import DataType


logger = logging.getLogger('vaex.tasks')
_task_register = {}


@vaex.encoding.register("task")
class task_encoder:
    @staticmethod
    def encode(encoding, task):
        return task.encode(encoding)

    @staticmethod
    def decode(encoding, spec, df):
        cls = _task_register[spec['task']]
        return cls.decode(encoding, spec, df)


def register(cls):
    _task_register[cls.name] = cls
    return cls


class Task(vaex.promise.Promise):
    """
    :type: signal_progress: Signal
    """

    def __init__(self, df=None, expressions=[], pre_filter=False, name="task"):
        vaex.promise.Promise.__init__(self)
        self.df = df
        self.expressions = expressions
        self.expressions_all = list(expressions)
        self.signal_progress = vaex.events.Signal("progress (float)")
        self.progress_fraction = 0
        self.signal_progress.connect(self._set_progress)
        self.cancelled = False
        self.name = name
        self.pre_filter = pre_filter
        self.result = None

    def _set_progress(self, fraction):
        self.progress_fraction = fraction
        return not self.cancelled  # don't cancel

    def cancel(self):
        self.cancelled = True

    @property
    def dimension(self):
        return len(self.expressions)

    @classmethod
    def create(cls):
        ret = Task()
        return ret

    def create_next(self):
        ret = Task(self.df, [])
        self.signal_progress.connect(ret.signal_progress.emit)
        return ret

    # def __repr__(self):
    #     from .encoding import Encoding
    #     encoding = Encoding()
    #     return json.dumps({key: repr(value) for key, value in self.encode(encoding).items()})


# only used for testing
@register
class TaskSum(Task):
    name = "sum-test"

    # def __init__(self, df, expression):
    #     super().__init__(df, expression)
    #     self.expression = expression

    def encode(self, encoding):
        return {'task': type(self).name, 'expression': self.expressions}

    @classmethod
    def decode(cls, encoding, spec, df):
        return cls(df, expression=[spec['expression']])


@register
class TaskMapReduce(Task):
    name = "map_reduce"

    def __init__(self, df, expressions, map, reduce, info=False, to_float=False,
                 to_numpy=True, skip_masked=False, ignore_filter=False, selection=None, pre_filter=False, name="task"):
        Task.__init__(self, df, expressions, name=name, pre_filter=pre_filter)
        self._map = map
        self._reduce = reduce
        self.info = info
        self.to_float = to_float
        self.to_numpy = to_numpy
        self.skip_masked = skip_masked
        self.ignore_filter = ignore_filter
        if self.pre_filter and self.ignore_filter:
            raise ValueError("Cannot pre filter and also ignore the filter")
        self.selection = selection

    def encode(self, encoding):
        return {'task': type(self).name, 'expressions': self.expressions, 'map': self._map, 'reduce': self._reduce,
                'info': self.info, 'to_float': self.to_float, 'to_numpy': self.to_numpy,  # 'ordered_reduce': self.ordered_reduce,
                'skip_masked': self.skip_masked, 'ignore_filter': self.ignore_filter, 'selection': self.selection, 'pre_filter': self.pre_filter,
                }


class StatOp(object):
    def __init__(self, code, fields, reduce_function=np.nansum, dtype=None):
        self.code = code
        self.fixed_fields = fields
        self.reduce_function = reduce_function
        self.dtype = dtype

    def init(self, grid):
        pass

    def fields(self, weights):
        return self.fixed_fields

    def reduce(self, grid, axis=0):
        value = self.reduce_function(grid, axis=axis)
        if self.dtype:
            return value.astype(self.dtype)
        else:
            return value

    def encode(self, encoding):
        return {'code': self.code, 'fields': self.fixed_fields, 'reduce_function': self.reduce_function.__name__}


class StatOpMinMax(StatOp):
    def __init__(self, code, fields):
        super(StatOpMinMax, self).__init__(code, fields)

    def init(self, grid):
        grid[..., 0] = np.inf
        grid[..., 1] = -np.inf

    def reduce(self, grid, axis=0):
        out = np.zeros(grid.shape[1:], dtype=grid.dtype)
        out[..., 0] = np.nanmin(grid[..., 0], axis=axis)
        out[..., 1] = np.nanmax(grid[..., 1], axis=axis)
        return out

    def encode(self, encoding):
        return {'code': self.code, 'fields': self.fixed_fields}


class StatOpCov(StatOp):
    def __init__(self, code, fields=None, reduce_function=np.sum):
        super(StatOpCov, self).__init__(code, fields, reduce_function=reduce_function)

    def fields(self, weights):
        N = len(weights)
        # counts, sums, cross product sums
        return N * 2 + N**2 * 2  # ((N+1) * N) // 2 *2

    def encode(self, encoding):
        return {'code': self.code, 'reduce_function': self.reduce_function.__name__}


class StatOpFirst(StatOp):
    def __init__(self, code):
        super(StatOpFirst, self).__init__(code, 2, reduce_function=self._reduce_function)

    def init(self, grid):
        grid[..., 0] = np.nan
        grid[..., 1] = np.inf

    def _reduce_function(self, grid, axis=0):
        values = grid[..., 0]
        order_values = grid[..., 1]
        indices = np.argmin(order_values, axis=0)

        # see e.g. https://stackoverflow.com/questions/46840848/numpy-how-to-use-argmax-results-to-get-the-actual-max?noredirect=1&lq=1
        # and https://jakevdp.github.io/PythonDataScienceHandbook/02.07-fancy-indexing.html
        if len(values.shape) == 2:  # no binby
            return values[indices, np.arange(values.shape[1])[:, None]][0]
        if len(values.shape) == 3:  # 1d binby
            return values[indices, np.arange(values.shape[1])[:, None], np.arange(values.shape[2])]
        if len(values.shape) == 4:  # 2d binby
            return values[indices, np.arange(values.shape[1])[:, None], np.arange(values.shape[2])[None, :, None], np.arange(values.shape[3])]
        else:
            raise ValueError('dimension %d not yet supported' % len(values.shape))

    def fields(self, weights):
        # the value found, and the value by which it is ordered
        return 2

    def encode(self, encoding):
        return {'code': self.code, 'reduce_function': self.reduce_function.__name__}


@vaex.encoding.register('_op')
class op_encoding:
    @staticmethod
    def encode(encoding, op):
        return op.encode(encoding)

    @staticmethod
    def decode(encoding, op_spec):
        op_spec = op_spec.copy()
        if 'reduce_function' in op_spec:
            op_spec['reduce_function'] = getattr(np, op_spec.pop('reduce_function'))
        cls = StatOp
        if op_spec['code'] == 2:
            cls = StatOpMinMax
        if op_spec['code'] == 5:
            cls = StatOpCov
        if op_spec['code'] == 6:
            cls = StatOpFirst
        return cls(**op_spec)


OP_ADD1 = StatOp(0, 1)
OP_COUNT = StatOp(1, 1)
OP_MIN_MAX = StatOpMinMax(2, 2)
OP_ADD_WEIGHT_MOMENTS_01 = StatOp(3, 2, np.nansum)
OP_ADD_WEIGHT_MOMENTS_012 = StatOp(4, 3, np.nansum)
OP_COV = StatOpCov(5)
OP_FIRST = StatOpFirst(6)


@register
class TaskStatistic(Task):
    name = "legacy_statistic"

    def encode(self, encoding):
        return {'task': type(self).name, 'expressions': self.expressions,
                'shape': self.shape, 'selections': self.selections, 'op': encoding.encode('_op', self.op), 'weights': self.weights,
                'dtype': encoding.encode('dtype', DataType(self.dtype)), 'minima': self.minima, 'maxima': self.maxima, 'edges': self.edges,
                'selection_waslist': self.selection_waslist}

    @classmethod
    def decode(cls, encoding, spec, df):
        spec = spec.copy()
        del spec['task']
        spec['op'] = encoding.decode('_op', spec['op'])
        spec['dtype'] = encoding.decode('dtype', spec['dtype'])
        selection_waslist = spec.pop('selection_waslist')
        if selection_waslist:
            spec['selection'] = spec.pop('selections')
        else:
            spec['selection'] = spec.pop('selections')[0]
        spec['limits'] = list(zip(spec.pop('minima'), spec.pop('maxima')))
        return cls(df, **spec)

    def __init__(self, df, expressions, shape, limits, masked=False, weights=[], weight=None, op=OP_ADD1, selection=None, edges=False,
                 dtype=np.dtype('f8')):
        if not isinstance(expressions, (tuple, list)):
            expressions = [expressions]
        # edges include everything outside at index 1 and -1, and nan's at index 0, so we add 3 to each dimension
        self.shape = tuple([k + 3 if edges else k for k in _expand_shape(shape, len(expressions))])
        self.limits = limits
        if weight is not None:  # shortcut for weights=[weight]
            assert weights == [], 'only provide weight or weights, not both'
            weights = [weight]
            del weight
        self.weights = weights
        self.selection_waslist, [self.selections, ] = vaex.utils.listify(selection)
        self.op = op
        self.edges = edges
        Task.__init__(self, df, expressions, name="statisticNd", pre_filter=df.filtered)
        # self.dtype = np.int64 if self.op == OP_ADD1 else np.float64 # TODO: use int64 fir count and ADD1
        self.dtype = dtype
        self.masked = masked

        self.fields = op.fields(weights)
        # self.shape_total = (self.df.executor.thread_pool.nthreads,) + (len(self.selections), ) + self.shape + (self.fields,)
        # self.grid = np.zeros(self.shape_total, dtype=self.dtype)
        # self.op.init(self.grid)
        self.minima = []
        self.maxima = []
        limits = np.array(self.limits)
        if len(limits) != 0:
            logger.debug("limits = %r", limits)
            assert limits.shape[-1] == 2, "expected last dimension of limits to have a length of 2 (not %d, total shape: %s), of the form [[xmin, xmin], ... [zmin, zmax]], not %s" %\
                                          (limits.shape[-1], limits.shape, limits)
            if len(limits.shape) == 1:  # short notation: [xmin, max], instead of [[xmin, xmax]]
                limits = [limits]
            logger.debug("limits = %r", limits)
            for limit in limits:
                vmin, vmax = limit
                self.minima.append(float(vmin))
                self.maxima.append(float(vmax))
        # if self.weight is not None:
        self.expressions_all.extend(weights)


@register
class TaskAggregations(Task):
    """Multiple aggregations on a single grid."""
    name = "aggregations"

    def __init__(self, df, grid):
        expressions = [binner.expression for binner in grid.binners]
        self.df = df
        self.parent_grid = grid
        self.aggregation_descriptions = []
        self.dtypes = {}
        Task.__init__(self, df, expressions, name="statisticNd", pre_filter=df.filtered)

    def encode(self, encoding):
        # TODO: get rid of dtypes
        return {'task': type(self).name,
                'grid': encoding.encode('grid', self.parent_grid),
                'aggregations': encoding.encode_list("aggregation", self.aggregation_descriptions),
                'dtypes': encoding.encode_dict("dtype", self.dtypes)
                }

    @classmethod
    def decode(cls, encoding, spec, df):
        grid = encoding.decode('grid', spec['grid'])
        task = cls(df, grid)
        aggs = encoding.decode_list('aggregation', spec['aggregations'])
        for agg in aggs:
            agg._prepare_types(df)
            task.add_aggregation_operation(agg)
        return task

    def add_aggregation_operation(self, aggregator_descriptor):
        task = Task(self.df, [], "--")

        def chain_reject(x):
            task.reject(x)
            return x

        def assign_subtask(values, index=len(self.aggregation_descriptions)):
            task.fulfill(values[index])
        self.then(assign_subtask, chain_reject)

        self.aggregation_descriptions.append((aggregator_descriptor))
        # THIS SHOULD BE IN THE SAME ORDER AS THE ABOVE TASKPART
        # it is up the the executor to remove duplicate expressions
        self.expressions_all.extend(aggregator_descriptor.expressions)
        # TODO: optimize/remove?
        self.dtypes = {expr: self.df.data_type(expr) for expr in self.expressions_all}
        return task

    def check(self):
        if not self.aggregation_descriptions:
            raise RuntimeError('Aggregation tasks started but nothing to do, maybe adding operations failed?')
