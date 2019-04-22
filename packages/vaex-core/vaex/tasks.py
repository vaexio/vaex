from functools import reduce
import logging
import numpy as np

import vaex.promise
from vaex.column import str_type

from .utils import (_ensure_strings_from_expressions,
    _ensure_string_from_expression,
    _ensure_list,
    _is_limit,
    _isnumber,
    _issequence,
    _is_string,
    _parse_reduction,
    _parse_n,
    _normalize_selection_name,
    _normalize,
    _parse_f,
    _expand,
    _expand_shape,
    _expand_limits,
    as_flat_float,
    as_flat_array,
    _split_and_combine_mask)

logger = logging.getLogger('vaex.tasks')

class Task(vaex.promise.Promise):
    """
    :type: signal_progress: Signal
    """

    def __init__(self, df=None, expressions=[], name="task"):
        vaex.promise.Promise.__init__(self)
        self.df = df
        self.expressions = expressions
        self.expressions_all = list(expressions)
        self.signal_progress = vaex.events.Signal("progress (float)")
        self.progress_fraction = 0
        self.signal_progress.connect(self._set_progress)
        self.cancelled = False
        self.name = name

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


class TaskBase(Task):
    def __init__(self, df, expressions, selection=None, to_float=False, dtype=np.float64, name="TaskBase"):
        if not isinstance(expressions, (tuple, list)):
            expressions = [expressions]
        # edges include everything outside at index 1 and -1, and nan's at index 0, so we add 3 to each dimension
        self.selection_waslist, [self.selections, ] = vaex.utils.listify(selection)
        Task.__init__(self, df, expressions, name=name)
        self.to_float = to_float
        self.dtype = dtype

    def map(self, thread_index, i1, i2, *blocks):
        class Info(object):
            pass
        info = Info()
        info.i1 = i1
        info.i2 = i2
        info.first = i1 == 0
        info.last = i2 == self.df.length_unfiltered()
        info.size = i2 - i1

        masks = [np.ma.getmaskarray(block) for block in blocks if np.ma.isMaskedArray(block)]
        blocks = [block.data if np.ma.isMaskedArray(block) else block for block in blocks]
        mask = None
        if masks:
            # find all 'rows', where all columns are present (not masked)
            mask = masks[0].copy()
            for other in masks[1:]:
                mask |= other
            # masked arrays mean mask==1 is masked, for vaex we use mask==1 is used
            # blocks = [block[~mask] for block in blocks]

        if self.to_float:
            blocks = [as_flat_float(block) for block in blocks]

        for i, selection in enumerate(self.selections):
            if selection or self.df.filtered:
                selection_mask = self.df.evaluate_selection_mask(selection, i1=i1, i2=i2, cache=True)  # TODO
                if selection_mask is None:
                    raise ValueError("performing operation on selection while no selection present")
                if mask is not None:
                    selection_mask = selection_mask[~mask]
                selection_blocks = [block[selection_mask] for block in blocks]
            else:
                selection_blocks = [block for block in blocks]
            little_endians = len([k for k in selection_blocks if k.dtype.byteorder in ["<", "="]])
            if not ((len(selection_blocks) == little_endians) or little_endians == 0):
                def _to_native(ar):
                    if ar.dtype.byteorder not in ["<", "="]:
                        dtype = ar.dtype.newbyteorder()
                        return ar.astype(dtype)
                    else:
                        return ar

                selection_blocks = [_to_native(k) for k in selection_blocks]
            subblock_weight = None
            if len(selection_blocks) == len(self.expressions) + 1:
                subblock_weight = selection_blocks[-1]
                selection_blocks = list(selection_blocks[:-1])
            self.map_processed(thread_index, i1, i2, mask, *blocks)
        return i2 - i1


class TaskMapReduce(Task):
    def __init__(self, df, expressions, map, reduce, converter=lambda x: x, info=False, to_float=False,
                 to_numpy=True, ordered_reduce=False, skip_masked=False, ignore_filter=False, name="task"):
        Task.__init__(self, df, expressions, name=name)
        self._map = map
        self._reduce = reduce
        self.converter = converter
        self.info = info
        self.ordered_reduce = ordered_reduce
        self.to_float = to_float
        self.to_numpy = to_numpy
        self.skip_masked = skip_masked
        self.ignore_filter = ignore_filter

    def map(self, thread_index, i1, i2, *blocks):
        if self.to_numpy:
            blocks = [block if isinstance(block, np.ndarray) else block.to_numpy() for block in blocks]
        if self.to_float:
            blocks = [as_flat_float(block) for block in blocks]
        if self.skip_masked:
            masks = [np.ma.getmaskarray(block) for block in blocks if np.ma.isMaskedArray(block)]
            blocks = [block.data if np.ma.isMaskedArray(block) else block for block in blocks]
            mask = None
            if masks:
                # find all 'rows', where all columns are present (not masked)
                mask = masks[0].copy()
                for other in masks[1:]:
                    mask |= other
                blocks = [block[~mask] for block in blocks]

        if not self.ignore_filter:
            selection = None
            if selection or self.df.filtered:
                selection_mask = self.df.evaluate_selection_mask(selection, i1=i1, i2=i2, cache=True)
                blocks = [block[selection_mask] for block in blocks]
        if self.info:
            return self._map(thread_index, i1, i2, *blocks)
        else:
            return self._map(*blocks)  # [self.map(block) for block in blocks]

    def reduce(self, results):
        if self.ordered_reduce:
            results.sort(key=lambda x: x[0])
            results = [k[1] for k in results]
        return self.converter(reduce(self._reduce, results))


class TaskApply(TaskBase):
    def __init__(self, df, expressions, f, info=False, to_float=False, name="apply", masked=False, dtype=np.float64):
        TaskBase.__init__(self, df, expressions, selection=None, to_float=to_float, name=name)
        self.f = f
        self.dtype = dtype
        self.data = np.zeros(df.length_unfiltered(), dtype=self.dtype)
        self.mask = None
        if masked:
            self.mask = np.zeros(df.length_unfiltered(), dtype=np.bool)
            self.array = np.ma.array(self.data, mask=self.mask, shrink=False)
        else:
            self.array = self.data
        self.info = info
        self.to_float = to_float

    def map_processed(self, thread_index, i1, i2, mask, *blocks):
        if self.to_float:
            blocks = [as_flat_float(block) for block in blocks]
        print(len(self.array), i1, i2)
        for i in range(i1, i2):
            print(i)
            if mask is None or mask[i]:
                v = [block[i - i1] for block in blocks]
                self.data[i] = self.f(*v)
                if mask is not None:
                    self.mask[i] = False
            else:
                self.mask[i] = True

            print(v)
            print(self.array, self.array.dtype)
        return None

    def reduce(self, results):
        return None


# import numba
# @numba.jit(nopython=True, nogil=True)
# def histogram_numba(x, y, weight, grid, xmin, xmax, ymin, ymax):
#    scale_x = 1./ (xmax-xmin);
#    scale_y = 1./ (ymax-ymin);
#    counts_length_y, counts_length_x = grid.shape
#    for i in range(len(x)):
#        value_x = x[i];
#        value_y = y[i];
#        scaled_x = (value_x - xmin) * scale_x;
#        scaled_y = (value_y - ymin) * scale_y;
#
#        if ( (scaled_x >= 0) & (scaled_x < 1) &  (scaled_y >= 0) & (scaled_y < 1) ) :
#            index_x = (int)(scaled_x * counts_length_x);
#            index_y = (int)(scaled_y * counts_length_y);
#            grid[index_y, index_x] += 1;


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


class StatOpCov(StatOp):
    def __init__(self, code, fields=None, reduce_function=np.sum):
        super(StatOpCov, self).__init__(code, fields, reduce_function=reduce_function)

    def fields(self, weights):
        N = len(weights)
        # counts, sums, cross product sums
        return N * 2 + N**2 * 2  # ((N+1) * N) // 2 *2

class StatOpFirst(StatOp):
    def __init__(self, code):
        super(StatOpFirst, self).__init__(code, 2, reduce_function=self._reduce_function)

    def init(self, grid):
        grid[..., 0] = np.nan
        grid[..., 1] = np.inf

    def _reduce_function(self, grid, axis=0):
        values = grid[...,0]
        order_values = grid[...,1]
        indices = np.argmin(order_values, axis=0)

        # see e.g. https://stackoverflow.com/questions/46840848/numpy-how-to-use-argmax-results-to-get-the-actual-max?noredirect=1&lq=1
        # and https://jakevdp.github.io/PythonDataScienceHandbook/02.07-fancy-indexing.html
        if len(values.shape) == 2:  # no binby
            return values[indices, np.arange(values.shape[1])[:,None]][0]
        if len(values.shape) == 3:  # 1d binby
            return values[indices, np.arange(values.shape[1])[:,None], np.arange(values.shape[2])]
        if len(values.shape) == 4:  # 2d binby
            return values[indices, np.arange(values.shape[1])[:,None], np.arange(values.shape[2])[None,:,None], np.arange(values.shape[3])]
        else:
            raise ValueError('dimension %d not yet supported' % len(values.shape))

    def fields(self, weights):
        # the value found, and the value by which it is ordered
        return 2


OP_ADD1 = StatOp(0, 1)
OP_COUNT = StatOp(1, 1)
OP_MIN_MAX = StatOpMinMax(2, 2)
OP_ADD_WEIGHT_MOMENTS_01 = StatOp(3, 2, np.nansum)
OP_ADD_WEIGHT_MOMENTS_012 = StatOp(4, 3, np.nansum)
OP_COV = StatOpCov(5)
OP_FIRST = StatOpFirst(6)


class TaskStatistic(Task):
    def __init__(self, df, expressions, shape, limits, masked=False, weights=[], weight=None, op=OP_ADD1, selection=None, edges=False):
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
        Task.__init__(self, df, expressions, name="statisticNd")
        #self.dtype = np.int64 if self.op == OP_ADD1 else np.float64 # TODO: use int64 fir count and ADD1
        self.dtype = np.float64
        self.masked = masked

        self.fields = op.fields(weights)
        self.shape_total = (self.df.executor.thread_pool.nthreads,) + (len(self.selections), ) + self.shape + (self.fields,)
        self.grid = np.zeros(self.shape_total, dtype=self.dtype)
        self.op.init(self.grid)
        self.minima = []
        self.maxima = []
        limits = np.array(self.limits)
        if len(limits) != 0:
            logger.debug("limits = %r", limits)
            assert limits.shape[-1] == 2, "expected last dimension of limits to have a length of 2 (not %d, total shape: %s), of the form [[xmin, xmin], ... [zmin, zmax]], not %s" % (limits.shape[-1], limits.shape, limits)
            if len(limits.shape) == 1:  # short notation: [xmin, max], instead of [[xmin, xmax]]
                limits = [limits]
            logger.debug("limits = %r", limits)
            for limit in limits:
                vmin, vmax = limit
                self.minima.append(float(vmin))
                self.maxima.append(float(vmax))
        # if self.weight is not None:
        self.expressions_all.extend(weights)

    def __repr__(self):
        name = self.__class__.__module__ + "." + self.__class__.__name__
        return "<%s(df=%r, expressions=%r, shape=%r, limits=%r, weights=%r, selections=%r, op=%r)> instance at 0x%x" % (name, self.df, self.expressions, self.shape, self.limits, self.weights, self.selections, self.op, id(self))


    def map(self, thread_index, i1, i2, *blocks):
        class Info(object):
            pass
        info = Info()
        info.i1 = i1
        info.i2 = i2
        info.first = i1 == 0
        info.last = i2 == self.df.length_unfiltered()
        info.size = i2 - i1

        masks = [np.ma.getmaskarray(block) for block in blocks if np.ma.isMaskedArray(block)]
        blocks = [block.data if np.ma.isMaskedArray(block) else block for block in blocks]
        mask = None

        #blocks = [as_flat_float(block) for block in blocks]
        if len(blocks) != 0:
            dtype = np.find_common_type([block.dtype for block in blocks], [])
            histogram2d = vaex.vaexfast.histogram2d
            if dtype.str in ">f8 <f8 =f8":
                statistic_function = vaex.vaexfast.statisticNd_f8
            elif dtype.str in ">f4 <f4 =f4":
                statistic_function = vaex.vaexfast.statisticNd_f4
                histogram2d = vaex.vaexfast.histogram2d_f4
            elif dtype.str in ">i8 <i8 =i8":
                dtype = np.dtype(np.float64)
                statistic_function = vaex.vaexfast.statisticNd_f8
            else:
                dtype = np.dtype(np.float32)
                statistic_function = vaex.vaexfast.statisticNd_f4
                histogram2d = vaex.vaexfast.histogram2d_f4
            #print(dtype, statistic_function, histogram2d)

        if masks:
            mask = masks[0].copy()
            for other in masks[1:]:
                mask |= other
            blocks = [block[~mask] for block in blocks]

        this_thread_grid = self.grid[thread_index]
        for i, selection in enumerate(self.selections):
            if selection or self.df.filtered:
                selection_mask = self.df.evaluate_selection_mask(selection, i1=i1, i2=i2, cache=True)  # TODO
                if selection_mask is None:
                    raise ValueError("performing operation on selection while no selection present")
                if mask is not None:
                    selection_mask = selection_mask[~mask]
                selection_blocks = [block[selection_mask] for block in blocks]
            else:
                selection_blocks = [block for block in blocks]
            little_endians = len([k for k in selection_blocks if k.dtype != str_type and k.dtype.byteorder in ["<", "="]])
            if not ((len(selection_blocks) == little_endians) or little_endians == 0):
                def _to_native(ar):
                    if ar.dtype == str_type:
                        return ar  # string are always fine
                    if ar.dtype.byteorder not in ["<", "="]:
                        dtype = ar.dtype.newbyteorder()
                        return ar.astype(dtype)
                    else:
                        return ar

                selection_blocks = [_to_native(k) for k in selection_blocks]
            subblock_weight = None
            subblock_weights = selection_blocks[len(self.expressions):]
            selection_blocks = list(selection_blocks[:len(self.expressions)])
            if len(selection_blocks) == 0 and subblock_weights == []:
                if self.op == OP_ADD1:  # special case for counting '*' (i.e. the number of rows)
                    if selection or self.df.filtered:
                        this_thread_grid[i][0] += np.sum(selection_mask)
                    else:
                        this_thread_grid[i][0] += i2 - i1
                else:
                    raise ValueError("Nothing to compute for OP %s" % self.op.code)
            # special case for counting string values etc
            elif len(selection_blocks) == 0 and len(subblock_weights) == 1 and self.op in [OP_COUNT]\
                    and (subblock_weights[0].dtype == str_type or subblock_weights[0].dtype.kind not in 'biuf'):
                weight = subblock_weights[0]
                mask = None
                if weight.dtype != str_type:
                    if weight.dtype.kind == 'O':
                        mask = vaex.strings.StringArray(weight).mask()
                else:
                    mask = weight.get_mask()
                if selection or self.df.filtered:
                    if mask is not None:
                        this_thread_grid[i][0] += np.sum(~mask)
                    else:
                        this_thread_grid[i][0] += np.sum(selection_mask)
                else:
                    if mask is not None:
                        this_thread_grid[i][0] += len(mask) - mask.sum()
                    else:
                        this_thread_grid[i][0] += len(weight)
            else:
                #blocks = list(blocks)  # histogramNd wants blocks to be a list
                # if False: #len(selection_blocks) == 2 and self.op == OP_ADD1:  # special case, slighty faster
                #     #print('fast case!')
                #     assert len(subblock_weights) <= 1
                #     histogram2d(selection_blocks[0], selection_blocks[1], subblock_weights[0] if len(subblock_weights) else None,
                #                 this_thread_grid[i,...,0],
                #                 self.minima[0], self.maxima[0], self.minima[1], self.maxima[1])
                # else:
                    selection_blocks = [as_flat_array(block, dtype) for block in selection_blocks]
                    subblock_weights = [as_flat_array(block, dtype) for block in subblock_weights]
                    statistic_function(selection_blocks, subblock_weights, this_thread_grid[i], self.minima, self.maxima, self.op.code, self.edges)
        return i2 - i1
        # return map(self._map, blocks)#[self.map(block) for block in blocks]

    def reduce(self, results):
        # for i in range(1, self.subspace.executor.thread_pool.nthreads):
        #   self.data[0] += self.data[i]
        # return self.data[0]
        # return self.data
        grid = self.op.reduce(self.grid)
        # If selection was a string, we just return the single selection
        return grid if self.selection_waslist else grid[0]


class TaskAggregate(Task):
    def __init__(self, df, grid):
        expressions = [binner.expression for binner in grid.binners]
        Task.__init__(self, df, expressions, name="statisticNd")

        self.df = df
        self.parent_grid = grid
        self.nthreads = self.df.executor.thread_pool.nthreads
        # for each thread, we have 1 grid and a set of binners
        self.grids = [vaex.superagg.Grid([binner.copy() for binner in grid.binners]) for i in range(self.nthreads)]
        self.aggregations = []
        # self.grids = []

    def add_aggregation_operation(self, aggregator_descriptor, selection=None, edges=False):
        selection_waslist = _issequence(selection)
        selections = _ensure_list(selection)
        def create_aggregator(thread_index):
            # for each selection, we have a separate aggregator, sharing the grid and binners
            return [aggregator_descriptor._create_operation(self.df, self.grids[thread_index]) for selection in selections]
        task = Task(self.df, [], "--")
        self.aggregations.append((aggregator_descriptor, selections, [create_aggregator(i) for i in range(self.nthreads)], selection_waslist, edges, task))
        self.expressions_all.extend(aggregator_descriptor.expressions)
        self.expressions_all = list(set(self.expressions_all))
        self.dtypes = {expr: self.df.dtype(expr) for expr in self.expressions_all}
        return task

    def map(self, thread_index, i1, i2, *blocks):
        if not self.aggregations:
            raise RuntimeError('Aggregation tasks started but nothing to do, maybe adding operations failed?')
        grid = self.grids[thread_index]
        def check_array(x, dtype):
            if dtype == str_type:
                x = vaex.column._to_string_sequence(x)
            else:
                x = vaex.utils.as_contiguous(x)
                if x.dtype.kind in "mM":
                    # we pass datetime as int
                    x = x.view('uint64')
            return x
        block_map = {expr: block for expr, block in zip(self.expressions_all, blocks)}
        # we need to make sure we keep some objects alive, since the c++ side does not incref
        # on set_data and set_data_mask
        references = []
        for binner in grid.binners:
            block = block_map[binner.expression]
            dtype = self.dtypes[binner.expression]
            block = check_array(block, dtype)
            if np.ma.isMaskedArray(block):
                block, mask = block.data, np.ma.getmaskarray(block)
                binner.set_data(block)
                binner.set_data_mask(mask)
                references.extend([block, mask])
            else:
                binner.set_data(block)
                references.extend([block])
        all_aggregators = []
        for agg_desc, selections, aggregation2d, selection_waslist, edges, task in self.aggregations:
            for selection_index, selection in enumerate(selections):
                agg = aggregation2d[thread_index][selection_index]
                all_aggregators.append(agg)
                selection_mask = None
                if selection or self.df.filtered:
                    selection_mask = self.df.evaluate_selection_mask(selection, i1=i1, i2=i2, cache=True)  # TODO
                if agg_desc.expressions:
                    assert len(agg_desc.expressions) in [1,2], "only length 1 or 2 supported for now"
                    dtype_ref = block = block_map[agg_desc.expressions[0]].dtype
                    for i, expression in enumerate(agg_desc.expressions):
                        block = block_map[agg_desc.expressions[i]]
                        dtype = self.dtypes[agg_desc.expressions[i]]
                        # we have data for the aggregator as well
                        if np.ma.isMaskedArray(block):
                            block, mask = block.data, np.ma.getmaskarray(block)
                            block = check_array(block, dtype)
                            agg.set_data(block, i)
                            references.extend([block])
                            if selection_mask is None:
                                selection_mask = ~mask
                            else:
                                selection_mask = selection_mask & ~mask
                        else:
                            block = check_array(block, dtype)
                            agg.set_data(block, i)
                            references.extend([block])
                # we only have 1 data mask, since it's locally combined
                if selection_mask is not None:
                    agg.set_data_mask(selection_mask)
                    references.extend([selection_mask])
        grid.bin(all_aggregators, i2-i1)

    def reduce(self, results):
        results = []
        for agg_desc, selections, aggregation2d, selection_waslist, edges, task in self.aggregations:
            grids = []
            for selection_index, selection in enumerate(selections):
                agg0 = aggregation2d[0][selection_index]
                agg0.reduce([k[selection_index] for k in aggregation2d[1:]])
                grid = np.asarray(agg0)
                if not edges:
                    grid = vaex.utils.extract_central_part(grid)
                grids.append(grid)
            result = np.asarray(grids) if selection_waslist else grids[0]
            dtype_out = vaex.utils.to_native_dtype(agg_desc.dtype_out)
            result = result.view(dtype_out)
            task.fulfill(result)
            results.append(result)
        return results
