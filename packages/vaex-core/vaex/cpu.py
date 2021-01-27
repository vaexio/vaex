"""Implementation all tasks parts for the cpu"""

from functools import reduce
import numpy as np

import vaex
import vaex.encoding
from .utils import as_flat_float, as_flat_array, _issequence, _ensure_list
from . import array_types
from .array_types import filter


_task_part_register = {}


def register(cls):
    assert cls is not None
    _task_part_register[cls.name] = cls
    return cls


def create_part_from_spec(df, spec):
    cls = _task_part_register[spec['task']]
    part_cls = cls.part_class
    del spec['task']
    return part_cls.from_spec(df, spec)


@vaex.encoding.register("task-part-cpu")
class encoder:
    @staticmethod
    def encode(encoding, task):
        return task.encode(encoding)

    @staticmethod
    def decode(encoding, spec, df):
        cls = _task_part_register[spec['task']]
        return cls.decode(encoding, spec, df)


class TaskPart:
    def __init__(self, df, expressions, name, pre_filter):
        self.df = df
        self.expressions = expressions
        self.name = name
        self.pre_filter = pre_filter


@register
class TaskPartSum:
    name = "sum-test"

    def __init__(self, expression):
        self.total = 0
        self.expression = expression

    @property
    def expressions(self):
        return [self.expression]

    def get_result(self):
        return self.total

    def process(self, thread_index, i1, i2, filter_mask, chunk):
        self.total += chunk.sum()

    def reduce(self, others):
        self.total += sum(other.total for other in others)

    @classmethod
    def decode(cls, encoding, spec, df):
        return cls(spec['expression'])


@register
class TaskPartMapReduce(TaskPart):
    name = "map_reduce"

    def __init__(self, df, expressions, map, reduce, converter=lambda x: x, info=False, to_float=False,
                 to_numpy=True, ordered_reduce=False, skip_masked=False, ignore_filter=False, selection=None, pre_filter=False, name="task"):
        TaskPart.__init__(self, df, expressions, name=name, pre_filter=pre_filter)

        self._map = map
        self._reduce = reduce
        self.converter = converter
        self.info = info
        self.ordered_reduce = ordered_reduce
        self.to_float = to_float
        self.to_numpy = to_numpy
        self.skip_masked = skip_masked
        self.ignore_filter = ignore_filter
        if self.pre_filter and self.ignore_filter:
            raise ValueError("Cannot pre filter and also ignore the filter")
        self.selection = selection
        self.values = []

    def process(self, thread_index, i1, i2, filter_mask, *blocks):
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
            selection = self.selection
            if self.pre_filter:
                if selection:
                    selection_mask = self.df.evaluate_selection_mask(selection, i1=i1, i2=i2, cache=True)
                    blocks = [filter(block, selection_mask) for block in blocks]
            else:
                if selection or self.df.filtered:
                    selection_mask = self.df.evaluate_selection_mask(selection, i1=i1, i2=i2, cache=True, pre_filtered=False)
                    if filter_mask is not None:
                        selection_mask = selection_mask & filter_mask
                    blocks = [filter(block, selection_mask) for block in blocks]
        if self.info:
            self.values.append(self._map(thread_index, i1, i2, *blocks))
        else:
            self.values.append(self._map(*blocks))

    def reduce(self, others):
        # if self.ordered_reduce:
        #     results.sort(key=lambda x: x[0])
        #     results = [k[1] for k in results]
        # return self.converter(reduce(self._reduce, results))
        values = self.values
        for other in others:
            values.extend(other.values)
        if values:
            self.values = reduce(self._reduce, values)

    def get_result(self):
        return self.values

    @classmethod
    def decode(cls, encoding, spec, df):
        spec = spec.copy()
        del spec['task']
        return cls(df, **spec)


@register
class TaskPartStatistic:
    name = "legacy_statistic"

    def __init__(self, df, shape, expressions, dtype, selections, op, weights, minima, maxima, edges, selection_waslist):
        self.df = df
        self.shape = shape
        self.dtype = dtype
        self.expressions = expressions
        self.op = op
        self.selections = selections
        self.fields = op.fields(weights)
        self.shape_total = (len(self.selections), ) + self.shape + (self.fields,)
        self.grid = np.zeros(self.shape_total, dtype=self.dtype.numpy)
        self.op.init(self.grid)
        self.minima = minima
        self.maxima = maxima
        self.edges = edges
        self.selection_waslist = selection_waslist

    def process(self, thread_index, i1, i2, filter_mask, *blocks):
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
        blocks = [np.asarray(k) for k in blocks]
        mask = None

        # blocks = [as_flat_float(block) for block in blocks]
        if len(blocks) != 0:
            dtype = np.find_common_type([block.dtype for block in blocks], [])
            if dtype.str in ">f8 <f8 =f8":
                statistic_function = vaex.vaexfast.statisticNd_f8
            elif dtype.str in ">f4 <f4 =f4":
                statistic_function = vaex.vaexfast.statisticNd_f4
            elif dtype.str in ">i8 <i8 =i8":
                dtype = np.dtype(np.float64)
                statistic_function = vaex.vaexfast.statisticNd_f8
            else:
                dtype = np.dtype(np.float32)
                statistic_function = vaex.vaexfast.statisticNd_f4
            # print(dtype, statistic_function, histogram2d)

        if masks:
            mask = masks[0].copy()
            for other in masks[1:]:
                mask |= other
            blocks = [block[~mask] for block in blocks]

        this_thread_grid = self.grid
        for i, selection in enumerate(self.selections):
            if selection:
                selection_mask = self.df.evaluate_selection_mask(selection, i1=i1, i2=i2, cache=True)  # TODO
                if selection_mask is None:
                    raise ValueError("performing operation on selection while no selection present")
                if mask is not None:
                    selection_mask = selection_mask[~mask]
                selection_blocks = [block[selection_mask] for block in blocks]
            else:
                selection_blocks = [block for block in blocks]
            little_endians = len([k for k in selection_blocks if k.dtype != str and k.dtype.byteorder in ["<", "="]])
            if not ((len(selection_blocks) == little_endians) or little_endians == 0):
                def _to_native(ar):
                    if ar.dtype == str:
                        return ar  # string are always fine
                    if ar.dtype.byteorder not in ["<", "="]:
                        dtype = ar.dtype.newbyteorder()
                        return ar.astype(dtype)
                    else:
                        return ar

                selection_blocks = [_to_native(k) for k in selection_blocks]
            # subblock_weight = None
            subblock_weights = selection_blocks[len(self.expressions):]
            selection_blocks = list(selection_blocks[:len(self.expressions)])
            if len(selection_blocks) == 0 and subblock_weights == []:
                if self.op == vaex.tasks.OP_ADD1:  # special case for counting '*' (i.e. the number of rows)
                    if selection or self.df.filtered:
                        this_thread_grid[i][0] += np.sum(selection_mask)
                    else:
                        this_thread_grid[i][0] += i2 - i1
                else:
                    raise ValueError("Nothing to compute for OP %s" % self.op.code)
            # special case for counting string values etc
            elif len(selection_blocks) == 0 and len(subblock_weights) == 1 and self.op in [vaex.tasks.OP_COUNT]\
                    and (subblock_weights[0].dtype == str or subblock_weights[0].dtype.kind not in 'biuf'):
                weight = subblock_weights[0]
                mask = None
                if weight.dtype != str:
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
                selection_blocks = [as_flat_array(block, dtype) for block in selection_blocks]
                subblock_weights = [as_flat_array(block, dtype) for block in subblock_weights]
                statistic_function(selection_blocks, subblock_weights, this_thread_grid[i], self.minima, self.maxima, self.op.code, self.edges)
        return i2 - i1
        # return map(self._map, blocks)#[self.map(block) for block in blocks]

    def reduce(self, others):
        grids = [self.grid] + [k.grid for k in others]
        self.grid = self.op.reduce(np.array(grids))
        # If selection was a string, we just return the single selection

    def get_result(self):
        return self.grid if self.selection_waslist else self.grid[0]

    @classmethod
    def decode(cls, encoding, spec, df):
        spec = spec.copy()
        del spec['task']
        spec['op'] = encoding.decode('_op', spec['op'])
        spec['dtype'] = encoding.decode('dtype', spec['dtype'])
        return cls(df, **spec)


@register
class TaskPartAggregations:
    name = "aggregations"

    def __init__(self, df, grid, aggregation_descriptions, dtypes, initial_values=None):
        self.df = df
        self.has_values = False
        self.dtypes = dtypes
        # self.expressions_all = expressions
        self.expressions = [binner.expression for binner in grid.binners]
        # TODO: selection and edges in descriptor?
        self.aggregation_descriptions = aggregation_descriptions
        for aggregator_descriptor in self.aggregation_descriptions:
            self.expressions.extend(aggregator_descriptor.expressions)
        # self.expressions = list(set(expressions))
        self.grid = vaex.superagg.Grid([binner.copy() for binner in grid.binners])

        def create_aggregator(aggregator_descriptor, selections, initial_values):
            # for each selection, we have a separate aggregator, sharing the grid and binners
            for i, selection in enumerate(selections):
                agg = aggregator_descriptor._create_operation(self.df, self.grid)
                if initial_values is not None:
                    print(np.asarray(agg))
                    print(initial_values[i])
                    # np.asarray(agg)[:] = initial_values[i]
                    np.copyto(np.asarray(agg), initial_values[i])
                yield agg

        self.aggregations = []
        for i, aggregator_descriptor in enumerate(self.aggregation_descriptions):
            selection = aggregator_descriptor.selection
            selection_waslist = _issequence(selection)
            selections = _ensure_list(selection)
            initial_values_i = initial_values[i] if initial_values else None
            self.aggregations.append((aggregator_descriptor, selections, list(create_aggregator(aggregator_descriptor, selections, initial_values_i)), selection_waslist))

    def process(self, thread_index, i1, i2, filter_mask, *blocks):
        # self.check()
        grid = self.grid

        def check_array(x, dtype):
            if vaex.array_types.is_string_type(dtype):
                x = vaex.column._to_string_sequence(x)
            else:
                x = vaex.utils.as_contiguous(x)
                if x.dtype.kind in "mM":
                    # we pass datetime as int
                    x = x.view('uint64')
            return x

        N = i2 - i1
        if filter_mask is not None:
            if blocks:
                N = len(blocks[0])
            else:
                N = filter_mask.sum()
        blocks = [array_types.to_numpy(block, strict=False) for block in blocks]
        for block in blocks:
            assert len(block) == N, f'Oops, got a block of length {len(block)} while it is expected to be of length {N} (at {i1}-{i2}, filter={filter_mask is not None})'
        block_map = {expr: block for expr, block in zip(self.expressions, blocks)}
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
                binner.clear_data_mask()
                references.extend([block])
        all_aggregators = []
        for agg_desc, selections, aggregation2d, selection_waslist in self.aggregations:
            for selection_index, selection in enumerate(selections):
                agg = aggregation2d[selection_index]
                all_aggregators.append(agg)
                selection_mask = None
                if not (selection is None or selection is False):
                    selection_mask = self.df.evaluate_selection_mask(selection, i1=i1, i2=i2, cache=True)  # TODO
                    # TODO: we probably want a way to avoid a to numpy conversion?
                    selection_mask = np.asarray(selection_mask)
                    references.append(selection_mask)
                    # some aggregators make a distiction between missing value and no value
                    # like nunique, they need to know if they should take the value into account or not
                    if hasattr(agg, 'set_selection_mask'):
                        agg.set_selection_mask(selection_mask)
                if agg_desc.expressions:
                    assert len(agg_desc.expressions) in [1, 2], "only length 1 or 2 supported for now"
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
                            references.append(selection_mask)
                        else:
                            block = check_array(block, dtype)
                            agg.set_data(block, i)
                            references.extend([block])
                # we only have 1 data mask, since it's locally combined
                if selection_mask is not None:
                    agg.set_data_mask(selection_mask)
                    references.extend([selection_mask])
                else:
                    agg.clear_data_mask()
        grid.bin(all_aggregators, N)
        self.has_values = True

    def reduce(self, others):
        for agg_index, (agg_desc, selections, aggregation, selection_waslist) in enumerate(self.aggregations):
            for selection_index, selection in enumerate(selections):
                agg0 = aggregation[selection_index]
                aggs = [other.aggregations[agg_index][2][selection_index] for other in others]
                agg0.reduce(aggs)

    def get_result(self):
        results = []
        for agg_index, (agg_desc, selections, aggregation, selection_waslist) in enumerate(self.aggregations):
            grids = []
            for selection_index, selection in enumerate(selections):
                grid = agg_desc.get_result(aggregation[selection_index])
                grids.append(grid)
            result = np.asarray(grids) if selection_waslist else grids[0]
            if agg_desc.dtype_out != str:
                dtype_out = agg_desc.dtype_out.to_native()
                result = result.view(dtype_out.numpy)
            result = result.copy()
            results.append(result)
        return results

    def get_values(self):
        values_outer = []
        for agg_index, (agg_desc, selections, aggregation, selection_waslist) in enumerate(self.aggregations):
            values = []
            for selection_index, selection in enumerate(selections):
                agg = aggregation[selection_index]
                values.append(np.asarray(agg))
            values_outer.append(values)
        return values_outer


    @classmethod
    def decode(cls, encoding, spec, df):
        # aggs = [vaex.agg._from_spec(agg_spec) for agg_spec in spec['aggregations']]
        aggs = encoding.decode_list('aggregation', spec['aggregations'])
        dtypes = encoding.decode_dict('dtype', spec['dtypes'])
        grid = encoding.decode('grid', spec['grid'])
        values = encoding.decode_list2('ndarray', spec['values']) if 'values' in spec else None
        # dtypes = {expr: _deserialize_type(type_spec) for expr, type_spec in spec['dtypes'].items()}
        for agg in aggs:
            agg._prepare_types(df)
        return cls(df, grid, aggs, dtypes, initial_values=values)

    def encode(self, encoding):
        # TODO: get rid of dtypes
        encoded = {'task': type(self).name,
                'grid': encoding.encode('grid', self.grid),
                'aggregations': encoding.encode_list("aggregation", self.aggregation_descriptions),
                'dtypes': encoding.encode_dict("dtype", self.dtypes)
                }
        if self.has_values:
            encoded['values'] = encoding.encode_list2('ndarray', self.get_values())
        return encoded
