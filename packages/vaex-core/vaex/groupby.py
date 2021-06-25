from functools import reduce
import logging
import operator
import numpy as np
import vaex
import collections
import six

import pyarrow as pa

try:
    collections_abc = collections.abc
except AttributeError:
    collections_abc = collections

__all__ = ['GroupBy', 'Grouper', 'BinnerTime']

logger = logging.getLogger("vaex.groupby")
_USE_DELAY = True


# pure Python to avoid int overflow
product = lambda l: reduce(operator.mul, l)


class BinnerBase:
    pass

class BinnerTime(BinnerBase):
    """Bins an expression in a specified datetime resolution.

    Note that different with :class:`Grouper`, all values between the minimum and maximum
    datetime (incremented by the resolution step) will be present in the dataframe.
    A normal groupby would only contain datatime values that are present in the original
    dataframe.

    Example:

    >>> import vaex
    >>> import numpy as np
    >>> t = np.arange('2015-01-01', '2015-02-01', dtype=np.datetime64)
    >>> y = np.arange(len(t))
    >>> df = vaex.from_arrays(t=t, y=y)
    >>> df.groupby(vaex.BinnerTime.per_week(df.t)).agg({'y' : 'sum'})
    #  t                      y
    0  2015-01-01 00:00:00   21
    1  2015-01-08 00:00:00   70
    2  2015-01-15 00:00:00  119
    3  2015-01-22 00:00:00  168
    4  2015-01-29 00:00:00   87

    """
    def __init__(self, expression, resolution='W', df=None, every=1):
        self.resolution = resolution
        self.expression = expression
        self.df = df or expression.ds
        self.sort_indices = None
        # make sure it's an expression
        self.expression = self.df[str(self.expression)]
        self.label = self.expression._label
        self.tmin, self.tmax = self.df[str(self.expression)].minmax()

        self.resolution_type = 'M8[%s]' % self.resolution
        dt = (self.tmax.astype(self.resolution_type) - self.tmin.astype(self.resolution_type))
        self.N = (dt.astype(int).item() + 1)
        # divide by every, and round up
        self.N = (self.N + every - 1) // every
        self.bin_values = np.arange(self.tmin.astype(self.resolution_type), self.tmax.astype(self.resolution_type)+1, every)
        # TODO: we modify the dataframe in place, this is not nice
        self.begin_name = self.df.add_variable('t_begin', self.tmin.astype(self.resolution_type), unique=True)
        # TODO: import integer from future?
        self.binby_expression = str(self.df['%s - %s' % (self.expression.astype(self.resolution_type), self.begin_name)].astype('int') // every)
        self.binner = self.df._binner_ordinal(self.binby_expression, self.N)

    @classmethod
    def per_day(cls, expression, df=None):
        return cls(expression, 'D', df)

    @classmethod
    def per_week(cls, expression, df=None):
        return cls(expression, 'W', df)

    @classmethod
    def per_month(cls, expression, df=None):
        return cls(expression, 'M', df)

    @classmethod
    def per_quarter(cls, expression, df=None, every=1):
        return cls(expression, 'M', df, every=3*every)

    @classmethod
    def per_year(cls, expression, df=None):
        return cls(expression, 'Y', df)


class Grouper(BinnerBase):
    """Bins an expression to a set of unique bins."""
    def __init__(self, expression, df=None, sort=False, pre_sort=True, row_limit=None, df_original=None):
        self.df = df or expression.ds
        # we prefer to calculate the set the original dataframe to have better cache hits, and modify df
        if df_original is None:
            df_original = self.df
        self.sort = sort
        self.expression = expression
        # make sure it's an expression
        self.expression = self.df[str(self.expression)]
        self.label = self.expression._label
        set = df_original._set(self.expression, unique_limit=row_limit)
        keys = set.keys()
        if self.sort:
            if pre_sort:
                sort_indices = np.argsort(keys)
                keys = np.array(keys)[sort_indices].tolist()
                set_dict = dict(zip(keys, range(len(keys))))
                set = type(set)(set_dict, set.count, set.nan_count, set.null_count)
                self.sort_indices = None
            else:
                self.sort_indices = np.argsort(keys)
                keys = np.array(keys)[self.sort_indices].tolist()
        else:
            self.sort_indices = None
        self.set = set

        # TODO: we modify the dataframe in place, this is not nice
        basename = 'set_%s' % vaex.utils._python_save_name(str(expression))
        self.setname = self.df.add_variable(basename, self.set, unique=True)

        self.bin_values = keys
        self.binby_expression = '_ordinal_values(%s, %s)' % (self.expression, self.setname)
        self.N = len(self.bin_values)
        if self.set.has_null:
            self.N += 1
            self.bin_values = [None] + self.bin_values
        if self.set.has_nan:
            self.N += 1
            self.bin_values = [np.nan] + self.bin_values
        if self.sort_indices is not None:
            if self.set.has_null and self.set.has_nan:
                self.sort_indices = np.concatenate([[0, 1], self.sort_indices + 2])
            elif self.set.has_null or self.set.has_nan:
                self.sort_indices = np.concatenate([[0], self.sort_indices + 1])
        self.bin_values = self.expression.dtype.create_array(self.bin_values)
        self.binner = self.df._binner_ordinal(self.binby_expression, self.N)


class GrouperCombined(Grouper):
    def __init__(self, expression, df, multipliers, parents, sort, row_limit=None):
        '''Will group by 1 expression, which is build up from multiple expressions.

        Used in the sparse/combined group by.
        '''
        super().__init__(expression, df, sort=sort, row_limit=row_limit)
        assert len(multipliers) == len(parents)

        assert multipliers[-1] == 1
        self.df = df
        self.label = 'SHOULD_NOT_BE_USED'
        self.expression = expression
        # efficient way to find the original bin values (parent.bin_value) from the 'compressed'
        # self.bin_values
        df = vaex.from_dict({'row': vaex.vrange(0, self.N, dtype='i8'), 'bin_value': self.bin_values})
        df[f'index_0'] = df['bin_value'] // multipliers[0]
        df[f'leftover_0'] = df[f'bin_value'] % multipliers[0]
        for i in range(1, len(multipliers)):
            df[f'index_{i}'] = df[f'leftover_{i-1}'] // multipliers[i]
            df[f'leftover_{i}'] = df[f'leftover_{i-1}'] % multipliers[i]
        columns = [f'index_{i}' for i in range(len(multipliers))]
        indices_parents = df.evaluate(columns)
        bin_values = {}
        for indices, parent in zip(indices_parents, parents):
            dtype = vaex.dtype_of(parent.bin_values)
            if dtype.is_struct:
                # collapse parent struct into our flat struct
                for field, ar in zip(parent.bin_values.type, parent.bin_values.flatten()):
                    bin_values[field.name] = ar.take(indices)
            else:
                bin_values[parent.label] = parent.bin_values.take(indices)
        self.bin_values = pa.StructArray.from_arrays(bin_values.values(), bin_values.keys())


class GrouperCategory(BinnerBase):
    def __init__(self, expression, df=None, sort=False, row_limit=None):
        self.df = df or expression.ds
        self.sort = sort
        # make sure it's an expression
        expression = self.df[str(expression)]
        self.expression_original = expression
        self.label = expression._label
        self.expression = expression.index_values() if expression.dtype.is_encoded else expression

        self.bin_values = self.df.category_labels(self.expression_original, aslist=False)
        if self.sort:
            # None will always be the first value
            if self.bin_values[0] is None:
                self.sort_indices = np.concatenate([[0], 1 + np.argsort(self.bin_values[1:])])
                self.bin_values = np.array(self.bin_values)[self.sort_indices].tolist()
            else:
                self.sort_indices = np.argsort(self.bin_values)
                self.bin_values = np.array(self.bin_values)[self.sort_indices].tolist()
        else:
            self.sort_indices = None
        if isinstance(self.bin_values, list):
            self.bin_values = pa.array(self.bin_values)

        self.N = self.df.category_count(self.expression_original)
        if row_limit is not None:
            if self.N > row_limit:
                raise vaex.RowLimitException(f'Resulting grouper has {self.N:,} unique combinations, which is larger than the allowed row limit of {row_limit:,}')
        self.min_value = self.df.category_offset(self.expression_original)
        # TODO: what do we do with null values for categories?
        # if self.set.has_null:
        #     self.N += 1
        #     keys += ['null']
        self.binner = self.df._binner_ordinal(self.expression, self.N, self.min_value)
        self.binby_expression = str(self.expression)


def _combine(df, groupers, sort, row_limit=None):
    groupers = groupers.copy()
    max_count_64bit = 2**63-1
    first = groupers.pop(0)
    combine_now = [first]
    combine_later = []
    counts = [first.N]

    # when does the cartesian product overflow 64 bits?
    next = groupers.pop(0)
    while (product(counts) * next.N < max_count_64bit):
        counts.append(next.N)
        combine_now.append(next)
        if groupers:
            next = groupers.pop(0)
        else:
            next = None
            break

    counts.append(1)
    # decreasing [40, 4, 1] for 2 groupers (N=10 and N=4)
    cumulative_counts = np.cumproduct(counts[::-1], dtype='i8').tolist()[::-1]
    assert len(combine_now) >= 2
    combine_later = ([next] if next else []) + groupers

    binby_expressions = [df[k.binby_expression] for k in combine_now]
    for i in range(0, len(binby_expressions)):
        binby_expression = binby_expressions[i]
        dtype = vaex.utils.required_dtype_for_max(cumulative_counts[i])
        binby_expression = binby_expression.astype(str(dtype))
        if isinstance(combine_now[i], GrouperCategory) and combine_now[i].min_value != 0:
            binby_expression -= combine_now[i].min_value
        if cumulative_counts[i+1] != 1:
            binby_expression = binby_expression * cumulative_counts[i+1]
        binby_expressions[i] = binby_expression
    expression = reduce(operator.add, binby_expressions)
    grouper = GrouperCombined(expression, df, multipliers=cumulative_counts[1:], parents=combine_now, sort=sort, row_limit=row_limit)
    if combine_later:
        # recursively add more of the groupers (because of 64 bit overflow)
        grouper = _combine(df, [grouper] + combine_later, sort=sort)
    return grouper


class GroupByBase(object):
    def __init__(self, df, by, sort=False, combine=False, expand=True, row_limit=None):
        '''Note that row_limit only works in combination with combine=True'''
        df_original = df
        df = df.copy()  # we're gonna mutate, so create a shallow copy
        self.df = df
        self.sort = sort
        self.expand = expand  # keep as pyarrow struct?

        if not isinstance(by, collections_abc.Iterable)\
            or isinstance(by, six.string_types):
            by = [by]

        self.by = []
        self.by_original = by
        for by_value in by:
            if not isinstance(by_value, BinnerBase):
                if df.is_category(by_value):
                    by_value = GrouperCategory(df[str(by_value)], sort=sort, row_limit=row_limit)
                else:
                    by_value = Grouper(df[str(by_value)], sort=sort, row_limit=row_limit, df_original=df_original)
            self.by.append(by_value)
        if combine is True and  len(self.by) >= 2:
            self.by = [_combine(self.df, self.by, sort=sort, row_limit=row_limit)]
            self.combine = True
        elif combine == 'auto' and len(self.by) >= 2:
            cells = product([grouper.N for grouper in self.by])
            dim = len(self.by)
            rows = df.length_unfiltered()  # we don't want to trigger a computation
            occupancy = rows/cells
            logger.debug('%s rows and %s grid cells => occupancy=%s', rows, cells, occupancy)
            # we want each cell to have a least 10x occupacy
            if occupancy < 10:
                logger.info(f'Combining {len(self.by)} groupers into 1')
                self.by = [_combine(self.df, self.by, sort=sort, row_limit=row_limit)]
                self.combine = True
            else:
                self.combine = False
        else:
            self.combine = False


        # binby may be an expression based on self.by.expression
        # if we want to have all columns, minus the columns grouped by
        # we should keep track of the original expressions, but binby
        self.groupby_expression = [str(by.expression) for by in self.by]
        self.binners = tuple(by.binner for by in self.by)
        self.shape = [by.N for by in self.by]
        self.dims = self.groupby_expression[:]

    @property
    def _coords1d(self):
        return [k.bin_values for k in self.by]

    def _agg(self, actions):
        df = self.df
        if isinstance(actions, collections_abc.Mapping):
            actions = list(actions.items())
        elif not isinstance(actions, collections_abc.Iterable)\
            or isinstance(actions, six.string_types):
            actions = [actions]


        grids = {}
        self.counts = None  # if we already do count *, we'd like to know so we don't have to do it ourselves!

        def add(aggregate, column_name=None, override_name=None):
            if column_name is None or override_name is not None:
                column_name = aggregate.pretty_name(override_name, df)
            aggregate.edges = True  # is this ok to override?
            values = df._agg(aggregate, self.binners, delay=_USE_DELAY)
            grids[column_name] = values
            if isinstance(aggregate, vaex.agg.AggregatorDescriptorBasic)\
                and aggregate.name == 'AggCount'\
                and aggregate.expression == "*"\
                and (aggregate.selection is None or aggregate.selection is False):
                self.counts = values

        for item in actions:
            override_name = None
            if isinstance(item, tuple):
                name, aggregates = item
            else:
                aggregates = item
                name = None

            if not isinstance(aggregates, collections_abc.Iterable)\
                or isinstance(aggregates, six.string_types):
                # not a list, or a string
                aggregates = [aggregates]
            else:
                # if we have a form of {'z': [.., ...]}
                # we want to have a name like z_<agg name>
                if name is not None:
                    override_name = name
            for aggregate in aggregates:
                if isinstance(aggregate, six.string_types) and aggregate == "count":
                    add(vaex.agg.count(), 'count' if name is None else name)
                else:
                    if isinstance(aggregate, six.string_types):
                        aggregate = vaex.agg.aggregates[aggregate]
                    if callable(aggregate):
                        if name is None:
                            # we use all columns
                            for column_name in df.get_column_names():
                                if column_name not in self.groupby_expression:
                                    add(aggregate(column_name), override_name=override_name)
                        else:
                            add(aggregate(name), name, override_name=override_name)
                    else:
                        add(aggregate, name, override_name=override_name)
        return grids

    @property
    def groups(self):
        for group, df in self:
            yield group

    def get_group(self, group):
        if self.combine:
            assert isinstance(group, int)
            filter_expression = self.df[str(self.by[0].binby_expression)] == group
        else:
            values = group
            filter_expressions = [self.df[expression] == value for expression, value in zip(self.groupby_expression, values)]
            filter_expression = filter_expressions[0]
            for expression in filter_expressions[1:]:
                filter_expression = filter_expression & expression
        return self.df[filter_expression]

    def __iter__(self):
        if self.combine:
            self.df.execute()
            values = self.by[0].bin_values
            for i, values in enumerate(zip(*values.flatten())):
                values = tuple(k.as_py() for k in values)
                yield values, self.get_group(i)
        else:
            count_agg = vaex.agg.count()
            counts = self.df._agg(count_agg, self.binners)
            mask = counts > 0
            values2d = np.array([coord[mask] for coord in np.meshgrid(*self._coords1d, indexing='ij')], dtype='O')
            for i in range(values2d.shape[1]):
                values = values2d[:,i]
                dff = self.get_group(values)
                yield tuple(values.tolist()), dff

    def __len__(self):
        count_agg = vaex.agg.count()
        counts = self.df._agg(count_agg, self.binners)
        mask = counts > 0
        return mask.sum()


class BinBy(GroupByBase):
    """Implementation of the binning and aggregation of data, see :method:`binby`."""
    def __init__(self, df, by):
        super(BinBy, self).__init__(df, by)

    def agg(self, actions, merge=False):
        import xarray as xr
        arrays = super(BinBy, self)._agg(actions)
        self.df.execute()
        if _USE_DELAY:
            arrays = {key: value.get() for key, value in arrays.items()}
        # take out the edges
        arrays = {key: vaex.utils.extract_central_part(value) for key, value in arrays.items()}

        # make sure we respect the sorting
        sorting = tuple(by.sort_indices if by.sort_indices is not None else slice(None) for by in self.by)
        arrays = {key: value[sorting] for key, value in arrays.items()}

        keys = list(arrays.keys())
        key0 = keys[0]
        if not isinstance(actions, collections_abc.Iterable)\
            or isinstance(actions, six.string_types):
            assert len(keys) == 1
            final_array = arrays[key0]
            coords = self._coords1d
            return xr.DataArray(final_array, coords=coords, dims=self.dims)
        else:
            final_array = np.zeros((len(arrays), ) + arrays[key0].shape)
            for i, value in enumerate(arrays.values()):
                final_array[i] = value
            coords = [list(arrays.keys())] + self._coords1d
            dims = ['statistic'] + self.dims
            return xr.DataArray(final_array, coords=coords, dims=dims)

class GroupBy(GroupByBase):
    """Implementation of the binning and aggregation of data, see :method:`groupby`."""
    def __init__(self, df, by, sort=False, combine=False, expand=True, row_limit=None):
        super(GroupBy, self).__init__(df, by, sort=sort, combine=combine, expand=expand, row_limit=row_limit)

    def agg(self, actions):
        # TODO: this basically forms a cartesian product, we can do better, use a
        # 'multistage' hashmap
        arrays = super(GroupBy, self)._agg(actions)
        has_non_existing_pairs = len(self.by) > 1
        # we don't want non-existing pairs (e.g. Amsterdam in France does not exist)
        counts = self.counts
         # nobody wanted to know count*, but we need it if we included non-existing pairs
        if has_non_existing_pairs and counts is None:
            # TODO: it seems this path is never tested
            count_agg = vaex.agg.count(edges=True)
            counts = self.df._agg(count_agg, self.binners, delay=_USE_DELAY)
        self.df.execute()
        if _USE_DELAY:
            arrays = {key: value.get() for key, value in arrays.items()}
            if has_non_existing_pairs:
                counts = counts.get()
        # take out the edges
        arrays = {key: vaex.utils.extract_central_part(value) for key, value in arrays.items()}
        if has_non_existing_pairs:
            counts = vaex.utils.extract_central_part(counts)

        # make sure we respect the sorting
        sorting = tuple(by.sort_indices if by.sort_indices is not None else slice(None) for by in self.by)
        arrays = {key: value[sorting] for key, value in arrays.items()}

        if self.combine and self.expand and isinstance(self.by[0], GrouperCombined):
            assert len(self.by) == 1
            values = self.by[0].bin_values
            columns = {field.name: ar for field, ar in zip(values.type, values.flatten())}
            for key, value in arrays.items():
                assert value.ndim == 1
                columns[key] = value
        else:
            if has_non_existing_pairs:
                counts = counts[sorting]
                mask = counts > 0
                coords = [coord[mask] for coord in np.meshgrid(*self._coords1d, indexing='ij')]
                columns = {by.label: coord for by, coord in zip(self.by, coords)}
                for key, value in arrays.items():
                    columns[key] = value[mask]
            else:
                columns = {by.label: coord for by, coord in zip(self.by, self._coords1d)}
                for key, value in arrays.items():
                    assert value.ndim == 1
                    columns[key] = value
        dataset_arrays = vaex.dataset.DatasetArrays(columns)
        dataset = DatasetGroupby(dataset_arrays, self.df, self.by_original, actions, combine=self.combine, expand=self.expand)
        df_grouped = vaex.from_dataset(dataset)
        return df_grouped


@vaex.dataset.register
class DatasetGroupby(vaex.dataset.DatasetDecorator):
    '''Wraps a resulting dataset from a dataframe groupby, so the groupby can be serialized'''
    snake_name = 'groupby'
    def __init__(self, original, df, by, agg, combine, expand):
        assert isinstance(original, vaex.dataset.DatasetArrays)
        super().__init__(original)
        self.df = df
        self.by = by
        self.agg = agg
        self.combine = combine
        self.expand = expand
        self._row_count = self.original.row_count
        self._create_columns()

    def _create_columns(self):
        # we know original is a DatasetArrays
        self._columns = self.original._columns.copy()
        self._ids = self.original._ids.copy()

    @property
    def _fingerprint(self):
        by = self.by
        by = str(by) if not isinstance(by, (list, tuple)) else list(map(str, by))
        id = vaex.cache.fingerprint(self.original.fingerprint, self.df.fingerprint(), by, self.agg, self.combine, self.expand)
        return f'dataset-{self.snake_name}-{id}'

    def chunk_iterator(self, *args, **kwargs):
        yield from self.original.chunk_iterator(*args, **kwargs)

    def hashed(self):
        return type(self)(self.original.hashed(), df=self.df, by=self.by, agg=self.agg, combine=self.combine, expand=self.expand)

    def _encode(self, encoding):
        by = self.by
        by = str(by) if not isinstance(by, (list, tuple)) else list(map(str, by))
        spec = {
            'dataframe': encoding.encode('dataframe', self.df),
            'by': by,
            'aggregation': encoding.encode_collection('aggregation', self.agg),
            'combine': self.combine,
            'expand': self.expand,
        }
        return spec

    @classmethod
    def _decode(cls, encoding, spec):
        df = encoding.decode('dataframe', spec.pop('dataframe'))
        by = spec.pop('by')
        agg = encoding.decode_collection('aggregation', spec.pop('aggregation'))
        dfg = df.groupby(by, agg=agg)
        return DatasetGroupby(dfg.dataset.original, df, by, agg, **spec)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['original']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.original = self.df.groupby(self.by, agg=self.agg).dataset.original
        # self._create_columns()

    def slice(self, start, end):
        if start == 0 and end == self.row_count:
            return self
        return vaex.dataset.DatasetSlicedArrays(self, start=start, end=end)
