from functools import reduce
import logging
import operator
import numpy as np
import vaex
import collections
import six

import pyarrow as pa
from vaex.delayed import delayed_args, delayed_dict, delayed_list
from vaex.utils import _ensure_list, _ensure_string_from_expression

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

class Binner(BinnerBase):
    def __init__(self, expression, vmin, vmax, bins, df=None, label=None):
        self.df = df or expression.df
        self.expression = self.df[str(expression)]
        self.label = label or self.expression._label
        self.vmin = vmin
        self.vmax = vmax
        self.N = bins
        self.binby_expression = str(expression)
        self.bin_values = self.df.bin_centers(expression, (self.vmin, self.vmax), bins)
        self.sort_indices = None
        self._promise = vaex.promise.Promise.fulfilled(None)

    def _create_binner(self, df):
        assert (df.dataset == self.df.dataset), "you passed a dataframe with a different dataset to the grouper/binned"
        self.df = df
        self.binner = self.df._binner_scalar(self.binby_expression, (self.vmin, self.vmax), self.N)

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
        self.every = every
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
        self._promise = vaex.promise.Promise.fulfilled(None)

    def _create_binner(self, df):
        # TODO: we modify the dataframe in place, this is not nice
        assert df.dataset == self.df.dataset, "you passed a dataframe with a different dataset to the grouper/binned"
        self.df = df
        self.begin_name = df.add_variable('t_begin', self.tmin.astype(self.resolution_type), unique=True)
        # TODO: import integer from future?
        self.binby_expression = str(df['%s - %s' % (self.expression.astype(self.resolution_type), self.begin_name)].astype('int') // self.every)
        self.binner = df._binner_ordinal(self.binby_expression, self.N)

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


class BinnerInteger(BinnerBase):
    '''Bins an expression into it's natural bin (i.e. 5 for the number 5)

    Useful for boolean, uint8, which only have a limited number of possibilities (2, 256)
    '''
    def __init__(self, expression, label=None, dropmissing=False):
        self.expression = expression
        self.df = expression.df
        self.dtype = self.expression.dtype
        self.label = label or self.expression._label
        if self.dtype.numpy == np.dtype('bool'):
            if dropmissing:
                self.binby_expression = str(self.expression)
                self.bin_values = np.array([False, True])
                self.N = 2
            else:
                self.binby_expression = f'fillmissing(astype({str(self.expression)}, "uint8"), 2)'
                self.bin_values = pa.array([False, True, None])
                self.N = 3
        elif self.dtype.numpy == np.dtype('uint8'):
            if dropmissing:
                self.binby_expression = str(self.expression)
                self.bin_values = np.arange(0, 256, dtype="uint8")
                self.N = 256
            else:
                self.binby_expression = f'fillmissing(astype({str(self.expression)}, "int16"), 256)'
                self.bin_values = pa.array(list(range(256)) + [None])
                self.N = 257
        else:
            raise TypeError(f'Only boolean and uint8 are supported, not {self.dtype}')
        self.sort_indices = None
        self._promise = vaex.promise.Promise.fulfilled(None)

    def _create_binner(self, df):
        assert df.dataset == self.df.dataset, "you passed a dataframe with a different dataset to the grouper/binned"
        self.df = df
        self.binner = self.df._binner_ordinal(self.binby_expression, self.N)


class Grouper(BinnerBase):
    """Bins an expression to a set of unique bins, like an SQL like groupby."""
    def __init__(self, expression, df=None, sort=False, pre_sort=True, row_limit=None, df_original=None, materialize_experimental=False, progress=None):
        self.df = df or expression.ds
        self.sort = sort
        self.pre_sort = pre_sort
        # we prefer to calculate the set the original dataframe to have better cache hits, and modify df
        if df_original is None:
            df_original = self.df
        self.sort = sort
        self.expression = expression
        # make sure it's an expression
        self.expression = self.df[_ensure_string_from_expression(self.expression)]
        self.label = self.expression._label
        self.progressbar = vaex.utils.progressbars(progress, title=f"grouper: {repr(self.label)}" )
        dtype = self.expression.dtype
        if materialize_experimental:
            set, values = df_original._set(self.expression, unique_limit=row_limit, return_inverse=True)
            # TODO: add column should have a unique argument
            self.df.add_column(f'__materialized_{self.label}', values)

            self.bin_values = set.key_array()
            if isinstance(self.bin_values, vaex.superstrings.StringList64):
                self.bin_values = pa.array(self.bin_values.to_numpy())
            self.binby_expression = 'bla'
            self.N = len(self.bin_values)
            self.min_value = 0
            self.binner = self.df._binner_ordinal('bla', self.N, self.min_value)
            self.sort_indices = None
        else:
            @vaex.delayed
            def process(set):
                self.bin_values = set.key_array()

                if isinstance(self.bin_values, vaex.superstrings.StringList64):
                    # TODO: find out why this more efficient path does not work
                    # col = vaex.column.ColumnStringArrow.from_string_sequence(self.bin_values)
                    # self.bin_values = pa.array(col)
                    self.bin_values = pa.array(self.bin_values.to_numpy())
                if vaex.dtype_of(self.bin_values) == int:
                    max_value = self.bin_values.max()
                    self.bin_values = self.bin_values.astype(vaex.utils.required_dtype_for_max(max_value))
                logger.debug('Constructed grouper for expression %s with %i values', str(expression), len(self.bin_values))

                if set.has_null and (dtype.is_primitive or dtype.is_datetime):
                    mask = np.zeros(shape=self.bin_values.shape, dtype="?")
                    mask[set.null_value] = 1
                    self.bin_values = np.ma.array(self.bin_values, mask=mask)
                if self.sort:
                    self.bin_values = vaex.array_types.to_arrow(self.bin_values)
                    indices = pa.compute.sort_indices(self.bin_values)#[offset:])
                    if pre_sort:
                        self.bin_values = pa.compute.take(self.bin_values, indices)
                        # arrow sorts with null last
                        null_value = -1 if not set.has_null else len(self.bin_values)-1
                        fingerprint = set.fingerprint + "-sorted"
                        if dtype.is_string:
                            bin_values = vaex.column.ColumnStringArrow.from_arrow(self.bin_values)
                            string_sequence = bin_values.string_sequence
                            set = type(set)(string_sequence, null_value, set.nan_count, set.null_count, fingerprint)
                        else:
                            set = type(set)(self.bin_values, null_value, set.nan_count, set.null_count, fingerprint)
                        self.sort_indices = None
                    else:
                        self.sort_indices = vaex.array_types.to_numpy(indices)
                        # the bin_values will still be pre sorted, maybe that is confusing (implementation detail)
                        self.bin_values = pa.compute.take(self.bin_values, self.sort_indices)
                else:
                    self.sort_indices = None
                self.set = set

                self.basename = 'set_%s' % vaex.utils._python_save_name(str(self.expression) + "_" + set.fingerprint)

                self.N = len(self.bin_values)
                # for datetimes, we converted to int
                if dtype.is_datetime:
                    self.bin_values = dtype.create_array(self.bin_values)
            self._promise = process(df_original._set(self.expression, unique_limit=row_limit, delay=True, progress=self.progressbar))

    def _create_binner(self, df):
        # TODO: we modify the dataframe in place, this is not nice
        assert df.dataset == self.df.dataset, "you passed a dataframe with a different dataset to the grouper/binned"
        self.df = df
        if self.basename not in self.df.variables:
            self.setname = df.add_variable(self.basename, self.set, unique=True)
        else:
            self.setname = self.basename
        self.binby_expression = '_ordinal_values(%s, %s)' % (self.expression, self.setname)
        self.binner = self.df._binner_ordinal(self.binby_expression, self.N)

class GrouperCombined(Grouper):
    def __init__(self, expression, df, multipliers, parents, sort, row_limit=None, progress=None):
        '''Will group by 1 expression, which is build up from multiple expressions.

        Used in the sparse/combined group by.
        '''
        super().__init__(expression, df, sort=sort, row_limit=row_limit, progress=progress)
        assert len(multipliers) == len(parents)

        assert multipliers[-1] == 1
        self.df = df
        self.label = 'SHOULD_NOT_BE_USED'
        self.expression = expression
        # efficient way to find the original bin values (parent.bin_value) from the 'compressed'
        # self.bin_values
        progressbar = self.progressbar.add("extract labels from sparse set")
        @vaex.delayed
        def process(_ignore):
            df = vaex.from_dict({'row': vaex.vrange(0, self.N, dtype='i8'), 'bin_value': self.bin_values})
            df[f'index_0'] = df['bin_value'] // multipliers[0]
            df[f'leftover_0'] = df[f'bin_value'] % multipliers[0]
            for i in range(1, len(multipliers)):
                df[f'index_{i}'] = df[f'leftover_{i-1}'] // multipliers[i]
                df[f'leftover_{i}'] = df[f'leftover_{i-1}'] % multipliers[i]
            columns = [f'index_{i}' for i in range(len(multipliers))]
            indices_parents = df.evaluate(columns, progress=progressbar)
            def compress(ar):
                if vaex.dtype_of(ar).kind == 'i':
                    ar = vaex.array_types.to_numpy(ar)
                    max_value = ar.max()
                    ar = ar.astype(vaex.utils.required_dtype_for_max(max_value))
                    return ar
            indices_parents = [compress(ar) for ar in indices_parents]
            bin_values = {}
            # NOTE: we can also use dict encoding instead of take
            for indices, parent in zip(indices_parents, parents):
                if sort:
                    assert parent.pre_sort, "cannot sort while parent not presorted"
                    assert parent.sort_indices is None
                dtype = vaex.dtype_of(parent.bin_values)
                if dtype.is_struct:
                    # collapse parent struct into our flat struct
                    for field, ar in zip(parent.bin_values.type, parent.bin_values.flatten()):
                        bin_values[field.name] = ar.take(indices)
                        # bin_values[field.name] = pa.DictionaryArray.from_arrays(indices, ar)
                else:
                    bin_values[parent.label] = parent.bin_values.take(indices)
                    # bin_values[parent.label] = pa.DictionaryArray.from_arrays(indices, parent.bin_values)
            return pa.StructArray.from_arrays(bin_values.values(), bin_values.keys())
        self._promise_parent = self._promise
        def key_function():
            fp = vaex.cache.fingerprint(expression.fingerprint())
            return f'groupby-combined-{fp}'
        lookup_originals = vaex.cache._memoize(process, key_function=key_function, delay=True)
        @vaex.delayed
        def set(bin_values):
            self.bin_values = bin_values
        self._promise = set(lookup_originals(self._promise_parent))


class GrouperCategory(BinnerBase):
    """Faster grouper that will use the fact that a column is categorical."""

    def __init__(self, expression, df=None, sort=False, row_limit=None, pre_sort=True):
        self.df = df or expression.ds
        self.sort = sort
        self.pre_sort = pre_sort
        # make sure it's an expression
        expression = self.df[str(expression)]
        self.expression_original = expression
        self.label = expression._label
        self.expression = expression.index_values() if expression.dtype.is_encoded else expression
        self.row_limit = row_limit

        self.min_value = self.df.category_offset(self.expression_original)
        self.bin_values = self.df.category_labels(self.expression_original, aslist=False)
        self.N = self.df.category_count(self.expression_original)
        dtype = self.expression.dtype
        if self.sort:
            # not pre-sorting is faster
            sort_indices = pa.compute.sort_indices(self.bin_values)
            self.bin_values = pa.compute.take(self.bin_values, sort_indices)
            if self.pre_sort:
                sort_indices = vaex.array_types.to_numpy(sort_indices)
                # TODO: this is kind of like expression.map
                from .hash import ordered_set_type_from_dtype

                ordered_set_type = ordered_set_type_from_dtype(dtype)
                fingerprint = self.expression.fingerprint() + "-grouper-sort-mapper"
                self.set = ordered_set_type(sort_indices + self.min_value, -1, 0, 0, fingerprint)
                self.min_value = 0
                self.sort_indices = None
                self.basename = "set_%s" % vaex.utils._python_save_name(str(self.expression) + "_" + self.set.fingerprint)
            else:
                self.sort_indices = sort_indices
        else:
            self.sort_indices = None
        if isinstance(self.bin_values, list):
            self.bin_values = pa.array(self.bin_values)

        if row_limit is not None:
            if self.N > row_limit:
                raise vaex.RowLimitException(f'Resulting grouper has {self.N:,} unique combinations, which is larger than the allowed row limit of {row_limit:,}')
        # TODO: what do we do with null values for categories?
        # if self.set.has_null:
        #     self.N += 1
        #     keys += ['null']
        self._promise = vaex.promise.Promise.fulfilled(None)

    def _create_binner(self, df):
        assert df.dataset == self.df.dataset, "you passed a dataframe with a different dataset to the grouper/binned"
        self.df = df
        if self.sort and self.pre_sort:
            if self.basename not in self.df.variables:
                self.setname = df.add_variable(self.basename, self.set, unique=True)
            else:
                self.setname = self.basename
            self.binby_expression = "_ordinal_values(%s, %s)" % (self.expression, self.setname)
            self.binner = self.df._binner_ordinal(self.binby_expression, self.N, 0)
        else:
            self.binby_expression = str(self.expression)
            self.binner = self.df._binner_ordinal(self.binby_expression, self.N, self.min_value)


class GrouperLimited(BinnerBase):
    """Group to a limited set of values, store the rest in an (optional) other bin"""

    def __init__(self, expression, values, keep_other=True, other_value=None, sort=False, label=None, df=None):
        self.df = df or expression.df
        self.sort = sort
        self.pre_sort = True
        self.expression = self.df[str(expression)]
        self.label = label or self.expression._label
        self.keep_other = keep_other
        if isinstance(values, pa.ChunkedArray):
            values = pa.concat_arrays(values.chunks)
        if sort:
            indices = pa.compute.sort_indices(values)
            values = pa.compute.take(values, indices)

        if self.keep_other:
            self.bin_values = pa.array(vaex.array_types.tolist(values) + [other_value])
            self.values = self.bin_values.slice(0, len(self.bin_values) - 1)
        else:
            raise NotImplementedError("not supported yet")
            # although we can support this, it will fail with _combine, because of
            # the mapping of the set to -1
            self.bin_values = pa.array(vaex.array_types.tolist(values))
            self.values = self.bin_values
        self.N = len(self.bin_values)
        dtype = vaex.dtype_of(self.values)
        set_type = vaex.hash.ordered_set_type_from_dtype(dtype)
        values_list = self.values.tolist()
        try:
            null_value = values_list.index(None)
            null_count = 1
        except ValueError:
            null_value = -1
            null_count = 0
        if vaex.dtype_of(self.values) == float:
            nancount = np.isnan(self.values).sum()
        else:
            nancount = 0

        fp = vaex.cache.fingerprint(values)
        fingerprint = f"set-grouper-fixed-{fp}"
        if dtype.is_string:
            values = vaex.column.ColumnStringArrow.from_arrow(self.values)
            string_sequence = values.string_sequence
            self.set = set_type(string_sequence, null_value, nancount, null_count, fingerprint)
        else:
            self.set = set_type(self.values, null_value, nancount, null_count, fingerprint)

        self.basename = "set_%s" % vaex.utils._python_save_name(str(self.expression) + "_" + self.set.fingerprint)
        self.binby_expression = expression
        self.sort_indices = None
        self._promise = vaex.promise.Promise.fulfilled(None)

    def _create_binner(self, df):
        assert df.dataset == self.df.dataset, "you passed a dataframe with a different dataset to the grouper/binned"
        self.df = df
        if self.basename not in self.df.variables:
            self.setname = df.add_variable(self.basename, self.set, unique=True)
        else:
            self.setname = self.basename
        # modulo N will map -1 (value not found) to N-1
        self.binby_expression = "_ordinal_values(%s, %s) %% %s" % (self.expression, self.setname, self.N)
        self.binner = self.df._binner_ordinal(self.binby_expression, self.N)

def _combine(df, groupers, sort, row_limit=None, progress=None):
    for grouper in groupers:
        if isinstance(grouper, Binner):
            raise NotImplementedError('Cannot combined Binner with other groupers yet')

    progressbar = vaex.utils.progressbars(progress, title="find sparse entries / compress")
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
    grouper = GrouperCombined(expression, df, multipliers=cumulative_counts[1:], parents=combine_now, sort=sort, row_limit=row_limit, progress=progressbar)
    if combine_later:
        @vaex.delayed
        def combine(_ignore):
            # recursively add more of the groupers (because of 64 bit overflow)
            # return 1
            grouper._create_binner(df)
            new_grouper = _combine(df, [grouper] + combine_later, sort=sort)
            return new_grouper
        return combine(grouper._promise)
    return grouper._promise.then(lambda x: grouper)


class GroupByBase(object):
    def __init__(self, df, by, sort=False, combine=False, expand=True, row_limit=None, copy=True, progress=None):
        '''Note that row_limit only works in combination with combine=True'''
        df_original = df
        if copy:
            df = df.copy() # we will mutate the df (Add variables), this will keep the original dataframe unchanged
        self.df = df
        self.sort = sort
        self.expand = expand  # keep as pyarrow struct?
        self.progressbar = vaex.utils.progressbars(progress, title="groupby/binby")
        self.progressbar_groupers = self.progressbar.add("groupers")
        self.progressbar_agg = self.progressbar.add("aggregation")

        if not isinstance(by, collections_abc.Iterable)\
            or isinstance(by, six.string_types):
            by = [by]

        self.by = []
        self.by_original = by
        for by_value in by:
            if not isinstance(by_value, BinnerBase):
                expression = df[_ensure_string_from_expression(by_value)]
                if df.is_category(by_value):
                    by_value = GrouperCategory(expression, sort=sort, row_limit=row_limit)
                else:
                    dtype = expression.dtype
                    if dtype == np.dtype('uint8') or dtype == np.dtype('bool'):
                        by_value = BinnerInteger(expression)  # doesn't modify, always sorted
                    else:
                        by_value = Grouper(expression, sort=sort, row_limit=row_limit, df_original=df_original, progress=self.progressbar_groupers)
            self.by.append(by_value)
        @vaex.delayed
        def possible_combine(*binner_promises):
            # because binners can be created from other dataframes (we make a copy)
            # we let it mutate *our* dataframe
            for binner in self.by:
                binner._create_binner(self.df)
            @vaex.delayed
            def set_combined(combined):
                combined._create_binner(self.df)
                self.by = [combined]
                self.combine = True
            if combine is True and len(self.by) >= 2:
                promise = set_combined(_combine(self.df, self.by, sort=sort, row_limit=row_limit, progress=self.progressbar_groupers))
            elif combine == 'auto' and len(self.by) >= 2:
                cells = product([grouper.N for grouper in self.by])
                dim = len(self.by)
                rows = df.length_unfiltered()  # we don't want to trigger a computation
                occupancy = rows/cells
                logger.debug('%s rows and %s grid cells => occupancy=%s', rows, cells, occupancy)
                # we want each cell to have a least 10x occupacy
                if occupancy < 10:
                    logger.info(f'Combining {len(self.by)} groupers into 1')
                    promise = set_combined(_combine(self.df, self.by, sort=sort, row_limit=row_limit, progress=self.progressbar_groupers))
                    self.combine = True
                else:
                    self.combine = False
                    promise = vaex.promise.Promise.fulfilled(None)
            else:
                self.combine = False
                promise = vaex.promise.Promise.fulfilled(None)
            @vaex.delayed
            def process(_ignore):
                self.groupby_expression = [str(by.expression) for by in self.by]
                self.binners = tuple(by.binner for by in self.by)
                self.shape = [by.N for by in self.by]
                self.dims = self.groupby_expression[:]
            return process(promise)

        self._promise_by = possible_combine(*[by._promise for by in self.by])

    @property
    def _coords1d(self):
        return [k.bin_values for k in self.by]

    def _agg(self, actions):
        return self._promise_by.then(lambda x: self._agg_impl(actions))

    def _agg_impl(self, actions):
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
            values = df._agg(aggregate, self.binners, delay=_USE_DELAY, progress=self.progressbar_agg)
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
        '''Get DataFrame containing a single group from the :class:`GroupBy`.

        Example:

        >>> import vaex
        >>> import vaex.ml
        >>> df = vaex.ml.datasets.load_titanic()
        >>> g1 = df.groupby(by='pclass')
        >>> df_group1 = g1.get_group(1)
        >>> df_group1.head(3)
          #  name                              pclass  sex         age     fare
          0  Allen, Miss. Elisabeth Walton          1  female  29       211.338
          1  Allison, Master. Hudson Trevor         1  male     0.9167  151.55
          2  Allison, Miss. Helen Loraine           1  female   2       151.55


        >>> df = vaex.ml.datasets.load_titanic()
        >>> g2 = df.groupby(by=['pclass', 'sex'])
        >>> df_group2 = g2.get_group([1, 'female'])
        >>> df_group2.head(3)
          #  name                                               pclass  sex       age     fare
          0  Allen, Miss. Elisabeth Walton                           1  female     29  211.338
          1  Allison, Miss. Helen Loraine                            1  female      2  151.55
          2  Allison, Mrs. Hudson J C (Bessie Waldo Daniels)         1  female     25  151.55

        :param group: A value or a list of values of the expressions used to create the groupby object.
            If `assume_sparse=True` when greating the groupby object, this param takes an int corresponding to the particular groupby combination.
        :rtype: DataFrame
        '''
        if self.combine:
            assert isinstance(group, int)
            filter_expression = self.df[str(self.by[0].binby_expression)] == group
        else:
            group = _ensure_list(group)
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
    def __init__(self, df, by, sort=False):
        super(BinBy, self).__init__(df, by, sort=sort)

    def agg(self, actions, merge=False, delay=False):
        import xarray as xr
        @vaex.delayed
        def aggregate(promise_by):
            arrays = super(BinBy, self)._agg(actions)
            return delayed_dict(arrays)

        @vaex.delayed
        def process(arrays):
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

        result = process(aggregate(self._promise_by))
        if delay:
            return result
        else:
            self.df.execute()
            return result.get()

class GroupBy(GroupByBase):
    """Implementation of the binning and aggregation of data, see :method:`groupby`."""
    def __init__(self, df, by, sort=False, combine=False, expand=True, row_limit=None, copy=True, progress=None):
        super(GroupBy, self).__init__(df, by, sort=sort, combine=combine, expand=expand, row_limit=row_limit, copy=copy, progress=progress)

    def agg(self, actions, delay=False):
        # TODO: this basically forms a cartesian product, we can do better, use a
        # 'multistage' hashmap
        @vaex.delayed
        def aggregate(promise_by):
            arrays = super(GroupBy, self)._agg(actions)
            # we don't want non-existing pairs (e.g. Amsterdam in France does not exist)
            # but also, e.g. GrouperInteger will always expect missing values
            # but they may not aways exist
            counts = self.counts
            # nobody wanted to know count*, but we need it if we included non-existing pairs
            if counts is None:
                # TODO: it seems this path is never tested
                count_agg = vaex.agg.count(edges=True)
                counts = self.df._agg(count_agg, self.binners, delay=_USE_DELAY, progress=self.progressbar_agg)
            arrays = delayed_dict(arrays)
            return counts, arrays

        @vaex.delayed
        def process(args):
            counts, arrays = args
            # arrays = {key: value.get() for key, value in arrays.items()}
            # take out the edges
            arrays = {key: vaex.utils.extract_central_part(value) for key, value in arrays.items()}
            counts = vaex.utils.extract_central_part(counts)

            # make sure we respect the sorting
            def sort(ar):
                for i, by in list(enumerate(self.by))[::-1]:
                    sort_indices = by.sort_indices
                    if sort_indices is not None:
                        # if sort_indices come from arrow, it will be uint64
                        # which np.take does not like
                        sort_indices = vaex.array_types.to_numpy(sort_indices)
                        if sort_indices.dtype == np.dtype("uint64"):
                            sort_indices = sort_indices.astype("int64")
                        ar = np.take(ar, sort_indices, axis=i)
                return ar

            arrays = {key: sort(value) for key, value in arrays.items()}

            if self.combine and self.expand and isinstance(self.by[0], GrouperCombined):
                assert len(self.by) == 1
                values = self.by[0].bin_values
                columns = {field.name: ar for field, ar in zip(values.type, values.flatten())}
                for key, value in arrays.items():
                    assert value.ndim == 1
                    columns[key] = value
            else:
                counts = sort(counts)
                mask = counts > 0
                columns = {}
                for by, indices in zip(self.by, np.where(mask)):
                    columns[by.label] = by.bin_values.take(indices)
                if mask.sum() == mask.size:
                    # if we want all, just take it all
                    # should be faster
                    for key, value in arrays.items():
                        columns[key] = value.ravel()
                else:
                    for key, value in arrays.items():
                        columns[key] = value[mask]
            dataset_arrays = vaex.dataset.DatasetArrays(columns)
            dataset = DatasetGroupby(dataset_arrays, self.df, self.by_original, actions, combine=self.combine, expand=self.expand, sort=self.sort)
            df_grouped = vaex.from_dataset(dataset)
            return df_grouped
        result = process(delayed_list(aggregate(self._promise_by)))
        if delay:
            return result
        else:
            self.df.execute()
            return result.get()


@vaex.dataset.register
class DatasetGroupby(vaex.dataset.DatasetDecorator):
    '''Wraps a resulting dataset from a dataframe groupby, so the groupby can be serialized'''
    snake_name = 'groupby'
    def __init__(self, original, df, by, agg, combine, expand, sort):
        assert isinstance(original, vaex.dataset.DatasetArrays)
        super().__init__(original)
        self.df = df
        self.by = by
        self.agg = agg
        self.sort = sort
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
        return type(self)(self.original.hashed(), df=self.df, by=self.by, agg=self.agg, combine=self.combine, expand=self.expand, sort=self.sort)

    def _encode(self, encoding):
        by = self.by
        by = str(by) if not isinstance(by, (list, tuple)) else list(map(str, by))
        spec = {
            'dataframe': encoding.encode('dataframe', self.df),
            'by': by,
            'aggregation': encoding.encode_collection('aggregation', self.agg),
            'combine': self.combine,
            'expand': self.expand,
            'sort': self.sort,
        }
        return spec

    @classmethod
    def _decode(cls, encoding, spec):
        df = encoding.decode('dataframe', spec.pop('dataframe'))
        by = spec.pop('by')
        agg = encoding.decode_collection('aggregation', spec.pop('aggregation'))
        sort = spec.pop('sort')
        dfg = df.groupby(by, agg=agg, sort=sort)
        return DatasetGroupby(dfg.dataset.original, df, by, agg, sort=sort, **spec)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['original']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.original = self.df.groupby(self.by, agg=self.agg, sort=self.sort).dataset.original
        # self._create_columns()

    def slice(self, start, end):
        if start == 0 and end == self.row_count:
            return self
        return vaex.dataset.DatasetSlicedArrays(self, start=start, end=end)
