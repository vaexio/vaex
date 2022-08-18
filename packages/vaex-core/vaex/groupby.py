from functools import reduce
import logging
import operator
import collections
import six

import numpy as np
import pyarrow as pa
from vaex.dataframe import DataFrame
from vaex.docstrings import docsubst

import vaex.array_types
from vaex.delayed import delayed_args, delayed_dict, delayed_list
from vaex.utils import _ensure_list, _ensure_string_from_expression
import vaex
import vaex.hash
import vaex.utils

try:
    collections_abc = collections.abc
except AttributeError:
    collections_abc = collections

__all__ = ['GroupBy', 'Grouper', 'BinnerTime']

logger = logging.getLogger("vaex.groupby")
_USE_DELAY = True
_EXPERIMENTAL_BINNER_HASH = False


# pure Python to avoid int overflow
product = lambda l: reduce(operator.mul, l, 1)


class BinnerBase:
    simpler = None  # a binner may realize there is a simpler fallback
    def extract_center(self, dim, ar):
        # gets rid of the nan and out of bound values
        slices = [
            slice(None, None),
        ] * vaex.array_types.ndim(ar)
        slices[dim] = slice(2, -1)
        return vaex.array_types.getitem(ar, tuple(slices))

class Binner(BinnerBase):
    dense = False
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
      #  t             y
      0  2015-01-01   21
      1  2015-01-08   70
      2  2015-01-15  119
      3  2015-01-22  168
      4  2015-01-29   87

    """

    dense = False

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

    def extract_center(self, dim, ar):
        # gets rid of the nan and out of bound values
        slices = [
            slice(None, None),
        ] * vaex.array_types.ndim(ar)
        slices[dim] = slice(0, -2) # remove null and nan
        return vaex.array_types.getitem(ar, tuple(slices))

class BinnerInteger(BinnerBase):
    '''Bins an expression into it's natural bin (i.e. 5 for the number 5)

    Useful for boolean, int8/uint8, which only have a limited number of possibilities (2, 256)
    '''
    # these are always true
    pre_sort = True
    sort = True

    def __init__(self, expression, label=None, dropmissing=False, min_value=None, max_value=None, dense=False, sort=False, ascending=True):
        self.expression = expression
        self.df = expression.df
        self.dtype = self.expression.dtype
        self.label = label or self.expression._label
        self.min_value = 0
        self.dropmissing = dropmissing
        def make_array_with_null(vmin, vmax, dtype='int64'):
            if sort and not ascending:
                values = np.arange(vmax - 1, vmin - 2, -1, dtype=dtype)
            else:
                values = np.arange(vmin, vmax + 1, dtype=dtype)
            mask = np.zeros(values.shape, dtype="?")
            mask[len(values) - 1] = 1
            values[len(values) - 1] = 0
            return np.ma.array(values, mask=mask, shrink=False)

        self.dense = dense
        if self.dtype.numpy == np.dtype('bool'):
            if sort and not ascending:
                self.bin_values = vaex.array_types.to_numpy(pa.array([False, True, None]))
            else:
                self.bin_values = vaex.array_types.to_numpy(pa.array([False, True, None]))
            self.N = 2
        elif self.dtype.numpy == np.dtype('uint8'):
            self.bin_values = make_array_with_null(0, 256)
            self.N = 256
        elif self.dtype.numpy == np.dtype('int8'):
            self.min_value = -128
            self.bin_values = make_array_with_null(-128, 128)
            self.N = 256
        elif min_value is not None and max_value is not None:
            self.min_value = min_value
            self.N = int(max_value - min_value + 1)
            self.bin_values = make_array_with_null(min_value, max_value + 1)
        else:
            raise TypeError(f"Only boolean, int8 and uint8 are supported, not {self.dtype}, or private min_value and max_value")
        if self.dropmissing:
            # if we remove the missing values, we are already sorted
            self.bin_values = self.bin_values[:-1]
        self.binby_expression = str(self.expression)
        # no need to invert if we don't sort, but sort is always implicit
        self.invert = sort and not ascending
        if self.invert:
            # invert the range, taking into account min_value
            self.combine_expression = (self.N - 1 - (self.df[self.binby_expression] - self.min_value) + self.min_value).fillmissing(self.N).expression
        else:
            self.combine_expression = self.df[self.binby_expression].fillmissing(self.N).expression
        # self.combine_expression.expression = self.combine_expression.expression
        self.sort_indices = None
        self._promise = vaex.promise.Promise.fulfilled(None)

    def extract_center(self, dim, ar):
        # gets rid of the nan and out of bound values
        # indices = n
        slices = [
            slice(None, None),
        ] * vaex.array_types.ndim(ar)
        if self.dropmissing:
            slices[dim] = slice(0, -2)  # remove null and nan
        else:
            slices[dim] = slice(0, -1)  # remove nan
        return vaex.array_types.getitem(ar, tuple(slices))

    def _create_binner(self, df: DataFrame):
        assert df.dataset == self.df.dataset, "you passed a dataframe with a different dataset to the grouper/binned"
        self.df = df
        self.binner = self.df._binner_ordinal(self.binby_expression, self.N, self.min_value, self.invert)


class Grouper(BinnerBase):
    """Bins an expression to a set of unique bins, like an SQL like groupby."""

    dense = True

    def __init__(self, expression, df=None, sort=False, ascending=True, pre_sort=True, row_limit=None, df_original=None, materialize_experimental=False, progress=None, allow_simplify=False):
        self.df = df or expression.ds
        self.sort = sort
        self.pre_sort = pre_sort
        # we prefer to calculate the set the original dataframe to have better cache hits, and modify df
        if df_original is None:
            df_original = self.df
        self.sort = sort
        self.expression = expression
        self.allow_simplify = allow_simplify
        # make sure it's an expression
        self.expression = self.df[_ensure_string_from_expression(self.expression)]
        self.label = self.expression._label
        self.progressbar = vaex.utils.progressbars(progress, title=f"grouper: {repr(self.label)}" )
        dtype = self.expression.dtype
        if materialize_experimental:
            set, values = df_original._set(self.expression, limit=row_limit, return_inverse=True)
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
            def process(hashmap_unique: vaex.hash.HashMapUnique):
                self.bin_values = hashmap_unique.keys()
                if self.allow_simplify and dtype == int and len(self.bin_values):
                    vmin = self.bin_values.min()
                    vmax = self.bin_values.max()
                    int_range = vmax - vmin + 1
                    # we allow for 25% unused 'slots'
                    bins = len(self.bin_values)
                    if int_range <= (bins * 4 / 3):
                        dense = bins == int_range
                        self.simpler = BinnerInteger(self.expression, min_value=vmin, max_value=vmax, dropmissing=not hashmap_unique.has_null, dense=dense, sort=sort, ascending=ascending)
                        return

                if vaex.dtype_of(self.bin_values) == int and len(self.bin_values):
                    min_value, max_value = self.bin_values.min(), self.bin_values.max()
                    self.bin_values = self.bin_values.astype(vaex.utils.required_dtype_for_range(min_value, max_value))
                logger.debug('Constructed grouper for expression %s with %i values', str(expression), len(self.bin_values))

                if self.sort:
                    if pre_sort:
                        hashmap_unique, self.bin_values = hashmap_unique.sorted(keys=self.bin_values, ascending=ascending, return_keys=True)
                        self.sort_indices = None
                    else:
                        indices = pa.compute.sort_indices(self.bin_values, sort_keys=[("x", "ascending" if ascending else "descending")])
                        self.sort_indices = vaex.array_types.to_numpy(indices)
                        # the bin_values will still be pre sorted, maybe that is confusing (implementation detail)
                        self.bin_values = pa.compute.take(self.bin_values, self.sort_indices)
                else:
                    self.sort_indices = None
                self.hashmap_unique = hashmap_unique

                self.basename = 'hashmap_unique_%s' % vaex.utils._python_save_name(str(self.expression) + "_" + hashmap_unique.fingerprint)

                self.N = len(self.bin_values)
                # for datetimes, we converted to int
                if dtype.is_datetime:
                    self.bin_values = dtype.create_array(self.bin_values)
            self._promise = process(df_original._hash_map_unique(self.expression, limit=row_limit, delay=True, progress=self.progressbar))

    def __repr__(self):
        return f"vaex.groupby.Grouper({str(self.expression)})"

    def _create_binner(self, df):
        # TODO: we modify the dataframe in place, this is not nice
        assert df.dataset == self.df.dataset, "you passed a dataframe with a different dataset to the grouper/binned"
        self.df = df
        if self.basename not in self.df.variables:
            self.hash_map_unique_name = df.add_variable(self.basename, self.hashmap_unique, unique=True)
        else:
            self.hash_map_unique_name = self.basename
        if _EXPERIMENTAL_BINNER_HASH:
            self.binby_expression = str(self.expression)
            self.binner = self.df._binner_hash(self.binby_expression, self.hashmap_unique)
        else:
            self.binby_expression = "_ordinal_values(%s, %s)" % (self.expression, self.hash_map_unique_name)
            self.binner = self.df._binner_ordinal(self.binby_expression, self.N)
        self.combine_expression = self.binby_expression

    def extract_center(self, dim, ar):
        slices = [
            slice(None, None),
        ] * vaex.array_types.ndim(ar)
        if _EXPERIMENTAL_BINNER_HASH:
            slices[dim] = slice(1, -1)
        else:
            slices[dim] = slice(0, -2)  # remove null and nan, actually null in grouper is part of the 'dictionary' (self.bin_values)
        return vaex.array_types.getitem(ar, tuple(slices))


class GrouperCombined(Grouper):
    dense = True

    def __init__(self, expression, df, multipliers, parents, sort, row_limit=None, progress=None):
        '''Will group by 1 expression, which is build up from multiple expressions.

        Used in the sparse/combined group by.
        '''
        super().__init__(expression, df, sort=sort, row_limit=row_limit, progress=progress, allow_simplify=False)
        assert len(multipliers) == len(parents)
        self.parents = parents

        assert multipliers[-1] == 1
        self.df = df
        for parent in parents:
            assert isinstance(parent, (Grouper, GrouperCategory, BinnerInteger, GrouperLimited)), "only (Grouper, GrouperCategory) supported for combining"
        self.label = 'SHOULD_NOT_BE_USED'
        self.expression = expression
        # efficient way to find the original bin values (parent.bin_value) from the 'compressed'
        # self.bin_values
        progressbar = self.progressbar.add("extract labels from sparse set")
        @vaex.delayed
        def process(_ignore):
            logger.info(f"extracing indices of parent groupers ({self.N:,} unique rows)")
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
            logger.info(f"extracing labels of parent groupers...")
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
            logger.info(f"extracing labels of parent groupers done")
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

    dense = True

    def __init__(self, expression, df=None, sort=False, ascending=True, row_limit=None, pre_sort=True):
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
        dtype = self.expression.dtype  # should be the dtype of the 'codes'
        if self.sort:
            # not pre-sorting is faster
            sort_indices = pa.compute.sort_indices(self.bin_values, sort_keys=[("x", "ascending" if ascending else "descending")])
            self.bin_values = pa.compute.take(self.bin_values, sort_indices)
            if self.pre_sort:
                # we will map from int to int
                sort_indices = vaex.array_types.to_numpy(sort_indices)
                fingerprint = self.expression.fingerprint() + "-grouper-sort-mapper"
                self.hash_map_unique = vaex.hash.HashMapUnique.from_keys(sort_indices + self.min_value, fingerprint=fingerprint)
                self.min_value = 0
                self.sort_indices = None
                self.basename = "hash_map_unique_%s" % vaex.utils._python_save_name(str(self.expression) + "_" + self.hash_map_unique.fingerprint)
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
                self.var_name = df.add_variable(self.basename, self.hash_map_unique, unique=True)
            else:
                self.var_name = self.basename
            self.binby_expression = "_ordinal_values(%s, %s)" % (self.expression, self.var_name)
            self.binner = self.df._binner_ordinal(self.binby_expression, self.N, 0)
        else:
            self.binby_expression = str(self.expression)
            self.binner = self.df._binner_ordinal(self.binby_expression, self.N, self.min_value)
        self.combine_expression = self.binby_expression

    def extract_center(self, dim, ar):
        slices = [
            slice(None, None),
        ] * vaex.array_types.ndim(ar)
        slices[dim] = slice(0, -2)  # again, null is in the dictionary
        return vaex.array_types.getitem(ar, tuple(slices))

class GrouperLimited(BinnerBase):
    """Group to a limited set of values, store the rest in an (optional) other bin"""

    dense = True

    def __init__(self, expression, values, keep_other=True, other_value=None, sort=False, ascending=True, label=None, df=None):
        self.df = df or expression.df
        self.sort = sort
        self.pre_sort = True
        self.expression = self.df[str(expression)]
        self.label = label or self.expression._label
        self.keep_other = keep_other
        if isinstance(values, pa.ChunkedArray):
            values = pa.concat_arrays(values.chunks)
        if sort:
            indices = pa.compute.sort_indices(values, sort_keys=[("x", "ascending" if ascending else "descending")])
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
        fp = vaex.cache.fingerprint(values)
        fingerprint = f"set-grouper-fixed-{fp}"
        self.hash_map_unique = vaex.hash.HashMapUnique.from_keys(self.values, fingerprint=fingerprint)

        self.basename = "hash_map_unique_%s" % vaex.utils._python_save_name(str(self.expression) + "_" + self.hash_map_unique.fingerprint)
        self.sort_indices = None
        self._promise = vaex.promise.Promise.fulfilled(None)

    def _create_binner(self, df):
        assert df.dataset == self.df.dataset, "you passed a dataframe with a different dataset to the grouper/binned"
        self.df = df
        if self.basename not in self.df.variables:
            self.var_name = df.add_variable(self.basename, self.hash_map_unique, unique=True)
        else:
            self.var_name = self.basename
        # modulo N will map -1 (value not found) to N-1
        self.binby_expression = "_ordinal_values(%s, %s) %% %s" % (self.expression, self.var_name, self.N)
        self.combine_expression = self.binby_expression
        self.binner = self.df._binner_ordinal(self.binby_expression, self.N)

    def extract_center(self, dim, ar):
        # gets rid of the nan and out of bound values
        # indices = n
        slices = [
            slice(None, None),
        ] * vaex.array_types.ndim(ar)
        slices[dim] = slice(0, -2)  # remove null values
        return vaex.array_types.getitem(ar, tuple(slices))


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

    for grouper in groupers:
        assert isinstance(grouper, (Grouper, GrouperCategory, BinnerInteger, GrouperLimited)), f"not supported binner {type(grouper)}"

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

    logger.info("combining groupers, now %r, next %r", combine_now, combine_later)
    combine_expressions = [df[k.combine_expression] for k in combine_now]
    for i in range(0, len(combine_expressions)):
        combine_expression = combine_expressions[i]
        dtype = vaex.utils.required_dtype_for_max(cumulative_counts[i])
        if isinstance(combine_now[i], (GrouperCategory, BinnerInteger)) and combine_now[i].min_value != 0:
            combine_expression -= combine_now[i].min_value
        combine_expression = combine_expression.astype(str(dtype))
        if cumulative_counts[i + 1] != 1:
            combine_expression = combine_expression * cumulative_counts[i + 1]
        combine_expressions[i] = combine_expression
    expression = reduce(operator.add, combine_expressions)
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


def grouper(df, column, sort=False, ascending=True, row_limit=None, df_original=None, progress=None):
    if df_original is None:
        df_original = df
    expression = df[_ensure_string_from_expression(column)]
    if df.is_category(column):
        by_value = GrouperCategory(expression, sort=sort, ascending=ascending, row_limit=row_limit)
    else:
        dtype = expression.dtype
        if dtype == np.dtype("uint8") or dtype == np.dtype("int8") or dtype == np.dtype("bool"):
            by_value = BinnerInteger(expression, sort=sort, ascending=ascending)  # always sorted, and pre_sorted
        else:
            # we cannot mix _combine with BinnerInteger yet
            by_value = Grouper(expression, sort=sort, ascending=ascending, row_limit=row_limit, df_original=df_original, progress=progress, allow_simplify=True)
    return by_value

class GroupByBase(object):
    def __init__(self, df, by, sort=False, ascending=True, combine=False, expand=True, row_limit=None, copy=True, progress=None):
        '''Note that row_limit only works in combination with combine=True'''
        df_original = df
        if copy:
            df = df.copy() # we will mutate the df (Add variables), this will keep the original dataframe unchanged
        self.df = df
        self.expand = expand  # keep as pyarrow struct?
        self.progressbar = vaex.utils.progressbars(progress)
        self.progressbar_groupers = self.progressbar.add("groupers")

        if by is None:
            by = []
        elif not isinstance(by, collections_abc.Iterable)\
            or isinstance(by, six.string_types):
            by = [by]

        if isinstance(ascending, (list, tuple)):
            assert len(ascending) == len(by), 'If "ascending" is a list, it must have the same number of elements as "by".'
        else:
            ascending = [ascending] * len(by)

        if isinstance(sort, (list, tuple)):
            assert len(sort) == len(by), 'If "sort" is a list, it must have the same number of elements as "by".'
        else:
            sort = [sort] * len(by)
        self.sort = sort
        self.ascending = ascending

        self.by = []
        self.by_original = by
        for by_value, sort, ascending in zip(by, self.sort, self.ascending):
            if not isinstance(by_value, BinnerBase):
                expression = df[_ensure_string_from_expression(by_value)]
                by_value = grouper(df, by_value, row_limit=row_limit, df_original=df_original, progress=self.progressbar_groupers, sort=sort, ascending=ascending)
            self.by.append(by_value)
        @vaex.delayed
        def possible_combine(*binner_promises):
            # if a binner realized there is a simpler way (e.g. grouper -> intbinner)
            self.by = [by.simpler if by.simpler is not None else by for by in self.by]
            # because binners can be created from other dataframes (we make a copy)
            # we let it mutate *our* dataframe
            for binner in self.by:
                binner._create_binner(self.df)
            cells = product([grouper.N for grouper in self.by])
            @vaex.delayed
            def set_combined(combined):
                combined._create_binner(self.df)
                self.by = [combined]
                self.combine = True
            if ((row_limit is not None) or (combine is True)) and len(self.by) >= 2 and cells > 0:
                promise = set_combined(_combine(self.df, self.by, sort=sort, row_limit=row_limit, progress=self.progressbar_groupers))
            elif combine == 'auto' and len(self.by) >= 2:
                # default assume we cannot combined
                self.combine = False
                promise = vaex.promise.Promise.fulfilled(None)
                # don't even try when one grouper has 0 options
                if cells > 0:
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
            @vaex.delayed
            def process(_ignore):
                self.dense = len(self.by) == 1 and self.by[0].dense
                self.groupby_expression = [str(by.expression) for by in self.by]
                self.binners = tuple(by.binner for by in self.by)
                self.shape = [by.N for by in self.by]
                self.dims = [by.label for by in self.by]
                self.progressbar_groupers(1)
            return process(promise)

        self._promise_by = self.progressbar_groupers.exit_on(possible_combine(*[by._promise for by in self.by]))

    @property
    def _coords1d(self):
        return [k.bin_values for k in self.by]

    def _agg(self, actions, progressbar_agg):
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
            values = df._agg(aggregate, self.binners, delay=_USE_DELAY, progress=progressbar_agg)
            grids[column_name] = values
            if isinstance(aggregate, vaex.agg.AggregatorDescriptorBasic)\
                and aggregate.name == 'AggCount'\
                and aggregate.expressions == []\
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
        >>> import vaex.datasets
        >>> df = vaex.datasets.titanic()
        >>> g1 = df.groupby(by='pclass')
        >>> df_group1 = g1.get_group(1)
        >>> df_group1.head(3)
          #    pclass  survived    name                            sex         age    sibsp    parch    ticket     fare  cabin    embarked    boat      body  home_dest
          0         1  True        Allen, Miss. Elisabeth Walton   female  29             0        0     24160  211.338  B5       S           2          nan  St Louis, MO
          1         1  True        Allison, Master. Hudson Trevor  male     0.9167        1        2    113781  151.55   C22 C26  S           11         nan  Montreal, PQ / Chesterville, ON
          2         1  False       Allison, Miss. Helen Loraine    female   2             1        2    113781  151.55   C22 C26  S           --         nan  Montreal, PQ / Chesterville, ON

        >>> df = vaex.datasets.titanic()
        >>> g2 = df.groupby(by=['pclass', 'sex'])
        >>> df_group2 = g2.get_group([1, 'female'])
        >>> df_group2.head(3)
          #    pclass  survived    name                                             sex       age    sibsp    parch    ticket     fare  cabin    embarked    boat      body  home_dest
          0         1  True        Allen, Miss. Elisabeth Walton                    female     29        0        0     24160  211.338  B5       S           2          nan  St Louis, MO
          1         1  False       Allison, Miss. Helen Loraine                     female      2        1        2    113781  151.55   C22 C26  S           --         nan  Montreal, PQ / Chesterville, ON
          2         1  False       Allison, Mrs. Hudson J C (Bessie Waldo Daniels)  female     25        1        2    113781  151.55   C22 C26  S           --         nan  Montreal, PQ / Chesterville, ON

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

    def _extract_center(self, array):
        # take out the edges
        for i, by in enumerate(self.by):
            array = by.extract_center(i, array)
        return array


class BinBy(GroupByBase):
    """Implementation of the binning and aggregation of data, see :method:`binby`."""

    def __init__(self, df, by, sort=False, ascending=True, progress=None, copy=True):
        super(BinBy, self).__init__(df, by, sort=sort, ascending=ascending, progress=progress, copy=copy)

    def agg(self, actions, merge=False, delay=False, progress=None):
        progressbar_agg = vaex.progress.tree(progress)
        import xarray as xr

        @vaex.delayed
        def process(arrays):
            # take out the edges
            arrays = {key: self._extract_center(value) for key, value in arrays.items()}
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

        result = process(self._agg(actions, progressbar_agg))
        progressbar_agg.exit_on(result)
        if delay:
            return result
        else:
            self.df.execute()
            return result.get()

class GroupBy(GroupByBase):
    """Implementation of the binning and aggregation of data, see :method:`groupby`."""

    def __init__(self, df, by, sort=False, ascending=True, combine=False, expand=True, row_limit=None, copy=True, progress=None):
        super(GroupBy, self).__init__(df, by, sort=sort, ascending=ascending, combine=combine, expand=expand, row_limit=row_limit, copy=copy, progress=progress)

    def agg(self, actions, delay=False, progress=None):
        progressbar = vaex.progress.tree(progress, title="aggregators")
        # TODO: this basically forms a cartesian product, we can do better, use a
        # 'multistage' hashmap
        @vaex.delayed
        def aggregate(promise_by):
            arrays = super(GroupBy, self)._agg(actions, progressbar)
            # we don't want non-existing pairs (e.g. Amsterdam in France does not exist)
            # but also, e.g. GrouperInteger will always expect missing values
            # but they may not aways exist
            counts = self.counts
            # nobody wanted to know count*, but we need it if we included non-existing pairs
            if counts is None and not self.dense:
                count_agg = vaex.agg.count(edges=True)
                counts = self.df._agg(count_agg, self.binners, delay=_USE_DELAY, progress=progressbar)
            arrays = delayed_dict(arrays)
            return counts, arrays

        @vaex.delayed
        def process(args):
            counts, arrays = args
            logger.info(f"aggregated on grid, constructing dataframe...")
            if counts is not None and vaex.array_types.is_scalar(counts):
                counts = np.array([counts])


            for name, array in list(arrays.items()):
                if array is not None and vaex.array_types.is_scalar(array):
                    array = np.array([array])
                    arrays[name] = array

            if counts is not None:
                for name, array in arrays.items():

                    shape1 = vaex.array_types.shape(array)
                    shape2 = vaex.array_types.shape(counts)
                    if shape1 != shape2:
                        raise RuntimeError(f'{array} {name} has shape {shape1} while we expected {shape2}')

            arrays = {key: self._extract_center(value) for key, value in arrays.items()}
            if not self.dense:
                counts = self._extract_center(counts)

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
                    assert vaex.array_types.ndim(value) == 1
                    columns[key] = value
            else:
                columns = {}
                if self.dense:
                    if len(self.by) == 1:
                        for by in self.by:
                            columns[by.label] = by.bin_values
                    else:
                        array0 = arrays[list(arrays)[0]]
                        # similar to the where, this creates indices like [0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]
                        indices = [k.ravel() for k in np.mgrid[[slice(0, n) for n in array0.shape]]]
                        for by, index in zip(self.by, indices):
                            columns[by.label] = vaex.array_types.take(by.bin_values, index)

                else:
                    counts = sort(counts)
                    mask = counts > 0
                    for by, indices in zip(self.by, np.where(mask)):
                        columns[by.label] = by.bin_values.take(indices)
                if self.dense or mask.sum() == mask.size:
                    # if we want all, just take it all
                    # should be faster
                    for key, value in arrays.items():
                        if vaex.array_types.ndim(value) > 1:
                            # only happens for numpy
                            columns[key] = value.ravel()
                        else:
                            columns[key] = value
                else:
                    for key, value in arrays.items():
                        columns[key] = vaex.array_types.filter(value, mask)
            logger.info(f"constructed dataframe")
            dataset_arrays = vaex.dataset.DatasetArrays(columns)
            dataset = DatasetGroupby(dataset_arrays, self.df, self.by_original, actions, combine=self.combine, expand=self.expand, sort=self.sort)
            df_grouped = vaex.from_dataset(dataset)
            return df_grouped
        result = process(delayed_list(aggregate(self._promise_by)))
        progressbar.exit_on(result)
        if delay:
            return result
        else:
            self.df.execute()
            return result.get()

    @docsubst
    def describe(self, expression=None):
        '''Return a dataframe with summary statistics per group for each of the expressions provided.

        Example:

        >>> import vaex
        >>> df = vaex.datasets.titanic()
        >>> df.groupby('pclass').describe('age')
          #    pclass    age_count    age_count_na    age_mean    age_std    age_min    age_max
          0         1          284              39     39.1599    14.5224     0.9167         80
          1         2          261              16     29.5067    13.6125     0.6667         70
          2         3          501             208     24.8164    11.9463     0.1667         74

        Short for:
        >>> df.groupby('pclass').agg({{'age': vaex.agg.describe('age')}}).struct.flatten()
          #    pclass    age_count    age_count_na    age_mean    age_std    age_min    age_max
          0         1          284              39     39.1599    14.5224     0.9167         80
          1         2          261              16     29.5067    13.6125     0.6667         70
          2         3          501             208     24.8164    11.9463     0.1667         74

        :param expression: {expression}
        :rtype: DataFrame
        '''
        if expression is None:
            expression = self.df.get_column_names()
        columns = vaex.utils._ensure_strings_from_expressions(expression)
        columns = vaex.utils._ensure_list(columns)
        aggs = {col: vaex.agg.describe(col) for col in columns}
        return self.agg(aggs).struct.flatten()


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
