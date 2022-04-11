import numpy as np
from vaex.dataframe import DataFrame

import vaex.dataset
from vaex.utils import _ensure_string_from_expression

@vaex.dataset.register
class DatasetJoin(vaex.dataset.DatasetDecorator):
    snake_name = "join"
    _no_serialize = '_columns _ids original _row_count _id _cached_fingerprint'.split()
    def __init__(self, original, left, right, on=None, left_on=None, right_on=None, lprefix='', rprefix='', lsuffix='', rsuffix='', how='left', allow_duplication=False, prime_growth=False, cardinality_other=None, slice_start=None, slice_end=None):
        super().__init__(original)
        self.left = left
        self.right = right
        # trigger length computation, otherwise it will be triggered in .slice, which can lead
        # to nested executions
        len(left)
        len(right)
        # len(left)
        # len(right)
        self.on = on
        self.left_on = left_on
        self.right_on = right_on
        self.lprefix = lprefix
        self.rprefix = rprefix
        self.lsuffix = lsuffix
        self.rsuffix = rsuffix
        self.how = how
        self.allow_duplication = allow_duplication
        self.prime_growth = prime_growth
        self.cardinality_other = cardinality_other
        self.slice_start = slice_start
        self.slice_end = slice_end
        self._row_count = self.original.row_count
        self._create_columns()

    def _create_columns(self):
        # we know original is a DatasetArrays
        self._columns = self.original._columns.copy()
        self._ids = self.original._ids.copy()

    @property
    def _fingerprint(self):
        # kwargs = self.__dict__.copy()
        # for skip in self._no_serialize:
        #     kwargs.pop(skip)
        fp = vaex.cache.fingerprint({
            'left': self.left.fingerprint(),
            'right': self.right.fingerprint(),
            'on': str(self.on) if self.on is not None else None,
            'left_on': str(self.left_on) if self.left_on is not None else None,
            'right_on': str(self.right_on) if self.right_on is not None else None,
            'lprefix': self.lprefix,
            'rprefix': self.rprefix,
            'lsuffix': self.lsuffix,
            'rsuffix': self.rsuffix,
            'how': self.how,
            'allow_duplication': self.allow_duplication,
            'prime_growth': self.prime_growth,
            'cardinality_other': self.cardinality_other,
        })
        return f'dataset-{self.snake_name}-{fp}'

    def chunk_iterator(self, *args, **kwargs):
        yield from self.original.chunk_iterator(*args, **kwargs)

    def hashed(self):
        kwargs = self.__dict__.copy()
        for skip in self._no_serialize:
            kwargs.pop(skip, None)
        return type(self)(self.original.hashed(), **kwargs)

    def slice(self, start, end):
        if start == 0 and end == self.row_count:
            return self
        kwargs = self.__dict__.copy()
        kwargs['slice_start'] = start
        kwargs['slice_end'] = end
        for skip in self._no_serialize:
            kwargs.pop(skip, None)
        left = kwargs.pop('left')
        right = kwargs.pop('right')
        dataset = self.original.slice(start, end)
        return type(self)(dataset, left=left, right=right, **kwargs)

    def _encode(self, encoding):
        # by = self.by
        # by = str(by) if not isinstance(by, (list, tuple)) else list(map(str, by))
        spec = {
            'left': encoding.encode('dataframe', self.left),
            'right': encoding.encode('dataframe', self.right),
            'on': str(self.on) if self.on is not None else None,
            'left_on': str(self.left_on) if self.left_on is not None else None,
            'right_on': str(self.right_on) if self.right_on is not None else None,
            'lprefix': self.lprefix,
            'rprefix': self.rprefix,
            'lsuffix': self.lsuffix,
            'rsuffix': self.rsuffix,
            'how': self.how,
            'allow_duplication': self.allow_duplication,
            'prime_growth': self.prime_growth,
            'cardinality_other': self.cardinality_other,
            'slice_start': self.slice_start,
            'slice_end': self.slice_end,
        }
        return spec

    @classmethod
    def _decode(cls, encoding, spec):
        spec = spec.copy()
        left = encoding.decode('dataframe', spec.pop('left'))
        right = encoding.decode('dataframe', spec.pop('right'))
        spec_no_slice = spec.copy()
        del spec_no_slice['slice_start']
        del spec_no_slice['slice_end']
        dfj = left.join(right, **spec_no_slice)
        start = spec['slice_start']
        end = spec['slice_end']
        dataset = dfj.dataset.original
        if start is not None and end is not None:
            dataset = dataset.slice(start, end)
        return DatasetJoin(dataset, left, right, **spec)
        

def join(df, other, on=None, left_on=None, right_on=None, lprefix='', rprefix='', lsuffix='', rsuffix='', how='left', allow_duplication=False, prime_growth=False, cardinality_other=None, inplace=False):
    # implementation of DataFrameLocal.join
    inner = False
    left: DataFrame = df
    right: DataFrame = other
    left_original = left.copy()
    right_original = right.copy()
    rprefix_original, lprefix_original = rprefix, lprefix
    rsuffix_original, lsuffix_original = rsuffix, lsuffix
    right_on_original, left_on_original = right_on, left_on
    if how == 'left':
        pass
    elif how == 'right':
        left, right = right, left
        lprefix, rprefix = rprefix, lprefix
        lsuffix, rsuffix = rsuffix, lsuffix
        left_on, right_on = right_on, left_on
    elif how == 'inner':
        inner = True
    else:
        raise ValueError('join type not supported: {}, only left and right'.format(how))
    left = left if inplace else left.copy()

    on = _ensure_string_from_expression(on)
    left_on = _ensure_string_from_expression(left_on)
    right_on = _ensure_string_from_expression(right_on)
    left_on = left_on or on
    right_on = right_on or on
    for name in right:
        if left_on and (rprefix + name + rsuffix == lprefix + left_on + lsuffix):
            continue  # it's ok when we join on the same column name
        if name in left and rprefix + name + rsuffix == lprefix + name + lsuffix:
            raise ValueError('column name collision: {} exists in both column, and no proper suffix given'
                                .format(name))

    right = right.extract()  # get rid of filters and active_range
    assert left.length_unfiltered() == left.length_original()
    N = left.length_unfiltered()
    N_other = len(right)
    if left_on is None and right_on is None:
        lookup = None
    else:
        df = left
        # we index the right side, this assumes right is smaller in size
        index = right._index(right_on, prime_growth=prime_growth, cardinality=cardinality_other)
        dtype = left.data_type(left_on)
        duplicates_right = index.has_duplicates

        if duplicates_right and not allow_duplication:
            raise ValueError('This join will lead to duplication of rows which is disabled, pass allow_duplication=True')

        # our max value for the lookup table is the row index number, so if we join a small
        # df with say 100 rows, we can do it with a int8
        lookup_dtype = vaex.utils.required_dtype_for_max(len(right))
        # we put in the max value to maximize triggering failures in the case of a bug (we don't want
        # to point to row 0 in case we do, we'd rather crash)
        lookup = np.full(left._length_original, np.iinfo(lookup_dtype).max, dtype=lookup_dtype)
        nthreads = df.executor.thread_pool.nthreads
        lookup_masked = [False] * nthreads  # does the lookup contain masked/-1 values?
        lookup_extra_chunks = []

        from vaex.column import _to_string_sequence
        def map(thread_index, i1, i2, selection_masks, blocks):
            ar = blocks[0]
            if vaex.array_types.is_string_type(dtype):
                previous_ar = ar
                ar = _to_string_sequence(ar)
            if dtype.is_datetime:
                ar = ar.view(np.int64)
            if np.ma.isMaskedArray(ar):
                mask = np.ma.getmaskarray(ar)
                found_masked = index.map_index_masked(ar.data, mask, lookup[i1:i2])
                lookup_masked[thread_index] = lookup_masked[thread_index] or found_masked
                if duplicates_right:
                    extra = index.map_index_duplicates(ar.data, mask, i1)
                    lookup_extra_chunks.append(extra)
            else:
                found_masked = index.map_index(ar, lookup[i1:i2])
                lookup_masked[thread_index] = lookup_masked[thread_index] or found_masked
                if duplicates_right:
                    extra = index.map_index_duplicates(ar, i1)
                    lookup_extra_chunks.append(extra)
        def reduce(a, b):
            pass
        left.map_reduce(map, reduce, [left_on], delay=False, name='fill looking', info=True, to_numpy=False, ignore_filter=True)
        if len(lookup_extra_chunks):
            # if the right has duplicates, we increase the left of left, and the lookup array
            lookup_left = np.concatenate([k[0] for k in lookup_extra_chunks])
            lookup_right = np.concatenate([k[1] for k in lookup_extra_chunks])
            left = left.concat(left.take(lookup_left))
            lookup = np.concatenate([lookup, lookup_right])

        if inner:
            left_mask_matched = lookup != -1  # all the places where we found a match to the right
            lookup = lookup[left_mask_matched]  # filter the lookup table to the right
            left_indices_matched = np.where(left_mask_matched)[0]  # convert mask to indices for the left
            # indices can still refer to filtered rows, so do not drop the filter
            left = left.take(left_indices_matched, filtered=False, dropfilter=False)
    direct_indices_map = {}  # for performance, keeps a cache of two levels of indirection of indices

    def mangle_name(prefix, name, suffix):
        if name.startswith('__'):
            return '__' + prefix + name[2:] + suffix
        else:
            return prefix + name + suffix

    # first, do renaming, so all column names are unique
    right_names = right.get_names(hidden=True)
    left_names = left.get_names(hidden=True)
    for name in right_names:
        if name in left_names:
            # find a unique name across both dataframe, including the new name for the left
            all_names = list(set(right_names + left_names))
            all_names.append(mangle_name(lprefix, name, lsuffix))  # we dont want to steal the left's name
            all_names.remove(name)  # we could even claim the original name
            new_name = mangle_name(rprefix, name, rsuffix)
            # we will not add this column twice when it is the join column
            if new_name != left_on:
                if new_name in all_names:  # it's still not unique
                    new_name = vaex.utils.find_valid_name(new_name, all_names)
                right.rename(name, new_name)
                right_names[right_names.index(name)] = new_name

            # and the same for the left
            all_names = list(set(right_names + left_names))
            all_names.remove(name)
            new_name = mangle_name(lprefix, name, lsuffix)
            if new_name in all_names:  # still not unique
                new_name = vaex.utils.find_valid_name(new_name, all_names)
            left.rename(name, new_name)
            left_names[left_names.index(name)] = new_name

    # now we add columns from the right, to the left
    right_names = right.get_names(hidden=True)
    left_names = left.get_names(hidden=True)
    right_columns = []
    for name in right_names:
        column_name = name
        if name == left_on and name in left_names:
            continue  # skip when it's the join column
        assert name not in left_names
        if name in right.variables:
            left.set_variable(name, right.variables[name])
        elif column_name in right.virtual_columns:
            left.add_virtual_column(name, right.virtual_columns[column_name])
        elif column_name in right.functions:
            if name in left.functions:
                raise NameError(f'Name collision for function {name}')
            left.functions[name] = right.functions[name]
        else:
            right_columns.append(name)
            # we already add the column name here to get the same order
            left.column_names.append(name)
            left._initialize_column(name)
    # merge the two datasets
    if right_columns:
        right_dataset = right.dataset.project(*right_columns)
        if lookup is not None:
            # if lookup is None, we do a row based join
            # and we only need to merge.
            # if we have an array of lookup indices, we 'take' those
            right_dataset = right_dataset.take(lookup, masked=any(lookup_masked))
        dataset = left.dataset.merged(right_dataset)
    else:
        dataset = left.dataset
    # row number etc should not have changed, we only append new columns
    # so no need to reset caches
    left._dataset = DatasetJoin(dataset, left_original, right_original,
        on=on, left_on=left_on_original, right_on=right_on_original,
        lprefix=lprefix_original, rprefix=rprefix_original, lsuffix=lsuffix_original, rsuffix=rsuffix_original,
        how=how, allow_duplication=allow_duplication, prime_growth=prime_growth, cardinality_other=cardinality_other
    )
    return left
