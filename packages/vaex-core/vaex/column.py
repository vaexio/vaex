import logging

import numpy as np

logger = logging.getLogger("vaex.column")

class Column(object):
    pass


class ColumnVirtualRange(Column):
    def __init__(self, start, stop, step=1, dtype=None):
        self.start = start
        self.stop = stop
        self.step = step
        self.dtype = np.dtype(dtype)
        self.shape = (self.__len__(),)

    def __len__(self):
        return (self.stop - self.start) // self.step

    def __getitem__(self,  slice):
        start, stop, step = slice.start, slice.stop, slice.step
        return np.arange(self.start + start, self.start + stop, step, dtype=self.dtype)

    def trim(self, i1, i2):
        return ColumnVirtualRange(self.start + i1 * self.step, self.start + i2 * self.step, self.step, self.dtype)


class ColumnSparse(object):
    def __init__(self, matrix, column_index):
        self.matrix = matrix
        self.column_index = column_index
        self.shape = self.matrix.shape[:1]
        self.dtype = self.matrix.dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, slice):
        # not sure if this is the fastest
        return self.matrix[slice, self.column_index].A[:,0]

class ColumnIndexed(Column):
    def __init__(self, df, indices, name):
        self.df = df
        self.indices = indices
        self.name = name
        self.dtype = self.df.dtype(name)
        self.shape = (len(indices),)

    def __len__(self):
        return len(self.indices)

    def trim(self, i1, i2):
        return ColumnIndexed(self.df, self.indices[i1:i2], self.name)

    def __getitem__(self, slice):
        start, stop, step = slice.start, slice.stop, slice.step
        start = start or 0
        stop = stop or len(self)
        assert step in [None, 1]
        indices = self.indices[start:stop]
        ar = self.df.columns[self.name][indices]
        if np.ma.isMaskedArray(indices):
            mask = self.indices.mask[start:stop]
            return np.ma.array(ar, mask=mask)
        else:
            return ar


class ColumnConcatenatedLazy(Column):
    def __init__(self, dfs, column_name):
        self.dfs = dfs
        self.column_name = column_name
        dtypes = [df.dtype(column_name) for df in dfs]
        self.is_masked = any([df.is_masked(column_name) for df in dfs])
        if self.is_masked:
            self.fill_value = dfs[0].columns[self.column_name].fill_value
        # np.datetime64 and find_common_type don't mix very well
        all_strings = all([dtype == str for dtype in dtypes])
        if all_strings:
            self.dtype = str
        else:
            if all([dtype.type == np.datetime64 for dtype in dtypes]):
                self.dtype = dtypes[0]
            else:
                if all([dtype == dtypes[0] for dtype in dtypes]):  # find common types doesn't always behave well
                    self.dtype = dtypes[0]
                if  any([dtype.kind in 'SU' for dtype in dtypes]):  # strings are also done manually
                    if all([dtype.kind in 'SU' for dtype in dtypes]):
                        index = np.argmax([dtype.itemsize for dtype in dtypes])
                        self.dtype = dtypes[index]
                    else:
                        index = np.argmax([df.columns[self.column_name].astype('O').astype('U').dtype.itemsize for df in dfs])
                        self.dtype = dfs[index].columns[self.column_name].astype('O').astype('U').dtype
                else:
                    self.dtype = np.find_common_type(dtypes, [])
                logger.debug("common type for %r is %r", dtypes, self.dtype)
        self.shape = (len(self), ) + self.dfs[0].evaluate(self.column_name, i1=0, i2=1).shape[1:]
        for i in range(1, len(dfs)):
            shape_i = (len(self), ) + self.dfs[i].evaluate(self.column_name, i1=0, i2=1).shape[1:]
            if self.shape != shape_i:
                raise ValueError("shape of of column %s, array index 0, is %r and is incompatible with the shape of the same column of array index %d, %r" % (self.column_name, self.shape, i, shape_i))

    def __len__(self):
        return sum(len(ds) for ds in self.dfs)

    def __getitem__(self, slice):
        start, stop, step = slice.start, slice.stop, slice.step
        start = start or 0
        stop = stop or len(self)
        assert step in [None, 1]
        dtype = self.dtype
        if dtype == str:
            dtype = 'O'  # we store the strings in a dtype=object array
        dfs = iter(self.dfs)
        current_df = next(dfs)
        offset = 0
        # print "#@!", start, stop, [len(df) for df in self.dfs]
        while start >= offset + len(current_df):
            # print offset
            offset += len(current_df)
            # try:
            current_df = next(dfs)
            # except StopIteration:
            # logger.exception("requested start:stop %d:%d when max was %d, offset=%d" % (start, stop, offset+len(current_df), offset))
            # raise
            #   break
        # this is the fast path, no copy needed
        if stop <= offset + len(current_df):
            if current_df.filtered:  # TODO this may get slow! we're evaluating everything
                warnings.warn("might be slow, you have concatenated dfs with a filter set")
            return current_df.evaluate(self.column_name, i1=start - offset, i2=stop - offset)
        else:
            if self.is_masked:
                copy = np.ma.empty(stop - start, dtype=dtype)
                copy.fill_value = self.fill_value
            else:
                copy = np.zeros(stop - start, dtype=dtype)
            copy_offset = 0
            # print("!!>", start, stop, offset, len(current_df), current_df.columns[self.column_name])
            while offset < stop:  # > offset + len(current_df):
                # print(offset, stop)
                if current_df.filtered:  # TODO this may get slow! we're evaluating everything
                    warnings.warn("might be slow, you have concatenated DataFrames with a filter set")
                part = current_df.evaluate(self.column_name, i1=start-offset, i2=min(len(current_df), stop - offset))
                # print "part", part, copy_offset,copy_offset+len(part)
                copy[copy_offset:copy_offset + len(part)] = part
                # print copy[copy_offset:copy_offset+len(part)]
                offset += len(current_df)
                copy_offset += len(part)
                start = offset
                if offset < stop:
                    current_df = next(dfs)
            return copy


class ColumnStringArrow(Column):
    """Column that unpacks the arrow string column on the fly"""
    def __init__(self, indices, bytes, length=None, offset=0):
        self.indices = indices
        self.offset = offset  # to avoid memory copies in trim
        self.bytes = bytes
        self.dtype = str
        self.length = length if length is not None else len(indices) - 1
        self.shape = (self.__len__(),)
        self.nbytes = self.bytes.nbytes + self.indices.nbytes

    def __len__(self):
        return self.length

    def __getitem__(self, slice):
        if isinstance(slice, np.ndarray):
            assert slice.dtype.kind != 'b'
            strings = []
            for i in slice:
                i1 = self.indices[i] - self.offset
                i2 = self.indices[i+1] - self.offset
                s1 = self.bytes[i1:i2]
                m1 = memoryview(s1)
                strings.append(bytes(m1).decode('utf8'))
            return np.array(strings, dtype='O') # would be nice if we could do without
        else:
            start, stop, step = slice.start, slice.stop, slice.step
            start = start or 0
            stop = stop or len(self)
            assert step in [None, 1]
            # i = np.arange(start, stop, step)
            # start = self.indices[i]
            # stop = self.indices[i+1]
            strings = []
            # for i1, i2 in zip(start, stop):
            #     strings
            for i in range(start, stop):#, step):
                i1 = self.indices[i] - self.offset
                i2 = self.indices[i+1] - self.offset
                s1 = self.bytes[i1:i2]
                m1 = memoryview(s1)
                strings.append(bytes(m1).decode('utf8'))
            return np.array(strings, dtype='O') # would be nice if we could do without

    def trim(self, i1, i2):
        indices = self.indices[i1:i2+1]
        bytes = self.bytes[indices[0]:indices[-1]]
        return ColumnStringArrow(indices, bytes, offset=indices[0])

    def _zeros_like(self):
        return ColumnStringArrow(np.zeros_like(self.indices), np.zeros_like(self.bytes), self.length)