import logging
import os

import six
import numpy as np

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if not on_rtd:
    import vaex.strings

logger = logging.getLogger("vaex.column")


class Column(object):
    def tolist(self):
        return self.to_numpy().tolist()


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

class ColumnNumpyLike(Column):
    """Wraps a numpy like object (like hdf5 dataset) into a column vaex understands"""
    def __init__(self, ar):
        self.ar = ar  # this should behave like a numpy array

    def __len__(self):
        return len(self.ar)

    def trim(self, i1, i2):
        return type(self)(self.ar[i1:i2])

    def __getitem__(self, slice):
        return self.ar[slice]

    def __setitem__(self, slice, value):
        self.ar[slice] = value

class ColumnIndexed(Column):
    def __init__(self, df, indices, name):
        self.df = df
        self.indices = indices
        self.name = name
        self.dtype = self.df.dtype(name)
        self.shape = (len(indices),)
        max_index = self.indices.max()
        assert max_index < self.df._length_unfiltered

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
        ar_unfiltered = self.df.columns[self.name]
        ar = ar_unfiltered[indices]
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
        any_strings = any([dtype == str_type for dtype in dtypes])
        if any_strings:
            self.dtype = str_type
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
        if dtype == str_type:
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

try:
    str_type = unicode
except:
    str_type = str

use_c_api = True

class ColumnString(Column):
    pass

class ColumnStringArray(Column):
    """Wraps a numpy array with dtype=object, containing all strings"""
    def __init__(self, array):
        # assert vaex.strings.all_strings(array, allow)
        self.array = array
        self.string_sequence = vaex.strings.StringArray(self.array)

    def __getitem__(self, slice):
        return ColumnString(self.array[slice])

    def to_numpy(self):
        return self.array

def _to_string(string_bytes):
    if six.PY2:
        return unicode(buffer(string_bytes))
    else:
        return bytes(memoryview(string_bytes)).decode('utf8')


def _is_stringy(x):
    if isinstance(x, ColumnString):
        return True
    elif isinstance(x, np.ndarray):
        if x.dtype.kind in 'US':
            return True
    return False


def _to_string_sequence(x):
    if isinstance(x, ColumnString):
        return x.string_sequence
    elif isinstance(x, np.ndarray):
        if x.dtype == 'O':
            return vaex.strings.StringArray(x)
        elif x.dtype.kind in 'US':
            x = x.astype('O')
            return vaex.strings.StringArray(x)
        else:
            raise ValueError('unsupported dtype ' +str(x.dtype))
    else:
        raise ValueError('not a ColumnString or ndarray: ' + str(x))

def _to_string_list_sequence(x):
    if isinstance(x, vaex.strings.StringListList):
        return x
    else:
        raise ValueError('not a StringListList')

def _asnumpy(ar):
    if isinstance(ar, np.ndarray):
        return ar
    else:
        return ar.to_numpy()


def _trim(column, i1, i2):
    if isinstance(column, np.ndarray):
        return column[i1:i2]
    else:
        return column.trim(i1, i2)

def _trim_bits(column, i1, i2):
    i1_bytes = i1 // 8
    offset = i1 - i1_bytes * 8
    i2_bytes = (i2 + 7) // 8
    if isinstance(column, np.ndarray):
        return column[i1_bytes:i2_bytes], offset
    else:
        return column.trim(i1_bytes, i2_bytes), offset

class ColumnStringArrow(ColumnString):
    """Column that unpacks the arrow string column on the fly"""
    def __init__(self, indices, bytes, length=None, offset=0, string_sequence=None, null_bitmap=None, null_offset=0, references=None):
        self._string_sequence = string_sequence
        self.indices = indices
        self.offset = offset  # to avoid memory copies in trim
        self.bytes = bytes
        self.dtype = str_type
        self.length = length if length is not None else len(indices) - 1
        self.shape = (self.__len__(),)
        self.nbytes = self.bytes.nbytes + self.indices.nbytes
        self.null_bitmap = null_bitmap
        self.null_offset = null_offset
        # references is to keep other objects alive, similar to pybind11's keep_alive
        self.references = references or []

    @property
    def string_sequence(self):
        if self._string_sequence is None:
            if self.indices.dtype.kind == 'i' and self.indices.dtype.itemsize == 8:
                string_type = vaex.strings.StringList64
            elif self.indices.dtype.kind == 'i' and self.indices.dtype.itemsize == 4:
                string_type = vaex.strings.StringList32
            else:
                raise ValueError('unsupported index type' + str(self.indices.dtype))
            if self.null_bitmap is not None:
                self._string_sequence = string_type(_asnumpy(self.bytes), _asnumpy(self.indices), self.length, self.offset, _asnumpy(self.null_bitmap), self.null_offset)
            else:
                self._string_sequence = string_type(_asnumpy(self.bytes), _asnumpy(self.indices), self.length, self.offset)
        return self._string_sequence

    def __len__(self):
        return self.length

    # TODO: in the future we might want to rely on these methods instead of special branches
    # in Expression for == and +. However, that gives different behaviour for dtype=object
    # (for instance you cannot use + when there is a None in it)
    def __eq__(self, other):
        if not isinstance(other, six.string_types):
            other = _to_string_sequence(other)
        return self.string_sequence.equals(other)

    def __add__(self, rhs):
        if not isinstance(rhs, six.string_types):
            rhs = _to_string_sequence(rhs)
        return self.from_string_sequence(self.string_sequence.concat(rhs))

    def __radd__(self, lhs):
        return self.from_string_sequence(self.string_sequence.concat_reverse(lhs))

    def __getitem__(self, slice):
        if isinstance(slice, int):
            return self.string_sequence.get(slice)
        elif isinstance(slice, np.ndarray):
            return type(self).from_string_sequence(self.string_sequence.index(slice))
        else:
            start, stop, step = slice.start, slice.stop, slice.step
            start = start or 0
            stop = stop or len(self)
            assert step in [None, 1]
            return self.trim(start, stop)

    def to_numpy(self):
        return self.string_sequence.to_numpy()

    def trim(self, i1, i2):
        byte_offset = self.indices[i1:i1+1][0] - self.offset
        byte_end = self.indices[i2:i2+1][0] - self.offset
        indices = _trim(self.indices, i1, i2+1)
        bytes = _trim(self.bytes, byte_offset, byte_end)
        null_bitmap = self.null_bitmap
        null_offset = self.null_offset
        if null_bitmap is not None:
            null_bitmap, null_offset = _trim_bits(self.null_bitmap, i1, i2)
        references = self.references + [self]
        return type(self)(indices, bytes, i2-i1, self.offset + byte_offset, null_bitmap=null_bitmap,
                null_offset=null_offset, references=references)

    @classmethod
    def from_string_sequence(cls, string_sequence):
        s = string_sequence
        return cls(s.indices, s.bytes, s.length, s.offset, string_sequence=s, null_bitmap=s.null_bitmap)

    def _zeros_like(self):
        return ColumnStringArrow(np.zeros_like(self.indices), np.zeros_like(self.bytes), self.length, null_bitmap=self.null_bitmap)

    def get_mask(self):
        return self.string_sequence.mask()
