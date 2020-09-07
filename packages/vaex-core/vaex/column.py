import logging
import os
import warnings

import six
import numpy as np
import pyarrow as pa

import vaex
from .array_types import supported_array_types, supported_arrow_array_types, string_types, is_string_type

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if not on_rtd:
    import vaex.strings

logger = logging.getLogger("vaex.column")



class Column(object):
    def tolist(self):
        return self.to_numpy().tolist()

    def to_arrow(self, type=None):
        return pa.array(self, type=type)


supported_column_types = (Column, ) + supported_array_types


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


class ColumnMaskedNumpy(Column):
    def __init__(self, data, mask):
        self.data = data
        self.mask = mask
        assert len(data) == len(mask)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,  slice):
        data = self.data[slice]
        mask = self.mask[slice]
        return np.ma.array(data, mask=mask, shrink=False)

    def trim(self, i1, i2):
        return ColumnMaskedNumpy(self.data.trim(i1, i2), self.mask.trim(i1, i2))


class ColumnSparse(Column):
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



class ColumnArrowLazyCast(Column):
    """Wraps an array like object and cast it lazily"""
    def __init__(self, ar, type):
        self.ar = ar  # this should behave like a numpy array
        self.type = type

    @property
    def dtype(self):
        return vaex.array_types.to_numpy_type(self.type)

    def __len__(self):
        return len(self.ar)

    def trim(self, i1, i2):
        return type(self)(self.ar[i1:i2], self.type)

    def __getitem__(self, slice):
        if self.ar.dtype == object and vaex.array_types.is_string_type(self.type):
            # this seem to be the only way to convert mixed str and nan to include nulls
            return pa.Array.from_pandas(self.ar[slice], type=self.type)
        return pa.array(self.ar[slice], type=self.type)


class ColumnIndexed(Column):
    def __init__(self, column, indices, masked=False):
        self.column = column
        self.indices = indices
        self.dtype = vaex.array_types.data_type(column)
        self.shape = (len(indices),)
        self.masked = masked
        # this check is too expensive
        # max_index = self.indices.max()
        # if not np.ma.is_masked(max_index):
        #     assert max_index < self.df._length_original

    @staticmethod
    def index(column, indices, direct_indices_map=None, masked=False):
        """Creates a new column indexed by indices which avoids nested indices

        :param df: Dataframe where column comes from
        :param column: Column object or numpy array
        :param name: name of column
        :param indices: ndarray with indices which rows to take
        :param direct_indices_map: cache of the nested indices (pass a dict for better performance), defaults to None
        :return: A column object which avoids two levels of indirection
        :rtype: ColumnIndexed
        """
        direct_indices_map = direct_indices_map if direct_indices_map is not None else {}
        if isinstance(column, ColumnIndexed):
            if id(column.indices) not in direct_indices_map:
                direct_indices = column.indices[indices]
                if masked:
                    # restored sentinel mask values
                    direct_indices[indices == -1] = -1
                direct_indices_map[id(column.indices)] = direct_indices
            else:
                direct_indices = direct_indices_map[id(column.indices)]
            return ColumnIndexed(column.column, direct_indices, masked=masked or column.masked)
        else:
            return ColumnIndexed(column, indices, masked=masked)

    def __len__(self):
        return len(self.indices)

    def trim(self, i1, i2):
        return ColumnIndexed(self.column, self.indices[i1:i2], masked=self.masked)

    def __arrow_array__(self, type=None):
        # TODO: without a copy we get a buserror
        # TODO2: weird, this path is not triggered anymore
        # values = self[:]
        # if hasattr(values, "to_arrow"):
        #     return values.to_arrow()
        # else:
        return pa.array(self)

    def to_numpy(self):
        return np.array(self[:])

    def __getitem__(self, slice):
        start, stop, step = slice.start, slice.stop, slice.step
        start = start or 0
        stop = stop or len(self)
        assert step in [None, 1]
        indices = self.indices[start:stop]
        ar_unfiltered = self.column
        if self.masked:
            mask = indices == -1
        if isinstance(ar_unfiltered, Column):
            # TODO: this is a workaround, since we do not require yet
            # that Column classes know how to deal with indices, we get
            # the minimal slice, and get those (not the most efficient)
            if self.masked:
                unmasked_indices = indices[~mask]
                i1, i2 = np.min(unmasked_indices), np.max(unmasked_indices)
            else:
                i1, i2 = np.min(indices), np.max(indices)
            ar_unfiltered = ar_unfiltered[i1:i2+1]
            if self.masked:
                indices = indices - i1
            else:
                indices = indices - i1
        if isinstance(ar_unfiltered, supported_arrow_array_types):
            take_indices = indices
            if self.masked:
                # arrow does not like the -1 index, so we set them to 0
                take_indices = indices.copy()
                take_indices[mask] = 0
            ar = ar_unfiltered.take(vaex.array_types.to_arrow(take_indices))
        else:
            ar = ar_unfiltered[indices]
        assert not np.ma.isMaskedArray(indices)
        if self.masked:
            # TODO: we probably want to keep this as arrow array if it originally was
            return np.ma.array(ar, mask=mask)
        else:
            return ar


class ColumnConcatenatedLazy(Column):
    def __init__(self, expressions, dtype=None):
        self.is_masked = any([e.is_masked for e in expressions])
        if self.is_masked:
            for expression in expressions:
                if expression.is_masked:
                    try:
                        # fast path
                        self.fill_value = expression[0:1].fill_value
                        break
                    except:  # noqa
                        # slower path (we have to evaluate everything)
                        self.fill_value = expression.values.fill_value
                        break
            else:
                raise ValueError('Concatenating expressions with masked values, but no fill value is found')
        if dtype is None:
            dtypes = [e.dtype for e in expressions]

            any_strings = any([is_string_type(dtype) for dtype in dtypes])
            if any_strings:
                self.dtype = pa.string()  # TODO: how do we know it should not be large_string?
            else:
                # np.datetime64/timedelta64 and find_common_type don't mix very well
                if all([dtype.type == np.datetime64 for dtype in dtypes]):
                    self.dtype = dtypes[0]
                elif all([dtype.type == np.timedelta64 for dtype in dtypes]):
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
            # make sure all expression are the same type
            self.expressions = [e if vaex.array_types.same_type(e.dtype, self.dtype) else e.astype(self.dtype) for e in expressions]
        else:
            # if dtype is given, we assume every expression/column is the same dtype
            self.dtype = dtype
            self.expressions = expressions[:]
        self.shape = (len(self), ) + self.expressions[0][0:1].to_numpy().shape[1:]

        for i in range(1, len(self.expressions)):
            expression = self.expressions[i]
            shape_i = (len(self), ) + expressions[i][0:1].to_numpy().shape[1:]
            if self.shape != shape_i:
                raise ValueError("shape of of expression %s, array index 0, is %r and is incompatible with the shape of the same column of array index %d, %r" % (self.expressions[0], self.shape, i, shape_i))

    def to_arrow(self, type=None):
        values = [e.values for e in self.expressions]
        chunks = [value if isinstance(value, pa.Array) else pa.array(value, type=type) for value in values]
        types = [chunk.type for chunk in chunks]

        # upcast if mixed types
        if pa.string() in types and pa.large_string() in types:
            def _arrow_string_upcast(array):
                if array.type == pa.large_string():
                    return array
                if array.type == pa.string():
                    import vaex.arrow.convert
                    column = vaex.arrow.convert.column_from_arrow_array(array)
                    column.indices = column.indices.astype(np.int64)
                    return pa.array(column)
                else:
                    raise ValueError('Not a string type: %r' % array)
            chunks = [_arrow_string_upcast(chunk) for chunk in chunks]

        return pa.chunked_array(chunks)

    def __len__(self):
        return sum(len(e.df) for e in self.expressions)

    def trim(self, i1, i2):
        start, stop = i1, i2
        i = 0  # the current expression
        offset = 0  # and the offset the current expression has wrt the dataframe
        # find the first expression we overlap with
        while start >= offset + len(self.expressions[i].df):
            offset += len(self.expressions[i].df)
            i += 1
        # do need more than 1?
        if stop > offset + len(self.expressions[i].df):
            # then add the first one
            expressions = [self.expressions[i][start-offset:]]
            offset += len(self.expressions[i].df)
            # we need more expression
            i += 1
            # keep adding complete expression till we find the it to be the last
            while stop > offset + len(self.expressions[i].df):
                offset += len(self.expressions[i].df)
                expressions.append(self.expressions[i])
                i += 1
            if stop > offset:  # add the tail part
                expressions.append(self.expressions[i][:stop-offset])
        else:
            # otherwise we only need a slice of the first
            expressions = [self.expressions[i][start-offset:stop-offset]]

        return ColumnConcatenatedLazy(expressions, self.dtype)

    def __arrow_array__(self, type=None):
        return pa.array(self[:], type=type)

    def __getitem__(self, slice):
        start, stop, step = slice.start, slice.stop, slice.step
        start = start or 0
        stop = stop or len(self)
        assert step in [None, 1]
        dtype = self.dtype
        if is_string_type(dtype):
            dtype = 'O'  # we store the strings in a dtype=object array
        expressions = iter(self.expressions)
        current_expression = next(expressions)
        offset = 0
        while start >= offset + len(current_expression.df):
            offset += len(current_expression.df)
            current_expression = next(expressions)
        # this is the fast path, no copy needed
        if stop <= offset + len(current_expression.df):
            if current_expression.df.filtered:  # TODO this may get slow! we're evaluating everything
                warnings.warn("might be slow, you have concatenated expressions with a filter set")
            values = current_expression.evaluate(i1=start - offset, i2=stop - offset, parallel=False)
        else:
            if self.is_masked:
                copy = np.ma.empty(stop - start, dtype=dtype)
                copy.fill_value = self.fill_value
            else:
                copy = np.zeros(stop - start, dtype=dtype)
            copy_offset = 0
            while offset < stop:  # > offset + len(current_expression):
                if current_expression.df.filtered:  # TODO this may get slow! we're evaluating everything
                    warnings.warn("might be slow, you have concatenated DataFrames with a filter set")
                part = current_expression.evaluate(i1=start-offset, i2=min(len(current_expression.df), stop - offset), parallel=False)
                copy[copy_offset:copy_offset + len(part)] = part
                offset += len(current_expression.df)
                copy_offset += len(part)
                start = offset
                if offset < stop:
                    current_expression = next(expressions)
            values = copy
        if is_string_type(dtype):
            return _to_string_column(values)
        else:
            return values


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
    if vaex.array_types.is_string(x):
        return True
    elif isinstance(x, ColumnString):
        return True
    elif isinstance(x, np.ndarray):
        if x.dtype.kind in 'US':
            return True
    return False


def _to_string_sequence(x, force=True):
    if isinstance(x, pa.ChunkedArray):
        # turn into pa.Array, TODO: do we want this, this may result in a big mem copy
        table = pa.Table.from_arrays([x], ["single"])
        table_concat = table.combine_chunks()
        column = table_concat.columns[0]
        assert column.num_chunks == 1
        x = column.chunk(0)

    if isinstance(x, ColumnString):
        return x.string_sequence
    elif isinstance(x, pa.Array):
        from vaex.arrow import convert
        return convert.column_from_arrow_array(x).string_sequence
    elif isinstance(x, np.ndarray):
        mask = None
        if np.ma.isMaskedArray(x):
            mask = np.ma.getmaskarray(x)
            x = x.data
        if x.dtype == 'O':
            return vaex.strings.StringArray(x) if mask is None else vaex.strings.StringArray(x, mask)
        elif x.dtype.kind in 'US':
            x = x.astype('O')
            return vaex.strings.StringArray(x) if mask is None else vaex.strings.StringArray(x, mask)
        else:
            # This path is only required because the str_pandas wrapper uses NaN for missing values
            # see pandas_wrapper in functions.py
            if force:
                length = len(x)
                bytes = np.zeros((0,), dtype=np.uint8)
                indices = np.zeros((length+1,), dtype=np.int64)
                null_bitmap = np.ones(((length + 7) // 8,), dtype=np.uint8)
                return vaex.strings.StringList64(bytes, indices, length, 0, null_bitmap, 0)
            else:
                ValueError('unsupported dtype ' +str(x.dtype))
    elif isinstance(x, (list, type)):
        return _to_string_sequence(np.array(x))
    else:
        raise ValueError('not a ColumnString or ndarray: ' + str(x))

def _to_string_column(x, force=True):
    ss = _to_string_sequence(x, force=force)
    if not isinstance(ss, (vaex.strings.StringList64, vaex.strings.StringList32)):
        ss = ss.to_arrow()
    return ColumnStringArrow.from_string_sequence(ss)

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
        self.length = length if length is not None else len(indices) - 1
        if indices.dtype.kind == 'i' and indices.dtype.itemsize == 8:
            self.dtype = pa.large_string()
        elif indices.dtype.kind == 'i' and indices.dtype.itemsize == 4:
            self.dtype = pa.string()
        else:
            raise ValueError('unsupported index type' + str(indices.dtype))
        self.shape = (self.__len__(),)
        self.nbytes = self.bytes.nbytes + self.indices.nbytes
        self.null_bitmap = null_bitmap
        self.null_offset = null_offset
        # references is to keep other objects alive, similar to pybind11's keep_alive
        self.references = references or []

        if not (self.indices.dtype.kind == 'i' and self.indices.dtype.itemsize in [4,8]):
            raise ValueError('unsupported index type' + str(self.indices.dtype))

    def __arrow_array__(self, type=None):
        indices = self.indices
        type = type or self.dtype
        if type == pa.string() and self.dtype == pa.large_string():
            indices = indices.astype(np.int32)  # downcast
        elif type == pa.large_string() and self.dtype == pa.string():
            type = pa.string()  # upcast
        # TODO: we dealloc the memory in the C++ extension, so we need to copy for now
        buffers = [None, pa.py_buffer(_asnumpy(indices).copy() - self.offset), pa.py_buffer(_asnumpy(self.bytes).view(np.uint8).copy()), ]
        if self.null_bitmap is not None:
            assert self.null_offset == 0 #self.offset
            buffers[0] = pa.py_buffer(self.null_bitmap.copy())
        arrow_array = pa.Array.from_buffers(type, self.length, buffers=buffers)
        return arrow_array

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
            if np.ma.isMaskedArray(slice):
                if slice.dtype == np.bool_:
                    ss = self.string_sequence.index(slice.data & ~slice.mask)
                else:
                    ss = self.string_sequence.index(slice.data, slice.mask)
            else:
                ss = self.string_sequence.index(slice)
            return type(self).from_string_sequence(ss)
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

    @classmethod
    def from_arrow(cls, ar):
        return cls.from_string_sequence(_to_string_sequence(ar))

    def _zeros_like(self):
        return ColumnStringArrow(np.zeros_like(self.indices), np.zeros_like(self.bytes), self.length, null_bitmap=self.null_bitmap)

    def get_mask(self):
        return self.string_sequence.mask()

    def astype(self, type):
        return self.to_numpy().astype(type)