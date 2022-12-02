"""
Implementation of the dataframe exchange protocol.

Public API
----------

from_dataframe : construct a vaex.dataframe.DataFrame from an input data frame which
                 implements the exchange protocol

For a background and spec, see:
   * https://data-apis.org/blog/dataframe_protocol_rfc/
   * https://data-apis.org/dataframe-protocol/latest/index.html

Notes
-----
- Interpreting a raw pointer (as in ``Buffer.ptr``) is annoying and unsafe to
  do in pure Python. It's more general but definitely less friendly than having
  ``to_arrow`` and ``to_numpy`` methods. So for the buffers which lack
  ``__dlpack__`` (e.g., because the column dtype isn't supported by DLPack),
  this is worth looking at again.
"""
import enum
import collections.abc
import ctypes
from typing import Any, Optional, Tuple, Dict, Iterable, Sequence

import vaex
import pyarrow
import pyarrow as pa
import numpy as np
import pandas as pd


# A typing protocol could be added later to let Mypy validate code using
# `from_dataframe` better.
DataFrameObject = Any
ColumnObject = Any


def from_dataframe_to_vaex(df: DataFrameObject, allow_copy: bool = True) -> vaex.dataframe.DataFrame:
    """
    Construct a vaex DataFrame from ``df`` if it supports ``__dataframe__``
    """
    if isinstance(df, vaex.dataframe.DataFrame):
        return df

    if not hasattr(df, "__dataframe__"):
        raise ValueError("`df` does not support __dataframe__")

    return _from_dataframe_to_vaex(df.__dataframe__(allow_copy=allow_copy))


def _from_dataframe_to_vaex(df: DataFrameObject) -> vaex.dataframe.DataFrame:
    """
    Note: we need to implement/test support for bit/byte masks, chunk handling, etc.
    """
    # Iterate through the chunks
    dataframe = []
    _buffers = []
    for chunk in df.get_chunks():

        # We need a dict of columns here, with each column being an expression.
        columns = dict()
        _k = _DtypeKind
        _buffers_chunks = []  # hold on to buffers, keeps memory alive
        for name in chunk.column_names():
            if not isinstance(name, str):
                raise ValueError(f"Column {name} is not a string")
            if name in columns:
                raise ValueError(f"Column {name} is not unique")

            col = chunk.get_column_by_name(name)
            if col.dtype[0] in (_k.INT, _k.UINT, _k.FLOAT, _k.BOOL):
                # Simple numerical or bool dtype, turn into arrow array
                columns[name], _buf = convert_column_to_ndarray(col)
            elif col.dtype[0] == _k.CATEGORICAL:
                columns[name], _buf = convert_categorical_column(col)
            elif col.dtype[0] == _k.STRING:
                columns[name], _buf = convert_string_column(col)
            else:
                raise NotImplementedError(f"Data type {col.dtype[0]} not handled yet")

            _buffers_chunks.append(_buf)

        dataframe.append(vaex.from_dict(columns))
        # chunk buffers are added to list of all buffers
        _buffers.append(_buffers_chunks)

    if df.num_chunks() == 1:
        _buffers = _buffers[0]

    df_new = vaex.concat(dataframe)
    df_new._buffers = _buffers
    return df_new


class _DtypeKind(enum.IntEnum):
    INT = 0
    UINT = 1
    FLOAT = 2
    BOOL = 20
    STRING = 21  # UTF-8
    DATETIME = 22
    CATEGORICAL = 23


def convert_column_to_ndarray(col: ColumnObject) -> pa.Array:
    """
    Convert an int, uint, float or bool column to an arrow array
    """
    if col.offset != 0:
        raise NotImplementedError("column.offset > 0 not handled yet")

    if col.describe_null[0] not in (0, 1, 3, 4):
        raise NotImplementedError("Null values represented as" "sentinel values not handled yet")

    _buffer, _dtype = col.get_buffers()["data"]
    x = buffer_to_ndarray(_buffer, _dtype)

    # If there are any missing data with mask, apply the mask to the data
    if col.describe_null[0] in (3, 4) and col.null_count > 0:
        mask_buffer, mask_dtype = col._get_validity_buffer()
        mask = buffer_to_ndarray(mask_buffer, mask_dtype)
        x = pa.array(x, mask=mask)
    else:
        x = pa.array(x)

    return x, _buffer


def buffer_to_ndarray(_buffer, _dtype) -> np.ndarray:
    # Handle the dtype
    kind = _dtype[0]
    bitwidth = _dtype[1]
    _k = _DtypeKind
    if _dtype[0] not in (_k.INT, _k.UINT, _k.FLOAT, _k.BOOL):
        raise RuntimeError("Not a boolean, integer or floating-point dtype")

    _ints = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}
    _uints = {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}
    _floats = {32: np.float32, 64: np.float64}
    _np_dtypes = {0: _ints, 1: _uints, 2: _floats, 20: {8: bool}}
    column_dtype = _np_dtypes[kind][bitwidth]

    # No DLPack, so need to construct a new ndarray from the data pointer
    # and size in the buffer plus the dtype on the column
    ctypes_type = np.ctypeslib.as_ctypes_type(column_dtype)
    data_pointer = ctypes.cast(_buffer.ptr, ctypes.POINTER(ctypes_type))

    # NOTE: `x` does not own its memory, so the caller of this function must
    #       either make a copy or hold on to a reference of the column or
    #       buffer! (not done yet, this is pretty awful ...)
    x = np.ctypeslib.as_array(data_pointer, shape=(_buffer.bufsize // (bitwidth // 8),))

    return x


def convert_categorical_column(col: ColumnObject) -> Tuple[pa.DictionaryArray, Any]:
    """
    Convert a categorical column to an arrow dictionary
    """
    catinfo = col.describe_categorical
    if not catinfo["is_dictionary"]:
        raise NotImplementedError("Non-dictionary categoricals not supported yet")
    assert catinfo["categories"] is not None  # sanity check

    if not col.describe_null[0] in (0, 2, 3, 4):
        raise NotImplementedError(
            "Only categorical columns with sentinel "
            "value and masks supported at the moment"
        )

    codes_buffer, codes_dtype = col.get_buffers()["data"]
    codes = buffer_to_ndarray(codes_buffer, codes_dtype)

    if col.describe_null[0] == 2:  # sentinel value
        sentinel = [col.describe_null[1]] * col.size
        mask = codes == sentinel
        indices = pa.array(codes, mask=mask)
    elif col.describe_null[0] in (3, 4) and col.null_count > 0:  # masked missing values
        mask_buffer, mask_dtype = col._get_validity_buffer()
        mask = buffer_to_ndarray(mask_buffer, mask_dtype)
        indices = pa.array(codes, mask=mask)
    else:
        indices = pa.array(codes)

    labels_buffer, labels_dtype = catinfo["categories"].get_buffers()["data"]
    if labels_dtype[0]  == _DtypeKind.STRING:
        labels, _ = convert_string_column(catinfo["categories"])
    else:
        labels = buffer_to_ndarray(labels_buffer, labels_dtype)

    values = pa.DictionaryArray.from_arrays(indices, labels)

    return values, codes_buffer


def convert_string_column(col: ColumnObject) -> Tuple[pa.Array, list]:
    """
    Convert a string column to a Arrow array.
    """
    # Missing
    if col.null_count > 0:
        if col.describe_null != (3, 0):
            raise TypeError("Only support arrow style mask data")

    # Retrieve the data buffers
    buffers = col.get_buffers()

    dbuffer, bdtype = buffers["data"]  # buffer containing the UTF-8 code units
    obuffer, odtype = buffers["offsets"]  #  buffer containing the index offsets demarcating the beginning and end of each string
    mbuffer, mdtype = buffers["validity"]  # buffer indicating the presence of missing values

    # Convert the buffers to NumPy arrays
    dt = (_DtypeKind.UINT, 8, None, None)  # note: in order to go from STRING to an equivalent ndarray, we claim that the buffer is uint8 (i.e., a byte array)
    dbuf = buffer_to_ndarray(dbuffer, dt)

    obuf = buffer_to_ndarray(obuffer, odtype)
    mbuf = buffer_to_ndarray(mbuffer, mdtype)

    # not sure what the best way to communicate the two types of strings is
    if obuffer._x.dtype == "int64":
        arrow_type = pa.large_utf8()
    elif obuffer._x.dtype == "int32":
        arrow_type = pa.utf8()
    else:
        raise TypeError(f"oops")
    length = obuf.size - 1

    buffers = [None, pa.py_buffer(obuf), pa.py_buffer(dbuf)]
    arrow_array = pa.Array.from_buffers(arrow_type, length, buffers)

    # Apply the mask
    if col.null_count > 0:
        arrow_array = pa.array(arrow_array.tolist(), mask=mbuf)

    return arrow_array, buffers


# Implementation of interchange protocol
# --------------------------------------


class _VaexBuffer:
    """
    Data in the buffer is guaranteed to be contiguous in memory.
    """

    def __init__(self, x: np.ndarray, allow_copy: bool = True) -> None:
        """
        Handle only regular columns (= numpy arrays) for now.
        """
        if not x.strides == (x.dtype.itemsize,):
            # The protocol does not support strided buffers, so a copy is
            # necessary. If that's not allowed, we need to raise an exception.
            if allow_copy:
                x = x.copy()
            else:
                raise RuntimeError("Exports cannot be zero-copy in the case " "of a non-contiguous buffer")

        # Store the numpy array in which the data resides as a private
        # attribute, so we can use it to retrieve the public attributes
        self._x = x

    @property
    def bufsize(self) -> int:
        """
        Buffer size in bytes
        """
        return self._x.size * self._x.dtype.itemsize

    @property
    def ptr(self) -> int:
        """
        Pointer to start of the buffer as an integer
        """
        return self._x.__array_interface__["data"][0]

    def __dlpack__(self):
        """
        DLPack not implemented in Vaex, so leave it out here
        """
        raise NotImplementedError("__dlpack__")

    def __dlpack_device__(self) -> Tuple[enum.IntEnum, int]:
        """
        Device type and device ID for where the data in the buffer resides.
        """

        class Device(enum.IntEnum):
            CPU = 1

        return (Device.CPU, None)

    def __repr__(self) -> str:
        return "VaexBuffer(" + str({"bufsize": self.bufsize, "ptr": self.ptr, "device": self.__dlpack_device__()[0].name}) + ")"


class _VaexColumn:
    """
    A column object, with only the methods and properties required by the
    interchange protocol defined.

    A column can contain one or more chunks. Each chunk can contain up to three
    buffers - a data buffer, a mask buffer (depending on null representation),
    and an offsets buffer (if variable-size binary; e.g., variable-length
    strings).

    Note: this Column object can only be produced by ``__dataframe__``, so
          doesn't need its own version or ``__column__`` protocol.

    """

    def __init__(self, column: vaex.expression.Expression, allow_copy: bool = True) -> None:
        """
        Note: assuming column is an expression.
        The values of an expression can be NumPy or Arrow.
        """
        if not isinstance(column, vaex.expression.Expression):
            raise NotImplementedError("Columns of type {} not handled " "yet".format(type(column)))

        # Store the column as a private attribute
        self._col = column
        self._allow_copy = allow_copy

    def size(self) -> int:
        """
        Size of the column, in elements.

        Corresponds to DataFrame.num_rows() if column is a single chunk;
        equal to size of this current chunk otherwise.

        Is a method rather than a property because it may cause a (potentially
        expensive) computation for some dataframe implementations.
        """
        return int(len(self._col.df))

    @property
    def offset(self) -> int:
        """
        Offset of first element. Always zero.
        """
        return 0

    @property
    def dtype(self) -> Tuple[enum.IntEnum, int, str, str]:
        """
        Dtype description as a tuple ``(kind, bit-width, format string, endianness)``

        Kind :

            - INT = 0
            - UINT = 1
            - FLOAT = 2
            - BOOL = 20
            - STRING = 21   # UTF-8
            - DATETIME = 22
            - CATEGORICAL = 23

        Bit-width : the number of bits as an integer
        Format string : data type description format string in Apache Arrow C
                        Data Interface format.
        Endianness : current only native endianness (``=``) is supported

        Notes:

            - Kind specifiers are aligned with DLPack where possible (hence the
              jump to 20, leave enough room for future extension)
            - Masks must be specified as boolean with either bit width 1 (for bit
              masks) or 8 (for byte masks).
            - Dtype width in bits was preferred over bytes
            - Endianness isn't too useful, but included now in case in the future
              we need to support non-native endianness
            - Went with Apache Arrow format strings over NumPy format strings
              because they're more complete from a dataframe perspective
            - Format strings are mostly useful for datetime specification, and
              for categoricals.
            - For categoricals, the format string describes the type of the
              categorical in the data buffer. In case of a separate encoding of
              the categorical (e.g. an integer to string mapping), this can
              be derived from ``self.describe_categorical``.
            - Data types not included: complex, Arrow-style null, binary, decimal,
              and nested (list, struct, map, union) dtypes.
        """
        dtype = self._col.dtype

        # Categorical
        # If it is internal, kind must be categorical (23)
        # If it is external (call from_dataframe), dtype must give type of the data
        if self._col.df.is_category(self._col):
            return (_DtypeKind.CATEGORICAL, 64, "u", "=")  # what should be the default??

        return self._dtype_from_vaexdtype(dtype)

    def _dtype_from_vaexdtype(self, dtype) -> Tuple[enum.IntEnum, int, str, str]:
        """
        See `self.dtype` for details
        """
        # Note: 'c' (complex) not handled yet (not in array spec v1).
        #       'b', 'B' (bytes), 'S', 'a', (old-style string) 'V' (void) not handled
        #       datetime, timedelta not implemented yet
        _k = _DtypeKind
        _np_kinds = {"i": _k.INT, "u": _k.UINT, "f": _k.FLOAT, "b": _k.BOOL, "U": _k.STRING, "M": _k.DATETIME, "m": _k.DATETIME}
        kind = _np_kinds.get(dtype.kind, None)

        if kind is None:
            # Check if it's a an Arrow dictonary
            if not isinstance(self._col.values, np.ndarray) and isinstance(self._col.values.type, pa.DictionaryType):
                kind = 23
            # Check if it's a string dtype (dtype.kind == 'O')
            elif dtype == "string":
                kind = 21
            else:
                raise ValueError(f"Data type {dtype} not supported by exchange protocol")

        if kind not in (_k.INT, _k.UINT, _k.FLOAT, _k.BOOL, _k.CATEGORICAL, _k.STRING):
            raise NotImplementedError(f"Data type {dtype} not handled yet")

        bitwidth = dtype.numpy.itemsize * 8
        if not isinstance(self._col.values, np.ndarray) and isinstance(self._col.values.type, pa.DictionaryType):
            format_str = self._col.index_values().dtype.numpy.str
        else:
            format_str = dtype.numpy.str
        endianness = dtype.byteorder if not kind == _k.CATEGORICAL else "="
        return (kind, bitwidth, format_str, endianness)

    @property
    def describe_categorical(self) -> Dict[str, Any]:
        """
        If the dtype is categorical, there are two options:
        - There are only values in the data buffer.
        - There is a separate non-categorical Column encoding categorical values.

        Raises TypeError if the dtype is not categorical

        Returns the dictionary with description on how to interpret the data buffer:
            - "is_ordered" : bool, whether the ordering of dictionary indices is
                             semantically meaningful.
            - "is_dictionary" : bool, whether a mapping of
                                categorical values to other objects exists
            - "categories" : Column representing the (implicit) mapping of indices to
                             category values (e.g. an array of cat1, cat2, ...).
                             None if not a dictionary-style categorical.

        TBD: are there any other in-memory representations that are needed?
        """
        if not self.dtype[0] == _DtypeKind.CATEGORICAL:
            raise TypeError(
                "describe_categorical only works on a column with "
                "categorical dtype!"
            )
        df = vaex.from_dict({"labels": self._col.df.category_labels(self._col)})
        labels = df["labels"]
        categories = _VaexColumn(labels)
        return {
            "is_ordered": False,
            "is_dictionary": True,
            "categories": categories,
        }

    @property
    def describe_null(self) -> Tuple[int, Any]:
        """
        Return the missing value (or "null") representation the column dtype
        uses, as a tuple ``(kind, value)``.

        Kind:

            - 0 : non-nullable
            - 1 : NaN/NaT
            - 2 : sentinel value
            - 3 : bit mask
            - 4 : byte mask

        Value : if kind is "sentinel value", the actual value.  If kind is a bit
        mask or a byte mask, the value (0 or 1) indicating a missing value. None
        otherwise.
        """
        _k = _DtypeKind
        kind = self.dtype[0]
        value = None
        if kind in (_k.INT, _k.UINT, _k.FLOAT, _k.BOOL, _k.STRING):
            if self._col.dtype.is_arrow:
                # arrow arrays always allow for null values
                # where 0 encodes a null/missing value
                null = 3
                value = 0
            elif self._col.is_masked:
                # masked arrays are always numpy.ma arrays
                null = 4
                value = 1
            else:
                # otherwise we have a normal numpy array
                null = 0
                value = None
        elif kind == _k.CATEGORICAL:
            if self._col.dtype.is_arrow:
                # arrow dictionary allows for null values
                # where 0 encodes a null/missing value
                null = 3
                value = 0
            else:
                # otherwise we have categorize() with a normal numpy array
                null = 0
                value = None
        else:
            raise NotImplementedError(f"Data type {self.dtype} not yet supported")

        return null, value

    @property
    def null_count(self) -> int:
        """
        Number of null elements. Should always be known.
        """
        # Whilst vaex-proper does not treat all NA values as null (i.e. NaNs),
        # the interchange protocol does, meaning we have to break away from vaex
        # semantics here and return a count of all NA values.
        # See https://github.com/vaexio/vaex/issues/2120
        return self._col.countna()

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Store specific metadata of the column.
        """
        return {}

    def num_chunks(self) -> int:
        """
        Return the number of chunks the column consists of.
        """
        if isinstance(self._col.values, pa.ChunkedArray):
            return self._col.values.num_chunks
        else:
            return 1

    def get_chunks(self, n_chunks: Optional[int] = None) -> Iterable["_VaexColumn"]:
        """
        Return an iterator yielding the chunks.

        See `DataFrame.get_chunks` for details on ``n_chunks``.
        """
        if n_chunks == None:
            size = self.size()
            n_chunks = self.num_chunks()
            i = self._col.df.evaluate_iterator(self._col, chunk_size=size // n_chunks)
            iterator = []
            for i1, i2, chunk in i:
                iterator.append(_VaexColumn(self._col[i1:i2]))
            return iterator
        elif self.num_chunks == 1:
            size = self.size()
            i = self._col.df.evaluate_iterator(self._col, chunk_size=size // n_chunks)
            iterator = []
            for i1, i2, chunk in i:
                iterator.append(_VaexColumn(self._col[i1:i2]))
            return iterator

        else:
            raise ValueError(f"Column {self._col.expression} is already chunked.")

    def get_buffers(self) -> Dict[str, Any]:
        """
        Return a dictionary containing the underlying buffers.

        The returned dictionary has the following contents:

            - "data": a two-element tuple whose first element is a buffer
                      containing the data and whose second element is the data
                      buffer's associated dtype.
            - "validity": a two-element tuple whose first element is a buffer
                          containing mask values indicating missing data and
                          whose second element is the mask value buffer's
                          associated dtype. None if the null representation is
                          not a bit or byte mask.
            - "offsets": a two-element tuple whose first element is a buffer
                         containing the offset values for variable-size binary
                         data (e.g., variable-length strings) and whose second
                         element is the offsets buffer's associated dtype. None
                         if the data buffer does not have an associated offsets
                         buffer.
        """
        buffers = {}
        buffers["data"] = self._get_data_buffer()
        try:
            buffers["validity"] = self._get_validity_buffer()
        except:
            buffers["validity"] = None

        try:
            buffers["offsets"] = self._get_offsets_buffer()
        except:
            buffers["offsets"] = None

        return buffers

    def _get_data_buffer(self) -> Tuple[_VaexBuffer, Any]:  # Any is for self.dtype tuple
        """
        Return the buffer containing the data and the buffer's associated dtype.
        """
        _k = _DtypeKind
        if self.dtype[0] in (_k.INT, _k.UINT, _k.FLOAT, _k.BOOL):
            # If arrow array is boolean .to_numpy changes values for some reason
            # For that reason data is transferred to numpy through .tolist
            if self.dtype[0] == _k.BOOL and isinstance(self._col.values, (pa.Array, pa.ChunkedArray)):
                buffer = _VaexBuffer(np.array(self._col.tolist(), dtype=bool))
            else:
                buffer = _VaexBuffer(self._col.to_numpy())
            dtype = self.dtype
        elif self.dtype[0] == _k.CATEGORICAL:
            # TODO: Use expression.codes (https://github.com/vaexio/vaex/pull/1503), when merged
            if isinstance(self._col.values, (pa.DictionaryArray)):
                buffer = _VaexBuffer(self._col.index_values().to_numpy())
                dtype = self._dtype_from_vaexdtype(self._col.index_values().dtype)
            else:
                codes = self._col.values
                indices = self._col
                offset = self._col.df.category_offset(self._col)
                if offset:
                    indices -= offset
                buffer = _VaexBuffer(indices.to_numpy())
                dtype = self._dtype_from_vaexdtype(self._col.dtype)
        elif self.dtype[0] == _k.STRING:
            bitmap_buffer, offsets, string_bytes = self._col.evaluate().buffers()

            if string_bytes is None:
                string_bytes = np.array([], dtype="uint8")
            else:
                string_bytes = np.frombuffer(string_bytes, "uint8", len(string_bytes))
            buffer = _VaexBuffer(string_bytes)

            # Define the dtype for the returned buffer
            dtype = (_k.STRING, 8, "u", "=")  # note: currently only support native endianness
        else:
            raise NotImplementedError(f"Data type {self._col.dtype} not handled yet")

        return buffer, dtype

    def _get_validity_buffer(self) -> Tuple[_VaexBuffer, Any]:
        """
        Return the buffer containing the mask values indicating missing data and
        the buffer's associated dtype.

        Raises RuntimeError if null representation is not a bit or byte mask.
        """
        null, invalid = self.describe_null

        _k = _DtypeKind
        if null == 3 or null == 4:  # arrow
            mask = self._col.ismissing()

            # if arrow use .tolist and then turn it into np.array
            # if numpy masked turn the mask into numpy
            data = np.array(mask.tolist()) if null == 3 else mask.to_numpy()

            buffer = _VaexBuffer(data)
            dtype = self._dtype_from_vaexdtype(mask.dtype)

            return buffer, dtype

        elif null == 0:
            msg = "This column is non-nullable so does not have a mask"
        elif null == 1:
            msg = "This column uses NaN as null so does not have a separate mask"
        else:
            raise NotImplementedError("See self.describe_null")

        raise RuntimeError(msg)

    def _get_offsets_buffer(self) -> Tuple[_VaexBuffer, Any]:
        """
        Return the buffer containing the offset values for variable-size binary
        data (e.g., variable-length strings) and the buffer's associated dtype.

        Raises RuntimeError if the data buffer does not have an associated
        offsets buffer.
        """
        _k = _DtypeKind
        if self.dtype[0] == _k.STRING:
            bitmap_buffer, offsets, string_bytes = self._col.evaluate().buffers()

            if self._col.evaluate().type == pyarrow.string():
                offsets = np.frombuffer(offsets, np.int32, len(offsets) // 4)
                dtype = (_k.INT, 32, "i", "=")
            else:
                offsets = np.frombuffer(offsets, np.int64, len(offsets) // 8)
                dtype = (_k.INT, 64, "l", "=")
            buffer = _VaexBuffer(offsets)

        else:
            raise RuntimeError("This column has a fixed-length dtype so does not have an offsets buffer")

        return buffer, dtype


class _VaexDataFrame:
    """
    A data frame class, with only the methods required by the interchange
    protocol defined.

    Instances of this (private) class are returned from
    ``vaex.dataframe.DataFrame.__dataframe__`` as objects with the methods and
    attributes defined on this class.
    """

    def __init__(self, df: vaex.dataframe.DataFrame, nan_as_null: bool = False, allow_copy: bool = True) -> None:
        """
        Constructor - an instance of this (private) class is returned from
        `vaex.dataframe.DataFrame.__dataframe__`.
        """
        self._df = df
        # ``nan_as_null`` is a keyword intended for the consumer to tell the
        # producer to overwrite null values in the data with ``NaN`` (or ``NaT``).
        # This currently has no effect; once support for nullable extension
        # dtypes is added, this value should be propagated to columns.
        self._nan_as_null = nan_as_null
        self._allow_copy = allow_copy

    def __dataframe__(self, nan_as_null : bool = False, allow_copy : bool = True) -> "_VaexDataFrame":
        return _VaexDataFrame(self._df, nan_as_null=nan_as_null, allow_copy=allow_copy)

    @property
    def metadata(self) -> Dict[str, Any]:
        return {}

    def num_columns(self) -> int:
        return len(self._df.columns)

    def num_rows(self) -> int:
        return len(self._df)

    def num_chunks(self) -> int:
        if isinstance(self.get_column(0)._col.values, pa.ChunkedArray):
            return self.get_column(0)._col.values.num_chunks
        else:
            return 1

    def column_names(self) -> Iterable[str]:
        return self._df.get_column_names()

    def get_column(self, i: int) -> _VaexColumn:
        return _VaexColumn(self._df[:, i], allow_copy=self._allow_copy)

    def get_column_by_name(self, name: str) -> _VaexColumn:
        return _VaexColumn(self._df[name], allow_copy=self._allow_copy)

    def get_columns(self) -> Iterable[_VaexColumn]:
        return [_VaexColumn(self._df[name], allow_copy=self._allow_copy) for name in self._df.columns]

    def select_columns(self, indices: Sequence[int]) -> "_VaexDataFrame":
        if not isinstance(indices, collections.abc.Sequence):
            raise ValueError("`indices` is not a sequence")

        names = []
        for i in indices:
            names.append(self._df[:, i].expression)

        return self.select_columns_by_name(names)

    def select_columns_by_name(self, names: Sequence[str]) -> "_VaexDataFrame":
        if not isinstance(names, collections.abc.Sequence):
            raise ValueError("`names` is not a sequence")

        return _VaexDataFrame(self._df[names], allow_copy=self._allow_copy)

    def get_chunks(self, n_chunks: Optional[int] = None) -> Iterable["_VaexDataFrame"]:
        """
        Return an iterator yielding the chunks.
        TODO: details on ``n_chunks``
        """
        n_chunks = n_chunks if n_chunks is not None else self.num_chunks()
        size = self.num_rows()
        chunk_size = (size + n_chunks - 1) // n_chunks
        column_names = self.column_names()
        i = self._df._future().evaluate_iterator(column_names, chunk_size=chunk_size)
        for i1, i2, chunk in i:
            yield _VaexDataFrame(vaex.from_items(*zip(column_names, chunk)))

