from typing import Sequence
import dask.base
import numpy as np
import pyarrow as pa

import vaex
import vaex.array_types

class DataType:
    """Wraps numpy and arrow data types in a uniform interface

    Examples:
    >>> import numpy as np
    >>> import pyarrow as pa
    >>> type1 = DataType(np.dtype('f8'))
    >>> type1
    float64
    >>> type2 = DataType(np.dtype('>f8'))
    >>> type2
    >f8
    >>> type1 in [float, int]
    True
    >>> type1 == type2
    False
    >>> type1 == pa.float64()
    True
    >>> type1 == pa.int64()
    False
    >>> DataType(np.dtype('f4'))
    float32
    >>> DataType(pa.float32())
    float32

    """
    def __init__(self, dtype):
        if isinstance(dtype, DataType):
            self.internal = dtype.internal
        else:
            if isinstance(dtype, pa.DataType):
                self.internal = dtype
            else:
                self.internal = np.dtype(dtype)

    def to_native(self):
        '''Removes non-native endianness'''
        return DataType(vaex.utils.to_native_dtype(self.internal))        

    def __hash__(self):
        return hash((self.__class__.__name__, self.internal))

    def __eq__(self, other):
        if self.is_encoded:
            return self.value_type == other
        if other is str:
            return self.is_string
        if other is float:
            return self.is_float
        if other is int:
            return self.is_integer
        if other is list:
            return self.is_list
        if other is dict:
            return self.is_struct
        if other is object:
            return self.is_object
        if isinstance(other, str):
            tester = 'is_' + other
            if hasattr(self, tester):
                return getattr(self, tester)
        if not isinstance(other, DataType):
            other = DataType(other)
        if other.is_primitive:
            if self.is_arrow:
                other = DataType(other.arrow)
            if self.is_numpy:
                other = DataType(other.numpy)
        return vaex.array_types.same_type(self.internal, other.internal)

    def __repr__(self):
        '''Standard representation for datatypes


        >>> dtype = DataType(pa.float64())
        >>> dtype.internal
        DataType(double)
        >>> dtype
        float64
        >>> DataType(pa.float32())
        float32
        >>> DataType(pa.dictionary(pa.int32(), pa.string()))
        dictionary<values=string, indices=int32, ordered=0>
        '''

        internal = self.internal
        if self.is_datetime:
            internal = self.numpy

        repr = str(internal)
        translate = {
            'datetime64': 'datetime64[ns]',  # for consistency, always add time unit
            'double': 'float64',  # this is what arrow does
            'float': 'float32',  # this is what arrow does
        }
        return translate.get(repr, repr)

    @property
    def name(self):
        '''Alias of dtype.numpy.name or str(dtype.arrow) if not primitive

        >>> DataType(np.dtype('f8')).name
        'float64'
        >>> DataType(np.dtype('>f4')).name
        'float32'
        >>> DataType(pa.float64()).name
        'float64'
        >>> DataType(pa.large_string()).name
        'large_string'
        >>> DataType(pa.string()).name
        'string'
        >>> DataType(pa.bool_()).name
        'bool'
        >>> DataType(np.dtype('?')).name
        'bool'
        >>> DataType(pa.dictionary(pa.int32(), pa.string())).name
        'dictionary<values=string, indices=int32, ordered=0>'
        '''
        return self.numpy.name if (self.is_primitive or self.is_datetime) else str(self.internal)

    @property
    def kind(self):
        return self.numpy.kind

    @property
    def numpy(self):
        '''Return the numpy equivalent type

        >>> DataType(pa.float64()).numpy == np.dtype('f8')
        True
        '''
        return vaex.array_types.to_numpy_type(self.internal)

    @property
    def arrow(self):
        '''Return the Apache Arrow equivalent type

        >>> DataType(np.dtype('f8')).arrow == pa.float64()
        True
        '''
        return vaex.array_types.to_arrow_type(self.internal)

    @property
    def is_arrow(self):
        '''Return True if it wraps an Arrow type

        >>> DataType(pa.string()).is_arrow
        True
        >>> DataType(pa.int32()).is_arrow
        True
        >>> DataType(np.dtype('f8')).is_arrow
        False
        '''
        return isinstance(self.internal, pa.DataType)

    @property
    def is_numpy(self):
        '''Return True if it wraps an NumPy dtype

        >>> DataType(np.dtype('f8')).is_numpy
        True
        >>> DataType(pa.string()).is_numpy
        False
        >>> DataType(pa.int32()).is_numpy
        False
        '''
        return isinstance(self.internal, np.dtype)

    @property
    def is_numeric(self):
        '''Tests if type is numerical (float, int)

        >>> DataType(np.dtype('f8')).is_numeric
        True
        >>> DataType(pa.float32()).is_numeric
        True
        >>> DataType(pa.large_string()).is_numeric
        False
        '''
        try:
            return self.kind in 'fiu'
        except NotImplementedError:
            return False

    @property
    def is_primitive(self):
        '''Tests if type is numerical (float, int, bool)

        >>> DataType(np.dtype('b')).is_primitive
        True
        >>> DataType(pa.bool_()).is_primitive
        True
        '''
        if self.is_arrow:
            return pa.types.is_primitive(self.internal)
        else:
            return self.kind in 'fiub'

    @property
    def is_datetime(self):
        """Tests if dtype is datetime (numpy) or timestamp (arrow)

        Date/Time:
        >>> date_type = DataType(np.dtype('datetime64'))
        >>> date_type
        datetime64[ns]
        >>> date_type == 'datetime'
        True

        Using Arrow:

        >>> date_type = DataType(pa.timestamp('ns'))
        >>> date_type
        datetime64[ns]
        >>> date_type == 'datetime'
        True
        >>> date_type = DataType(pa.large_string())
        >>> date_type.is_datetime
        False
        """
        if self.is_arrow:
            return pa.types.is_timestamp(self.internal)
        else:
            return self.kind in 'M'

    @property
    def is_timedelta(self):
        '''Test if timedelta

        >>> dtype = DataType(np.dtype('timedelta64'))
        >>> dtype
        timedelta64
        >>> dtype == 'timedelta'
        True
        >>> dtype.is_timedelta
        True
        >>> date_type = DataType(pa.large_string())
        >>> date_type.is_timedelta
        False
        '''
        if self.is_arrow:
            return isinstance(self.arrow, pa.DurationType)
        else:
            return self.kind in 'm'

    @property
    def is_temporal(self):
        '''Alias of (is_datetime or is_timedelta)'''
        return self.is_datetime or self.is_timedelta

    @property
    def is_float(self):
        '''Test if a float (float32 or float64)

        >>> dtype = DataType(np.dtype('float32'))
        >>> dtype
        float32
        >>> dtype == 'float'
        True
        >>> dtype == float
        True
        >>> dtype.is_float
        True

        Using Arrow:
        >>> DataType(pa.float32()) == float
        True
        '''
        return self.is_primitive and vaex.array_types.to_numpy_type(self.internal).kind in 'f'

    @property
    def is_unsigned(self):
        '''Test if an (unsigned) integer

        >>> dtype = DataType(np.dtype('uint32'))
        >>> dtype
        uint32
        >>> dtype == 'unsigned'
        True
        >>> dtype.is_unsigned
        True

        Using Arrow:
        >>> DataType(pa.uint32()).is_unsigned
        True
        '''
        return self.is_primitive and vaex.array_types.to_numpy_type(self.internal).kind in 'u'

    @property
    def is_signed(self):
        '''Test if a (signed) integer

        >>> dtype = DataType(np.dtype('int32'))
        >>> dtype
        int32
        >>> dtype == 'signed'
        True

        Using Arrow:
        >>> DataType(pa.int32()).is_signed
        True
        '''
        return self.is_primitive and vaex.array_types.to_numpy_type(self.internal).kind in 'i'

    @property
    def is_integer(self):
        '''Test if an (unsigned or signed) integer

        >>> DataType(np.dtype('uint32')) == 'integer'
        True
        >>> DataType(np.dtype('int8')) == int
        True
        >>> DataType(np.dtype('int16')).is_integer
        True

        Using Arrow:
        >>> DataType(pa.uint32()).is_integer
        True
        >>> DataType(pa.int16()) == int
        True

        '''
        return self.is_primitive and vaex.array_types.to_numpy_type(self.internal).kind in 'iu'

    @property
    def is_string(self):
        '''Test if an (arrow) string or large_string

        >>> DataType(pa.string()) == str
        True
        >>> DataType(pa.large_string()) == str
        True
        >>> DataType(pa.large_string()).is_string
        True
        >>> DataType(pa.large_string()) == 'string'
        True
        '''
        return vaex.array_types.is_string_type(self.internal)

    @property
    def is_list(self):
        '''Test if an (arrow) list or large_string

        >>> DataType(pa.list_(pa.string())) == list
        True
        >>> DataType(pa.large_list(pa.string())) == list
        True
        >>> DataType(pa.list_(pa.string())).is_list
        True
        >>> DataType(pa.list_(pa.string())) == 'list'
        True
        '''
        return self.is_arrow and (pa.types.is_list(self.internal) or pa.types.is_large_list(self.internal))

    @property
    def is_struct(self) -> bool:
        '''Test if an (arrow) struct

        >>> DataType(pa.struct([pa.field('a', pa.utf8())])) == dict
        True
        >>> DataType(pa.struct([pa.field('a', pa.utf8())])).is_struct
        True
        >>> DataType(pa.struct([pa.field('a', pa.utf8())])) == 'struct'
        True
        '''
        return self.is_arrow and pa.types.is_struct(self.internal)

    @property
    def is_object(self):
        '''Test if a NumPy dtype=object (avoid if possible)'''
        return self.is_numpy and self.internal == object

    @property
    def is_encoded(self):
        '''Test if an (arrow) dictionary type (encoded data)

        >>> DataType(pa.dictionary(pa.int32(), pa.string())) == str
        True
        >>> DataType(pa.dictionary(pa.int32(), pa.string())).is_encoded
        True
        '''
        return self.is_arrow and (pa.types.is_dictionary(self.internal))

    @property
    def value_type(self):
        '''Return the DataType of the list values or values of an encoded type

        >>> DataType(pa.list_(pa.string())).value_type
        string
        >>> DataType(pa.list_(pa.float64())).value_type
        float64
        >>> DataType(pa.dictionary(pa.int32(), pa.string())).value_type
        string
        '''
        if not (self.is_list or self.is_encoded):
            raise TypeError( f'{self} is not a list or encoded type')
        return DataType(self.internal.value_type)

    @property
    def index_type(self):
        '''Return the DataType of the index of an encoded type, or simple the type

        >>> DataType(pa.string()).index_type
        string
        >>> DataType(pa.dictionary(pa.int32(), pa.string())).index_type
        int32
        '''
        type = self.internal
        if self.is_encoded:
            type = self.internal.index_type
        return DataType(type)

    def upcast(self):
        '''Cast to the higest data type matching the type

        >>> DataType(np.dtype('uint32')).upcast()
        uint64
        >>> DataType(np.dtype('int8')).upcast()
        int64
        >>> DataType(np.dtype('float32')).upcast()
        float64

        Using Arrow
        >>> DataType(pa.float32()).upcast()
        float64
        '''
        return DataType(vaex.array_types.upcast(self.internal))

    @property
    def byteorder(self):
        return self.numpy.byteorder

    def create_array(self, values : Sequence):
        '''Create an array from a sequence with the same dtype

        If values is a list containing None, it will map to a masked array (numpy) or null values (arrow)

        >>> DataType(np.dtype('float32')).create_array([1., 2.5, None, np.nan])
        masked_array(data=[1.0, 2.5, --, nan],
                     mask=[False, False,  True, False],
               fill_value=1e+20)
        >>> DataType(pa.float32()).create_array([1., 2.5, None, np.nan])  # doctest:+ELLIPSIS
        <pyarrow.lib.FloatArray object at ...>
        [
          1,
          2.5,
          null,
          nan
        ]
        '''
        if self.is_arrow:
            if vaex.array_types.is_arrow_array(values):
                return values
            else:
                return pa.array(values, type=self.arrow)
        else:
            if isinstance(values, np.ndarray):
                return values.astype(self.internal, copy=False)
            mask = [k is None for k in values]
            if any(mask):
                values = [values[0] if k is None else k for k in values]
                return np.ma.array(values, mask=mask)
            else:
                return np.array(values)
            return np.asarray(values, dtype=self.numpy)

@dask.base.normalize_token.register(DataType)
def normalize_DataType(t):
    return type(t).__name__, t.internal
