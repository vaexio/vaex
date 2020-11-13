import vaex
import vaex.array_types
import numpy as np
import pyarrow as pa


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
    float64
    >>> type1 in [float, int]
    True

    """
    def __init__(self, internal):
        self.internal = internal

    def to_native(self):
        '''Removes non-native endianness'''
        return DataType(vaex.utils.to_native_dtype(self.internal))        

    def __eq__(self, other):
        if other == str:
            return self.is_string
        if other == float:
            return self.is_float
        if other == int:
            return self.is_integer
        if isinstance(other, str):
            tester = 'is_' + other
            if hasattr(self, tester):
                return getattr(self, tester)
        if not isinstance(other, DataType):
            other = DataType(other)
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
        '''

        internal = self.internal
        if isinstance(internal, np.dtype):
            if internal.byteorder == ">":
                internal = internal.newbyteorder()
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
        '''Alias of dtype.numpy.name'''
        return self.numpy.name

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
    def is_numeric(self):
        '''Tests if type is numerical (float, int)

        >>> DataType(np.dtype('f8')).is_numeric
        True
        >>> DataType(pa.float32()).is_numeric
        True
        '''
        return self.kind in 'fiu'

    @property
    def is_primitive(self):
        '''Tests if type is numerical (float, int, bool)

        >>> DataType(np.dtype('b')).is_primitive
        True
        >>> DataType(pa.bool_()).is_primitive
        True
        '''
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
        """
        if self.is_string:
            return False
        return vaex.array_types.to_numpy_type(self.internal).kind in 'M'

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
        '''
        return vaex.array_types.to_numpy_type(self.internal).kind in 'm'

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
        return vaex.array_types.to_numpy_type(self.internal).kind in 'f'

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
        return vaex.array_types.to_numpy_type(self.internal).kind in 'u'

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
        return vaex.array_types.to_numpy_type(self.internal).kind in 'i'

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
        return vaex.array_types.to_numpy_type(self.internal).kind in 'iu'

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