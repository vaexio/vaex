"""This module contains struct dtype related expression functionality.

"""

import functools

import pyarrow as pa
import vaex.expression
from vaex import register_function


def _assert_struct_dtype(func):
    """Decorator to ensure that struct functions are only applied to expressions containing
    struct dtype. Otherwise, provide helpful error message.

    """

    @functools.wraps(func)
    def wrapper(struct, *args, **kwargs):
        vaex.expression.StructOperations.assert_struct_dtype(struct)
        return func(struct, *args, **kwargs)

    return wrapper


def _check_valid_struct_fields(struct, fields):
    """Ensure that fields do exist for given struct and provide helpful error message otherwise.

    """

    valid_fields = {x.name for x in struct.type}
    non_existant_fields = {field for field in fields if field not in valid_fields}
    if non_existant_fields:
        raise ValueError(f"Invalid field names provided: {non_existant_fields}. "
                         f"Valid field names are '{valid_fields}'")


@register_function(scope="struct")
@_assert_struct_dtype
def struct_get(x, field):
    """Return a single field from a struct array. You may also use the shorthand notation `df.name[:, 'field']`.

    :param {str, int} field: A string or integer identifying a struct field.
    :returns: an expression containing a struct field.

    Example:

    >>> import vaex
    >>> import pyarrow as pa
    >>> array = pa.StructArray.from_arrays(arrays=[[1,2], ["a", "b"]], names=["col1", "col2"])
    >>> df = vaex.from_arrays(array=array)
    >>> df
    # 	array
    0	{'col1': 1, 'col2': 'a'}
    1	{'col1': 2, 'col2': 'b'}

    >>> df.array.struct.get("col1")
    Expression = struct_get(array, 'col1')
    Length: 2 dtype: int64 (expression)
    -----------------------------------
    0  1
    1  2

    >>> df.array[:, 'col1']
    Expression = struct_get(array, 'col1')
    Length: 2 dtype: int64 (expression)
    -----------------------------------
    0  1
    1  2

    """

    _check_valid_struct_fields(x, [field])
    return x.field(field)


@register_function(scope="struct")
@_assert_struct_dtype
def struct_project(x, fields):
    """Project one or more fields of a struct array to a new struct array. You may also use the shorthand notation
    `df.name[:, ['field1', 'field2']]`.

    :param list field: A list of strings or integers identifying one or more fields.
    :returns: an expression containing a struct array.

    Example:

    >>> import vaex
    >>> import pyarrow as pa
    >>> array = pa.StructArray.from_arrays(arrays=[[1,2], ["a", "b"], [3, 4]], names=["col1", "col2", "col3"])
    >>> df = vaex.from_arrays(array=array)
    >>> df
    # 	array
    0	{'col1': 1, 'col2': 'a', 'col3': 3}
    1	{'col1': 2, 'col2': 'b', 'col3': 4}

    >>> df.array.struct.project(["col3", "col1"])
    Expression = struct_project(array, ['col3', 'col1'])
    Length: 2 dtype: struct<col3: int64, col1: int64> (expression)
    --------------------------------------------------------------
    0  {'col3': 3, 'col1': 1}
    1  {'col3': 4, 'col1': 2}

    >>> df.array[:, ["col3", "col1"]]
    Expression = struct_project(array, ['col3', 'col1'])
    Length: 2 dtype: struct<col3: int64, col1: int64> (expression)
    --------------------------------------------------------------
    0  {'col3': 3, 'col1': 1}
    1  {'col3': 4, 'col1': 2}

    """

    _check_valid_struct_fields(x, fields)
    arrays = [x.field(field) for field in fields]
    return pa.StructArray.from_arrays(arrays=arrays, names=fields)