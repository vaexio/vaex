"""This module contains struct dtype related expression functionality.

"""

from typing import List

from vaex.utils import _ensure_list
from .expression import Expression
from .dataframe import DataFrame
import functools
import json

import pyarrow as pa
import vaex
from vaex import register_function



class DataFrameAccessorStruct:
    def __init__(self, df):
        self.df : DataFrame = df


    def flatten(self, column=None, recursive=True, join_char='_') -> DataFrame:
        """Returns a DataFrame where each struct column is turned into individual columns.

        Example:
        >>> import vaex
        >>> import pyarrow as pa
        >>> array = pa.StructArray.from_arrays(arrays=[[1,2], ["a", "b"], [3, 4]], names=["col1", "col2", "col3"])
        >>> df = vaex.from_arrays(array=array)
        >>> df
          #  array
          0  {'col1': 1, 'col2': 'a', 'col3': 3}
          1  {'col1': 2, 'col2': 'b', 'col3': 4}
        >>> df.struct.flatten()
          #    array_col1  array_col2      array_col3
          0             1  a                        3
          1             2  b                        4

        :param str or list[str] column: Column or list of columns to use (default is all).
        :param bool recursive: Keep expanding already expanded columns or not.
        """
        df : DataFrame = self.df.copy()
        filter_columns =  set(self.df.get_column_names() if column is None else _ensure_list(column))
        queue : List[str] = self.df.get_column_names()
        column_names = []  # re-order afterwards
        while queue:
            column = queue.pop(0)
            expression : Expression = df[column]
            if column in filter_columns and expression.data_type().is_struct:
                df._hide_column(column)
                # loop in reverse, so the first items ends up at the start of the queue
                for name, projected_expression in reversed(expression.struct.items()):
                    projected_name = f'{column}{join_char}{name}'
                    df[projected_name] = projected_expression
                    queue.insert(0, projected_name)
                    if recursive:
                        filter_columns.add(projected_name)
            else:
                column_names.append(column)
        return df[column_names]

def assert_struct_dtype(struct):
    """Helper function to ensure that struct operations are only applied to struct arrays."""

    try:
        dtype = struct.type  # arrow type
    except AttributeError:
        dtype = struct.dtype  # numpy type

    dtype = vaex.datatype.DataType(dtype)
    if not dtype.is_struct:
        raise TypeError(
            f"Struct functions needs to be applied on struct dtype, got '{dtype}' instead.")


def format_struct_item_vaex_style(struct_item):
    """Provides consistent vaex formatting output for struct items
    handling structs with duplicated field labels for which conversion
    methods (e.g. `as_py()`) are currently (4.0.1) not well supported
    by pyarrow.

    """

    if is_struct_dtype_with_duplicated_field_labels(struct_item.type):
        return format_duplicated_struct_item_vaex_style(struct_item)
    else:
        return struct_item.as_py()


def format_duplicated_struct_item_vaex_style(struct_item):
    """Provides consistent vaex formatting output for struct item with
    duplicated labels.

    """

    mapping = {idx: dtype.name for idx, dtype in enumerate(struct_item.type)}
    values = [f"'{label}': {struct_item[idx].as_py()}"
              for idx, label in mapping.items()]

    return f"{{{', '.join(values)}}}"


def is_struct_dtype_with_duplicated_field_labels(dtype):
    """Check if struct item has duplicated field labels.

    """

    labels = {field.name for field in dtype}
    return len(labels) < dtype.num_fields


def _get_struct_field_label(dtype, identifier):
    """Return the string field label for given field identifier
    which can either be an integer for position based access or
    a string label directly.

    """

    if isinstance(identifier, str):
        return identifier

    return dtype[identifier].name


def assert_struct_dtype_argument(func):
    """Decorator to ensure that struct functions are only applied to expressions containing
    struct dtype. Otherwise, provide helpful error message.

    """

    @functools.wraps(func)
    def wrapper(struct, *args, **kwargs):
        assert_struct_dtype(struct)
        return func(struct, *args, **kwargs)

    return wrapper


def _check_valid_struct_fields(struct, fields):
    """Ensure that fields do exist for given struct and provide helpful error message otherwise.

    """

    # check for existing/valid fields
    valid_field_labels = {x.name for x in struct.type}
    valid_field_indices = set(range(struct.type.num_fields))
    valid_field_lookups = valid_field_labels.union(valid_field_indices)

    non_existant_fields = {field for field in fields
                           if field not in valid_field_lookups}
    if non_existant_fields:
        raise LookupError(
            f"Invalid field lookup provided: {non_existant_fields}. "
            f"Valid field lookups are '{valid_field_lookups}'")

    # check for duplicated field names and provide helpful error message
    labels = [field.name for field in struct.type]
    duplis = {x for x in labels if labels.count(x) > 1}
    duplicated_label_lookup = duplis.intersection(fields)
    if duplicated_label_lookup:
        raise LookupError(f"Invalid field lookup due to duplicated field "
                          f"labels '{duplicated_label_lookup}'. Please use "
                          f"index position based lookup for fields with "
                          f"duplicated labels to uniquely identify relevant "
                          f"field. To get index positions for field labels, "
                          f"please use `{{idx: key for idx, key in enumerate(df.array.struct)}}`.")


@register_function(scope="struct")
@assert_struct_dtype_argument
def struct_get(x, field):
    """Return a single field from a struct array. You may also use the shorthand notation `df.name[:, 'field']`.

    Please note, in case of duplicated field labels, a field can't be uniquely identified. Please
    use index position based access instead. To get corresponding field indices, please use
    `{{idx: key for idx, key in enumerate(df.array.struct)}}`.

    :param {str, int} field: A string (label) or integer (index position) identifying a struct field.
    :returns: an expression containing a struct field.

    Example:

    >>> import vaex
    >>> import pyarrow as pa
    >>> array = pa.StructArray.from_arrays(arrays=[[1,2], ["a", "b"]], names=["col1", "col2"])
    >>> df = vaex.from_arrays(array=array)
    >>> df
      #  array
      0  {'col1': 1, 'col2': 'a'}
      1  {'col1': 2, 'col2': 'b'}

    >>> df.array.struct.get("col1")
    Expression = struct_get(array, 'col1')
    Length: 2 dtype: int64 (expression)
    -----------------------------------
    0  1
    1  2

    >>> df.array.struct.get(0)
    Expression = struct_get(array, 0)
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
@assert_struct_dtype_argument
def struct_project(x, fields):
    """Project one or more fields of a struct array to a new struct array. You may also use the shorthand notation
    `df.name[:, ['field1', 'field2']]`.

    :param list field: A list of strings (label) or integers (index position) identifying one or more fields.
    :returns: an expression containing a struct array.

    Example:

    >>> import vaex
    >>> import pyarrow as pa
    >>> array = pa.StructArray.from_arrays(arrays=[[1,2], ["a", "b"], [3, 4]], names=["col1", "col2", "col3"])
    >>> df = vaex.from_arrays(array=array)
    >>> df
      #  array
      0  {'col1': 1, 'col2': 'a', 'col3': 3}
      1  {'col1': 2, 'col2': 'b', 'col3': 4}

    >>> df.array.struct.project(["col3", "col1"])
    Expression = struct_project(array, ['col3', 'col1'])
    Length: 2 dtype: struct<col3: int64, col1: int64> (expression)
    --------------------------------------------------------------
    0  {'col3': 3, 'col1': 1}
    1  {'col3': 4, 'col1': 2}

    >>> df.array.struct.project([2, 0])
    Expression = struct_project(array, [2, 0])
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
    fields = [_get_struct_field_label(x.type, field) for field in fields]

    return pa.StructArray.from_arrays(arrays=arrays, names=fields)
