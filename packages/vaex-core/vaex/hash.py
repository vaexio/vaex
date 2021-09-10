import os
import copyreg
import vaex
import pyarrow as pa

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if not on_rtd:
    from .superutils import *
    from . import superutils

    ordered_set = tuple([cls for name, cls in vars(superutils).items() if name.startswith('ordered_set')])

    def pickle(x):
        keys = x.key_array()
        return type(x), (keys, x.null_value, x.nan_count, x.null_count)
    for cls in ordered_set:
        copyreg.pickle(cls, pickle)


    def create_set_string(keys, *other):
        keys = vaex.column.ColumnStringArrow.from_arrow(keys)
        return ordered_set_string(keys.string_sequence, *other)

    def pickle_set_string(x):
        keys = x.key_array()
        keys = pa.array(keys.to_numpy(), type=pa.large_utf8())
        return create_set_string, (keys, x.null_value, x.nan_count, x.null_count)
    copyreg.pickle(ordered_set_string, pickle_set_string)


def counter_type_from_dtype(dtype, transient=True):
    return vaex.utils.find_type_from_dtype(vaex.hash, 'counter_', dtype, transient=transient, support_non_native=False)


def ordered_set_type_from_dtype(dtype, transient=True):
    return vaex.utils.find_type_from_dtype(vaex.hash, 'ordered_set_', dtype, transient=transient, support_non_native=False)


def index_type_from_dtype(dtype, transient=True, prime_growth=False):
    name = 'index_hash_'
    if prime_growth:
        name += "_prime_growth"
    return vaex.utils.find_type_from_dtype(vaex.hash, name, dtype, transient=transient, support_non_native=False)
