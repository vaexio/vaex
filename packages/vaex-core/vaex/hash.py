import os
import copyreg
import vaex


on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if not on_rtd:
    from .superutils import *
    from . import superutils

    ordered_set = tuple([cls for name, cls in vars(superutils).items() if name.startswith('ordered_set')])

    def pickle(x):
        return type(x), (x.extract(), x.count, x.nan_count, x.null_count)
    for cls in ordered_set:
        copyreg.pickle(cls, pickle)


def counter_type_from_dtype(dtype, transient=True):
    return vaex.utils.find_type_from_dtype(vaex.hash, 'counter_', dtype, transient=transient, support_non_native=False)


def ordered_set_type_from_dtype(dtype, transient=True):
    return vaex.utils.find_type_from_dtype(vaex.hash, 'ordered_set_', dtype, transient=transient, support_non_native=False)


def index_type_from_dtype(dtype, transient=True, prime_growth=False):
    name = 'index_hash_'
    if prime_growth:
        name += "_prime_growth"
    return vaex.utils.find_type_from_dtype(vaex.hash, name, dtype, transient=transient, support_non_native=False)
