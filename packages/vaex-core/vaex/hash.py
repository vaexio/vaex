import os
from .column import str_type


on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if not on_rtd:
    from .superutils import *
    from . import superutils
    ordered_set = tuple([cls for name, cls in vars(superutils).items() if name.startswith('ordered_set')])


def counter_type_from_dtype(dtype, transient=True):
    if dtype == str_type:
        if transient:
            postfix = 'string'
        else:
            postfix = 'string' # view not support atm
    else:
        postfix = str(dtype)
        if postfix == '>f8':
            postfix = 'float64'
    name = 'counter_' + postfix
    return globals()[name]

def ordered_set_type_from_dtype(dtype, transient=True):
    if dtype == str_type:
        if transient:
            postfix = 'string'
        else:
            postfix = 'string' #  not support atm
    else:
        postfix = str(dtype)
        if postfix == '>f8':
            postfix = 'float64'
    name = 'ordered_set_' + postfix
    return globals()[name]

# from numpy import *
# import IPython
# IPython.embed()