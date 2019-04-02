from .superutils import *
from .column import str_type

def counter_type_from_dtype(dtype):
    if dtype == str_type:
        postfix = 'string'
    else:
        postfix = str(dtype)
        if postfix == '>f8':
            postfix = 'float64'
    name = 'counter_' + postfix
    return globals()[name]

# from numpy import *
# import IPython
# IPython.embed()